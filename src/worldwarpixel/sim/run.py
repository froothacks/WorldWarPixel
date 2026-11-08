import asyncio
import contextlib
import hashlib
import random
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator

from worldwarpixel.personas.roster import load_roster
from worldwarpixel.sim.agents import AgentSpec, Observation, Policy, seed_agents, seed_roster
from worldwarpixel.sim.board import Board, BoardError
from worldwarpixel.sim.canvas import PALETTE, Canvas, PixelWrite
from worldwarpixel.sim.dispatch import DISPATCH_ID, DISPATCH_NAME, Dispatcher
from worldwarpixel.sim.events import REJECTED, EventStore
from worldwarpixel.sim.hub import Hub
from worldwarpixel.sim.narrator import Narrator, NarratorModel
from worldwarpixel.sim.societies import Societies, random_names
from worldwarpixel.sim.speech import Speech, SpeechSynth
from worldwarpixel.sim.tools import ToolError, ToolSet, TurnBudget, apply_writes
from worldwarpixel.sim.world import World

RunState = Literal["created", "running", "paused", "stopped"]

MIN_PIXEL_INTERVAL = 0.01
MAX_AGENTS = 10_000
"""Agent ids are uint16 in the owners map (65535 is reserved for god mode). 10k keeps the agent
list cheap to ship to every browser; actual throughput is bounded by max_concurrency (LLM) or
by one core for the random policy (~100 turns/s)."""

LIST_FIELDS = ("id", "name", "country", "color", "home_x", "home_y", "cooldown", "headline")


def _close_if_supported(value: object | None) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


class RunConfig(BaseModel):
    width: int = Field(default=512, ge=16, le=4096)
    height: int = Field(default=320, ge=16, le=4096)
    agents: int = Field(default=100, ge=1, le=MAX_AGENTS)
    cooldown: float = Field(
        default=0.5,
        gt=0,
        description="Seconds between the starts of an agent's turns (randomised by jitter)",
    )
    pixel_interval: float = Field(
        default=0.01,
        ge=MIN_PIXEL_INTERVAL,
        le=5,
        description=(
            "Seconds between an agent's successive pixels (at least 10 ms so batches animate in)"
        ),
    )
    jitter: float = Field(default=0.3, ge=0, le=1, description="Cooldown jitter fraction")
    pixels_per_turn: int = Field(default=200, ge=1, le=1024)
    posts_per_turn: int = Field(default=1, ge=0, le=10)
    dispatch_interval: float | None = Field(
        default=None,
        ge=0,
        le=3600,
        description="Seconds between World News dispatches (0 = off; None = off in free-for-all, "
        "else 30)",
    )
    post_cooldown_turns: int = Field(
        default=3, ge=0, le=100, description="Turns an agent sits out posting after it posts"
    )
    max_concurrency: int = Field(default=1000, ge=1, le=1000, description="Turns in flight at once")
    seed: int | None = None
    policy: Literal["llm", "llm-guided", "random"] = Field(
        default="llm",
        description="llm: free-for-all; llm-guided: adds event framing, examples and hints",
    )
    model: str = Field(
        default="gpt-6-luna",
        description="Model preset id from GET /models (e.g. gpt-6-luna, qwen, glm)",
    )

    @field_validator("model", mode="before")
    @classmethod
    def _default_model(cls, v: Any) -> Any:
        # Runs archived before model presets stored model=null.
        return "gpt-6-luna" if v is None else v

    roster: Literal["world", "hackathon"] = Field(
        default="world",
        description=(
            "Who plays: 'world' seats `agents` people per country by population; 'hackathon' "
            "is one agent per attendee of the bundled hackathon roster (ignores `agents`)"
        ),
    )
    narrator_interval: float = Field(
        default=60.0, ge=5, le=3600, description="Seconds between narrator updates"
    )
    societies: int = Field(
        default=0,
        ge=0,
        le=6,
        description="Split agents into this many societies, each with its own board (0 = off)",
    )
    society_theme: str | None = Field(
        default=None,
        max_length=300,
        description="Optional free-text nudge for what the generated society names are about",
    )
    society_names: list[str] | None = Field(
        default=None,
        max_length=6,
        description="Starting society names; random (or model-generated) when omitted",
    )
    narrator_canvas_image: bool = Field(
        default=True, description="Send the narrator a downsampled PNG of the canvas"
    )
    speak_narrator: bool = Field(
        default=True, description="Read narrator updates aloud on the speech panel"
    )


GOD_ID = 65535
"""Owner id for pixels and posts made by the human viewer in god mode."""
GOD_NAME = "The Hand of God"
ANNOUNCEMENT_WINDOW = 600.0  # seconds an announcement stays in every agent's observation
AGENT_VOTE_WINDOW = 600.0  # seconds before an agent may vote on the same thread again
MEMORY_TURNS = 3  # own past turns shown to an agent
CONTACTS_SHOWN = 5
MENTION = re.compile(r"@(\w+(?:-\w+)*)")
MAX_GOD_PIXELS = 50_000
PREFETCH_LEAD = 1.0
"""Start an agent's next model call when this many seconds of queued pixels remain, so the
next batch lands as the current one finishes draining ("premoving")."""


class Run:
    """One simulation: a canvas, a board and a set of agents each on its own cooldown loop."""

    def __init__(
        self,
        run_id: str,
        config: RunConfig,
        log: EventStore,
        policy: Policy,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        narrator_model: NarratorModel | None = None,
        agents: list[AgentSpec] | None = None,
        speech: SpeechSynth | None = None,
    ) -> None:
        self.id = run_id
        self.config = config
        self.log = log
        self.policy = policy
        self._clock = clock
        self._sleep = sleep
        self.state: RunState = "created"
        self.interrupted = False
        """Rebuilt from the log of a run a server shutdown interrupted; it is due to resume."""
        self.resume_paused = False
        self.created_at = clock()
        self._agent_votes: dict[tuple[int, int], float] = {}
        self.started_at: float | None = None
        self.canvas = Canvas(config.width, config.height)
        self.board = Board()
        self._announcements: list[dict[str, Any]] = []
        self.hub = Hub()
        self.rng = random.Random(config.seed)
        if agents is None and config.roster != "world":
            roster = load_roster(config.roster)
            agents = seed_roster(
                roster.people, roster.event, config.width, config.height, config.cooldown, self.rng
            )
            self.config = config = config.model_copy(update={"agents": len(agents)})
        self.agents = agents or seed_agents(
            config.agents,
            config.width,
            config.height,
            config.cooldown,
            self.rng,
            by_country=config.policy != "llm",
        )
        self.world = World(self.agents, self.board)
        self.societies: Societies | None = None
        if config.societies:
            names = [n for n in config.society_names or [] if n.strip()][: config.societies]
            rng = random.Random(f"societies-{config.seed}") if config.seed is not None else None
            names += random_names(config.societies - len(names), rng or random.Random(), names)
            self.config = config = config.model_copy(update={"society_names": names})
            self.societies = Societies(
                config.societies, switch_cooldown=3 * config.cooldown, names=names
            )
            self.societies.assign([a.id for a in self.agents])
        self.dispatcher = Dispatcher(
            self.board,
            self.world,
            self.canvas,
            self.agents,
            publish=lambda post: self._on_write(DISPATCH_ID, "post", post.to_dict()),
            interval=(
                config.dispatch_interval
                if config.dispatch_interval is not None
                else 0.0
                if config.policy == "llm"
                else 30.0
            ),
        )
        self._agent_turns: dict[int, int] = dict.fromkeys(self.by_id, 0)
        self._last_post_turn: dict[int, int] = {}
        self.status: dict[int, str] = dict.fromkeys(self.by_id, "idle")
        self.thoughts: dict[int, deque[dict[str, Any]]] = {
            a.id: deque(maxlen=20) for a in self.agents
        }
        self.contacts: dict[int, dict[int, float]] = {a.id: {} for a in self.agents}
        """Who each agent has recently talked with (shared threads, @mentions, invitations)."""
        self.stats = {"pixels": 0, "posts": 0, "turns": 0, "errors": 0}
        self._running = asyncio.Event()
        self._sem = asyncio.Semaphore(config.max_concurrency)
        self._tasks: list[asyncio.Task[None]] = []
        self.speech = Speech(
            self, speech, self._running, narrate=config.speak_narrator, clock=clock
        )
        self.narrator = Narrator(
            self,
            narrator_model if config.policy != "random" else None,
            self._running,
            interval=config.narrator_interval,
            include_image=config.narrator_canvas_image,
            clock=clock,
            sleep=sleep,
            on_update=self.speech.narrate,
        )

    @property
    def by_id(self) -> dict[int, AgentSpec]:
        return {a.id: a for a in self.agents}

    def find_agent(self, name: str) -> int | None:
        """Resolve a name as agents write it: full name, or a unique first name or prefix."""
        key = " ".join(name.strip().lstrip("@").split()).lower()
        if not key:
            return None
        names = [(a.id, a.name.lower()) for a in self.agents]
        for hits in (
            [i for i, n in names if n == key],
            [i for i, n in names if n.split()[0] == key],
            [i for i, n in names if n.startswith(key)],
        ):
            if len(hits) == 1:
                return hits[0]
        return None

    def meet(self, a: int, b: int) -> None:
        if a == b or a not in self.contacts or b not in self.contacts:
            return
        now = self._clock()
        for x, y in ((a, b), (b, a)):
            seen = self.contacts[x]
            seen[y] = now
            if len(seen) > 16:
                for old in sorted(seen, key=seen.__getitem__)[:-8]:
                    del seen[old]

    def agent_name(self, agent_id: int) -> str:
        if agent_id == DISPATCH_ID:
            return DISPATCH_NAME
        if agent_id == GOD_ID:
            return GOD_NAME
        agent = self.by_id.get(agent_id)
        return agent.name if agent else f"agent {agent_id}"

    # -- god mode (the human viewer intervening) ------------------------------------------

    def god_paint(self, pixels: list[tuple[int, int, int]], erase: bool = False) -> int:
        """Paint (or erase to blank, unowned) pixels as the human; returns how many landed."""
        batch = pixels[:MAX_GOD_PIXELS]
        bad = [c for _, _, c in batch if not 0 <= int(c) < len(PALETTE)]
        if bad:
            raise ValueError(f"colour {bad[0]} is not in 0..{len(PALETTE) - 1}")
        n = 0
        with self.log.batch():
            for x, y, color in batch:
                if not self.canvas.in_bounds(x, y):
                    continue
                prev = self.canvas.owner(x, y)
                c, owner = (0, 0) if erase else (int(color), GOD_ID)
                self.canvas.write(PixelWrite(x, y, c, owner))
                payload = {"x": x, "y": y, "color": c, "prev_owner": prev, "owner": owner}
                self._on_write(GOD_ID, "pixel", payload)
                n += 1
            if n:
                self._record("god.paint", GOD_ID, {"pixels": n, "erase": erase})
        return n

    def god_clear_canvas(self) -> None:
        self.canvas.clear()
        self._record("god.clear_canvas", GOD_ID, {})
        self.hub.resync()
        self.hub.publish("canvas.cleared")

    def god_clear_board(self) -> None:
        self.board.clear()
        self._agent_votes.clear()
        self.dispatcher.reset_threads()
        self._announcements.clear()
        self._record("god.clear_board", GOD_ID, {})
        self.hub.publish("board.cleared")

    def swap_model(self, policy: Policy, narrator_model: NarratorModel | None, model: str) -> None:
        """Switch the LLM preset mid-run; agents and the narrator pick it up on next use."""
        old_components = (self.policy, self.narrator.model)
        new_components = (policy, narrator_model)
        closed: set[int] = set()
        for component in old_components:
            if (
                component is not None
                and not any(component is new for new in new_components)
                and id(component) not in closed
            ):
                closed.add(id(component))
                _close_if_supported(component)
        self.policy = policy
        self.narrator.set_model(narrator_model)
        self.config = self.config.model_copy(update={"model": model})
        self._record("god.model_changed", GOD_ID, {"model": model})

    def god_announce(self, title: str, body: str) -> dict[str, Any]:
        """Post a thread as the human; agents see it on the board like any other thread."""
        post = self.board.post_thread(GOD_ID, title, body, self._clock())
        self._on_write(GOD_ID, "post", post.to_dict())
        self._announcements.append(
            {"thread_id": post.thread_id, "title": title, "body": body, "ts": post.created_at}
        )
        return post.to_dict()

    def vote_thread(self, thread_id: int, voter: str, value: int) -> int:
        """A viewer's up/down click on a thread; it shifts where the board ranks it."""
        ballot = hashlib.sha256(f"{self.id}:{voter}".encode()).hexdigest()[:16]
        votes = self.board.vote(thread_id, ballot, value)
        self._record(
            "board.vote",
            None,
            {"thread_id": thread_id, "voter": ballot, "value": value, "votes": votes},
        )
        self.hub.publish("thread.voted", thread_id=thread_id, votes=votes)
        return votes

    def agent_vote(self, agent_id: int, thread_id: int, up: bool) -> int:
        """An agent's vote counts like one viewer click, at most once per thread per window."""
        now = self._clock()
        last = self._agent_votes.get((agent_id, thread_id))
        if last is not None and now - last < AGENT_VOTE_WINDOW:
            raise ToolError(f"you already voted on thread {thread_id} recently")
        try:
            votes = self.board.vote(thread_id, f"agent:{agent_id}", 1 if up else -1)
        except BoardError as e:
            raise ToolError(str(e)) from e
        self._agent_votes[(agent_id, thread_id)] = now
        self._record(
            "board.vote",
            agent_id,
            {
                "thread_id": thread_id,
                "voter": f"agent:{agent_id}",
                "value": 1 if up else -1,
                "votes": votes,
            },
        )
        self.hub.publish("thread.voted", thread_id=thread_id, votes=votes)
        return votes

    def god_reply(self, thread_id: int, body: str) -> dict[str, Any]:
        """Reply in an existing thread as the human; pinned for agents like an announcement."""
        post = self.board.reply(GOD_ID, thread_id, body, self._clock())
        self._on_write(GOD_ID, "post", post.to_dict())
        root = self.board.thread(thread_id)[0]
        self._announcements.append(
            {
                "thread_id": thread_id,
                "title": root.title,
                "body": body,
                "ts": post.created_at,
                "reply": True,
            }
        )
        return post.to_dict()

    def recent_announcements(self, window: float = ANNOUNCEMENT_WINDOW) -> list[dict[str, Any]]:
        """God announcements from the last `window` seconds, newest last, for the observation."""
        cutoff = self._clock() - window
        self._announcements = [a for a in self._announcements if a["ts"] >= cutoff]
        return self._announcements[-3:]

    def _announcement_digest(self) -> list[dict[str, Any]]:
        """Recent announcements as pinned board entries, so every prompt version shows them."""
        out: list[dict[str, Any]] = []
        for a in self.recent_announcements():
            try:
                replies = len(self.board.thread(a["thread_id"])) - 1
            except (KeyError, IndexError):
                continue
            if a.get("reply"):
                title = (
                    f'📢 {GOD_NAME} replied in "{a["title"]}": {a["body"]} '
                    "(answer this in that thread before anything else)"
                )
            else:
                title = f"📢 ANNOUNCEMENT: {a['title']} — {a['body']}"
            out.append(
                {
                    "thread_id": a["thread_id"],
                    "title": title,
                    "author": GOD_NAME,
                    "replies": replies,
                    "latest": None,
                    "tag": "announcement",
                }
            )
        return out

    def _feed(
        self, agent: AgentSpec, now: float, society: int | None = None
    ) -> list[dict[str, Any]]:
        announcements = self._announcement_digest()
        dispatches = self.dispatcher.feed_entries(now)
        excluded = {cast(int, entry["thread_id"]) for entry in [*announcements, *dispatches]}
        digest = [
            entry
            for entry in self.board.digest(limit=40, name_of=self.agent_name, society=society)
            if cast(int, entry["thread_id"]) not in excluded and entry["author"] != DISPATCH_NAME
        ]
        first_name = agent.name.split()[0].casefold()
        mentions = []
        hot = []
        for order, entry in enumerate(digest):
            latest = entry.get("latest")
            latest_body = latest.get("body", "") if isinstance(latest, dict) else ""
            text = f"{entry.get('title') or ''} {latest_body}".casefold()
            if (
                agent.country.casefold() in text
                or f"@{first_name}" in text
                or (entry["author"] == agent.name and cast(int, entry["replies"]) > 0)
            ):
                mentions.append({**entry, "tag": "mention"})
            else:
                hot.append((order, entry))
        hot.sort(key=lambda item: (-cast(int, item[1]["replies"]), item[0]))
        return [*announcements, *dispatches, *mentions[:3], *[entry for _, entry in hot[:6]]]

    def god_set_pace(
        self,
        cooldown: float | None,
        pixels_per_turn: int | None,
        pixel_interval: float | None = None,
    ) -> None:
        """Change how fast agents act. Cooldown and px/turn apply from each agent's next turn;
        the pixel interval applies to the very next pixel of every queue."""
        if cooldown is not None:
            for agent in self.agents:
                agent.cooldown = cooldown
            self.config.cooldown = cooldown
        if pixels_per_turn is not None:
            self.config.pixels_per_turn = pixels_per_turn
        if pixel_interval is not None:
            self.config.pixel_interval = pixel_interval
        self._record(
            "god.pace",
            GOD_ID,
            {
                "cooldown": cooldown,
                "pixels_per_turn": pixels_per_turn,
                "pixel_interval": pixel_interval,
            },
        )
        self.hub.publish("run.state", state=self.state, stats=self.stats)

    # -- lifecycle -------------------------------------------------------------------------

    async def start(self, *, resumed: bool = False) -> None:
        """Start the agents; ``resumed`` continues a run rebuilt from the log after a restart."""
        if self.state != ("stopped" if resumed else "created"):
            raise RuntimeError(f"run is {self.state}")
        if not resumed:
            self.started_at = self._clock()
            roster = [a.to_dict() for a in self.agents]
            self._record(
                "run.created", None, {"config": self.config.model_dump(), "agents": roster}
            )
        self._set_state("running")
        self._running.set()
        self.hub.start()
        self._tasks = [asyncio.create_task(self._agent_loop(a)) for a in self.agents]
        self._tasks.append(asyncio.create_task(self.speech.run()))
        self.narrator.start()

    def pause(self) -> None:
        if self.state == "running":
            self._running.clear()
            self._set_state("paused")

    def resume(self) -> None:
        if self.state == "paused":
            self._running.set()
            self._set_state("running")

    async def stop(self) -> None:
        if self.state == "stopped":
            if self.interrupted:  # stopping it while it waits to resume keeps it stopped
                self.interrupted = False
                self._record("run.state", None, {"state": "stopped"})
            return
        await self._halt()
        self._set_state("stopped")
        await self.hub.stop()

    async def interrupt(self) -> None:
        """Stop for a server shutdown, logged as ``interrupted`` so the next server resumes it."""
        if self.state in ("stopped", "created"):
            return
        self._record("run.state", None, {"state": "interrupted", "was": self.state})
        await self._halt()
        self.state = "stopped"
        await self.hub.stop()

    async def _halt(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._tasks = []
        await self.narrator.stop()
        closed: set[int] = set()
        for component in (self.policy, self.narrator.model):
            if component is not None and id(component) not in closed:
                closed.add(id(component))
                _close_if_supported(component)
        self._running.set()

    def _set_state(self, state: RunState) -> None:
        self.state = state
        self._record("run.state", None, {"state": state})
        self.hub.publish("run.state", state=state, stats=self.stats)

    # -- agent loop ------------------------------------------------------------------------

    async def _agent_loop(self, agent: AgentSpec) -> None:
        queue: asyncio.Queue[PixelWrite] = asyncio.Queue()
        pending: dict[tuple[int, int], tuple[int, int]] = {}
        painter = asyncio.create_task(self._painter(agent, queue, pending))
        try:
            await self._sleep(self.rng.random() * agent.cooldown)
            last_turn = self.started_at or self._clock()
            while True:
                await self._running.wait()
                now = self._clock()
                await self._turn(agent, now - last_turn, queue, pending)
                last_turn = now
                j = self.config.jitter
                await self._sleep(agent.cooldown * (1 + self.rng.uniform(-j, j)))
                while queue.qsize() * self.config.pixel_interval > PREFETCH_LEAD:
                    await self._sleep(self.config.pixel_interval)
        finally:
            painter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await painter

    async def _painter(
        self,
        agent: AgentSpec,
        queue: asyncio.Queue[PixelWrite],
        pending: dict[tuple[int, int], tuple[int, int]],
    ) -> None:
        """Drip an agent's queued pixels onto the canvas one every `pixel_interval` seconds."""
        while True:
            w = await queue.get()
            await self._running.wait()
            apply_writes(self.canvas, [w], lambda kind, p: self._on_write(agent.id, kind, p))
            if pending.get((w.x, w.y)) == (w.color, w.agent_id):
                del pending[(w.x, w.y)]
            if queue.empty() and self.status.get(agent.id) == "painting":
                self._set_status(agent.id, "idle")
            # Sleep in short slices so a god-mode change to pixel_interval applies within 250 ms.
            t0 = time.monotonic()
            while (left := self.config.pixel_interval - (time.monotonic() - t0)) > 0:
                await self._sleep(min(left, 0.25))

    async def _turn(
        self,
        agent: AgentSpec,
        elapsed: float,
        queue: asyncio.Queue[PixelWrite] | None = None,
        pending: dict[tuple[int, int], tuple[int, int]] | None = None,
    ) -> None:
        self.dispatcher.maybe_tick(self._clock())
        turn = self._agent_turns[agent.id] + 1
        self._agent_turns[agent.id] = turn
        posts = self.config.posts_per_turn
        if turn <= self._last_post_turn.get(agent.id, -self.config.post_cooldown_turns - 1) + (
            self.config.post_cooldown_turns
        ):
            posts = 0
        budget = TurnBudget(self.config.pixels_per_turn, posts)
        initial_posts = budget.posts
        sink = lambda kind, payload: self._on_write(agent.id, kind, payload)  # noqa: E731

        def writer(writes: list[PixelWrite]) -> None:
            if queue is None or (self.config.pixel_interval <= 0 and queue.empty()):
                apply_writes(self.canvas, writes, sink)
                return
            for w in writes:
                queue.put_nowait(w)
                if pending is not None:
                    pending[(w.x, w.y)] = (w.color, w.agent_id)

        tools = ToolSet(
            self.canvas,
            self.board,
            agent.id,
            budget,
            sink,
            self._clock,
            writer,
            societies=_SocietyAccess(self, agent.id) if self.societies else None,
            vote=lambda thread_id, up: self.agent_vote(agent.id, thread_id, up),
        )
        society = self.societies.of(agent.id) if self.societies else None
        obs = Observation(
            elapsed=elapsed,
            budget=budget,
            canvas_width=self.canvas.width,
            canvas_height=self.canvas.height,
            board=self._feed(agent, self._clock(), society),
        )
        now = self._clock()
        obs.memory = self.recall(agent.id)
        if self.societies is not None and society is not None:
            obs.society = {
                **self.societies.get(society).to_dict(),
                "members": self.societies.size(society),
                "murmur": self.societies.murmur(society, now),
                **self._social_facts(agent.id, society, now),
            }
            obs.societies = [
                {**s.to_dict(), "members": self.societies.size(s.id)}
                for s in self.societies.societies
            ]
        self._set_status(agent.id, "thinking")
        async with self._sem:
            try:
                thought = await self.policy.act(agent, tools, obs)
            except ToolError as e:
                thought = f"{REJECTED}: {e}"
                self.stats["errors"] += 1
        if budget.posts < initial_posts:
            self._last_post_turn[agent.id] = turn
        self.stats["turns"] += 1
        self.narrator.observe_turn(agent.id)
        self.note_thought(agent.id, thought)
        self._set_status(agent.id, "idle" if queue is None or queue.empty() else "painting")

    def recall(self, agent_id: int, turns: int = MEMORY_TURNS) -> list[str]:
        """An agent's own recent thoughts, as shown back to it."""
        now = self._clock()
        return [
            f"{_ago(now - t['ts'])} ago: {_recall(t['thought'])}"
            for t in list(self.thoughts[agent_id])[-turns:]
        ]

    def note_thought(self, agent_id: int, thought: str) -> None:
        entry = {"ts": self._clock(), "thought": thought}
        self.thoughts[agent_id].append(entry)
        self._record("agent.thought", agent_id, entry)
        self.hub.publish("agent.thought", agent_id=agent_id, **entry)

    # -- societies -------------------------------------------------------------------------

    def _social_facts(self, agent_id: int, society: int, now: float) -> dict[str, list[str]]:
        """Plain facts about the agent's people: where its contacts are, who moved, invitations."""
        assert self.societies is not None
        socs = self.societies

        def where(s: int) -> str:
            return "here" if s == society else f'in {s} "{socs.get(s).name}"'

        seen = self.contacts.get(agent_id, {})
        recent = sorted(seen, key=seen.__getitem__, reverse=True)[:CONTACTS_SHOWN]
        moves = []
        for m in socs.recent_moves(society, now):
            if m.agent_id == agent_id:
                continue
            who = self.agent_name(m.agent_id)
            if m.dst == society:
                moves.append(f'{who} arrived from {m.src} "{socs.get(m.src).name}"')
            else:
                moves.append(f'{who} left for {m.dst} "{socs.get(m.dst).name}"')
        return {
            "contacts": [f"{self.agent_name(c)} ({where(socs.of(c))})" for c in recent],
            "moves": moves,
            "invites": [
                f'{self.agent_name(i.src)} invited you to {i.society} "{socs.get(i.society).name}"'
                for i in socs.pending(agent_id, now)
            ],
        }

    def society_summary(self) -> list[dict[str, Any]]:
        return self.societies.summary(self._clock()) if self.societies else []

    def society_threads(self, society_id: int, limit: int = 10) -> list[dict[str, object]]:
        return self.board.digest(limit=limit, name_of=self.agent_name, society=society_id)

    def set_murmur(self, society_id: int, murmur: str, transcript: str) -> dict[str, Any]:
        """A human voice reached this society; members notice `murmur` in their next turns."""
        if self.societies is None:
            raise ValueError("this run has no societies")
        s = self.societies.set_murmur(society_id, murmur, transcript, self._clock())
        payload = {"society": s.id, "murmur": murmur, "transcript": transcript}
        self._record("society.murmur", None, payload)
        self.hub.publish("society.murmur", **payload)
        return payload

    def _set_status(self, agent_id: int, status: str) -> None:
        self.status[agent_id] = status
        self.hub.publish("agent.status", agent_id=agent_id, status=status)

    def _on_write(self, agent_id: int, kind: str, payload: dict[str, Any]) -> None:
        self._record(kind, agent_id, payload)
        self.narrator.observe_write(agent_id, kind, payload)
        if kind == "pixel":
            self.stats["pixels"] += 1
            self.world.observe_pixel(
                agent_id,
                int(payload.get("prev_owner", 0)),
                int(payload["x"]),
                int(payload["y"]),
            )
            owner = int(payload.get("owner", agent_id))
            self.hub.push_pixel(payload["x"], payload["y"], payload["color"], owner)
        elif kind == "post":
            self.stats["posts"] += 1
            self.world.observe_post(payload)
            for p in self.board.thread(int(payload["thread_id"]))[-6:]:
                self.meet(agent_id, p.author_id)
            for word in MENTION.findall(str(payload.get("body", ""))):
                if (other := self.find_agent(word)) is not None:
                    self.meet(agent_id, other)
            self.hub.publish("post.created", post=payload, author=self.agent_name(agent_id))
        elif kind == "action":
            self.hub.publish("agent.action", agent_id=agent_id, **payload)

    def _record(self, kind: str, agent_id: int | None, payload: dict[str, Any]) -> None:
        self.log.append(self.id, self._clock(), kind, agent_id, payload)

    def record(self, kind: str, agent_id: int | None, payload: dict[str, Any]) -> None:
        """Append an event on behalf of a collaborator (the narrator)."""
        self._record(kind, agent_id, payload)

    # -- read model ------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        public_stats = getattr(self.policy, "public_stats", None)
        return {
            "id": self.id,
            "state": self.state,
            "config": self.config.model_dump(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "stats": dict(self.stats),
            "agents": len(self.agents),
            "threads": len(self.board.threads()),
            "clients": self.hub.client_count,
            "llm": public_stats() if public_stats else None,
            "narrator": dict(self.narrator.stats),
            "societies": self.society_summary(),
        }

    def agent_list(self) -> list[dict[str, Any]]:
        """Compact rows for the sidebar; persona text and card come from ``agent_detail``."""
        return [
            {
                **{k: getattr(a, k) for k in LIST_FIELDS},
                "society": self.societies.of(a.id) if self.societies else None,
                "affiliation": a.persona_card.get("affiliation", "") if a.headline else "",
                "avatar": a.persona_card.get("avatar"),
                "status": self.status[a.id],
                "last_thought": self.thoughts[a.id][-1] if self.thoughts[a.id] else None,
            }
            for a in self.agents
        ]

    def agent_detail(self, agent_id: int) -> dict[str, Any] | None:
        agent = self.by_id.get(agent_id)
        if agent is None:
            return None
        describe = getattr(self.policy, "describe", None)
        prompt = (
            describe(
                agent,
                self.canvas.width,
                self.canvas.height,
                self.config.pixels_per_turn,
                self.config.posts_per_turn,
            )
            if describe
            else None
        )
        return {
            **agent.to_dict(),
            "status": self.status[agent_id],
            "thoughts": list(self.thoughts[agent_id]),
            "system_prompt": prompt,
        }

    def snapshot(self) -> dict[str, Any]:
        """The JSON half of what a client needs on connect (the pixels go as a binary frame)."""
        return {
            "type": "snapshot",
            "run": self.summary(),
            "palette": list(PALETTE),
            "agents": self.agent_list(),
            "threads": self.board.listing(limit=50, name_of=self.agent_name),
            "narrator": self.narrator.state(),
        }


class _SocietyAccess:
    """`SocietyAccess` for one agent's turn, backed by the run."""

    def __init__(self, run: Run, agent_id: int) -> None:
        self._run = run
        self._agent_id = agent_id
        assert run.societies is not None
        self._societies = run.societies

    def current(self) -> int:
        return self._societies.of(self._agent_id)

    def peek(self, society_id: int) -> dict[str, Any]:
        s = self._societies.get(society_id)
        return {
            **s.to_dict(),
            "members": self._societies.size(society_id),
            "threads": [
                {k: d[k] for k in ("thread_id", "title", "author", "replies", "latest")}
                for d in self._run.society_threads(society_id, limit=6)
            ],
        }

    def join(self, society_id: int) -> dict[str, Any]:
        run = self._run
        m = self._societies.join(self._agent_id, society_id, run._clock())
        payload = {"agent": run.agent_name(m.agent_id), "from": m.src, "to": m.dst}
        run._record("society.join", m.agent_id, payload)
        run.hub.publish("society.joined", agent_id=m.agent_id, **payload)
        return {"society": m.dst, "name": self._societies.get(m.dst).name}

    def invite(self, name: str) -> dict[str, Any]:
        run = self._run
        target = run.find_agent(name)
        if target is None:
            raise ValueError(f"nobody here is called {name!r}")
        if target == self._agent_id:
            raise ValueError("you cannot invite yourself")
        inv = self._societies.invite(self._agent_id, target, run._clock())
        run.meet(self._agent_id, target)
        payload = {
            "agent": run.agent_name(self._agent_id),
            "target_id": target,
            "target": run.agent_name(target),
            "society": inv.society,
        }
        run._record("society.invite", self._agent_id, payload)
        run.hub.publish("society.invited", agent_id=self._agent_id, **payload)
        return {"invited": payload["target"], "society": inv.society}

    def rename(self, name: str) -> dict[str, Any]:
        run = self._run
        s = self._societies.rename(self.current(), name)
        payload = {"society": s.id, "name": s.name}
        run._record("society.renamed", self._agent_id, payload)
        run.hub.publish("society.renamed", agent_id=self._agent_id, **payload)
        return payload


def _ago(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"


def _recall(thought: str, limit: int = 200) -> str:
    """Shorten a remembered thought but keep its trailing "[what I did]" summary."""
    text, sep, actions = thought.rpartition(" [")
    if not (sep and actions.endswith("]")):
        return thought[:limit]
    return f"{text[:limit]}{sep}{actions}"
