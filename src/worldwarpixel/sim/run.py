import asyncio
import contextlib
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from worldwarpixel.sim.agents import AgentSpec, Observation, Policy, seed_agents
from worldwarpixel.sim.board import Board
from worldwarpixel.sim.canvas import PALETTE, Canvas, PixelWrite
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.hub import Hub
from worldwarpixel.sim.narrator import Narrator, NarratorModel
from worldwarpixel.sim.tools import ToolError, ToolSet, TurnBudget

RunState = Literal["created", "running", "paused", "stopped"]

MAX_AGENTS = 10_000
"""Agent ids are uint16 in the owners map (65535 is reserved for god mode). 10k keeps the agent
list cheap to ship to every browser; actual throughput is bounded by max_concurrency (LLM) or
by one core for the random policy (~100 turns/s)."""

LIST_FIELDS = ("id", "name", "country", "color", "home_x", "home_y", "cooldown")


class RunConfig(BaseModel):
    width: int = Field(default=512, ge=16, le=4096)
    height: int = Field(default=320, ge=16, le=4096)
    agents: int = Field(default=100, ge=1, le=MAX_AGENTS)
    cooldown: float = Field(default=10.0, gt=0, description="Seconds an agent sleeps between turns")
    jitter: float = Field(default=0.3, ge=0, le=1, description="Cooldown jitter fraction")
    pixels_per_turn: int = Field(default=40, ge=1, le=1024)
    posts_per_turn: int = Field(default=1, ge=0, le=10)
    max_concurrency: int = Field(default=10, ge=1, le=1000, description="Turns in flight at once")
    seed: int | None = None
    policy: Literal["llm", "random"] = "llm"
    narrator_interval: float = Field(
        default=60.0, ge=5, le=3600, description="Seconds between narrator updates"
    )
    narrator_canvas_image: bool = Field(
        default=True, description="Send the narrator a downsampled PNG of the canvas"
    )


GOD_ID = 65535
"""Owner id for pixels and posts made by the human viewer in god mode."""
GOD_NAME = "The Hand of God"
ANNOUNCEMENT_WINDOW = 600.0  # seconds an announcement stays in every agent's observation
MAX_GOD_PIXELS = 50_000


class Run:
    """One simulation: a canvas, a board and a set of agents each on its own cooldown loop."""

    def __init__(
        self,
        run_id: str,
        config: RunConfig,
        log: EventLog,
        policy: Policy,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        narrator_model: NarratorModel | None = None,
    ) -> None:
        self.id = run_id
        self.config = config
        self.log = log
        self.policy = policy
        self._clock = clock
        self._sleep = sleep
        self.state: RunState = "created"
        self.created_at = clock()
        self.started_at: float | None = None
        self.canvas = Canvas(config.width, config.height)
        self.board = Board()
        self._announcements: list[dict[str, Any]] = []
        self.hub = Hub()
        self.rng = random.Random(config.seed)
        self.agents = seed_agents(
            config.agents, config.width, config.height, config.cooldown, self.rng
        )
        self.status: dict[int, str] = dict.fromkeys(self.by_id, "idle")
        self.thoughts: dict[int, deque[dict[str, Any]]] = {
            a.id: deque(maxlen=20) for a in self.agents
        }
        self.stats = {"pixels": 0, "posts": 0, "turns": 0, "errors": 0}
        self._running = asyncio.Event()
        self._sem = asyncio.Semaphore(config.max_concurrency)
        self._tasks: list[asyncio.Task[None]] = []
        self.narrator = Narrator(
            self,
            narrator_model if config.policy == "llm" else None,
            self._running,
            interval=config.narrator_interval,
            include_image=config.narrator_canvas_image,
            clock=clock,
            sleep=sleep,
        )

    @property
    def by_id(self) -> dict[int, AgentSpec]:
        return {a.id: a for a in self.agents}

    def agent_name(self, agent_id: int) -> str:
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
        self._announcements.clear()
        self._record("god.clear_board", GOD_ID, {})
        self.hub.publish("board.cleared")

    def god_announce(self, title: str, body: str) -> dict[str, Any]:
        """Post a thread as the human; agents see it on the board like any other thread."""
        post = self.board.post_thread(GOD_ID, title, body, self._clock())
        self._on_write(GOD_ID, "post", post.to_dict())
        self._announcements.append(
            {"thread_id": post.thread_id, "title": title, "body": body, "ts": post.created_at}
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
            out.append(
                {
                    "thread_id": a["thread_id"],
                    "title": (
                        f"📢 ANNOUNCEMENT: {a['title']} — {a['body']} "
                        "(everyone sees this; react however your persona would)"
                    ),
                    "author": GOD_NAME,
                    "replies": replies,
                    "latest": None,
                }
            )
        return out

    def god_set_pace(self, cooldown: float | None, pixels_per_turn: int | None) -> None:
        """Change how fast agents act; takes effect from each agent's next turn."""
        if cooldown is not None:
            for agent in self.agents:
                agent.cooldown = cooldown
            self.config.cooldown = cooldown
        if pixels_per_turn is not None:
            self.config.pixels_per_turn = pixels_per_turn
        self._record("god.pace", GOD_ID, {"cooldown": cooldown, "pixels_per_turn": pixels_per_turn})
        self.hub.publish("run.state", state=self.state, stats=self.stats)

    # -- lifecycle -------------------------------------------------------------------------

    async def start(self) -> None:
        if self.state != "created":
            raise RuntimeError(f"run is {self.state}")
        self.started_at = self._clock()
        self._record("run.created", None, {"config": self.config.model_dump()})
        self._set_state("running")
        self._running.set()
        self.hub.start()
        self._tasks = [asyncio.create_task(self._agent_loop(a)) for a in self.agents]
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
            return
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._tasks = []
        await self.narrator.stop()
        self._running.set()
        self._set_state("stopped")
        await self.hub.stop()

    def _set_state(self, state: RunState) -> None:
        self.state = state
        self._record("run.state", None, {"state": state})
        self.hub.publish("run.state", state=state, stats=self.stats)

    # -- agent loop ------------------------------------------------------------------------

    async def _agent_loop(self, agent: AgentSpec) -> None:
        await self._sleep(self.rng.random() * agent.cooldown)
        last_turn = self.started_at or self._clock()
        while True:
            await self._running.wait()
            now = self._clock()
            await self._turn(agent, now - last_turn)
            last_turn = now
            j = self.config.jitter
            await self._sleep(agent.cooldown * (1 + self.rng.uniform(-j, j)))

    async def _turn(self, agent: AgentSpec, elapsed: float) -> None:
        budget = TurnBudget(self.config.pixels_per_turn, self.config.posts_per_turn)
        tools = ToolSet(
            self.canvas,
            self.board,
            agent.id,
            budget,
            lambda kind, payload: self._on_write(agent.id, kind, payload),
            self._clock,
        )
        obs = Observation(
            elapsed=elapsed,
            budget=budget,
            canvas_width=self.canvas.width,
            canvas_height=self.canvas.height,
            board=self._announcement_digest() + self.board.digest(name_of=self.agent_name),
        )
        self._set_status(agent.id, "thinking")
        async with self._sem:
            try:
                thought = await self.policy.act(agent, tools, obs)
            except ToolError as e:
                thought = f"Tool call rejected: {e}"
                self.stats["errors"] += 1
        self.stats["turns"] += 1
        self.narrator.observe_turn(agent.id)
        entry = {"ts": self._clock(), "thought": thought}
        self.thoughts[agent.id].append(entry)
        self._record("agent.thought", agent.id, entry)
        self.hub.publish("agent.thought", agent_id=agent.id, **entry)
        self._set_status(agent.id, "idle")

    def _set_status(self, agent_id: int, status: str) -> None:
        self.status[agent_id] = status
        self.hub.publish("agent.status", agent_id=agent_id, status=status)

    def _on_write(self, agent_id: int, kind: str, payload: dict[str, Any]) -> None:
        self._record(kind, agent_id, payload)
        self.narrator.observe_write(agent_id, kind, payload)
        if kind == "pixel":
            self.stats["pixels"] += 1
            owner = int(payload.get("owner", agent_id))
            self.hub.push_pixel(payload["x"], payload["y"], payload["color"], owner)
        elif kind == "post":
            self.stats["posts"] += 1
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
        }

    def agent_list(self) -> list[dict[str, Any]]:
        """Compact rows for the sidebar; persona text and card come from ``agent_detail``."""
        return [
            {
                **{k: getattr(a, k) for k in LIST_FIELDS},
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
