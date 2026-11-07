from collections import Counter
from collections.abc import Callable
from typing import Any

from worldwarpixel.sim.agents import AgentSpec
from worldwarpixel.sim.board import Board, Post
from worldwarpixel.sim.canvas import Canvas
from worldwarpixel.sim.world import World

DISPATCH_ID = 65534
DISPATCH_NAME = "World News"
WAR_MIN_PX = 50
TALK_MIN = 3
MAX_PER_TICK = 4


class Dispatcher:
    def __init__(
        self,
        board: Board,
        world: World,
        canvas: Canvas,
        agents: list[AgentSpec],
        publish: Callable[[Post], None],
        interval: float = 30.0,
        window: float = 120.0,
    ) -> None:
        self.board = board
        self.world = world
        self.canvas = canvas
        self.agents = {agent.id: agent for agent in agents}
        self.publish = publish
        self.interval = interval
        self.window = window
        self._threads: dict[tuple[Any, ...], int] = {}
        self._thread_details: dict[int, tuple[str, list[str]]] = {}
        self._previous_raids = Counter(world.raids)
        self._previous_talk = Counter(world.talk)
        self._previous_leader = self._territory_leader()
        self._last_tick: float | None = None

    def _territory_leader(self) -> int | None:
        owned: Counter[int] = Counter()
        for owner, count in Counter(self.canvas.owners).items():
            country = self.world.country_of.get(owner)
            if country is not None:
                owned[country] += count
        if not owned:
            return None
        return min(owned, key=lambda country: (-owned[country], country))

    def _baseline(self, now: float) -> None:
        self._previous_raids = Counter(self.world.raids)
        self._previous_talk = Counter(self.world.talk)
        self._previous_leader = self._territory_leader()
        self._last_tick = now

    def maybe_tick(self, now: float) -> list[Post]:
        if self.interval <= 0:
            return []
        if self._last_tick is None:
            self._baseline(now)
            return []
        if now - self._last_tick < self.interval:
            return []
        return self.tick(now)

    def _post(
        self,
        key: tuple[Any, ...],
        title: str,
        body: str,
        now: float,
        kind: str,
        countries: list[str],
    ) -> Post:
        thread_id = self._threads.get(key)
        if thread_id is not None and self.board.thread(thread_id):
            post = self.board.reply(DISPATCH_ID, thread_id, body, now)
        else:
            post = self.board.post_thread(DISPATCH_ID, title, body, now)
            self._threads[key] = post.thread_id
            self._thread_details[post.thread_id] = (kind, countries)
        self.publish(post)
        return post

    def tick(self, now: float) -> list[Post]:
        secs = (
            max(0, int(now - self._last_tick))
            if self._last_tick is not None
            else max(0, int(self.interval))
        )
        raids = Counter(self.world.raids)
        talk = Counter(self.world.talk)
        delta_raids = Counter(
            {key: max(0, value - self._previous_raids[key]) for key, value in raids.items()}
        )
        delta_talk = Counter(
            {key: max(0, value - self._previous_talk[key]) for key, value in talk.items()}
        )
        self._previous_raids = raids
        self._previous_talk = talk
        self._last_tick = now

        events: list[tuple[tuple[Any, ...], str, str, str, list[str]]] = []
        pairs = {
            (min(a, b), max(a, b))
            for a, b in self._previous_raids.keys() | delta_raids.keys()
            if a != b
        }
        # Include all currently observed raid pairs, then rank their interval deltas.
        pairs |= {(min(a, b), max(a, b)) for a, b in self.world.raids if a != b}
        war_pairs = []
        for lo, hi in pairs:
            ab = delta_raids[(lo, hi)]
            ba = delta_raids[(hi, lo)]
            total = ab + ba
            if total >= WAR_MIN_PX:
                war_pairs.append((total, lo, hi, ab, ba))
        war_pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
        for _, lo, hi, ab, ba in war_pairs[:2]:
            country_a, country_b = self.world.countries[lo], self.world.countries[hi]
            near = self.world.raid_at.get((lo, hi))
            location = f", most recently near ({near[0]}, {near[1]})" if near is not None else ""
            events.append(
                (
                    ("war", lo, hi),
                    f"⚔️ {country_a} vs {country_b}",
                    f"{country_a} painted over {ab} px of {country_b} and {country_b} {ba} px of "
                    f"{country_a} in the last {secs} s{location}.",
                    "war",
                    [country_a, country_b],
                )
            )

        talk_pairs = []
        for (lo, hi), count in delta_talk.items():
            if count >= TALK_MIN and delta_raids[(lo, hi)] + delta_raids[(hi, lo)] < WAR_MIN_PX:
                talk_pairs.append((count, lo, hi))
        talk_pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
        if talk_pairs:
            count, lo, hi = talk_pairs[0]
            country_a, country_b = self.world.countries[lo], self.world.countries[hi]
            events.append(
                (
                    ("talk", lo, hi),
                    f"🤝 {country_a} and {country_b} are talking",
                    f"{count} replies and @mentions between {country_a} and {country_b} "
                    f"in the last {secs} s.",
                    "talk",
                    [country_a, country_b],
                )
            )

        leader = self._territory_leader()
        previous_leader = self._previous_leader
        if leader is not None and previous_leader is not None and leader != previous_leader:
            leader_name = self.world.countries[leader]
            previous_name = self.world.countries[previous_leader]
            pct = (
                100
                * sum(
                    count
                    for owner, count in Counter(self.canvas.owners).items()
                    if self.world.country_of.get(owner) == leader
                )
                / max(1, self.canvas.width * self.canvas.height)
            )
            events.append(
                (
                    ("territory",),
                    "👑 Territory race",
                    f"{leader_name} now holds the most of the canvas ({pct:.0f}%), overtaking "
                    f"{previous_name}.",
                    "territory",
                    [leader_name, previous_name],
                )
            )
        self._previous_leader = leader

        posts: list[Post] = []
        for key, title, body, kind, countries in events[:MAX_PER_TICK]:
            posts.append(self._post(key, title, body, now, kind, countries))
        return posts

    def feed_entries(self, now: float) -> list[dict[str, Any]]:
        entries: list[tuple[float, dict[str, Any]]] = []
        for thread_id, (kind, countries) in self._thread_details.items():
            thread = self.board.thread(thread_id)
            if not thread or self.board.last_activity(thread_id) < now - self.window:
                continue
            root = thread[0]
            latest = thread[-1]
            latest_author = (
                DISPATCH_NAME
                if latest.author_id == DISPATCH_ID
                else self.agents[latest.author_id].name
                if latest.author_id in self.agents
                else f"agent {latest.author_id}"
            )
            entries.append(
                (
                    self.board.last_activity(thread_id),
                    {
                        "thread_id": thread_id,
                        "title": root.title,
                        "author": DISPATCH_NAME,
                        "replies": self.board.reply_count(thread_id),
                        "latest": {"author": latest_author, "body": latest.body},
                        "tag": "dispatch",
                        "kind": kind,
                        "countries": countries,
                    },
                )
            )
        entries.sort(key=lambda entry: entry[0], reverse=True)
        return [entry for _, entry in entries[:3]]

    def reset_threads(self) -> None:
        self._threads.clear()
        self._thread_details.clear()
