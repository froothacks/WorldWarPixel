"""Country-level read model: who talks to whom on the board and who raids whose pixels."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from worldwarpixel.sim.agents import AgentSpec
from worldwarpixel.sim.board import MENTION_RE, Board


class World:
    """Aggregates per-agent writes into country x country relations.

    ``talk`` counts replies and @mentions between agents of different countries (undirected);
    ``raids`` counts pixels one country painted over another's (directed attacker -> victim).
    Both are cumulative for the run, so clearing the canvas or board does not erase history.
    """

    def __init__(self, agents: Iterable[AgentSpec], board: Board) -> None:
        agents = list(agents)
        self.countries = sorted({a.country for a in agents})
        self._index = {c: i for i, c in enumerate(self.countries)}
        self.country_of = {a.id: self._index[a.country] for a in agents}
        self.agents_per = Counter(self.country_of.values())
        self._by_first: dict[str, list[AgentSpec]] = defaultdict(list)
        for a in agents:
            self._by_first[a.name.split()[0].lower()].append(a)
        self._board = board
        self.pixels_by: Counter[int] = Counter()
        self.posts_by: Counter[int] = Counter()
        self.talk: Counter[tuple[int, int]] = Counter()
        self.raids: Counter[tuple[int, int]] = Counter()
        self.agent_talk: Counter[tuple[int, int]] = Counter()

    # -- observe ---------------------------------------------------------------------------

    def observe_pixel(self, agent_id: int, prev_owner: int) -> None:
        me = self.country_of.get(agent_id)
        if me is None:
            return
        self.pixels_by[me] += 1
        them = self.country_of.get(prev_owner)
        if them is not None and them != me:
            self.raids[(me, them)] += 1

    def observe_post(self, post: dict[str, Any]) -> None:
        author = int(post["author_id"])
        me = self.country_of.get(author)
        if me is None:
            return
        self.posts_by[me] += 1
        targets: set[int] = set()
        parent_id = post.get("parent_id")
        thread_id = int(post["thread_id"])
        target_post = self._board.get(int(parent_id)) if parent_id is not None else None
        if target_post is None and thread_id != int(post["id"]):
            target_post = self._board.get(thread_id)
        if target_post is not None:
            targets.add(target_post.author_id)
        text = f"{post.get('title') or ''} {post.get('body', '')}"
        for handle in MENTION_RE.findall(text):
            hit = self._resolve(handle, thread_id)
            if hit is not None:
                targets.add(hit)
        targets.discard(author)
        for other in targets:
            them = self.country_of.get(other)
            if them is None:
                continue
            self.agent_talk[(min(author, other), max(author, other))] += 1
            if them != me:
                self.talk[(min(me, them), max(me, them))] += 1

    def _resolve(self, handle: str, thread_id: int) -> int | None:
        """``@Priya`` or ``@Priya_Sharma`` -> agent id; ambiguous first names prefer someone who
        already posted in the thread, otherwise a name shared by one country still counts."""
        first = handle.replace("_", " ").split()[0].lower()
        candidates = self._by_first.get(first, [])
        if len(candidates) == 1:
            return candidates[0].id
        if not candidates:
            return None
        full = handle.replace("_", " ").lower()
        exact = [a for a in candidates if a.name.lower() == full]
        if len(exact) == 1:
            return exact[0].id
        in_thread = {p.author_id for p in self._board.thread(thread_id)}
        present = [a for a in candidates if a.id in in_thread]
        if len(present) == 1:
            return present[0].id
        if len({a.country for a in candidates}) == 1:
            return candidates[0].id
        return None

    # -- read ------------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        nodes = [
            {
                "country": c,
                "agents": self.agents_per[i],
                "pixels": self.pixels_by[i],
                "posts": self.posts_by[i],
            }
            for i, c in enumerate(self.countries)
        ]
        pairs: dict[tuple[int, int], dict[str, Any]] = {}
        for (a, b), n in self.talk.items():
            pairs.setdefault((a, b), self._edge(a, b))["talk"] = n
        for (atk, vic), n in self.raids.items():
            key = (min(atk, vic), max(atk, vic))
            edge = pairs.setdefault(key, self._edge(*key))
            edge["raids_ab" if atk == key[0] else "raids_ba"] = n
        edges = sorted(
            pairs.values(), key=lambda e: e["talk"] + e["raids_ab"] + e["raids_ba"], reverse=True
        )
        return {"nodes": nodes, "edges": edges}

    def _edge(self, a: int, b: int) -> dict[str, Any]:
        return {
            "a": self.countries[a],
            "b": self.countries[b],
            "talk": 0,
            "raids_ab": 0,
            "raids_ba": 0,
        }
