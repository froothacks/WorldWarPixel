"""Societies: groups of agents, each with its own slice of the message board.

Every agent belongs to exactly one society and sees that society's threads each turn. It may
peek at another society's threads and move there. Humans influence a society only through its
*murmur*: a short unattributed line its members notice in their turn prompt.
"""

import json
import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

SOCIETY_COLORS = ("#ff4500", "#2450a4", "#00a368", "#ffd635", "#811e9f", "#51e9f4")
SOCIETY_NAMES = ("Red", "Blue", "Green", "Yellow", "Purple", "Cyan")
"""Starting names of runs logged before names were generated (no numbers, so models don't mix
them up with 0-based ids)."""
NAME_WORDS = (
    (
        "Amber",
        "Quiet",
        "Velvet",
        "Copper",
        "Hollow",
        "Bright",
        "Drifting",
        "Paper",
        "Silver",
        "Midnight",
        "Salt",
        "Little",
        "Wild",
        "Lantern",
        "Electric",
        "Sleepy",
        "Golden",
        "Restless",
        "Glass",
        "Wandering",
        "Neon",
        "Mossy",
        "Crooked",
        "Lucky",
    ),
    (
        "Orchard",
        "Tide",
        "Comets",
        "Foxes",
        "Harbor",
        "Moths",
        "Garden",
        "Signal",
        "Static",
        "Owls",
        "Ferns",
        "Kites",
        "Pebbles",
        "Sparrows",
        "Embers",
        "Lighthouse",
        "Meadow",
        "Otters",
        "Thunder",
        "Marbles",
        "Ravens",
        "Cactus",
        "Willows",
        "Lanterns",
    ),
)
"""Random starting names are an adjective and a noun from these lists."""
MURMUR_WINDOW = 900.0
"""Seconds a murmur stays in members' prompts before it fades."""
MIGRATION_WINDOW = 60.0
INVITE_WINDOW = 900.0
"""Seconds an invitation stays in the invitee's prompt."""
MOVES_WINDOW = 600.0
"""Seconds members keep hearing about who arrived or left."""
MAX_NAME = 40


class SocietyError(ValueError):
    pass


def random_names(count: int, rng: random.Random, taken: list[str] | None = None) -> list[str]:
    """``count`` distinct random two-word names, avoiding those in ``taken``."""
    seen = {n.lower() for n in taken or []}
    names: list[str] = []
    while len(names) < count:
        name = f"{rng.choice(NAME_WORDS[0])} {rng.choice(NAME_WORDS[1])}"
        if name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


def naming_messages(count: int, theme: str | None) -> list[dict[str, str]]:
    """Chat messages asking a model to name ``count`` freshly formed societies."""
    nudge = (
        f"What the names should be about: {theme.strip()}"
        if theme and theme.strip()
        else "Any style or subject you like; surprise us."
    )
    return [
        {
            "role": "system",
            "content": "You name groups of people in a shared online world where everyone "
            "paints on one big canvas and talks on message boards.",
        },
        {
            "role": "user",
            "content": f"Invent names for {count} groups that are just forming. Each name is "
            f"1-3 words, at most {MAX_NAME} characters, and clearly different from the others. "
            f"{nudge}\nReply with only a JSON array of {count} strings.",
        },
    ]


def parse_names(text: str, count: int) -> list[str]:
    """Up to ``count`` distinct names from a model reply (a JSON array, or one name per line)."""
    found = re.search(r"\[.*\]", text, re.S)
    raw: list[Any] = []
    if found:
        try:
            parsed = json.loads(found.group(0))
            raw = parsed if isinstance(parsed, list) else []
        except ValueError:
            raw = []
    if not raw:
        raw = text.splitlines()
    names: list[str] = []
    for item in raw:
        name = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", str(item)).strip(" \t\"'`,")
        name = " ".join(name.split())[:MAX_NAME].strip()
        if name and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    return names[:count]


@dataclass(slots=True)
class Society:
    id: int
    name: str
    color: str
    murmur: str | None = None
    murmur_at: float = 0.0
    transcript: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "color": self.color}


@dataclass(slots=True)
class Migration:
    ts: float
    agent_id: int
    src: int
    dst: int


@dataclass(slots=True)
class Invite:
    ts: float
    src: int
    society: int


@dataclass
class Societies:
    """Membership, names, murmurs and the migration history of one run."""

    count: int
    switch_cooldown: float
    names: list[str] | None = None
    societies: list[Society] = field(init=False)
    members: dict[int, int] = field(default_factory=dict)
    last_switch: dict[int, float] = field(default_factory=dict)
    migrations: deque[Migration] = field(default_factory=lambda: deque(maxlen=500))
    moves: int = 0
    """Total joins ever, so readers can tell what is new after the deque wraps."""
    invites: dict[int, dict[int, Invite]] = field(default_factory=dict)
    """Invitee -> inviter -> their latest invitation."""

    def __post_init__(self) -> None:
        self.societies = [
            Society(
                i,
                self.names[i] if self.names and i < len(self.names) else SOCIETY_NAMES[i % 6],
                SOCIETY_COLORS[i % len(SOCIETY_COLORS)],
            )
            for i in range(self.count)
        ]

    def assign(self, agent_ids: list[int]) -> None:
        """Deal agents round-robin (the caller shuffles them) so societies start even."""
        for i, aid in enumerate(agent_ids):
            self.members[aid] = i % self.count

    def get(self, society_id: int) -> Society:
        if not 0 <= society_id < self.count:
            raise SocietyError(f"society {society_id} does not exist (0..{self.count - 1})")
        return self.societies[society_id]

    def of(self, agent_id: int) -> int:
        return self.members[agent_id]

    def size(self, society_id: int) -> int:
        return sum(1 for s in self.members.values() if s == society_id)

    def join(self, agent_id: int, society_id: int, now: float) -> Migration:
        self.get(society_id)
        src = self.members[agent_id]
        if src == society_id:
            raise SocietyError(f"you are already in society {society_id}")
        wait = self.last_switch.get(agent_id, -1e18) + self.switch_cooldown - now
        if wait > 0:
            raise SocietyError(f"you moved recently; you can move again in {wait:.0f} s")
        self.members[agent_id] = society_id
        self.last_switch[agent_id] = now
        pending = self.invites.get(agent_id, {})
        for src in [k for k, i in pending.items() if i.society == society_id]:
            del pending[src]
        m = Migration(now, agent_id, src, society_id)
        self.migrations.append(m)
        self.moves += 1
        return m

    def invite(self, src: int, target: int, now: float) -> Invite:
        """`src` invites `target` into the society `src` is in now."""
        society = self.members[src]
        if self.members[target] == society:
            raise SocietyError("they are already in your society")
        inv = Invite(now, src, society)
        self.invites.setdefault(target, {})[src] = inv
        return inv

    def pending(self, target: int, now: float, limit: int = 3) -> list[Invite]:
        """Live invitations to a society `target` is not in, newest last."""
        here = self.members[target]
        live = [
            i
            for i in self.invites.get(target, {}).values()
            if now - i.ts < INVITE_WINDOW and i.society != here
        ]
        return sorted(live, key=lambda i: i.ts)[-limit:]

    def recent_moves(self, society_id: int, now: float, limit: int = 5) -> list[Migration]:
        """Arrivals and departures of this society in the last MOVES_WINDOW seconds."""
        moves = [
            m for m in self.migrations if now - m.ts < MOVES_WINDOW and society_id in (m.src, m.dst)
        ]
        return moves[-limit:]

    def rename(self, society_id: int, name: str) -> Society:
        name = " ".join(name.split())[:MAX_NAME]
        if not name:
            raise SocietyError("name must not be empty")
        society = self.get(society_id)
        society.name = name
        return society

    def set_murmur(self, society_id: int, murmur: str, transcript: str, now: float) -> Society:
        society = self.get(society_id)
        society.murmur, society.transcript, society.murmur_at = murmur, transcript, now
        return society

    def murmur(self, society_id: int, now: float) -> str | None:
        s = self.get(society_id)
        return s.murmur if s.murmur and now - s.murmur_at < MURMUR_WINDOW else None

    def flow(self, society_id: int, now: float) -> tuple[int, int]:
        """(joined, left) in the last MIGRATION_WINDOW seconds."""
        recent = [m for m in self.migrations if now - m.ts < MIGRATION_WINDOW]
        return (
            sum(1 for m in recent if m.dst == society_id),
            sum(1 for m in recent if m.src == society_id),
        )

    def summary(self, now: float) -> list[dict[str, Any]]:
        out = []
        for s in self.societies:
            joined, left = self.flow(s.id, now)
            out.append(
                {
                    **s.to_dict(),
                    "members": self.size(s.id),
                    "joined": joined,
                    "left": left,
                    "murmur": self.murmur(s.id, now),
                    "transcript": s.transcript,
                }
            )
        return out
