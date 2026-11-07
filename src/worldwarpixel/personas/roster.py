"""Real attendee rosters: one agent per person, persona built from their public profile.

A roster file (``data/<file>.json``) lists people with an affiliation (school or company; profile
info only, nobody is put on a team), a one-line headline, LinkedIn and other public links,
highlights, how confident the profile match is, and a short playstyle. ``persona_text`` turns
that into the "You are ..." line the LLM policy uses; ``persona_card`` is what the drawer shows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

RosterId = Literal["hackathon"]
ROSTER_FILES: dict[str, str] = {"hackathon": "neo_hackathon.json"}


@dataclass(frozen=True, slots=True)
class Person:
    name: str
    affiliation: str
    headline: str
    linkedin: str | None = None
    links: tuple[str, ...] = ()
    highlights: tuple[str, ...] = ()
    confidence: str = "low"
    playstyle: str = ""


@dataclass(frozen=True, slots=True)
class Roster:
    id: str
    title: str
    event: str
    note: str
    people: tuple[Person, ...]


def data_path(roster: str) -> Path:
    return Path(str(resources.files("worldwarpixel") / "data" / ROSTER_FILES[roster]))


@lru_cache(maxsize=len(ROSTER_FILES))
def load_roster(roster: str) -> Roster:
    raw = json.loads(data_path(roster).read_text(encoding="utf-8"))
    people = tuple(
        Person(
            name=p["name"],
            affiliation=p["affiliation"],
            headline=p["headline"],
            linkedin=p.get("linkedin"),
            links=tuple(p.get("links", ())),
            highlights=tuple(p.get("highlights", ())),
            confidence=p.get("confidence", "low"),
            playstyle=p.get("playstyle", ""),
        )
        for p in raw["people"]
    )
    return Roster(raw["id"], raw["title"], raw["event"], raw.get("note", ""), people)


def roster_summaries() -> list[dict[str, Any]]:
    return [
        {"id": key, "title": r.title, "event": r.event, "size": len(r.people)}
        for key in ROSTER_FILES
        for r in [load_roster(key)]
    ]


def persona_text(person: Person, event: str) -> str:
    """Second-person persona: the policy prefixes it with "You are "."""
    text = f"{person.name}, {person.headline}."
    if person.highlights:
        text += " Known for: " + "; ".join(person.highlights) + "."
    text += f" You are at {event}, playing as yourself."
    if person.playstyle:
        text += f" {person.playstyle}"
    return text


def persona_card(person: Person, event: str) -> dict[str, Any]:
    card: dict[str, Any] = {
        "source": "roster",
        "event": event,
        "affiliation": person.affiliation,
        "headline": person.headline,
        "linkedin": person.linkedin,
        "links": list(person.links) or None,
        "highlights": list(person.highlights) or None,
        "confidence": person.confidence,
        "playstyle": person.playstyle or None,
    }
    return {k: v for k, v in card.items() if v is not None}
