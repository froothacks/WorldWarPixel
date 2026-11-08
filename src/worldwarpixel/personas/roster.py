"""Real attendee rosters: one agent per person, persona built from their public profile.

A roster file (``data/<file>.json``) lists people with an affiliation (school or company; profile
info only, nobody is put on a team), a one-line headline, LinkedIn and other public links,
highlights, how confident the profile match is, a short playstyle and a ``personality`` read
(work style, leadership, conflict and canvas approach) inferred from public signals.
``persona_text`` turns that into the "You are ..." line the LLM policy uses; ``persona_card`` is
what the drawer shows.
Public hackathon project teams are background relationships, not assigned canvas alliances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

RosterId = Literal["hackathon"]
ROSTER_FILES: dict[str, str] = {"hackathon": "neo_hackathon.json"}


@dataclass(frozen=True, slots=True)
class HackathonProject:
    title: str
    url: str
    members: tuple[str, ...]
    team_notes: str = ""


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
    personality: dict[str, Any] = field(default_factory=dict)
    avatar: str | None = None
    projects: tuple[HackathonProject, ...] = ()


@dataclass(frozen=True, slots=True)
class Roster:
    id: str
    title: str
    event: str
    note: str
    people: tuple[Person, ...]
    projects: tuple[HackathonProject, ...] = ()


def data_path(roster: str) -> Path:
    return Path(str(resources.files("worldwarpixel") / "data" / ROSTER_FILES[roster]))


@lru_cache(maxsize=len(ROSTER_FILES))
def load_roster(roster: str) -> Roster:
    raw = json.loads(data_path(roster).read_text(encoding="utf-8"))
    projects = tuple(
        HackathonProject(
            title=p["title"],
            url=p["url"],
            members=tuple(p["members"]),
            team_notes=p.get("team_notes", ""),
        )
        for p in raw.get("projects", ())
    )
    people = tuple(
        Person(
            name=p["name"],
            affiliation=p["affiliation"],
            headline=p["headline"],
            avatar=p.get("avatar"),
            linkedin=p.get("linkedin"),
            links=tuple(p.get("links", ())),
            highlights=tuple(p.get("highlights", ())),
            confidence=p.get("confidence", "low"),
            playstyle=p.get("playstyle", ""),
            personality=dict(p.get("personality") or {}),
            projects=tuple(project for project in projects if p["name"] in project.members),
        )
        for p in raw["people"]
    )
    return Roster(raw["id"], raw["title"], raw["event"], raw.get("note", ""), people, projects)


def roster_summaries() -> list[dict[str, Any]]:
    return [
        {"id": key, "title": r.title, "event": r.event, "size": len(r.people)}
        for key in ROSTER_FILES
        for r in [load_roster(key)]
    ]


STYLE_LABELS = (
    ("leadership", "leadership:"),
    ("social_approach", "with others:"),
    ("conflict_style", "in conflicts:"),
    ("decision_style", "decisions:"),
)


def persona_text(person: Person, event: str) -> str:
    """Second-person persona: the policy prefixes it with "You are "."""
    text = f"{person.name}, {person.headline}."
    if person.highlights:
        text += " Known for: " + "; ".join(person.highlights) + "."
    text += f" You are at {event}, playing as yourself."
    for project in person.projects:
        teammates = [name for name in project.members if name != person.name]
        text += f" Your hackathon project: {project.title}."
        if teammates:
            text += " Publicly listed project teammates: " + ", ".join(teammates) + "."
        else:
            text += " No other project teammates are publicly listed."
        if project.team_notes:
            text += f" {project.team_notes}"
    if person.playstyle:
        text += f" {person.playstyle}"
    traits = person.personality
    if traits.get("summary"):
        text += f" How you work: {traits['summary']}"
    styles = [
        f"{label} {traits[key].replace('_', ' ')}"
        for key, label in STYLE_LABELS
        if traits.get(key) and traits[key] != "unknown"
    ]
    if styles:
        text += " Style: " + "; ".join(styles) + "."
    if traits.get("canvas_approach"):
        text += f" On a shared canvas you would likely: {traits['canvas_approach']}"
    return text


def persona_card(person: Person, event: str) -> dict[str, Any]:
    card: dict[str, Any] = {
        "source": "roster",
        "event": event,
        "affiliation": person.affiliation,
        "headline": person.headline,
        "avatar": person.avatar,
        "linkedin": person.linkedin,
        "links": list(person.links) or None,
        "highlights": list(person.highlights) or None,
        "confidence": person.confidence,
        "playstyle": person.playstyle or None,
        "personality": person.personality or None,
        "hackathon_projects": [
            {
                "title": project.title,
                "url": project.url,
                "teammates": [name for name in project.members if name != person.name],
                "team_notes": project.team_notes,
            }
            for project in person.projects
        ]
        or None,
    }
    return {k: v for k, v in card.items() if v is not None}
