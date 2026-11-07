"""Turn a respondent into a persona card (structured) and persona text (prompt opening).

The text is written as the respondent's OWN answers ("you place yourself at 4 on a 1 (left) to
10 (right) scale"), never as claims about what people from their country believe, and stays
around five sentences so it fits a small model's context.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from worldwarpixel.personas import schema as S
from worldwarpixel.personas.schema import Respondent

# Q290 labels that say nothing useful about the person.
GENERIC_ETHNIC = {"", "other", "others", "general", "not pertaining to indigenous groups"}
WORKING = {1, 2, 3}
RETIRED = 4


def _label(table: dict[int, str], code: int | None) -> str | None:
    return table.get(code) if code is not None else None


def persona_card(r: Respondent, country: str, survey_country: str) -> dict[str, Any]:
    """Structured persona; keys with unknown values are omitted."""
    extra = r.extra or {}
    hobbies = extra.get("hobbies")
    occupation = extra.get("occupation") or _label(S.OCCUPATION, r.occupation)
    if not extra.get("occupation") and r.occupation in S.OCCUPATION_EXAMPLES:
        occupation = f"{occupation} ({S.OCCUPATION_EXAMPLES[r.occupation]})"
    card: dict[str, Any] = {
        "source": r.source,
        "country": country,
        "survey_country": survey_country,
        "age": r.age,
        "sex": _label(S.SEX, r.sex),
        "ethnic_group": r.ethnic_group if r.ethnic_group.lower() not in GENERIC_ETHNIC else None,
        "settlement": _label(S.TOWN_SIZE, r.town_size) or _label(S.URBAN, r.urban),
        "marital_status": _label(S.MARITAL, r.marital),
        "children": r.children,
        "education": _label(S.EDUCATION, r.education),
        "employment": (_label(S.EMPLOYMENT, r.employment) or "").removeprefix("are ") or None,
        "occupation": occupation,
        "income_decile": r.income_decile,
        "social_class": _label(S.SOCIAL_CLASS, r.social_class),
        "religion": _label(S.RELIGION, r.religion),
        "religiosity": _label(S.RELIGIOUS_PERSON, r.religious_person),
        "attendance": _label(S.ATTENDANCE, r.attendance),
        "god_importance": r.god_importance,
        "political_interest": _label(S.POLITICAL_INTEREST, r.political_interest),
        "left_right": r.left_right,
        "trust": _label(S.TRUST, r.trust),
        "democracy_importance": r.democracy_importance,
        "democratic_system": _label(S.GOOD_BAD, r.democratic_system),
        "strong_leader": _label(S.GOOD_BAD, r.strong_leader),
        "income_equality": r.income_equality,
        "env_vs_growth": _label(S.ENV_VS_GROWTH, r.env_vs_growth),
        "happiness": _label(S.HAPPINESS, r.happiness),
        "life_satisfaction": r.life_satisfaction,
        "national_pride": _label(S.NATIONAL_PRIDE, r.national_pride),
        "hobbies": list(hobbies) if hobbies else None,
    }
    return {k: v for k, v in card.items() if v is not None}


def _join(parts: Sequence[str | None]) -> str:
    kept = [p for p in parts if p]
    if len(kept) <= 1:
        return kept[0] if kept else ""
    return ", ".join(kept[:-1]) + " and " + kept[-1]


def _sentence(lead: str, parts: Sequence[str | None]) -> str:
    body = _join(parts)
    return f"{lead} {body}." if body else ""


THE_COUNTRIES = frozenset(
    {
        "United States",
        "United Kingdom",
        "United Arab Emirates",
        "Netherlands",
        "Philippines",
        "DR Congo",
        "Czech Republic",
        "Dominican Republic",
        "Central African Republic",
        "Maldives",
        "Bahamas",
        "Gambia",
        "Comoros",
        "Marshall Islands",
        "Solomon Islands",
        "Republic of Congo",
        "Congo",
        "Isle of Man",
        "Cayman Islands",
        "Falkland Islands",
        "Faroe Islands",
        "Cook Islands",
        "Turks and Caicos Islands",
        "British Virgin Islands",
        "United States Virgin Islands",
        "Northern Mariana Islands",
        "Channel Islands",
        "Holy See",
        "Seychelles",
        "Vatican",
    }
)


def with_article(country: str) -> str:
    return f"the {country}" if country in THE_COUNTRIES else country


def persona_text(name: str, r: Respondent, country: str) -> str:
    """Persona opening, meant to follow "You are " in the system prompt. About five sentences."""
    card = persona_card(r, country, country)
    extra = r.extra or {}

    article = "an" if r.age in (11, 18) or 80 <= (r.age or 0) <= 89 else "a"
    who = f"{article} {r.age}-year-old {card.get('sex', 'person')}"
    intro = f"{name}, {who} from {with_article(country)}"
    if card.get("ethnic_group"):
        intro += f" who identifies as {card['ethnic_group']}"
    if card.get("settlement"):
        intro += f", living in {card['settlement']}"
    intro += "."

    work: list[str | None] = []
    if r.education is not None:
        work.append(("have " if r.education == 0 else "finished ") + S.EDUCATION[r.education])
    job = extra.get("occupation") or _label(S.OCCUPATION, r.occupation)
    if r.employment in WORKING and job and r.occupation != 0:
        work.append(
            f"{S.EMPLOYMENT[r.employment]} {'as a' if extra.get('occupation') else 'in'} {job}"
        )
    elif r.employment == RETIRED and job and r.occupation != 0:
        work.append(f"are retired after {job}")
    elif card.get("employment"):
        work.append(card["employment"])
    if r.income_decile:
        step = f"your household is on step {r.income_decile} of 10 on the income ladder"
        if card.get("social_class"):
            step += f" ({card['social_class']}, as you see it)"
        work.append(step)

    faith: list[str | None] = []
    if r.religion == 0:
        faith.append("belong to no religion")
    elif card.get("religion"):
        faith.append(f"are {card['religion']}")
    if card.get("religiosity"):
        faith.append(f"call yourself {card['religiosity']}")
    if card.get("attendance") and r.religion != 0:
        faith.append(f"attend services {card['attendance']}")
    if r.god_importance:
        faith.append(f"God matters {r.god_importance}/10 to you")

    politics: list[str | None] = []
    if card.get("political_interest"):
        politics.append(f"are {card['political_interest']}")
    if r.left_right:
        politics.append(f"place yourself at {r.left_right} on a 1 (left) to 10 (right) scale")
    if card.get("trust"):
        politics.append(f"say {card['trust']}")
    if r.democracy_importance:
        politics.append(f"rate living in a democracy {r.democracy_importance}/10 in importance")
    if card.get("strong_leader"):
        politics.append(f"call a strong leader who ignores parliament {card['strong_leader']}")

    life: list[str | None] = [
        f"are {card['happiness']}" if card.get("happiness") else None,
        card.get("national_pride"),
    ]
    if card.get("marital_status"):
        status = card["marital_status"]
        if r.children:
            status += f" with {r.children} child{'ren' if r.children != 1 else ''}"
        elif r.children == 0:
            status += " with no children"
        life.append(status)

    hobbies = card.get("hobbies")
    fun = (
        f"Outside work you enjoy {_join([h[0].lower() + h[1:] for h in hobbies[:3]])}."
        if hobbies
        else ""
    )

    return " ".join(
        s
        for s in (
            intro,
            _sentence("You", work),
            _sentence("You", faith),
            _sentence("You", politics),
            _sentence("You", life),
            fun,
        )
        if s
    )
