"""The respondent record every persona is built from, plus WVS Wave 7 code books.

A ``Respondent`` is one survey answer sheet reduced to the handful of variables we render. Values
keep the WVS numeric codes (see the WVS7 codebook, v6.0) with ``None`` for missing/not asked, so
the WVS importer and the synthetic generator produce the same shape and the renderer has one code
path. ``source`` says where a record came from: ``wvs7`` (real respondent) or ``synthetic``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

Source = Literal["wvs7", "synthetic"]


@dataclass(frozen=True, slots=True)
class Respondent:
    source: Source
    country_iso3: str  # country the survey was fielded in
    weight: float = 1.0  # W_WEIGHT: post-stratification weight
    age: int | None = None  # Q262
    sex: int | None = None  # Q260: 1 male, 2 female
    education: int | None = None  # Q275: ISCED 2011 level 0-8
    employment: int | None = None  # Q279
    occupation: int | None = None  # Q281
    income_decile: int | None = None  # Q288: 1-10
    social_class: int | None = None  # Q287: 1 upper .. 5 lower
    religion: int | None = None  # Q289: major denomination groups
    religious_person: int | None = None  # Q173: 1 religious, 2 not, 3 atheist
    attendance: int | None = None  # Q171: 1 more than weekly .. 7 never
    god_importance: int | None = None  # Q164: 1-10
    left_right: int | None = None  # Q240: 1 left .. 10 right
    trust: int | None = None  # Q57: 1 most people can be trusted, 2 need to be careful
    democracy_importance: int | None = None  # Q250: 1-10
    strong_leader: int | None = None  # Q235: 1 very good .. 4 very bad
    democratic_system: int | None = None  # Q238: 1 very good .. 4 very bad
    political_interest: int | None = None  # Q199: 1 very .. 4 not at all
    income_equality: int | None = None  # Q106: 1 incomes more equal .. 10 larger differences
    env_vs_growth: int | None = None  # Q111: 1 protect environment, 2 economic growth
    happiness: int | None = None  # Q46: 1 very happy .. 4 not at all
    life_satisfaction: int | None = None  # Q49: 1-10
    national_pride: int | None = None  # Q254: 1 very proud .. 4 not at all, 5 not a national
    marital: int | None = None  # Q273
    children: int | None = None  # Q274
    urban: int | None = None  # H_URBRURAL: 1 urban, 2 rural
    town_size: int | None = None  # G_TOWNSIZE2: 1 <5k .. 5 500k+
    ethnic_group: str = ""  # Q290 label, country-specific ("" if unknown)
    extra: dict[str, Any] = field(default_factory=dict)  # optional enrichment (hobbies, job)


INT_FIELDS = tuple(
    f.name
    for f in fields(Respondent)
    if f.name not in ("source", "country_iso3", "weight", "ethnic_group", "extra")
)
CSV_COLUMNS = ("source", "country_iso3", "weight", *INT_FIELDS, "ethnic_group")


def write_respondents(rows: list[Respondent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow(
                [r.source, r.country_iso3, f"{r.weight:g}"]
                + ["" if getattr(r, k) is None else getattr(r, k) for k in INT_FIELDS]
                + [r.ethnic_group]
            )


def read_respondents(path: Path) -> list[Respondent]:
    out: list[Respondent] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = row["source"]
            if source not in ("wvs7", "synthetic"):
                raise ValueError(f"unknown respondent source {source!r} in {path}")
            ints = {k: int(row[k]) for k in INT_FIELDS if row.get(k) not in (None, "")}
            out.append(
                Respondent(
                    source=source,  # type: ignore[arg-type]
                    country_iso3=row["country_iso3"],
                    weight=float(row.get("weight") or 1.0),
                    ethnic_group=row.get("ethnic_group", ""),
                    **ints,  # type: ignore[arg-type]
                )
            )
    return out


# -- WVS7 code books (codebook v6.0; negative codes are missing) ----------------------------

SEX = {1: "man", 2: "woman"}
EDUCATION = {
    0: "no formal schooling",
    1: "primary school",
    2: "lower secondary school",
    3: "upper secondary school",
    4: "a post-secondary vocational course",
    5: "a short-cycle college diploma",
    6: "a bachelor's degree",
    7: "a master's degree",
    8: "a doctorate",
}
EMPLOYMENT = {
    1: "work full time",
    2: "work part time",
    3: "are self-employed",
    4: "are retired",
    5: "run the household",
    6: "are a student",
    7: "are unemployed",
    8: "are not in regular work",
}
OCCUPATION = {
    0: "never had a job",
    1: "a professional or technical job",
    2: "a senior administrative job",
    3: "a clerical job",
    4: "a sales job",
    5: "a service job",
    6: "a skilled trade",
    7: "a semi-skilled job",
    8: "an unskilled job",
    9: "farm work",
    10: "running a farm",
}
OCCUPATION_EXAMPLES = {
    1: "teacher, nurse, engineer, accountant",
    2: "executive, senior official",
    3: "clerk, secretary, civil servant",
    4: "shopkeeper, sales agent",
    5: "waiter, barber, police officer, caretaker",
    6: "electrician, mechanic, seamstress",
    7: "driver, bricklayer, baker",
    8: "labourer, porter, cleaner",
}
SOCIAL_CLASS = {
    1: "upper class",
    2: "upper middle class",
    3: "lower middle class",
    4: "working class",
    5: "lower class",
}
RELIGION = {
    0: "no religion",
    1: "Roman Catholic",
    2: "Protestant",
    3: "Orthodox Christian",
    4: "Jewish",
    5: "Muslim",
    6: "Hindu",
    7: "Buddhist",
    8: "Evangelical or other Christian",
    9: "another religion",
}
RELIGIOUS_PERSON = {1: "a religious person", 2: "not a religious person", 3: "an atheist"}
ATTENDANCE = {
    1: "more than once a week",
    2: "once a week",
    3: "once a month",
    4: "only on holy days",
    5: "once a year",
    6: "less than once a year",
    7: "practically never",
}
TRUST = {1: "most people can be trusted", 2: "you need to be very careful with people"}
GOOD_BAD = {1: "very good", 2: "fairly good", 3: "fairly bad", 4: "very bad"}
POLITICAL_INTEREST = {
    1: "very interested in politics",
    2: "somewhat interested in politics",
    3: "not very interested in politics",
    4: "not at all interested in politics",
}
ENV_VS_GROWTH = {
    1: "protecting the environment over economic growth",
    2: "economic growth over protecting the environment",
}
HAPPINESS = {1: "very happy", 2: "quite happy", 3: "not very happy", 4: "not at all happy"}
NATIONAL_PRIDE = {
    1: "very proud of your nationality",
    2: "quite proud of your nationality",
    3: "not very proud of your nationality",
    4: "not at all proud of your nationality",
    5: "not a citizen of the country you live in",
}
MARITAL = {
    1: "married",
    2: "living with a partner",
    3: "divorced",
    4: "separated",
    5: "widowed",
    6: "single",
}
URBAN = {1: "urban", 2: "rural"}
TOWN_SIZE = {
    1: "a village of under 5,000 people",
    2: "a small town",
    3: "a town of 20,000-100,000 people",
    4: "a city of 100,000-500,000 people",
    5: "a big city of over 500,000 people",
}
