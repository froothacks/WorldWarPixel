"""Import the World Values Survey Wave 7 cross-national file into our compact respondent CSV.

WVS microdata is free after registration but may not be redistributed, so nothing here is
committed: download the file yourself, run ``uv run worldwarpixel-personas import-wvs <file>``
and the result lands in the git-ignored ``data/`` directory. See the README for the steps.

Input: ``WVS_Cross-National_Wave_7_csv_v6_0.csv`` (or any later CSV release; comma or semicolon
separated; headers may be bare variable names or "Q262 Age"-style). The per-country CSV files
on the same site work too. Variable numbers follow the WVS7 codebook v6.0.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from importlib import resources
from pathlib import Path
from typing import Any

from worldwarpixel.personas.schema import Respondent, write_respondents

# Respondent field -> WVS7 variable name.
VARIABLES: dict[str, str] = {
    "age": "Q262",
    "sex": "Q260",
    "education": "Q275",
    "employment": "Q279",
    "occupation": "Q281",
    "income_decile": "Q288",
    "social_class": "Q287",
    "religion": "Q289",
    "religious_person": "Q173",
    "attendance": "Q171",
    "god_importance": "Q164",
    "left_right": "Q240",
    "trust": "Q57",
    "democracy_importance": "Q250",
    "strong_leader": "Q235",
    "democratic_system": "Q238",
    "political_interest": "Q199",
    "income_equality": "Q106",
    "env_vs_growth": "Q111",
    "happiness": "Q46",
    "life_satisfaction": "Q49",
    "national_pride": "Q254",
    "marital": "Q273",
    "children": "Q274",
    "urban": "H_URBRURAL",
    "town_size": "G_TOWNSIZE2",
}
COUNTRY_VAR = "B_COUNTRY_ALPHA"
WEIGHT_VAR = "W_WEIGHT"
ETHNIC_VAR = "Q290"
# WVS7 fields Northern Ireland separately from Great Britain; we treat both as the UK.
COUNTRY_ALIASES = {"NIR": "GBR"}
LABELS_FILE = "wvs7_q290_labels.tsv"


def ethnic_labels() -> dict[int, str]:
    """Q290 code -> label, from the WVS7 codebook annex (country prefix like "IN: " stripped)."""
    path = resources.files("worldwarpixel") / "data" / LABELS_FILE
    out: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            label = row["label"].split(": ", 1)[-1].strip()
            out[int(row["code"])] = label
    return out


def _code(value: str | None) -> int | None:
    """WVS numeric code, with negative codes (don't know, no answer, not asked) as None."""
    if value is None or value.strip() == "":
        return None
    try:
        n = int(float(value))
    except ValueError:
        return None
    return n if n >= 0 else None


def _normalise_header(names: Iterable[str]) -> list[str]:
    return [n.strip().lstrip("\ufeff").split(" ")[0].split(":")[0] for n in names]


def iter_wvs_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        delimiter = ";" if first.count(";") > first.count(",") else ","
        header = _normalise_header(next(csv.reader([first], delimiter=delimiter)))
        missing = {COUNTRY_VAR, WEIGHT_VAR} - set(header)
        if missing:
            raise ValueError(f"{path} lacks required WVS columns {sorted(missing)}")
        for values in csv.reader(f, delimiter=delimiter):
            if values:
                yield dict(zip(header, values, strict=False))


def respondent_from_row(row: dict[str, str], labels: dict[int, str]) -> Respondent | None:
    iso3 = row.get(COUNTRY_VAR, "").strip().upper()
    if not iso3:
        return None
    iso3 = COUNTRY_ALIASES.get(iso3, iso3)
    ints: dict[str, Any] = {field: _code(row.get(var)) for field, var in VARIABLES.items()}
    if ints["age"] is None or ints["sex"] is None:
        return None
    ethnic = _code(row.get(ETHNIC_VAR))
    try:
        weight = float(row.get(WEIGHT_VAR) or 1.0)
    except ValueError:
        weight = 1.0
    return Respondent(
        source="wvs7",
        country_iso3=iso3,
        weight=weight if weight > 0 else 1.0,
        ethnic_group=labels.get(ethnic, "") if ethnic is not None else "",
        **ints,
    )


def import_wvs(src: Path, dest: Path) -> int:
    """Convert a WVS7 CSV release into ``dest`` (our respondent CSV). Returns the row count."""
    labels = ethnic_labels()
    rows = [r for r in map(lambda x: respondent_from_row(x, labels), iter_wvs_rows(src)) if r]
    if not rows:
        raise ValueError(f"no usable respondents found in {src}")
    write_respondents(rows, dest)
    return len(rows)
