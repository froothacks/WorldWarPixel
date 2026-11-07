"""Optional enrichment from NVIDIA Nemotron-Personas (CC BY 4.0, Hugging Face).

Gives agents a concrete occupation and hobbies on top of the survey answers. Only the
English-language releases are used (the Japanese/Brazilian/German/... sets are written in their
own languages). ``uv run worldwarpixel-personas fetch-nemotron`` samples rows through the public
datasets-server API into ``data/nemotron/<ISO3>.csv`` (git-ignored; the files are large).
At seeding time a respondent is matched to a row of the same sex and a similar age.

Attribution: NVIDIA Nemotron-Personas datasets, https://huggingface.co/nvidia, CC BY 4.0.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import random
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

from worldwarpixel.personas.schema import Respondent

API = "https://datasets-server.huggingface.co/rows"
# ISO3 -> (dataset, split)
DATASETS: dict[str, tuple[str, str]] = {
    "USA": ("nvidia/Nemotron-Personas-USA", "train"),
    "IND": ("nvidia/Nemotron-Personas-India", "en_IN"),
    "SGP": ("nvidia/Nemotron-Personas-Singapore", "train"),
    "GBR": ("nvidia/Nemotron-Personas-UK", "train"),
}
DEFAULT_ROWS = 2000
PAGE = 100
ENV_VAR = "NEMOTRON_DIR"
DEFAULT_DIR = Path("data/nemotron")
COLUMNS = ("sex", "age", "occupation", "hobbies")
SEX = {"male": 1, "m": 1, "man": 1, "female": 2, "f": 2, "woman": 2}


def data_dir() -> Path:
    return Path(os.environ.get(ENV_VAR) or DEFAULT_DIR)


def _hobbies(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(h) for h in raw]
    if isinstance(raw, str) and raw.startswith("["):
        try:
            return [str(h) for h in ast.literal_eval(raw)]
        except (ValueError, SyntaxError):
            return []
    return []


def _occupation(raw: object) -> str:
    text = str(raw or "").replace("_", " ").strip()
    return "" if not text or text.lower().startswith("no occupation") else text


def fetch(iso3: str, rows: int = DEFAULT_ROWS, dest: Path | None = None) -> tuple[Path, int]:
    dataset, split = DATASETS[iso3]
    dest = dest or data_dir() / f"{iso3}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for offset in range(0, rows, PAGE):
            q = urllib.parse.urlencode(
                {
                    "dataset": dataset,
                    "config": "default",
                    "split": split,
                    "offset": offset,
                    "length": min(PAGE, rows - offset),
                }
            )
            with urllib.request.urlopen(f"{API}?{q}", timeout=120) as resp:
                page = json.load(resp)
            for item in page.get("rows", []):
                row = item["row"]
                sex = SEX.get(str(row.get("sex", "")).lower())
                hobbies = _hobbies(row.get("hobbies_and_interests_list"))
                if sex is None or not hobbies:
                    continue
                w.writerow(
                    [sex, row.get("age"), _occupation(row.get("occupation")), json.dumps(hobbies)]
                )
                written += 1
            if not page.get("rows"):
                break
    return dest, written


@lru_cache(maxsize=8)
def _load(iso3: str) -> list[tuple[int, int, str, tuple[str, ...]]]:
    path = data_dir() / f"{iso3}.csv"
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append(
                    (
                        int(row["sex"]),
                        int(row["age"]),
                        row["occupation"],
                        tuple(json.loads(row["hobbies"])),
                    )
                )
            except (ValueError, KeyError):
                continue
    return out


def enrich(r: Respondent, iso3: str, rng: random.Random) -> Respondent:
    """Attach ``extra={"occupation", "hobbies"}`` from a same-sex, similar-age Nemotron persona."""
    rows = _load(iso3)
    if not rows or r.sex is None or r.age is None:
        return r
    near = [x for x in rows if x[0] == r.sex and abs(x[1] - r.age) <= 3] or [
        x for x in rows if x[0] == r.sex
    ]
    if not near:
        return r
    _, _, occupation, hobbies = rng.choice(near)
    extra = dict(r.extra)
    extra["hobbies"] = list(hobbies[:3])
    if occupation:
        extra["occupation"] = occupation
    return Respondent(**{**{f: getattr(r, f) for f in r.__dataclass_fields__}, "extra": extra})
