"""Respondent pool: real WVS7 records when imported, synthetic fallback otherwise."""

from __future__ import annotations

import logging
import os
import random
from bisect import bisect_right
from functools import lru_cache
from itertools import accumulate
from pathlib import Path

from worldwarpixel.personas import synthetic
from worldwarpixel.personas.schema import Respondent, read_respondents

log = logging.getLogger(__name__)

ENV_VAR = "WVS_RESPONDENTS_PATH"
DEFAULT_WVS_PATH = Path("data/wvs7_respondents.csv")


class RespondentPool:
    """Respondents grouped by survey country, sampled with their survey weights."""

    def __init__(self, respondents: list[Respondent]) -> None:
        groups: dict[str, list[Respondent]] = {}
        for r in respondents:
            groups.setdefault(r.country_iso3, []).append(r)
        self._groups = groups
        self._cumulative = {
            iso3: list(accumulate(r.weight for r in rows)) for iso3, rows in groups.items()
        }

    @property
    def countries(self) -> frozenset[str]:
        return frozenset(self._groups)

    @property
    def sources(self) -> frozenset[str]:
        return frozenset(r.source for rows in self._groups.values() for r in rows)

    def size(self, iso3: str) -> int:
        return len(self._groups.get(iso3, ()))

    def sample(self, iso3: str, rng: random.Random) -> Respondent:
        rows = self._groups[iso3]
        cum = self._cumulative[iso3]
        return rows[bisect_right(cum, rng.random() * cum[-1])]

    def merge(self, other: RespondentPool) -> RespondentPool:
        """This pool plus ``other``'s records for countries this pool lacks."""
        rows = [r for rows in self._groups.values() for r in rows]
        rows += [r for iso3, grp in other._groups.items() if iso3 not in self._groups for r in grp]
        return RespondentPool(rows)


def wvs_path() -> Path:
    return Path(os.environ.get(ENV_VAR) or DEFAULT_WVS_PATH)


def load_pool(path: Path | None = None) -> RespondentPool:
    """WVS7 respondents from ``path`` (default ``$WVS_RESPONDENTS_PATH`` or
    ``data/wvs7_respondents.csv``), topped up with synthetic records for any survey country the
    file lacks. Without the file, the pool is entirely synthetic."""
    fallback = RespondentPool(read_respondents(synthetic.data_path()))
    path = path or wvs_path()
    if not path.exists():
        log.warning(
            "%s not found; agents use the SYNTHETIC persona dataset. "
            "See README 'Persona data' to import WVS Wave 7.",
            path,
        )
        return fallback
    pool = RespondentPool(read_respondents(path))
    log.info("loaded %d WVS7 respondents from %s", sum(pool.size(c) for c in pool.countries), path)
    return pool.merge(fallback)


@lru_cache(maxsize=1)
def _cached_pool(path: Path, mtime_ns: int | None) -> RespondentPool:
    return load_pool(path)


def default_pool() -> RespondentPool:
    """The pool for the configured respondent file, re-read when the file appears or changes."""
    path = wvs_path()
    mtime = path.stat().st_mtime_ns if path.exists() else None
    return _cached_pool(path, mtime)
