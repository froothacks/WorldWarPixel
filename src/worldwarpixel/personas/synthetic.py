"""SYNTHETIC fallback respondents so the app runs without the WVS download.

These are NOT survey data. Each WVS7 country gets a few dozen records drawn from rough,
hand-written country priors (median age, religion mix, religiosity, trust, left-right lean,
education, urbanisation, support for democracy, national pride, happiness) with noise, in the
same shape as a real respondent. Every record is tagged ``source="synthetic"`` and the UI shows
that tag. Regenerate with ``uv run worldwarpixel-personas build-synthetic``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from worldwarpixel.personas.schema import Respondent, write_respondents

DATA_FILE = "synthetic_respondents.csv"
PER_COUNTRY = 30


@dataclass(frozen=True, slots=True)
class Prior:
    median_age: int
    religions: tuple[tuple[int, float], ...]  # (Q289 code, share)
    religiosity: float  # share saying they are a religious person
    trust: float  # share saying most people can be trusted
    left_right: float  # mean self-placement, 1 left .. 10 right
    education: float  # 0 (mostly primary) .. 1 (mostly tertiary)
    urban: float  # share living in urban areas
    democracy: float  # 0 .. 1 support for democratic government
    pride: float  # 0 .. 1 national pride
    happiness: float  # 0 .. 1


def _p(
    age: int,
    rel: str,
    religiosity: float,
    trust: float,
    lr: float,
    edu: float,
    urban: float,
    demo: float,
    pride: float,
    happy: float,
) -> Prior:
    religions = tuple((int(c), float(s)) for c, s in (x.split(":") for x in rel.split(",")))
    return Prior(age, religions, religiosity, trust, lr, edu, urban, demo, pride, happy)


# Rough, hand-written priors per WVS7 country (see module docstring). Religion codes: 0 none,
# 1 Catholic, 2 Protestant, 3 Orthodox, 4 Jewish, 5 Muslim, 6 Hindu, 7 Buddhist, 8 other
# Christian, 9 other.
PRIORS: dict[str, Prior] = {
    "AND": _p(44, "1:.7,0:.28,9:.02", 0.45, 0.35, 5.2, 0.6, 1.0, 0.85, 0.75, 0.8),
    "ARG": _p(32, "1:.6,0:.25,8:.13,9:.02", 0.6, 0.2, 5.3, 0.5, 0.92, 0.8, 0.7, 0.7),
    "ARM": _p(37, "3:.92,0:.06,9:.02", 0.8, 0.15, 5.6, 0.6, 0.63, 0.7, 0.8, 0.5),
    "AUS": _p(
        38,
        "0:.4,1:.2,2:.2,8:.08,7:.03,5:.03,6:.03,9:.03",
        0.4,
        0.5,
        5.3,
        0.7,
        0.86,
        0.85,
        0.75,
        0.8,
    ),
    "BGD": _p(27, "5:.9,6:.09,7:.01", 0.95, 0.2, 5.5, 0.25, 0.4, 0.75, 0.9, 0.7),
    "BOL": _p(26, "1:.7,8:.2,0:.08,9:.02", 0.8, 0.15, 5.0, 0.4, 0.7, 0.75, 0.75, 0.65),
    "BRA": _p(34, "1:.5,8:.3,0:.12,9:.08", 0.85, 0.1, 5.8, 0.45, 0.87, 0.8, 0.6, 0.7),
    "CAN": _p(
        41, "1:.3,2:.15,0:.35,8:.08,5:.04,6:.02,9:.06", 0.45, 0.5, 5.0, 0.75, 0.82, 0.85, 0.75, 0.8
    ),
    "CHL": _p(36, "1:.5,8:.17,0:.3,9:.03", 0.55, 0.15, 5.0, 0.55, 0.88, 0.8, 0.7, 0.7),
    "CHN": _p(40, "0:.85,7:.08,9:.04,2:.02,5:.01", 0.15, 0.6, 5.5, 0.45, 0.65, 0.65, 0.9, 0.7),
    "COL": _p(32, "1:.65,8:.2,0:.12,9:.03", 0.8, 0.07, 5.8, 0.45, 0.82, 0.75, 0.8, 0.75),
    "CYP": _p(38, "3:.7,5:.2,1:.03,0:.05,9:.02", 0.7, 0.2, 5.8, 0.65, 0.67, 0.8, 0.75, 0.7),
    "CZE": _p(44, "0:.75,1:.2,2:.03,9:.02", 0.25, 0.3, 5.5, 0.6, 0.74, 0.75, 0.6, 0.7),
    "ECU": _p(29, "1:.7,8:.18,0:.1,9:.02", 0.85, 0.1, 5.5, 0.45, 0.65, 0.7, 0.8, 0.7),
    "EGY": _p(25, "5:.9,3:.1", 0.95, 0.15, 5.7, 0.4, 0.43, 0.7, 0.9, 0.55),
    "ETH": _p(20, "3:.45,5:.35,2:.18,9:.02", 0.95, 0.2, 5.5, 0.2, 0.23, 0.75, 0.85, 0.5),
    "DEU": _p(46, "0:.4,1:.25,2:.25,5:.05,9:.05", 0.4, 0.45, 4.9, 0.7, 0.78, 0.9, 0.5, 0.75),
    "GBR": _p(
        41, "0:.5,2:.2,1:.08,8:.08,5:.06,6:.02,9:.06", 0.35, 0.4, 5.0, 0.7, 0.84, 0.85, 0.6, 0.75
    ),
    "GRC": _p(46, "3:.85,0:.12,9:.03", 0.7, 0.1, 5.5, 0.6, 0.8, 0.8, 0.75, 0.55),
    "GTM": _p(24, "1:.45,8:.4,0:.13,9:.02", 0.9, 0.1, 5.8, 0.3, 0.52, 0.65, 0.8, 0.75),
    "HKG": _p(46, "0:.65,7:.12,2:.08,1:.06,9:.09", 0.25, 0.4, 5.3, 0.65, 1.0, 0.75, 0.5, 0.6),
    "IND": _p(29, "6:.78,5:.14,8:.03,9:.05", 0.85, 0.3, 6.0, 0.35, 0.36, 0.7, 0.9, 0.7),
    "IDN": _p(31, "5:.87,2:.07,1:.03,6:.02,7:.01", 0.95, 0.1, 5.8, 0.4, 0.58, 0.75, 0.9, 0.8),
    "IRN": _p(33, "5:.95,0:.03,9:.02", 0.7, 0.15, 5.2, 0.5, 0.76, 0.6, 0.7, 0.5),
    "IRQ": _p(22, "5:.96,9:.04", 0.9, 0.1, 5.5, 0.35, 0.71, 0.6, 0.75, 0.5),
    "JPN": _p(49, "0:.6,7:.3,9:.08,8:.02", 0.2, 0.35, 5.5, 0.7, 0.92, 0.75, 0.6, 0.7),
    "JOR": _p(24, "5:.95,8:.03,9:.02", 0.9, 0.15, 5.6, 0.45, 0.91, 0.65, 0.85, 0.55),
    "KAZ": _p(32, "5:.65,3:.2,0:.12,9:.03", 0.6, 0.3, 5.5, 0.6, 0.58, 0.6, 0.85, 0.7),
    "KEN": _p(20, "2:.35,1:.2,8:.3,5:.1,9:.05", 0.95, 0.1, 5.8, 0.35, 0.3, 0.75, 0.85, 0.6),
    "KGZ": _p(27, "5:.85,3:.1,0:.03,9:.02", 0.8, 0.3, 5.5, 0.5, 0.37, 0.6, 0.85, 0.7),
    "LBN": _p(30, "5:.6,1:.2,3:.1,8:.08,9:.02", 0.7, 0.1, 5.3, 0.55, 0.89, 0.65, 0.6, 0.4),
    "LBY": _p(29, "5:.98,9:.02", 0.9, 0.1, 5.5, 0.45, 0.81, 0.55, 0.8, 0.5),
    "MAC": _p(40, "0:.6,7:.2,1:.08,9:.12", 0.3, 0.35, 5.3, 0.6, 1.0, 0.65, 0.6, 0.65),
    "MYS": _p(
        31, "5:.62,7:.18,8:.08,1:.03,6:.06,9:.03", 0.85, 0.2, 5.8, 0.5, 0.78, 0.7, 0.85, 0.75
    ),
    "MDV": _p(31, "5:.99,9:.01", 0.95, 0.15, 5.5, 0.45, 0.42, 0.65, 0.85, 0.7),
    "MEX": _p(30, "1:.75,8:.12,0:.1,9:.03", 0.75, 0.1, 5.6, 0.4, 0.81, 0.7, 0.8, 0.75),
    "MNG": _p(30, "7:.5,0:.4,5:.03,8:.03,9:.04", 0.6, 0.2, 5.0, 0.55, 0.69, 0.75, 0.9, 0.6),
    "MAR": _p(30, "5:.99,9:.01", 0.9, 0.15, 5.5, 0.3, 0.65, 0.65, 0.85, 0.6),
    "MMR": _p(30, "7:.88,8:.06,5:.04,9:.02", 0.95, 0.2, 5.3, 0.3, 0.32, 0.75, 0.85, 0.65),
    "NLD": _p(43, "0:.55,1:.2,2:.15,5:.05,9:.05", 0.35, 0.6, 5.2, 0.7, 0.93, 0.9, 0.55, 0.85),
    "NZL": _p(
        38,
        "0:.45,2:.2,1:.1,8:.1,6:.03,5:.01,7:.02,9:.09",
        0.35,
        0.55,
        5.2,
        0.7,
        0.87,
        0.9,
        0.75,
        0.8,
    ),
    "NIC": _p(27, "1:.45,8:.38,0:.15,9:.02", 0.85, 0.1, 5.3, 0.3, 0.6, 0.65, 0.8, 0.7),
    "NGA": _p(18, "5:.5,2:.2,8:.22,1:.08", 0.98, 0.1, 5.8, 0.4, 0.54, 0.75, 0.75, 0.75),
    "PAK": _p(21, "5:.97,8:.02,6:.01", 0.95, 0.2, 5.8, 0.3, 0.38, 0.65, 0.9, 0.7),
    "PER": _p(30, "1:.7,8:.2,0:.07,9:.03", 0.8, 0.07, 5.5, 0.45, 0.79, 0.7, 0.75, 0.65),
    "PHL": _p(26, "1:.78,8:.14,5:.06,9:.02", 0.9, 0.05, 6.2, 0.45, 0.48, 0.75, 0.9, 0.8),
    "PRI": _p(44, "1:.55,8:.3,0:.12,9:.03", 0.8, 0.15, 5.8, 0.6, 0.94, 0.8, 0.8, 0.75),
    "ROU": _p(43, "3:.85,1:.05,8:.06,0:.02,9:.02", 0.8, 0.15, 5.8, 0.5, 0.55, 0.75, 0.7, 0.6),
    "RUS": _p(40, "3:.6,0:.3,5:.07,9:.03", 0.55, 0.25, 5.5, 0.65, 0.75, 0.55, 0.7, 0.6),
    "SRB": _p(44, "3:.85,1:.05,5:.03,0:.05,9:.02", 0.7, 0.2, 5.5, 0.55, 0.57, 0.65, 0.7, 0.6),
    "SGP": _p(
        43, "7:.3,8:.1,1:.07,0:.2,5:.15,6:.05,9:.13", 0.55, 0.35, 5.5, 0.7, 1.0, 0.75, 0.85, 0.8
    ),
    "SVK": _p(42, "1:.6,0:.25,2:.08,9:.07", 0.6, 0.2, 5.3, 0.6, 0.54, 0.75, 0.6, 0.7),
    "KOR": _p(45, "0:.5,2:.2,7:.17,1:.11,9:.02", 0.35, 0.3, 5.5, 0.75, 0.81, 0.8, 0.65, 0.65),
    "TWN": _p(44, "0:.3,7:.3,9:.3,8:.05,1:.02,2:.03", 0.4, 0.3, 5.0, 0.7, 0.8, 0.8, 0.7, 0.7),
    "TJK": _p(23, "5:.97,0:.02,9:.01", 0.85, 0.3, 5.5, 0.4, 0.28, 0.55, 0.9, 0.75),
    "THA": _p(41, "7:.93,5:.05,8:.01,9:.01", 0.8, 0.3, 5.8, 0.4, 0.53, 0.7, 0.9, 0.8),
    "TUN": _p(34, "5:.98,0:.02", 0.85, 0.15, 5.3, 0.4, 0.7, 0.65, 0.8, 0.5),
    "TUR": _p(34, "5:.95,0:.04,9:.01", 0.8, 0.12, 6.0, 0.4, 0.77, 0.7, 0.85, 0.55),
    "UKR": _p(44, "3:.7,0:.15,1:.1,8:.03,9:.02", 0.7, 0.3, 5.3, 0.65, 0.7, 0.75, 0.75, 0.5),
    "URY": _p(36, "1:.4,0:.45,8:.1,9:.05", 0.4, 0.2, 4.8, 0.5, 0.96, 0.9, 0.75, 0.75),
    "USA": _p(
        39, "2:.3,1:.22,0:.25,8:.12,4:.02,5:.01,9:.08", 0.6, 0.35, 5.6, 0.7, 0.83, 0.8, 0.8, 0.75
    ),
    "UZB": _p(29, "5:.93,3:.04,0:.02,9:.01", 0.85, 0.3, 5.5, 0.5, 0.5, 0.6, 0.95, 0.85),
    "VEN": _p(29, "1:.6,8:.2,0:.15,9:.05", 0.8, 0.1, 5.3, 0.5, 0.88, 0.75, 0.8, 0.65),
    "VNM": _p(33, "0:.75,7:.15,1:.07,9:.03", 0.4, 0.3, 5.3, 0.45, 0.4, 0.75, 0.95, 0.75),
    "ZWE": _p(19, "2:.3,8:.45,1:.1,0:.1,9:.05", 0.95, 0.1, 5.5, 0.35, 0.32, 0.8, 0.8, 0.55),
}


def _clamp(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, round(x)))


def _pick(rng: random.Random, options: dict[int, float]) -> int:
    return rng.choices(list(options), weights=list(options.values()))[0]


def synthesise(iso3: str, prior: Prior, rng: random.Random, ethnic: list[str]) -> Respondent:
    p = prior
    age = _clamp(rng.gauss(p.median_age + 10, 15), 18, 85)
    sex = rng.choice((1, 2))
    education = _clamp(rng.gauss(1 + 6 * p.education, 1.6), 0, 8)
    if age >= 64:
        employment = _pick(rng, {4: 0.75, 1: 0.1, 3: 0.1, 5: 0.05})
    elif age <= 23:
        employment = _pick(rng, {6: 0.4, 1: 0.3, 2: 0.1, 7: 0.15, 5: 0.05})
    else:
        employment = _pick(
            rng, {1: 0.45, 2: 0.1, 3: 0.17, 5: 0.15 if sex == 2 else 0.01, 7: 0.1, 8: 0.02}
        )
    if education >= 6:
        occupation = _pick(rng, {1: 0.55, 2: 0.2, 3: 0.2, 4: 0.05})
    elif education >= 3:
        occupation = _pick(rng, {3: 0.25, 4: 0.2, 5: 0.25, 6: 0.25, 1: 0.05})
    else:
        occupation = _pick(rng, {7: 0.3, 8: 0.3, 9: 0.2, 10: 0.1, 5: 0.1})
    if employment == 6 and rng.random() < 0.7:
        occupation = 0
    rural = rng.random() >= p.urban
    if rural and education < 3 and rng.random() < 0.5:
        occupation = rng.choice((9, 10))
    income = _clamp(rng.gauss(4 + education * 0.4, 2), 1, 10)
    social_class = _clamp(5.5 - income * 0.45 + rng.gauss(0, 0.6), 1, 5)
    religion = _pick(rng, dict(p.religions))
    religious = rng.random() < (p.religiosity if religion else 0.15)
    religious_person = 1 if religious else _pick(rng, {2: 0.8, 3: 0.2})
    attendance = _pick(
        rng, {1: 0.2, 2: 0.35, 3: 0.25, 4: 0.2} if religious else {5: 0.2, 6: 0.3, 7: 0.5}
    )
    god = _clamp(rng.gauss(8.5 if religious else 3.5, 1.8), 1, 10)
    democracy = _clamp(rng.gauss(5 + 5 * p.democracy, 1.8), 1, 10)
    marital = (
        _pick(rng, {6: 0.8, 1: 0.12, 2: 0.08})
        if age < 25
        else _pick(rng, {1: 0.55, 5: 0.3, 3: 0.08, 6: 0.07})
        if age > 60
        else _pick(rng, {1: 0.6, 2: 0.1, 6: 0.2, 3: 0.07, 4: 0.03})
    )
    children = (
        0
        if marital == 6 and rng.random() < 0.85
        else _clamp(rng.gauss(2 if age > 30 else 0.7, 1.3), 0, 7)
    )
    return Respondent(
        source="synthetic",
        country_iso3=iso3,
        weight=1.0,
        age=age,
        sex=sex,
        education=education,
        employment=employment,
        occupation=occupation,
        income_decile=income,
        social_class=social_class,
        religion=religion,
        religious_person=religious_person,
        attendance=attendance,
        god_importance=god,
        left_right=_clamp(rng.gauss(p.left_right, 2.0), 1, 10),
        trust=1 if rng.random() < p.trust else 2,
        democracy_importance=democracy,
        strong_leader=_pick(
            rng,
            {1: 0.1, 2: 0.25, 3: 0.35, 4: 0.3}
            if p.democracy > 0.7
            else {1: 0.25, 2: 0.35, 3: 0.25, 4: 0.15},
        ),
        democratic_system=_pick(
            rng,
            {1: 0.55, 2: 0.35, 3: 0.08, 4: 0.02}
            if p.democracy > 0.7
            else {1: 0.35, 2: 0.4, 3: 0.18, 4: 0.07},
        ),
        political_interest=_pick(rng, {1: 0.12, 2: 0.33, 3: 0.33, 4: 0.22}),
        income_equality=_clamp(rng.gauss(5.5, 2.5), 1, 10),
        env_vs_growth=1 if rng.random() < 0.55 else 2,
        happiness=_pick(
            rng, {1: 0.45 * p.happiness + 0.1, 2: 0.5, 3: 0.3 * (1 - p.happiness) + 0.05, 4: 0.05}
        ),
        life_satisfaction=_clamp(rng.gauss(4 + 5 * p.happiness, 2), 1, 10),
        national_pride=_pick(
            rng, {1: 0.75 * p.pride + 0.1, 2: 0.4, 3: 0.3 * (1 - p.pride) + 0.05, 4: 0.05}
        ),
        marital=marital,
        children=children,
        urban=2 if rural else 1,
        town_size=_pick(rng, {1: 0.7, 2: 0.3}) if rural else _pick(rng, {3: 0.3, 4: 0.3, 5: 0.4}),
        ethnic_group=(ethnic[0] if rng.random() < 0.6 else rng.choice(ethnic)) if ethnic else "",
    )


def build_synthetic(seed: int = 7, per_country: int = PER_COUNTRY) -> list[Respondent]:
    from worldwarpixel.personas.wpp import load_countries
    from worldwarpixel.personas.wvs import ethnic_labels

    numeric = {c.iso3: c.m49 for c in load_countries()}
    by_country: dict[int, list[str]] = {}
    for code, label in ethnic_labels().items():
        by_country.setdefault(code // 1000, []).append(label)
    rng = random.Random(seed)
    out: list[Respondent] = []
    for iso3, prior in sorted(PRIORS.items()):
        ethnic = [
            lab
            for lab in by_country.get(numeric.get(iso3, -1), [])
            if lab.lower() not in ("other", "others")
        ]
        out.extend(synthesise(iso3, prior, rng, ethnic) for _ in range(per_country))
    return out


def data_path() -> Path:
    return Path(str(resources.files("worldwarpixel") / "data" / DATA_FILE))


def write_synthetic(path: Path | None = None, seed: int = 7) -> Path:
    path = path or data_path()
    write_respondents(build_synthetic(seed), path)
    return path
