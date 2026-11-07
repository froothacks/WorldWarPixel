import csv
import random
from collections import Counter
from pathlib import Path

import pytest

from worldwarpixel.personas import nemotron, synthetic, wvs
from worldwarpixel.personas.allocation import allocate_seats
from worldwarpixel.personas.pool import RespondentPool, load_pool
from worldwarpixel.personas.proxy import (
    PROXY_OVERRIDES,
    SUBREGION_DEFAULTS,
    WVS7_COUNTRIES,
    survey_country,
)
from worldwarpixel.personas.render import persona_card, persona_text
from worldwarpixel.personas.schema import Respondent, read_respondents, write_respondents
from worldwarpixel.personas.wpp import load_countries
from worldwarpixel.sim.agents import seed_agents
from worldwarpixel.sim.names import random_name


def test_largest_remainder_sums_and_rounds_by_remainder() -> None:
    seats = allocate_seats({"a": 50.5, "b": 30.3, "c": 19.2}, 10)
    assert sum(seats.values()) == 10
    assert seats == {"a": 5, "b": 3, "c": 2}
    assert allocate_seats({"a": 1, "b": 1}, 0) == {}
    with pytest.raises(ValueError):
        allocate_seats({"a": 0}, 3)


def test_allocation_omits_zero_seat_keys_and_is_deterministic() -> None:
    weights = {f"k{i}": 1 + i for i in range(50)}
    one, two = allocate_seats(weights, 7), allocate_seats(weights, 7)
    assert one == two and sum(one.values()) == 7 and all(v > 0 for v in one.values())


def test_wpp_2024_gives_expected_country_seats() -> None:
    countries = load_countries()
    assert len(countries) > 200
    assert sum(c.population for c in countries) > 8_000_000_000
    seats = allocate_seats({c.iso3: c.population for c in countries}, 100)
    assert seats["IND"] == 18 and seats["CHN"] == 17 and seats["USA"] == 4
    assert seats["IDN"] in (3, 4) and sum(seats.values()) == 100


def test_every_wpp_country_has_a_wvs_survey_proxy() -> None:
    assert set(PROXY_OVERRIDES.values()) <= WVS7_COUNTRIES
    assert set(SUBREGION_DEFAULTS.values()) <= WVS7_COUNTRIES
    for c in load_countries():
        assert survey_country(c.iso3, c.subregion) in WVS7_COUNTRIES, c
    assert survey_country("IND", "Southern Asia") == "IND"
    assert survey_country("NPL", "Southern Asia") != "NPL"


def test_weighted_sampling_follows_survey_weights() -> None:
    heavy = Respondent(source="synthetic", country_iso3="XXX", weight=9.0, age=30, sex=1)
    light = Respondent(source="synthetic", country_iso3="XXX", weight=1.0, age=60, sex=2)
    pool = RespondentPool([heavy, light])
    rng = random.Random(0)
    picks = Counter(pool.sample("XXX", rng).age for _ in range(2000))
    assert 0.85 < picks[30] / 2000 < 0.95
    assert pool.size("XXX") == 2 and pool.countries == {"XXX"}


def test_synthetic_fallback_is_labelled_and_covers_all_survey_countries() -> None:
    pool = load_pool(Path("/nonexistent/wvs7.csv"))
    assert pool.sources == {"synthetic"}
    assert pool.countries >= WVS7_COUNTRIES
    r = pool.sample("CHN", random.Random(1))
    assert r.source == "synthetic" and r.age and r.sex in (1, 2)


def test_build_synthetic_is_reproducible() -> None:
    a = synthetic.build_synthetic(seed=3, per_country=2)
    b = synthetic.build_synthetic(seed=3, per_country=2)
    assert a == b and len(a) == 2 * len(WVS7_COUNTRIES)


def test_wvs_import_handles_negative_codes_and_semicolons(tmp_path: Path) -> None:
    src = tmp_path / "WVS_Cross-National_Wave_7.csv"
    header = ["B_COUNTRY_ALPHA", "W_WEIGHT", "Q262", "Q260", "Q240", "Q57", "Q289", "Q290", "Q275"]
    src.write_text(
        ";".join(header)
        + "\n"
        + ";".join(["IND", "1.25", "34", "2", "4", "-2", "5", "356017", "3"])
        + "\n"
        + ";".join(["NIR", "0.8", "51", "1", "-1", "1", "1", "-4", "6"])
        + "\n"
        + ";".join(["CHN", "1.0", "-5", "1", "5", "2", "0", "-4", "4"])
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.csv"
    assert wvs.import_wvs(src, dest) == 2
    rows = read_respondents(dest)
    ind, nir = rows
    assert ind.source == "wvs7" and ind.left_right == 4 and ind.trust is None
    assert ind.weight == 1.25 and ind.ethnic_group
    assert nir.country_iso3 == "GBR" and nir.left_right is None
    pool = load_pool(dest)
    assert pool.sources == {"wvs7", "synthetic"}
    assert pool.size("IND") == 1 and pool.size("CHN") > 0


def test_wvs_import_requires_country_and_weight_columns(tmp_path: Path) -> None:
    src = tmp_path / "bad.csv"
    src.write_text("Q262,Q260\n30,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="B_COUNTRY_ALPHA"):
        wvs.import_wvs(src, tmp_path / "out.csv")


def test_respondent_csv_round_trip(tmp_path: Path) -> None:
    r = Respondent(
        source="wvs7",
        country_iso3="BRA",
        weight=0.7,
        age=29,
        sex=2,
        left_right=8,
        ethnic_group="Pardo",
    )
    path = tmp_path / "r.csv"
    write_respondents([r], path)
    assert read_respondents(path) == [r]


SAMPLE = Respondent(
    source="wvs7",
    country_iso3="IND",
    weight=1.0,
    age=41,
    sex=1,
    education=3,
    employment=1,
    occupation=5,
    income_decile=4,
    social_class=4,
    religion=6,
    religious_person=1,
    attendance=2,
    god_importance=9,
    left_right=4,
    trust=2,
    democracy_importance=9,
    strong_leader=2,
    democratic_system=1,
    political_interest=3,
    happiness=2,
    national_pride=1,
    marital=1,
    children=2,
    urban=2,
    town_size=2,
    ethnic_group="Hindu (Other Backward Castes)",
)


def test_persona_text_is_the_respondents_own_answers_and_compact() -> None:
    text = persona_text("Rohan Mehta", SAMPLE, "India")
    assert text.startswith("Rohan Mehta, a 41-year-old man from India")
    assert "place yourself at 4 on a 1 (left) to 10 (right) scale" in text
    assert "Hindu (Other Backward Castes)" in text
    sentences = [s for s in text.split(". ") if s]
    assert 3 <= len(sentences) <= 6
    assert len(text.split()) < 150
    for claim in ("Indians", "people in India", "most people", "they believe"):
        assert claim not in text


def test_persona_text_skips_missing_answers() -> None:
    sparse = Respondent(source="synthetic", country_iso3="NGA", age=23, sex=2)
    text = persona_text("Ngozi Okafor", sparse, "Nigeria")
    assert text.startswith("Ngozi Okafor, a 23-year-old woman from Nigeria")
    assert "None" not in text and "left" not in text
    assert persona_text(
        "Li Wei", Respondent(source="synthetic", country_iso3="CHN", age=18, sex=1), "China"
    ).startswith("Li Wei, an 18-year-old man from China")
    assert "from the United States" in persona_text(
        "Sam Hill",
        Respondent(source="synthetic", country_iso3="USA", age=30, sex=1),
        "United States",
    )


def test_persona_card_fields() -> None:
    card = persona_card(SAMPLE, "Nepal", "India")
    assert (
        card["source"] == "wvs7"
        and card["country"] == "Nepal"
        and card["survey_country"] == "India"
    )
    assert card["sex"] == "man" and card["left_right"] == 4 and card["income_decile"] == 4
    assert (
        card["religion"] == "Hindu" and card["education"] and card["employment"] == "work full time"
    )
    assert card["trust"] and card["democracy_importance"] == 9
    assert "hobbies" not in card


def test_nemotron_enrich_without_data_is_noop_and_with_data_matches_sex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(nemotron.ENV_VAR, str(tmp_path))
    nemotron._load.cache_clear()
    assert nemotron.enrich(SAMPLE, "USA", random.Random(0)) == SAMPLE
    with (tmp_path / "USA.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(nemotron.COLUMNS)
        w.writerow([1, 40, "Train conductor", '["model trains", "chess"]'])
        w.writerow([2, 41, "Nurse", '["knitting"]'])
    nemotron._load.cache_clear()
    enriched = nemotron.enrich(SAMPLE, "USA", random.Random(0))
    assert enriched.extra == {"hobbies": ["model trains", "chess"], "occupation": "Train conductor"}
    card = persona_card(enriched, "United States", "United States")
    assert card["hobbies"] == ["model trains", "chess"] and card["occupation"] == "Train conductor"
    assert "model trains" in persona_text("Sam Hill", enriched, "United States")
    nemotron._load.cache_clear()


def test_random_name_is_sex_aware_unique_and_falls_back_by_region() -> None:
    used: set[str] = set()
    names = [random_name("India", random.Random(i), used, sex=2) for i in range(30)]
    assert len(set(names)) == 30 and all(" " in n for n in names)
    assert random_name("Tuvalu", random.Random(0), set(), sex=1, subregion="Polynesia")
    assert random_name("Nowhere", random.Random(0), set(), subregion="Nowhere").count(" ") >= 1


def test_seed_agents_is_representative_named_and_reproducible() -> None:
    agents = seed_agents(100, 512, 320, 10.0, random.Random(42))
    again = seed_agents(100, 512, 320, 10.0, random.Random(42))
    assert [a.to_dict() for a in agents] == [a.to_dict() for a in again]
    assert [a.id for a in agents] == list(range(1, 101))
    by_country = Counter(a.country for a in agents)
    assert (
        by_country["India"] == 18 and by_country["China"] == 17 and by_country["United States"] == 4
    )
    assert len({a.name for a in agents}) == 100
    for a in agents:
        assert not a.name.startswith("Agent") and a.name != a.country and " " in a.name
        assert a.persona.startswith(f"{a.name}, a")
        assert a.persona_card["country"] == a.country
        assert a.persona_card["source"] in ("wvs7", "synthetic")
        assert 0 <= a.home_x < 512 and 0 <= a.home_y < 320
    small = seed_agents(3, 64, 64, 1.0, random.Random(0))
    assert [a.country for a in small] and len(small) == 3
