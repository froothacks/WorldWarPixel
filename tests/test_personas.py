import csv
import random
from collections import Counter
from pathlib import Path

import pytest

from worldwarpixel.personas import nemotron, synthetic, wvs
from worldwarpixel.personas import roster as roster_mod
from worldwarpixel.personas.allocation import allocate_seats
from worldwarpixel.personas.pool import RespondentPool, load_pool
from worldwarpixel.personas.proxy import (
    PROXY_OVERRIDES,
    SUBREGION_DEFAULTS,
    WVS7_COUNTRIES,
    survey_country,
)
from worldwarpixel.personas.render import persona_card, persona_text
from worldwarpixel.personas.roster import load_roster
from worldwarpixel.personas.roster import persona_card as roster_card
from worldwarpixel.personas.roster import persona_text as persona_text_roster
from worldwarpixel.personas.schema import Respondent, read_respondents, write_respondents
from worldwarpixel.personas.wpp import load_countries
from worldwarpixel.sim.agents import seed_agents, seed_roster
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
    assert text.startswith("Rohan Mehta, a 41-year-old man")
    assert "Your home country is India." in text
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
    assert text.startswith("Ngozi Okafor, a 23-year-old woman")
    assert "from Nigeria" not in text and "Nigeria" in text
    assert "None" not in text and "left" not in text
    assert persona_text(
        "Li Wei", Respondent(source="synthetic", country_iso3="CHN", age=18, sex=1), "China"
    ).startswith("Li Wei, an 18-year-old man")
    assert "home country is the United States" in persona_text(
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
    assert nemotron.enrich(SAMPLE, "USA", random.Random(0)) == SAMPLE
    with (tmp_path / "USA.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(nemotron.COLUMNS)
        w.writerow([1, 40, "Train conductor", '["model trains", "chess"]'])
        w.writerow([2, 41, "Nurse", '["knitting"]'])
    enriched = nemotron.enrich(SAMPLE, "USA", random.Random(0))
    assert enriched.extra == {"hobbies": ["model trains", "chess"], "occupation": "Train conductor"}
    card = persona_card(enriched, "United States", "United States")
    assert card["hobbies"] == ["model trains", "chess"] and card["occupation"] == "Train conductor"
    assert "model trains" in persona_text("Sam Hill", enriched, "United States")


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
    india = {a.color for a in agents if a.country == "India"}
    loose = seed_agents(100, 512, 320, 10.0, random.Random(42), by_country=False)
    assert len(india) == 1 and len({a.color for a in loose if a.country == "India"}) > 1


def test_default_pool_picks_up_an_import_without_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldwarpixel.personas import pool as pool_mod

    target = tmp_path / "wvs7.csv"
    monkeypatch.setenv(pool_mod.ENV_VAR, str(target))
    assert pool_mod.default_pool().sources == {"synthetic"}
    write_respondents(
        [Respondent(source="wvs7", country_iso3="CHN", weight=2.0, age=33, sex=1)], target
    )
    assert pool_mod.default_pool().sources == {"wvs7", "synthetic"}


def test_hackathon_roster_is_unique_people_with_profiles() -> None:
    roster = load_roster("hackathon")
    names = [p.name for p in roster.people]
    assert len(names) == 86 and len(set(names)) == 86
    assert all(
        p.affiliation and p.headline and p.confidence in {"high", "medium", "low"}
        for p in roster.people
    )
    assert all(
        p.linkedin is None or p.linkedin.startswith("https://www.linkedin.com/in/")
        for p in roster.people
    )


def test_seed_roster_is_one_individual_agent_per_person() -> None:
    roster = load_roster("hackathon")
    agents = seed_roster(roster.people, roster.event, 512, 320, 1.0, random.Random(3))
    assert sorted(a.name for a in agents) == sorted(p.name for p in roster.people)
    assert sorted(a.id for a in agents) == list(range(1, len(agents) + 1))
    for a in agents:
        assert a.country == a.name and a.persona.startswith(a.name) and a.headline
        assert a.persona_card["source"] == "roster" and "team" not in a.persona_card


def test_roster_card_omits_missing_fields() -> None:
    person = load_roster("hackathon").people[0]
    card = roster_card(person, "an event")
    assert None not in card.values() and card["event"] == "an event"


def test_roster_persona_uses_the_personality_read() -> None:
    person = load_roster("hackathon").people[0]
    text = persona_text_roster(person, "an event")
    assert person.personality["summary"] in text and person.personality["canvas_approach"] in text
    assert "_" not in text.split("Style:")[1].split(".")[0]


@pytest.mark.parametrize(
    ("name", "project", "slug"),
    [
        ("Leon Fattakhov", "World War Pixel", "world-war-pixel"),
        ("Gino Chiaranaipanich", "I Patched The Pixels Myself", "i-patched-the-pixels-myself"),
        ("Leah Vashevko", "Warped", "warped"),
        ("Nick Del Aguila", "Flaid", "flaid"),
    ],
)
def test_roster_project_context_reaches_personas_and_profile_cards(
    name: str, project: str, slug: str
) -> None:
    roster = load_roster("hackathon")
    person = next(p for p in roster.people if p.name == name)
    highlight = next(
        h for h in person.highlights if h.startswith(f"Neo Hackathon 2026 — {project}: ")
    )
    url = f"https://devpost.com/software/{slug}"
    assert url in person.links
    assert highlight in persona_text_roster(person, roster.event)
    card = roster_card(person, roster.event)
    assert highlight in card["highlights"] and url in card["links"]


@pytest.mark.parametrize("name", ["Maddie Bernheim", "Clara Bae", "Pavla Bobosikova", "Vincent Po"])
def test_roster_includes_confirmed_hackathon_organizer_context(name: str) -> None:
    roster = load_roster("hackathon")
    person = next(p for p in roster.people if p.name == name)
    highlight = (
        "Neo Hackathon 2026 organizer: Helped run events, food, and other hackathon logistics."
    )
    assert highlight in person.highlights
    assert highlight in persona_text_roster(person, roster.event)
    assert highlight in roster_card(person, roster.event)["highlights"]


def test_roster_includes_ali_partovis_hackathon_support() -> None:
    roster = load_roster("hackathon")
    person = next(p for p in roster.people if p.name == "Ali Partovi")
    highlight = (
        "Neo Hackathon 2026 supporter: As Neo's founder, made the event possible "
        "and encouraged and supported hackers throughout."
    )
    assert highlight in person.highlights
    assert highlight in persona_text_roster(person, roster.event)
    assert highlight in roster_card(person, roster.event)["highlights"]


@pytest.mark.parametrize(
    ("name", "title", "teammates"),
    [
        (
            "Leon Fattakhov",
            "World War Pixel",
            {"Advait Maybhate", "Megan Chun", "Christoph Moser"},
        ),
        (
            "Gino Chiaranaipanich",
            "I Patched The Pixels Myself",
            {"Tina Mai", "Alexander (Hongyi) Huang"},
        ),
        ("Micah Bozic", "Arcana", {"Ken Zhou", "Sofia Zaozerska", "khkim37382 Kim"}),
        ("Nick Del Aguila", "Flaid", {"Derrick Ha"}),
        ("Hunter Pruett", "Autonomous Fulfillment Center", set()),
    ],
)
def test_roster_project_teammates_reach_personas_and_cards(
    name: str, title: str, teammates: set[str]
) -> None:
    roster = load_roster("hackathon")
    person = next(p for p in roster.people if p.name == name)
    project = next(p for p in person.projects if p.title == title)
    assert set(project.members) == {name, *teammates}
    text = persona_text_roster(person, roster.event)
    assert f"Your hackathon project: {title}." in text
    assert all(teammate in text for teammate in teammates)
    card = roster_card(person, roster.event)["hackathon_projects"][0]
    assert card["title"] == title and card["url"] == project.url
    assert set(card["teammates"]) == teammates
    if not teammates:
        assert "No other project teammates are publicly listed." in text


def test_project_catalog_is_sourced_and_links_all_confirmed_roster_members() -> None:
    roster = load_roster("hackathon")
    assert len(roster.projects) == 17 and sum(bool(p.projects) for p in roster.people) == 40
    assert len({project.url for project in roster.projects}) == len(roster.projects)
    for project in roster.projects:
        assert project.members and len(project.members) == len(set(project.members))
        assert project.url.startswith("https://devpost.com/software/")
        for person in roster.people:
            assert (project in person.projects) == (person.name in project.members)
            if project in person.projects:
                assert project.url in person.links


def test_project_catalog_preserves_unknown_and_private_identities() -> None:
    roster = load_roster("hackathon")
    for name in ("Alexander Huang", "Kyuhyun Kim", "Maddie Bernheim", "Ali Partovi"):
        person = next(p for p in roster.people if p.name == name)
        assert not person.projects
        assert "hackathon_projects" not in roster_card(person, roster.event)
    hayden = next(p for p in roster.people if p.name == "Hayden Tedrake")
    project = hayden.projects[0]
    assert set(project.members) == {"Hayden Tedrake", "Aleksander Garbuz"}
    assert "private teammate whose identity is not public" in project.team_notes
    assert project.team_notes in persona_text_roster(hayden, roster.event)
    card = roster_card(hayden, roster.event)["hackathon_projects"][0]
    assert card["teammates"] == ["Aleksander Garbuz"]
    assert card["team_notes"] == project.team_notes


def test_roster_loads_legacy_data_without_project_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "roster.json"
    path.write_text(
        '{"id":"old","title":"Old","event":"an event",'
        '"people":[{"name":"A","affiliation":"B","headline":"C"}]}'
    )
    monkeypatch.setattr(roster_mod, "data_path", lambda _: path)
    load_roster.cache_clear()
    try:
        roster = load_roster("hackathon")
        assert not roster.projects and not roster.people[0].projects
    finally:
        load_roster.cache_clear()
