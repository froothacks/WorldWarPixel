import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from worldwarpixel.personas import nemotron
from worldwarpixel.personas.allocation import allocate_seats
from worldwarpixel.personas.pool import RespondentPool, default_pool
from worldwarpixel.personas.proxy import survey_country
from worldwarpixel.personas.render import persona_card, persona_text
from worldwarpixel.personas.wpp import Country, load_countries
from worldwarpixel.sim.canvas import PALETTE
from worldwarpixel.sim.names import random_name
from worldwarpixel.sim.tools import ToolError, ToolSet, TurnBudget


@dataclass(slots=True)
class AgentSpec:
    id: int
    name: str
    country: str
    color: int
    home_x: int
    home_y: int
    cooldown: float
    persona: str = ""
    persona_card: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Observation:
    elapsed: float
    budget: TurnBudget
    canvas_width: int
    canvas_height: int
    board: list[dict[str, Any]]


class Policy(Protocol):
    """Decides what an agent does on one turn and returns its explicit 'thought'."""

    async def act(self, agent: AgentSpec, tools: ToolSet, obs: Observation) -> str: ...


def seed_agents(
    count: int,
    width: int,
    height: int,
    cooldown: float,
    rng: random.Random,
    pool: RespondentPool | None = None,
    countries: Sequence[Country] | None = None,
) -> list[AgentSpec]:
    """Agents representative of the world: seats per country from UN WPP 2024 population
    (largest remainder), each seat filled by a weighted-sampled WVS7 respondent (or a synthetic
    stand-in) whose answers become the agent's persona. Agents cluster per country on the canvas.
    """
    pool = pool or default_pool()
    countries = countries or load_countries()
    by_iso3 = {c.iso3: c for c in countries}
    seats = allocate_seats({c.iso3: c.population for c in countries}, count)
    # Skip palette entries 0 and 1 (white and light grey): invisible on the blank canvas.
    visible = len(PALETTE) - 2
    spread = max(8, min(width, height) // 12)
    agents: list[AgentSpec] = []
    used: set[str] = set()
    for i, iso3 in enumerate(sorted(seats, key=lambda k: (-seats[k], k))):
        country = by_iso3[iso3]
        source = survey_country(iso3, country.subregion)
        if source not in pool.countries:
            source = next(iter(sorted(pool.countries)))
        cx, cy, color = rng.randrange(width), rng.randrange(height), 2 + i % visible
        for _ in range(seats[iso3]):
            respondent = nemotron.enrich(pool.sample(source, rng), iso3, rng)
            name = random_name(country.name, rng, used, respondent.sex, country.subregion)
            survey_name = by_iso3[source].name if source in by_iso3 else source
            agents.append(
                AgentSpec(
                    id=len(agents) + 1,
                    name=name,
                    country=country.name,
                    color=color,
                    home_x=min(max(cx + rng.randint(-spread, spread), 0), width - 1),
                    home_y=min(max(cy + rng.randint(-spread, spread), 0), height - 1),
                    cooldown=cooldown,
                    persona=persona_text(name, respondent, country.name),
                    persona_card=persona_card(respondent, country.name, survey_name),
                )
            )
    rng.shuffle(agents)
    for i, agent in enumerate(agents, start=1):
        agent.id = i
    return agents


class RandomWalkPolicy:
    """No LLM: paints a drifting blob in the agent's colour and occasionally posts."""

    def __init__(self, seed: int, post_chance: float = 0.05) -> None:
        self._rng = random.Random(seed)
        self._post_chance = post_chance

    async def act(self, agent: AgentSpec, tools: ToolSet, obs: Observation) -> str:
        rng = self._rng
        px: list[tuple[int, int, int]] = []
        x, y = agent.home_x, agent.home_y
        for _ in range(obs.budget.pixels):
            x = min(max(x + rng.randint(-2, 2), 0), obs.canvas_width - 1)
            y = min(max(y + rng.randint(-2, 2), 0), obs.canvas_height - 1)
            px.append((x, y, agent.color))
        agent.home_x, agent.home_y = x, y
        tools.place_pixels(px)
        thought = f"Painting near ({x}, {y}) in colour {agent.color}."
        if obs.budget.posts and rng.random() < self._post_chance:
            thought += " " + self._post(agent, tools, obs.board, rng)
        return thought

    @staticmethod
    def _post(
        agent: AgentSpec, tools: ToolSet, board: Sequence[dict[str, Any]], rng: random.Random
    ) -> str:
        try:
            news = [
                d
                for d in board
                if d.get("tag") == "dispatch" and agent.country in d.get("countries", [])
            ]
            mentions = [d for d in board if d.get("tag") == "mention"]
            if news or mentions:
                thread = (news or mentions)[0]
                tid = int(str(thread["thread_id"]))
                x, y = agent.home_x, agent.home_y
                other = next(
                    (
                        country
                        for country in thread.get("countries", [])
                        if country != agent.country
                    ),
                    "our rivals",
                )
                match thread.get("kind"):
                    case "war":
                        body = rng.choice(
                            [
                                f"{agent.country} is not backing down near ({x}, {y}).",
                                f"Allies of {agent.country}: help us hold ({x}, {y})!",
                                f"{agent.country} will accept a truce if {other} stops at "
                                f"({x}, {y}).",
                            ]
                        )
                    case "talk":
                        body = (
                            f"{agent.country} agrees. Let's keep building together near ({x}, {y})."
                        )
                    case "territory":
                        body = f"{agent.country} is coming for that top spot from ({x}, {y})."
                    case "art":
                        body = f"Nice work. {agent.country} will leave it alone."
                    case _:
                        body = f"{agent.country} hears you. We are at ({x}, {y})."
                tools.reply(tid, body)
                return f"Replied in thread {tid}."
            if rng.random() < 0.2:
                tools.post_thread(
                    f"{agent.country} claims ({agent.home_x}, {agent.home_y})",
                    "Anyone nearby want to coordinate colours?",
                )
                return "Started a thread to claim my area."
            return "Stayed quiet."
        except ToolError as e:
            return f"Post failed: {e}"
