import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from worldwarpixel.sim.canvas import PALETTE
from worldwarpixel.sim.tools import ToolError, ToolSet, TurnBudget

# Placeholder population until the WPP + WVS seeding lands: (country, share of world population).
PLACEHOLDER_COUNTRIES: tuple[tuple[str, float], ...] = (
    ("India", 17.8),
    ("China", 17.4),
    ("United States", 4.2),
    ("Indonesia", 3.5),
    ("Pakistan", 3.0),
    ("Nigeria", 2.8),
    ("Brazil", 2.6),
    ("Bangladesh", 2.1),
    ("Russia", 1.8),
    ("Ethiopia", 1.6),
    ("Mexico", 1.6),
    ("Japan", 1.5),
    ("Egypt", 1.4),
    ("Philippines", 1.4),
    ("DR Congo", 1.3),
    ("Vietnam", 1.2),
    ("Iran", 1.1),
    ("Turkey", 1.1),
    ("Germany", 1.0),
    ("Rest of world", 31.6),
)


@dataclass(slots=True)
class AgentSpec:
    id: int
    name: str
    country: str
    color: int
    home_x: int
    home_y: int
    cooldown: float

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
    count: int, width: int, height: int, cooldown: float, rng: random.Random
) -> list[AgentSpec]:
    """Dummy agents: a country drawn from the placeholder shares, clustered per country."""
    names, weights = zip(*PLACEHOLDER_COUNTRIES, strict=True)
    centres = {
        c: (rng.randrange(width), rng.randrange(height), 1 + i % (len(PALETTE) - 1))
        for i, c in enumerate(names)
    }
    agents: list[AgentSpec] = []
    for i in range(1, count + 1):
        country = rng.choices(names, weights)[0]
        cx, cy, color = centres[country]
        spread = max(8, min(width, height) // 12)
        agents.append(
            AgentSpec(
                id=i,
                name=f"Agent {i}",
                country=country,
                color=color,
                home_x=min(max(cx + rng.randint(-spread, spread), 0), width - 1),
                home_y=min(max(cy + rng.randint(-spread, spread), 0), height - 1),
                cooldown=cooldown,
            )
        )
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
            if board and rng.random() < 0.7:
                tid = int(str(board[rng.randrange(len(board))]["thread_id"]))
                tools.reply(tid, f"{agent.country} here, we're building near my corner.")
                return f"Replied in thread {tid}."
            tools.post_thread(
                f"{agent.country} claims ({agent.home_x}, {agent.home_y})",
                "Anyone nearby want to coordinate colours?",
            )
            return "Started a thread to claim my area."
        except ToolError as e:
            return f"Post failed: {e}"
