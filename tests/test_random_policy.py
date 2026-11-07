import asyncio
import random
from typing import Any

import pytest

from worldwarpixel.sim.agents import AgentSpec, Observation, RandomWalkPolicy
from worldwarpixel.sim.board import Board
from worldwarpixel.sim.canvas import Canvas
from worldwarpixel.sim.dispatch import DISPATCH_ID
from worldwarpixel.sim.tools import ToolSet, TurnBudget


def make_tools(
    board: Board, posts: int = 1, agent_id: int = 1
) -> tuple[ToolSet, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    tools = ToolSet(
        Canvas(16, 16),
        board,
        agent_id,
        TurnBudget(posts=posts),
        lambda kind, payload: events.append((kind, payload)),
        lambda: 10.0,
    )
    return tools, events


def test_random_policy_replies_to_a_war_dispatch_about_its_country() -> None:
    board = Board()
    thread = board.post_thread(DISPATCH_ID, "⚔️ India vs Japan", "India raided Japan.", 1.0)
    tools, events = make_tools(board)
    agent = AgentSpec(1, "Priya Raman", "India", 2, 4, 5, 1.0)
    entry = {
        "thread_id": thread.id,
        "tag": "dispatch",
        "kind": "war",
        "countries": ["India", "Japan"],
    }

    policy = RandomWalkPolicy(seed=1)
    thought = policy._post(agent, tools, [entry], random.Random(1))

    assert thought == f"Replied in thread {thread.id}."
    assert board.reply_count(thread.id) == 1
    assert events[0][1]["body"].endswith("near (4, 5).")


def test_random_policy_does_not_reply_to_an_unrelated_dispatch() -> None:
    board = Board()
    thread = board.post_thread(DISPATCH_ID, "⚔️ Japan vs Brazil", "Fighting elsewhere.", 1.0)
    tools, events = make_tools(board)
    agent = AgentSpec(1, "Priya Raman", "India", 2, 4, 5, 1.0)
    entry = {
        "thread_id": thread.id,
        "tag": "dispatch",
        "kind": "war",
        "countries": ["Japan", "Brazil"],
    }

    RandomWalkPolicy(seed=42)._post(agent, tools, [entry], random.Random(42))

    assert board.reply_count(thread.id) == 0
    assert events == []


def test_random_policy_replies_to_a_dispatch_only_once() -> None:
    board = Board()
    thread = board.post_thread(DISPATCH_ID, "⚔️ India vs Japan", "India raided Japan.", 1.0)
    tools, _ = make_tools(board, posts=2)
    agent = AgentSpec(1, "Priya Raman", "India", 2, 4, 5, 1.0)
    entry = {
        "thread_id": thread.id,
        "tag": "dispatch",
        "kind": "war",
        "countries": ["India", "Japan"],
    }
    policy = RandomWalkPolicy(seed=1)

    policy._post(agent, tools, [entry], random.Random(1))
    policy._post(agent, tools, [entry], random.Random(42))

    assert board.reply_count(thread.id) == 1
    assert len(board.threads()) == 1


def test_random_policy_does_not_reply_to_a_mention() -> None:
    board = Board()
    thread = board.post_thread(2, "A note for India", "India, coordinate with us.", 1.0)
    tools, _ = make_tools(board)
    agent = AgentSpec(1, "Priya Raman", "India", 2, 4, 5, 1.0)
    obs = Observation(
        elapsed=1.0,
        budget=tools.budget,
        canvas_width=16,
        canvas_height=16,
        board=[{"thread_id": thread.id, "tag": "mention"}],
    )

    asyncio.run(RandomWalkPolicy(seed=1, post_chance=1.0).act(agent, tools, obs))

    assert board.reply_count(thread.id) == 0


def test_random_policy_claims_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    board = Board()
    agent = AgentSpec(1, "Priya Raman", "India", 2, 4, 5, 1.0)
    policy = RandomWalkPolicy(seed=1, post_chance=1.0)
    monkeypatch.setattr(policy._rng, "random", lambda: 0.0)

    for _ in range(2):
        tools, _ = make_tools(board, posts=1, agent_id=agent.id)
        obs = Observation(1.0, tools.budget, 16, 16, [])
        asyncio.run(policy.act(agent, tools, obs))

    claims = [thread for thread in board.threads() if thread.title and "claims" in thread.title]
    assert len(claims) == 1


def test_random_policy_claim_limit_is_per_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    board = Board()
    policy = RandomWalkPolicy(seed=1, post_chance=1.0)
    monkeypatch.setattr(policy._rng, "random", lambda: 0.0)
    agents = [
        AgentSpec(1, "Priya Raman", "India", 2, 4, 5, 1.0),
        AgentSpec(2, "Yuki Sato", "Japan", 3, 8, 9, 1.0),
    ]

    for agent in agents:
        tools, _ = make_tools(board, posts=1, agent_id=agent.id)
        obs = Observation(1.0, tools.budget, 16, 16, [])
        asyncio.run(policy.act(agent, tools, obs))

    claims = [thread for thread in board.threads() if thread.title and "claims" in thread.title]
    assert {
        thread.title.split(" claims ", maxsplit=1)[0]
        for thread in claims
        if thread.title is not None
    } == {"India", "Japan"}
