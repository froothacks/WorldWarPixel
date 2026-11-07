import asyncio
import random
from typing import Any

from worldwarpixel.sim.agents import AgentSpec, Observation, RandomWalkPolicy
from worldwarpixel.sim.board import Board
from worldwarpixel.sim.canvas import Canvas
from worldwarpixel.sim.dispatch import DISPATCH_ID
from worldwarpixel.sim.tools import ToolSet, TurnBudget


def make_tools(board: Board, posts: int = 1) -> tuple[ToolSet, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    tools = ToolSet(
        Canvas(16, 16),
        board,
        1,
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
