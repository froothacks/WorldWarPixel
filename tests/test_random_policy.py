import random
from typing import Any

from worldwarpixel.sim.agents import AgentSpec, RandomWalkPolicy
from worldwarpixel.sim.board import Board
from worldwarpixel.sim.canvas import Canvas
from worldwarpixel.sim.dispatch import DISPATCH_ID
from worldwarpixel.sim.tools import ToolSet, TurnBudget


def make_tools(board: Board) -> tuple[ToolSet, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    tools = ToolSet(
        Canvas(16, 16),
        board,
        1,
        TurnBudget(posts=1),
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

    thought = RandomWalkPolicy._post(agent, tools, [entry], random.Random(1))

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

    RandomWalkPolicy._post(agent, tools, [entry], random.Random(42))

    assert board.reply_count(thread.id) == 0
    assert events == []
