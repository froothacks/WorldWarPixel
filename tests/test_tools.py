from typing import Any

import pytest

from worldwarpixel.sim.board import Board
from worldwarpixel.sim.canvas import Canvas
from worldwarpixel.sim.tools import ToolError, ToolSet, TurnBudget


def make(budget: TurnBudget | None = None) -> tuple[ToolSet, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    tools = ToolSet(
        Canvas(32, 32),
        Board(),
        agent_id=1,
        budget=budget or TurnBudget(),
        sink=lambda k, p: events.append((k, p)),
        clock=lambda: 123.0,
    )
    return tools, events


def test_pixel_budget_is_enforced_per_turn() -> None:
    tools, events = make(TurnBudget(pixels=3))
    assert tools.place_pixels([(0, 0, 1), (1, 0, 1)]) == {"placed": 2, "pixels_left": 1}
    with pytest.raises(ToolError, match="budget exceeded"):
        tools.place_pixels([(2, 0, 1), (3, 0, 1)])
    assert len(events) == 2
    tools.place_pixels([(2, 0, 1)])
    assert tools.budget.pixels == 0


def test_fill_rect_charges_every_cell() -> None:
    tools, events = make(TurnBudget(pixels=8))
    tools.fill_rect(0, 0, 2, 2, 5)
    assert [e[1] for e in events] == [
        {"x": 0, "y": 0, "color": 5},
        {"x": 1, "y": 0, "color": 5},
        {"x": 0, "y": 1, "color": 5},
        {"x": 1, "y": 1, "color": 5},
    ]
    with pytest.raises(ToolError, match="budget exceeded"):
        tools.fill_rect(0, 0, 3, 3, 5)
    with pytest.raises(ToolError, match="off the canvas"):
        tools.fill_rect(31, 31, 2, 1, 5)


def test_invalid_pixels_are_rejected_before_writing() -> None:
    tools, events = make()
    with pytest.raises(ToolError, match="off the canvas"):
        tools.place_pixels([(0, 0, 1), (40, 0, 1)])
    with pytest.raises(ToolError, match="color"):
        tools.place_pixels([(0, 0, 16)])
    assert events == []


def test_view_limits() -> None:
    tools, _ = make(TurnBudget(views=1))
    assert tools.view_canvas(0, 0, 2, 1) == {"x": 0, "y": 0, "rows": ["00"]}
    with pytest.raises(ToolError, match="budget exceeded"):
        tools.view_canvas(0, 0, 2, 1)
    with pytest.raises(ToolError, match="view size"):
        tools.view_canvas(0, 0, 65, 1)


def test_posting_shares_one_budget_and_wraps_board_errors() -> None:
    tools, events = make(TurnBudget(posts=2))
    res = tools.post_thread("Hello", "World")
    assert res["thread_id"] == 1
    assert events[-1][0] == "post" and events[-1][1]["created_at"] == 123.0
    with pytest.raises(ToolError, match="does not exist"):
        tools.reply(99, "hi")
    tools.reply(1, "hi")
    with pytest.raises(ToolError, match="budget exceeded"):
        tools.reply(1, "again")
    assert len(tools.read_thread(1)["posts"]) == 2
    with pytest.raises(ToolError, match="does not exist"):
        tools.read_thread(5)
