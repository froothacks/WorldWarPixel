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
    res = tools.place_pixels([(2, 0, 1), (3, 0, 1)])
    assert res["placed"] == 1 and res["dropped"] == 1 and tools.budget.pixels == 0
    assert len(events) == 3
    with pytest.raises(ToolError, match="no pixel budget"):
        tools.place_pixels([(4, 0, 1)])


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


def test_posts_must_not_parrot_the_organiser_or_reuse_a_title() -> None:
    board = Board()
    tools = ToolSet(Canvas(8, 8), board, 1, TurnBudget(0, 5), lambda k, p: None, lambda: 0.0)
    tid = tools.post_thread("Hearts everywhere", "Join me at (3, 3).")["thread_id"]
    with pytest.raises(ToolError, match="already exists"):
        tools.post_thread("hearts EVERYWHERE", "me too")
    with pytest.raises(ToolError, match="own words"):
        tools.post_thread("📢 ANNOUNCEMENT: Hearts", "copying")
    with pytest.raises(ToolError, match="own words"):
        tools.reply(tid, "📢 ANNOUNCEMENT: Hearts — paint hearts")
    assert tools.reply(tid, "On my way.")["post_id"]
    assert tools.budget.posts == 3, "rejected posts cost nothing"


def test_duplicate_replies_and_world_news_copy_are_rejected_without_spending_budget() -> None:
    tools, _ = make(TurnBudget(posts=4))
    thread_id = tools.post_thread("A plan", "Hold this corner.")["thread_id"]
    tools.reply(thread_id, "Let's hold the line.")
    with pytest.raises(ToolError, match="already said in this thread"):
        tools.reply(thread_id, "  LET'S  HOLD \n THE   LINE. ")
    with pytest.raises(ToolError, match="do not copy the 🗞 news"):
        tools.reply(thread_id, "🗞 A and B are talking")
    assert tools.budget.posts == 2
