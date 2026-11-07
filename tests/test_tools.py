from collections.abc import Callable
from typing import Any

import pytest

from worldwarpixel.sim.board import Board
from worldwarpixel.sim.canvas import Canvas, PixelWrite
from worldwarpixel.sim.tools import ToolError, ToolSet, TurnBudget


def make(
    budget: TurnBudget | None = None,
    blocked: Callable[[int, int], bool] | None = None,
) -> tuple[ToolSet, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    tools = ToolSet(
        Canvas(32, 32),
        Board(),
        agent_id=1,
        budget=budget or TurnBudget(),
        sink=lambda k, p: events.append((k, p)),
        clock=lambda: 123.0,
        blocked=blocked,
    )
    return tools, events


def test_pixel_budget_is_enforced_per_turn() -> None:
    tools, events = make(TurnBudget(pixels=3))
    assert tools.place_pixels([(0, 0, 1), (1, 0, 1)]) == {
        "placed": 2,
        "pixels_left": 1,
        "last": [1, 0],
    }
    res = tools.place_pixels([(2, 0, 1), (3, 0, 1)])
    assert res["placed"] == 1 and res["dropped"] == 1 and tools.budget.pixels == 0
    assert len(events) == 3
    with pytest.raises(ToolError, match="no pixel budget"):
        tools.place_pixels([(4, 0, 1)])


def test_blocked_pixels_do_not_spend_budget_and_join_dropped_note() -> None:
    tools, events = make(TurnBudget(pixels=2), blocked=lambda x, _: x == 0)
    out = tools.place_pixels([(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)])
    assert out == {
        "placed": 2,
        "pixels_left": 0,
        "last": [2, 0],
        "blocked": 1,
        "dropped": 1,
        "note": (
            "1 pixels are inside a region held by the debate winner; they were not placed "
            "only 2 fit the budget; 1 pixels were dropped"
        ),
    }
    assert [event[1]["x"] for event in events] == [1, 2]

    all_blocked, no_events = make(TurnBudget(pixels=2), blocked=lambda _x, _y: True)
    blocked_out = all_blocked.place_pixels([(0, 0, 1), (1, 0, 1)])
    assert blocked_out["placed"] == 0 and blocked_out["blocked"] == 2
    assert blocked_out["pixels_left"] == 2 and no_events == []


def test_fill_rect_charges_every_cell() -> None:
    tools, events = make(TurnBudget(pixels=8))
    tools.fill_rect(0, 0, 2, 2, 5)
    assert [e[1] for e in events] == [
        {"x": 0, "y": 0, "color": 5, "prev_owner": 0},
        {"x": 1, "y": 0, "color": 5, "prev_owner": 0},
        {"x": 0, "y": 1, "color": 5, "prev_owner": 0},
        {"x": 1, "y": 1, "color": 5, "prev_owner": 0},
    ]
    with pytest.raises(ToolError, match="budget exceeded"):
        tools.fill_rect(0, 0, 3, 3, 5)
    with pytest.raises(ToolError, match="off the canvas"):
        tools.fill_rect(31, 31, 2, 1, 5)
    with pytest.raises(ToolError, match="budget exceeded"):
        tools.fill_rect(0, 0, 32, 32, 5)


def test_fill_rect_checks_budget_before_building_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    tools, _ = make(TurnBudget(pixels=8))
    monkeypatch.setattr(tools._canvas, "in_bounds", lambda _x, _y: True)

    with pytest.raises(ToolError, match="budget"):
        tools.fill_rect(0, 0, 4096, 4096, 1)


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


def test_stencil_skips_pixels_already_queued_for_painting() -> None:
    canvas = Canvas(16, 16)
    queued: list[PixelWrite] = []
    pending: dict[tuple[int, int], tuple[int, int]] = {}

    def writer(writes: list[PixelWrite]) -> None:
        for w in writes:
            queued.append(w)
            pending[(w.x, w.y)] = (w.color, w.agent_id)

    def tools() -> ToolSet:
        budget = TurnBudget(2, 1)
        return ToolSet(canvas, Board(), 1, budget, lambda k, p: None, lambda: 0.0, writer, pending)

    stencil = [(0, 0, 3), (1, 0, 3), (2, 0, 3)]
    assert tools().paint_stencil(stencil)["remaining"] == 1
    # Nothing has landed on the canvas yet, but the next turn must not re-select the queued cells.
    assert canvas.get(0, 0) == 0
    out = tools().paint_stencil(stencil)
    assert out["placed"] == 1 and out["done"]
    assert [(w.x, w.y) for w in queued] == [(0, 0), (1, 0), (2, 0)]


def test_blocked_stencil_stays_incomplete_and_keeps_blocked_note() -> None:
    tools, events = make(TurnBudget(pixels=4), blocked=lambda x, _: x < 2)
    out = tools.paint_stencil([(0, 0, 3), (1, 0, 3), (2, 0, 3)])

    assert out["placed"] == 1 and out["blocked"] == 2
    assert out["remaining"] == 2 and not out["done"]
    assert "2 pixels are inside a region held by the debate winner" in out["note"]
    assert "2 px still to paint" in out["note"]
    assert [event[1]["x"] for event in events] == [2]


def test_blocked_stencil_skips_held_cells_to_paint_open_cells() -> None:
    tools, events = make(TurnBudget(pixels=2), blocked=lambda x, _: x < 2)
    out = tools.paint_stencil([(0, 0, 3), (1, 0, 3), (2, 0, 3), (3, 0, 3)])

    assert out["placed"] == 2 and out["blocked"] == 2
    assert out["remaining"] == 2 and not out["done"]
    assert [event[1]["x"] for event in events] == [2, 3]
    assert tools._canvas.get(2, 0) == tools._canvas.get(3, 0) == 3


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
