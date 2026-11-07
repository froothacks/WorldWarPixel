from worldwarpixel.sim.agents import AgentSpec
from worldwarpixel.sim.board import Board
from worldwarpixel.sim.canvas import Canvas, PixelWrite
from worldwarpixel.sim.dispatch import (
    DISPATCH_ID,
    Dispatcher,
)
from worldwarpixel.sim.world import World


def agents() -> list[AgentSpec]:
    return [
        AgentSpec(1, "A One", "A", 2, 0, 0, 1.0),
        AgentSpec(2, "B Two", "B", 3, 1, 0, 1.0),
        AgentSpec(3, "C Three", "C", 4, 2, 0, 1.0),
    ]


def make(
    *,
    interval: float = 30.0,
    window: float = 120.0,
    canvas: Canvas | None = None,
) -> tuple[Board, World, Canvas, Dispatcher]:
    board = Board()
    world = World(agents(), board)
    canvas = canvas or Canvas(16, 1)
    dispatcher = Dispatcher(board, world, canvas, agents(), lambda _post: None, interval, window)
    return board, world, canvas, dispatcher


def test_war_posts_and_replies_in_the_same_thread() -> None:
    board, world, _, dispatcher = make()
    for _ in range(50):
        world.observe_pixel(1, 2, 4, 0)
    [post] = dispatcher.tick(30)
    assert post.author_id == DISPATCH_ID
    assert post.title == "⚔️ A vs B"
    assert (
        post.body
        == "A painted over 50 px of B and B 0 px of A in the last 30 s, most recently near (4, 0)."
    )
    for _ in range(50):
        world.observe_pixel(1, 2, 5, 0)
    [reply] = dispatcher.tick(60)
    assert reply.title is None and reply.thread_id == post.thread_id
    assert len(board.threads()) == 1 and board.reply_count(post.thread_id) == 1


def test_war_below_threshold_is_ignored() -> None:
    _, world, _, dispatcher = make()
    for _ in range(49):
        world.observe_pixel(1, 2)
    assert dispatcher.tick(30) == []


def test_talk_dispatch_uses_the_highest_cross_country_delta() -> None:
    board, world, _, dispatcher = make()
    thread = board.post_thread(2, "Border talk", "Can we coordinate?", 1)
    world.observe_post(thread.to_dict())
    for i in range(3):
        reply = board.reply(1, thread.id, f"@B reply {i}", i + 2)
        world.observe_post(reply.to_dict())
    [post] = dispatcher.tick(30)
    assert post.title == "🤝 A and B are talking"
    assert post.body == "3 replies and @mentions between A and B in the last 30 s."


def test_territory_dispatch_requires_a_leader_change() -> None:
    board, _, canvas, dispatcher = make()
    for x in range(9):
        canvas.write(PixelWrite(x, 0, 2, 1))
    for x in range(9, 16):
        canvas.write(PixelWrite(x, 0, 3, 2))
    assert dispatcher.tick(0) == []
    for x in (0, 1):
        canvas.write(PixelWrite(x, 0, 3, 2))
    [post] = dispatcher.tick(30)
    assert post.title == "👑 Territory race"
    assert post.body == "B now holds the most of the canvas (56%), overtaking A."
    assert board.threads() == [post]


def test_interval_zero_disables_scheduler() -> None:
    _, world, _, dispatcher = make(interval=0)
    for _ in range(60):
        world.observe_pixel(1, 2)
    assert dispatcher.maybe_tick(0) == []
    assert dispatcher.maybe_tick(100) == []


def test_reset_threads_starts_new_threads_after_board_clear() -> None:
    board, world, _, dispatcher = make()
    for _ in range(50):
        world.observe_pixel(1, 2)
    [first] = dispatcher.tick(30)
    board.clear()
    dispatcher.reset_threads()
    for _ in range(50):
        world.observe_pixel(1, 2)
    [second] = dispatcher.tick(60)
    assert second.id > first.id and second.title == first.title
    assert len(board.threads()) == 1


def test_feed_entries_keep_recent_dispatch_metadata_and_limit_to_three() -> None:
    board, world, canvas, dispatcher = make(window=200)
    for x in range(9):
        canvas.write(PixelWrite(x, 0, 2, 1))
    for x in range(9, 16):
        canvas.write(PixelWrite(x, 0, 3, 2))
    dispatcher.tick(0)

    for _ in range(50):
        world.observe_pixel(1, 2, 4, 0)
    dispatcher.tick(30)

    thread = board.post_thread(3, "Conversation", "Coordinate?", 31)
    world.observe_post(thread.to_dict())
    for i in range(3):
        reply = board.reply(1, thread.id, f"@C reply {i}", 32 + i)
        world.observe_post(reply.to_dict())
    dispatcher.tick(60)

    for x in (0, 1):
        canvas.write(PixelWrite(x, 0, 3, 2))
    dispatcher.tick(90)

    dispatcher.tick(120)

    entries = dispatcher.feed_entries(120)
    assert len(entries) == 3
    assert [entry["kind"] for entry in entries] == ["territory", "talk", "war"]
    assert all(entry["tag"] == "dispatch" for entry in entries)
    assert entries[0]["countries"] == ["B", "A"]
    assert entries[1]["countries"] == ["A", "C"]
    assert entries[2]["countries"] == ["A", "B"]
    assert entries[0]["latest"]["author"] == "World News"
    assert dispatcher.feed_entries(321) == []
