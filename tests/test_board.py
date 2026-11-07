import pytest

from worldwarpixel.sim.board import Board, BoardError


def test_threads_and_replies() -> None:
    b = Board()
    root = b.post_thread(1, "Sun in the corner", "Who's in?", now=1.0)
    r1 = b.reply(2, root.id, "Me", now=2.0)
    r2 = b.reply(3, root.id, "Also me", now=3.0, parent_id=r1.id)
    assert [p.id for p in b.thread(root.id)] == [root.id, r1.id, r2.id]
    assert r2.parent_id == r1.id and r2.thread_id == root.id
    assert b.reply_count(root.id) == 2
    assert b.thread(999) == []


def test_digest_orders_by_latest_activity() -> None:
    b = Board()
    old = b.post_thread(1, "old", "x", now=1.0)
    new = b.post_thread(2, "new", "y", now=2.0)
    b.reply(3, old.id, "bump", now=3.0)
    digest = b.digest(name_of=lambda i: f"a{i}")
    assert [d["thread_id"] for d in digest] == [old.id, new.id]
    assert digest[0]["latest"] == {"author": "a3", "body": "bump"}
    assert digest[1]["latest"] is None


def test_validation() -> None:
    b = Board(max_body=5)
    with pytest.raises(BoardError):
        b.post_thread(1, "", "body", now=0)
    with pytest.raises(BoardError):
        b.post_thread(1, "t", "too long", now=0)
    with pytest.raises(BoardError):
        b.reply(1, 42, "hi", now=0)
    root = b.post_thread(1, "t", "ok", now=0)
    other = b.post_thread(1, "t2", "ok", now=0)
    with pytest.raises(BoardError):
        b.reply(1, root.id, "hi", now=0, parent_id=other.id)
    assert b.mentions_in("hey @Agent_3 and @agent.4") == ["Agent_3", "agent.4"]
