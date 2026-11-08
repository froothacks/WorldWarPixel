import json
import logging
from pathlib import Path

import pytest

from worldwarpixel.config import Settings
from worldwarpixel.sim.board import Post
from worldwarpixel.sim.events import EventLog, RunMeta, open_event_log


def test_append_and_read_in_order() -> None:
    log = EventLog()
    a = log.append("r1", 1.0, "pixel", 3, {"x": 1, "y": 2, "color": 4})
    b = log.append(
        "r1",
        2.0,
        "post",
        3,
        {
            "id": 1,
            "thread_id": 1,
            "parent_id": None,
            "author_id": 3,
            "title": "hello",
            "body": "hi",
            "created_at": 2.0,
        },
    )
    log.append("r2", 3.0, "pixel", 1, {})
    events = log.read("r1")
    assert [e.id for e in events] == [a, b]
    assert events[0].payload == {"x": 1, "y": 2, "color": 4}
    assert events[0].agent_id == 3
    assert log.read("r1", after_id=a) == [events[1]]
    assert log.count("r1") == 2
    assert log.run_ids() == ["r1", "r2"]
    log.close()


def test_slow_event_read_warning_obeys_threshold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    log = EventLog()
    log.append("r", 1.0, "pixel", 1, {"x": 1, "y": 2, "color": 3})
    logger_name = "worldwarpixel.sim.events"

    monkeypatch.setattr("worldwarpixel.sim.events.SLOW_READ_SECONDS", 0)
    with caplog.at_level(logging.WARNING, logger=logger_name):
        assert len(log.read("r")) == 1
    assert any(
        record.name == logger_name and "slow event read run=r" in record.getMessage()
        for record in caplog.records
    )

    caplog.clear()
    monkeypatch.setattr("worldwarpixel.sim.events.SLOW_READ_SECONDS", float("inf"))
    with caplog.at_level(logging.WARNING, logger=logger_name):
        log.read("r")
    assert not caplog.records
    log.close()


def test_batch_defers_the_commit_until_the_block_ends() -> None:
    log = EventLog()
    with log.batch():
        for i in range(5):
            log.append("b", float(i), "pixel", 1, {"x": i, "y": 0, "color": 1})
    assert log.count("b") == 5
    log.append(
        "b",
        9.0,
        "post",
        1,
        {
            "id": 1,
            "thread_id": 1,
            "parent_id": None,
            "author_id": 1,
            "title": "t",
            "body": "body",
            "created_at": 9.0,
        },
    )
    assert log.count("b") == 6


def test_run_totals_track_appends_without_replacing_config() -> None:
    log = EventLog()
    config = {"width": 16, "height": 16, "agents": 2}
    log.append("r", 2.0, "run.created", None, {"config": config})
    log.append("r", 1.0, "run.created", None, {"config": {"width": 32}})
    log.append("r", 3.0, "pixel", 1, {})
    log.append(
        "r",
        4.0,
        "post",
        1,
        {
            "id": 1,
            "thread_id": 1,
            "parent_id": None,
            "author_id": 1,
            "title": "t",
            "body": "body",
            "created_at": 4.0,
        },
    )
    log.append("r", 5.0, "agent.thought", 1, {"thought": "Tool call rejected: bad input"})
    log.append("r", 6.0, "agent.thought", 1, {"thought": "Painting a sun"})
    log.append("r", 7.0, "narrator.update", None, {"kind": "update"})
    log.append("r", 8.0, "narrator.update", None, {"kind": "answer"})
    log.append("r", 9.0, "run.deleted", None, {})
    before_status = log.run_totals()
    log.append("r", 10.0, "agent.status", 1, {"status": "idle"})

    expected = (
        "r",
        1.0,
        1,
        1,
        2,
        1,
        1,
        1,
        True,
        json.dumps(config, separators=(",", ":")),
        9.0,
    )
    assert before_status == [expected]
    assert log.run_totals() == before_status
    log.close()


def test_run_last_ts_tracks_max_counted_event_timestamp() -> None:
    log = EventLog()
    log.append("r", 2.0, "run.created", None, {"config": {"width": 16}})
    log.append("r", 8.0, "pixel", 1, {})
    log.append("r", 5.0, "pixel", 1, {})

    [row] = log.run_totals()
    assert row[-1] == 8.0
    log.close()


def test_uncounted_only_run_gets_last_ts_from_its_first_upsert() -> None:
    log = EventLog()
    log.append("r", 1.0, "agent.status", 1, {"status": "idle"})
    log.append("r", 2.0, "agent.status", 1, {"status": "thinking"})

    [row] = log.run_totals()
    assert row[-1] == 1.0
    log.close()


def test_post_and_vote_projections_follow_appends_and_board_clears() -> None:
    log = EventLog()
    first = {
        "id": 1,
        "thread_id": 1,
        "parent_id": None,
        "author_id": 1,
        "title": "first",
        "body": "root",
        "created_at": 1.0,
        "society": 2,
    }
    reply = {
        "id": 2,
        "thread_id": 1,
        "parent_id": 1,
        "author_id": 2,
        "title": None,
        "body": "reply",
        "created_at": 2.0,
    }
    log.append("r", 1.0, "post", 1, first)
    log.append("r", 2.0, "post", 2, reply)
    log.append("r", 3.0, "board.vote", None, {"thread_id": 1, "value": 1, "votes": 1})
    log.append("r", 4.0, "board.vote", 2, {"thread_id": 1, "value": -1, "votes": 0})
    assert log.posts("r") == [
        (Post(1, 1, 1, None, "first", "root", 1.0), 2),
        (Post(2, 2, 1, 1, None, "reply", 2.0), None),
    ]
    assert [post.id for post, _ in log.posts("r", thread_id=1)] == [1, 2]
    assert log.thread_votes("r") == {1: 0}

    log.append("r", 5.0, "god.clear_board", None, {})
    assert log.posts("r") == []
    assert log.thread_votes("r") == {}
    after_clear = {
        "id": 3,
        "thread_id": 3,
        "parent_id": None,
        "author_id": 1,
        "title": "after",
        "body": "new root",
        "created_at": 6.0,
    }
    log.append("r", 6.0, "post", 1, after_clear)
    log.append("r", 7.0, "board.vote", 1, {"thread_id": 3, "value": 1, "votes": 1})
    assert [post.id for post, _ in log.posts("r")] == [3]
    assert log.thread_votes("r") == {3: 1}
    log.close()


def test_run_meta_and_canvas_only_reads() -> None:
    log = EventLog()
    config = {"width": 16, "height": 12}
    log.append("r", 1.0, "run.created", None, {"config": config})
    assert log.run_meta("r") == RunMeta(1.0, json.dumps(config, separators=(",", ":")), False)
    assert log.run_meta("missing") is None
    log.append("r", 2.0, "pixel", 1, {"x": 0, "y": 0, "color": 1})
    log.append("r", 3.0, "agent.status", 1, {})
    log.append("r", 4.0, "god.clear_canvas", None, {})
    log.append(
        "r",
        5.0,
        "post",
        1,
        {
            "id": 1,
            "thread_id": 1,
            "parent_id": None,
            "author_id": 1,
            "title": "t",
            "body": "body",
            "created_at": 5.0,
        },
    )
    log.append("r", 6.0, "pixel", 1, {"x": 1, "y": 0, "color": 2})

    first = log.read("r", limit=1, canvas_only=True)
    second = log.read("r", after_id=first[-1].id, limit=1, canvas_only=True)
    third = log.read("r", after_id=second[-1].id, limit=1, canvas_only=True)
    assert [event.kind for event in first + second + third] == [
        "pixel",
        "god.clear_canvas",
        "pixel",
    ]
    assert log.read("r", after_id=third[-1].id, canvas_only=True) == []
    plan = log._conn.execute(
        "EXPLAIN QUERY PLAN SELECT id, run_id, ts, kind, agent_id, payload FROM events"
        " WHERE run_id = ? AND id > ? AND kind IN ('pixel', 'god.clear_canvas')"
        " ORDER BY run_id, id LIMIT ?",
        ("r", 0, 10),
    ).fetchall()
    assert any("events_run_canvas" in str(row[3]) for row in plan)

    log.append("r", 7.0, "run.deleted", None, {})
    assert log.run_meta("r") == RunMeta(1.0, json.dumps(config, separators=(",", ":")), True)
    log.close()


def test_canvas_span_and_events_extract_canvas_rows() -> None:
    log = EventLog()
    log.append("r", 1.0, "run.created", None, {"config": {"width": 4, "height": 2}})
    pixel_id = log.append("r", 2.0, "pixel", 1, {"x": 1, "y": 0, "color": 3})
    log.append("r", 3.0, "agent.status", 1, {"status": "idle"})
    clear_id = log.append("r", 4.0, "god.clear_canvas", None, {})
    last_id = log.append("r", 5.0, "pixel", 2, {"x": 2, "y": 1, "color": 7})
    log.append("other", 6.0, "pixel", 1, {"x": 0, "y": 0, "color": 1})

    assert log.canvas_span("r") == (2.0, 5.0, last_id)
    assert log.canvas_span("missing") is None
    assert log.canvas_events("r", 0, 1) == [(pixel_id, 2.0, "pixel", 1, 0, 3)]
    assert log.canvas_events("r", pixel_id, 1) == [
        (clear_id, 4.0, "god.clear_canvas", None, None, None)
    ]
    assert log.canvas_events("r", clear_id, 10) == [(last_id, 5.0, "pixel", 2, 1, 7)]
    log.close()


def test_run_totals_read_the_runs_table() -> None:
    log = EventLog()
    log.append("r", 1.0, "run.created", None, {"config": {"width": 16}})
    log.append("r", 2.0, "pixel", 1, {})
    expected = log.run_totals()

    log._conn.execute("DELETE FROM events")
    log._conn.commit()

    assert log.run_totals() == expected
    log.close()


def test_open_event_log_uses_sqlite_by_default(tmp_path: Path) -> None:
    log = open_event_log(Settings(event_log_path=str(tmp_path / "events.db"), database_url=None))
    assert isinstance(log, EventLog)
    log.close()


def test_open_event_log_uses_postgres_when_database_url_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldwarpixel.sim import events_pg

    class FakeBuffered:
        def __init__(self, database_url: str, **kwargs: object) -> None:
            self.database_url = database_url
            self.kwargs = kwargs

    monkeypatch.setattr(events_pg, "BufferedPostgresEventLog", FakeBuffered)
    log = open_event_log(Settings(database_url="postgresql://example"))
    assert getattr(log, "database_url", None) == "postgresql://example"
    assert getattr(log, "kwargs", {}).get("flush_interval") == 0.1
