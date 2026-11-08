import json
import logging
from pathlib import Path

import pytest

from worldwarpixel.config import Settings
from worldwarpixel.sim.events import EventLog, open_event_log


def test_append_and_read_in_order() -> None:
    log = EventLog()
    a = log.append("r1", 1.0, "pixel", 3, {"x": 1, "y": 2, "color": 4})
    b = log.append("r1", 2.0, "post", 3, {"body": "hi"})
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
    log.append("b", 9.0, "post", 1, {})
    assert log.count("b") == 6


def test_run_totals_track_appends_without_replacing_config() -> None:
    log = EventLog()
    config = {"width": 16, "height": 16, "agents": 2}
    log.append("r", 2.0, "run.created", None, {"config": config})
    log.append("r", 1.0, "run.created", None, {"config": {"width": 32}})
    log.append("r", 3.0, "pixel", 1, {})
    log.append("r", 4.0, "post", 1, {})
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
    )
    assert before_status == [expected]
    assert log.run_totals() == before_status
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

    class FakePostgres:
        def __init__(self, database_url: str) -> None:
            self.database_url = database_url

    monkeypatch.setattr(events_pg, "PostgresEventLog", FakePostgres)
    log = open_event_log(Settings(database_url="postgresql://example"))
    assert getattr(log, "database_url", None) == "postgresql://example"
