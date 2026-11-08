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


def test_batch_defers_the_commit_until_the_block_ends() -> None:
    log = EventLog()
    with log.batch():
        for i in range(5):
            log.append("b", float(i), "pixel", 1, {"x": i, "y": 0, "color": 1})
    assert log.count("b") == 5
    log.append("b", 9.0, "post", 1, {})
    assert log.count("b") == 6


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
