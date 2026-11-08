import json
import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.events_migrate import migrate
from worldwarpixel.sim.events_pg import PostgresEventLog

PG_URL = os.environ.get("DATABASE_URL_TEST")
pytestmark = pytest.mark.skipif(PG_URL is None, reason="DATABASE_URL_TEST is not set")


@pytest.fixture
def pg_url() -> str:
    assert PG_URL is not None
    PostgresEventLog(PG_URL).close()
    with psycopg.connect(PG_URL) as conn:
        conn.execute("TRUNCATE events RESTART IDENTITY")
    return PG_URL


@pytest.fixture
def pg_log(pg_url: str) -> Iterator[PostgresEventLog]:
    log = PostgresEventLog(pg_url)
    try:
        yield log
    finally:
        log.close()


def test_append_read_count_and_run_ids(pg_log: PostgresEventLog) -> None:
    a = pg_log.append("r1", 1.0, "pixel", 3, {"x": 1, "y": 2, "color": 4})
    b = pg_log.append("r1", 2.0, "post", 3, {"body": "hi"})
    pg_log.append("r2", 3.0, "pixel", 1, {})

    events = pg_log.read("r1")
    assert [e.id for e in events] == [a, b]
    assert events[0].payload == {"x": 1, "y": 2, "color": 4}
    assert events[0].agent_id == 3
    assert pg_log.read("r1", after_id=a) == [events[1]]
    assert pg_log.count("r1") == 2
    assert pg_log.run_ids() == ["r1", "r2"]


def test_batch_is_a_transaction_visible_to_other_connections(pg_log: PostgresEventLog) -> None:
    reader = pg_log.reader()
    try:
        assert reader is not pg_log
        with pg_log.batch():
            pg_log.append("b", 1.0, "pixel", 1, {"x": 0, "y": 0, "color": 1})
            assert reader.count("b") == 0
        assert reader.count("b") == 1
    finally:
        reader.close()


def test_run_totals_match_the_sqlite_shape(pg_log: PostgresEventLog) -> None:
    config = {"width": 16, "height": 16, "agents": 2}
    pg_log.append("r", 1.0, "run.created", None, {"config": config, "agents": []})
    pg_log.append("r", 2.0, "pixel", 1, {"x": 1, "y": 1, "color": 2})
    pg_log.append("r", 3.0, "post", 1, {"body": "hi"})
    pg_log.append("r", 4.0, "agent.thought", 1, {"thought": "Tool call rejected: bad json"})
    pg_log.append("r", 5.0, "agent.thought", 1, {"thought": "Painting a sun"})
    pg_log.append("r", 6.0, "narrator.update", None, {"kind": "update", "text": "Hi"})
    pg_log.append("r", 7.0, "narrator.update", None, {"kind": "answer", "text": "Yes"})

    [row] = pg_log.run_totals("Tool call rejected")
    assert row[:2] == ("r", 1.0)
    assert row[2:8] == (1, 1, 2, 1, 1, 1)
    assert row[8] is False
    assert row[9] is not None and json.loads(row[9]) == config


def test_migrate_sqlite_to_pg_is_idempotent(pg_url: str, tmp_path: Path) -> None:
    sqlite_path = tmp_path / "events.db"
    source = EventLog(sqlite_path)
    source.append("m", 1.0, "run.created", None, {"config": {"width": 16}})
    source.append("m", 2.0, "pixel", 1, {"x": 1, "y": 2, "color": 3})
    source.close()

    assert migrate(sqlite_path, pg_url) == 2
    assert migrate(sqlite_path, pg_url) == 2

    target = PostgresEventLog(pg_url)
    try:
        assert [e.id for e in target.read("m")] == [1, 2]
        assert target.count("m") == 2
        assert target.append("m", 3.0, "post", 1, {}) == 3
    finally:
        target.close()
