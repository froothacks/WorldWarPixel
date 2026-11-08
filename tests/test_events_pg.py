import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest

from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.events_migrate import migrate
from worldwarpixel.sim.events_pg import BufferedPostgresEventLog, PostgresEventLog

PG_URL = os.environ.get("DATABASE_URL_TEST")
pytestmark = pytest.mark.skipif(PG_URL is None, reason="DATABASE_URL_TEST is not set")


@pytest.fixture
def pg_url() -> str:
    assert PG_URL is not None
    with psycopg.connect(PG_URL) as conn:
        conn.execute("DROP TABLE IF EXISTS events, runs, schema_migrations CASCADE")
    PostgresEventLog(PG_URL).close()
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


def test_batch_commits_after_reads_without_a_later_append(pg_url: str) -> None:
    log = PostgresEventLog(pg_url)
    try:
        log.append("r", 1.0, "run.created", None, {})
        log.count("r")
        with log.batch():
            log.append("r", 2.0, "pixel", 1, {})
    finally:
        log.close()

    check = PostgresEventLog(pg_url)
    try:
        assert check.count("r") == 2
    finally:
        check.close()


def test_run_totals_match_the_sqlite_shape(pg_log: PostgresEventLog) -> None:
    config = {"width": 16, "height": 16, "agents": 2}
    pg_log.append("r", 1.0, "run.created", None, {"config": config, "agents": []})
    pg_log.append("r", 2.0, "pixel", 1, {"x": 1, "y": 1, "color": 2})
    pg_log.append("r", 3.0, "post", 1, {"body": "hi"})
    pg_log.append("r", 4.0, "agent.thought", 1, {"thought": "Tool call rejected: bad json"})
    pg_log.append("r", 5.0, "agent.thought", 1, {"thought": "Painting a sun"})
    pg_log.append("r", 6.0, "narrator.update", None, {"kind": "update", "text": "Hi"})
    pg_log.append("r", 7.0, "narrator.update", None, {"kind": "answer", "text": "Yes"})

    [row] = pg_log.run_totals()
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
        [row] = target.run_totals()
        assert row[:9] == ("m", 1.0, 1, 0, 0, 0, 0, 0, False)
        assert row[9] is not None and json.loads(row[9]) == {"width": 16}
        assert target.append("m", 3.0, "post", 1, {}) == 3
    finally:
        target.close()


def test_migrate_sqlite_to_pg_rejects_conflicting_ids(pg_url: str, tmp_path: Path) -> None:
    target = PostgresEventLog(pg_url)
    try:
        target.append("pg", 9.0, "run.created", None, {"config": {"width": 8}})
    finally:
        target.close()

    sqlite_path = tmp_path / "events.db"
    source = EventLog(sqlite_path)
    source.append("sqlite", 1.0, "run.created", None, {"config": {"width": 16}})
    source.close()

    with pytest.raises(ValueError, match="already exists in Postgres"):
        migrate(sqlite_path, pg_url)

    check = PostgresEventLog(pg_url)
    try:
        assert check.count("pg") == 1
        assert check.count("sqlite") == 0
    finally:
        check.close()


def test_events_reject_unknown_run_ids(pg_url: str) -> None:
    with psycopg.connect(pg_url) as conn, pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute(
            "INSERT INTO events (run_id, ts, kind, payload) VALUES (%s, %s, %s, %s)",
            ("unknown", 1.0, "pixel", "{}"),
        )


def test_legacy_postgres_events_are_backfilled_and_constrained(pg_url: str) -> None:
    config = {"width": 16, "height": 16, "agents": 2}
    with psycopg.connect(pg_url) as conn:
        conn.execute("DROP TABLE IF EXISTS events, runs, schema_migrations CASCADE")
        conn.execute(
            """
            CREATE TABLE events (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                run_id TEXT NOT NULL,
                ts DOUBLE PRECISION NOT NULL,
                kind TEXT NOT NULL,
                agent_id INTEGER,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX events_run ON events (run_id, id)")
        rows = [
            ("legacy", 1.0, "run.created", None, {"config": config}),
            ("legacy", 2.0, "pixel", 1, {}),
            ("legacy", 3.0, "agent.thought", 1, {"thought": "Tool call rejected: bad input"}),
            ("legacy", 4.0, "narrator.update", None, {"kind": "answer"}),
        ]
        for run_id, ts, kind, agent_id, payload in rows:
            conn.execute(
                "INSERT INTO events (run_id, ts, kind, agent_id, payload)"
                " VALUES (%s, %s, %s, %s, %s)",
                (run_id, ts, kind, agent_id, json.dumps(payload)),
            )

    log = PostgresEventLog(pg_url)
    try:
        [row] = log.run_totals()
        assert row[:9] == ("legacy", 1.0, 1, 0, 1, 1, 0, 1, False)
        assert row[9] is not None and json.loads(row[9]) == config
    finally:
        log.close()

    with psycopg.connect(pg_url) as conn:
        [constraint] = conn.execute(
            "SELECT condeferrable, condeferred FROM pg_constraint WHERE conname = 'events_run_fk'"
        ).fetchall()
    assert constraint == (True, False)


def test_concurrent_postgres_migrations_are_applied_once(pg_url: str) -> None:
    with psycopg.connect(pg_url) as conn:
        conn.execute("DROP TABLE IF EXISTS events, runs, schema_migrations CASCADE")

    def construct_and_close(_: int) -> list[str]:
        log = PostgresEventLog(pg_url)
        try:
            return log.run_ids()
        finally:
            log.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(construct_and_close, range(2))) == [[], []]

    with psycopg.connect(pg_url) as conn:
        assert conn.execute(
            "SELECT version, COUNT(*) FROM schema_migrations GROUP BY version ORDER BY version"
        ).fetchall() == [(1, 1), (2, 1), (3, 1)]


def test_buffered_writes_flush_in_order(pg_url: str) -> None:
    log = BufferedPostgresEventLog(pg_url, flush_interval=0.05, batch_size=3)
    for i in range(7):
        log.append("b", float(i), "pixel", 1, {"x": i})
    log.close()

    check = PostgresEventLog(pg_url)
    try:
        events = check.read("b")
        assert [e.payload["x"] for e in events] == list(range(7))
        assert [e.id for e in events] == sorted(e.id for e in events)
    finally:
        check.close()


def test_buffered_batch_still_flushes_in_order(pg_url: str) -> None:
    log = BufferedPostgresEventLog(pg_url, flush_interval=0.05, batch_size=100)
    with log.batch():
        for i in range(4):
            log.append("c", float(i), "post", 2, {"body": str(i)})
    log.close()

    check = PostgresEventLog(pg_url)
    try:
        events = check.read("c")
        assert [e.payload["body"] for e in events] == ["0", "1", "2", "3"]
    finally:
        check.close()


def test_postgres_tracks_interrupted_runs() -> None:
    assert PG_URL is not None
    log = PostgresEventLog(PG_URL)
    try:
        run_id = f"int-{os.getpid()}-{id(log)}"
        log.append(run_id, 1.0, "run.state", None, {"state": "running"})
        log.append(run_id, 2.0, "pixel", 1, {"x": 0, "y": 0, "color": 1})
        log.append(run_id, 3.0, "run.state", None, {"state": "interrupted", "was": "running"})
        assert run_id in log.interrupted_runs()
        log.append(run_id, 4.0, "pixel", 1, {"x": 0, "y": 0, "color": 1})
        assert run_id in log.interrupted_runs()
        log.append(run_id, 5.0, "run.state", None, {"state": "running"})
        assert run_id not in log.interrupted_runs()
    finally:
        log.close()
