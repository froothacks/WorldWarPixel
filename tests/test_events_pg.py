import json
import os
import time
from collections.abc import Callable, Iterator
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
        conn.execute(
            "DROP TABLE IF EXISTS posts, thread_votes, events, runs, schema_migrations CASCADE"
        )
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
    b = pg_log.append(
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
    pg_log.append("r2", 3.0, "pixel", 1, {})

    events = pg_log.read("r1")
    assert [e.id for e in events] == [a, b]
    assert events[0].payload == {"x": 1, "y": 2, "color": 4}
    assert events[0].agent_id == 3
    assert pg_log.read("r1", after_id=a) == [events[1]]
    assert pg_log.count("r1") == 2
    assert pg_log.run_ids() == ["r1", "r2"]


def test_repeated_reads_do_not_create_prepared_statements(pg_log: PostgresEventLog) -> None:
    pg_log.append("r", 1.0, "pixel", 1, {"x": 0, "y": 0, "color": 1})

    for _ in range(20):
        assert len(pg_log.read("r")) == 1

    result = pg_log._conn.execute("SELECT count(*) FROM pg_prepared_statements").fetchone()
    assert result is not None and result[0] == 0


def test_postgres_projections_run_meta_and_canvas_reads(pg_log: PostgresEventLog) -> None:
    config = {"width": 16, "height": 12}
    pg_log.append("r", 1.0, "run.created", None, {"config": config})
    post = {
        "id": 1,
        "thread_id": 1,
        "parent_id": None,
        "author_id": 1,
        "title": "title",
        "body": "body",
        "created_at": 2.0,
        "society": 2,
    }
    pg_log.append("r", 2.0, "post", 1, post)
    pg_log.append("r", 3.0, "board.vote", 1, {"thread_id": 1, "value": 1, "votes": 1})
    pg_log.append("r", 4.0, "board.vote", 1, {"thread_id": 1, "value": -1, "votes": 0})
    pg_log.append("r", 5.0, "agent.status", 1, {})
    pg_log.append("r", 6.0, "pixel", 1, {"x": 0, "y": 0, "color": 1})
    pg_log.append("r", 7.0, "god.clear_canvas", None, {})

    meta = pg_log.run_meta("r")
    assert meta is not None
    assert meta.created_at == 1.0 and not meta.deleted
    assert meta.config is not None and json.loads(meta.config) == config
    assert pg_log.run_meta("missing") is None
    assert [entry[0].id for entry in pg_log.posts("r")] == [1]
    assert pg_log.posts("r", thread_id=1)[0][1] == 2
    assert pg_log.thread_votes("r") == {1: 0}
    assert [event.kind for event in pg_log.read("r", canvas_only=True)] == [
        "pixel",
        "god.clear_canvas",
    ]
    assert [event.kind for event in pg_log.read("r", limit=1, canvas_only=True)] == ["pixel"]
    assert [event.kind for event in pg_log.read("r", after_id=6, limit=1, canvas_only=True)] == [
        "god.clear_canvas"
    ]

    pg_log.append("r", 8.0, "god.clear_board", None, {})
    assert pg_log.posts("r") == []
    assert pg_log.thread_votes("r") == {}
    pg_log.append("r", 9.0, "run.deleted", None, {})
    assert pg_log.run_meta("r").deleted  # type: ignore[union-attr]


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
    pg_log.append(
        "r",
        3.0,
        "post",
        1,
        {
            "id": 1,
            "thread_id": 1,
            "parent_id": None,
            "author_id": 1,
            "title": "hello",
            "body": "hi",
            "created_at": 3.0,
        },
    )
    pg_log.append("r", 4.0, "agent.thought", 1, {"thought": "Tool call rejected: bad json"})
    pg_log.append("r", 5.0, "agent.thought", 1, {"thought": "Painting a sun"})
    pg_log.append("r", 6.0, "narrator.update", None, {"kind": "update", "text": "Hi"})
    pg_log.append("r", 7.0, "narrator.update", None, {"kind": "answer", "text": "Yes"})

    [row] = pg_log.run_totals()
    assert row[:2] == ("r", 1.0)
    assert row[2:8] == (1, 1, 2, 1, 1, 1)
    assert row[8] is False
    assert row[9] is not None and json.loads(row[9]) == config


def test_append_many_matches_sequential_appends(pg_log: PostgresEventLog) -> None:
    config = {"width": 16, "height": 16, "agents": 2}
    mixed_rows: list[tuple[float, str, int | None, dict[str, object]]] = [
        (10.0, "run.created", None, {"config": config}),
        (1.0, "pixel", 1, {"x": 1, "y": 1, "color": 2}),
        (2.0, "pixel", 1, {"x": 2, "y": 1, "color": 2}),
        (3.0, "agent.thought", 1, {"thought": "Painting a sun"}),
        (4.0, "agent.thought", 1, {"thought": "Tool call rejected: bad json"}),
        (5.0, "narrator.update", None, {"kind": "update", "text": "A new mural"}),
        (6.0, "narrator.update", None, {"kind": "answer", "text": "We will defend it"}),
        (
            7.0,
            "post",
            1,
            {
                "id": 1,
                "thread_id": 1,
                "parent_id": None,
                "author_id": 1,
                "title": "Before clear",
                "body": "This post will be removed",
                "created_at": 7.0,
                "society": 1,
            },
        ),
        (8.0, "board.vote", 1, {"thread_id": 1, "value": 1, "votes": 3}),
        (9.0, "run.state", None, {"state": "running"}),
        (10.0, "run.state", None, {"state": "stopped"}),
        (10.5, "run.created", None, {"config": {"width": 99}}),
        (11.0, "run.deleted", None, {}),
        (12.0, "god.clear_board", None, {}),
        (
            13.0,
            "post",
            1,
            {
                "id": 2,
                "thread_id": 2,
                "parent_id": None,
                "author_id": 1,
                "title": "After clear",
                "body": "This post remains",
                "created_at": 13.0,
                "society": 2,
            },
        ),
        (14.0, "board.vote", 1, {"thread_id": 2, "value": 1, "votes": 4}),
    ]

    def rows_for(
        run_id: str,
    ) -> list[tuple[str, float, str, int | None, dict[str, object]]]:
        return [(run_id, ts, kind, agent_id, payload) for ts, kind, agent_id, payload in mixed_rows]

    pg_log.append_many(rows_for("batch"))
    for row in rows_for("sequential"):
        pg_log.append(*row)

    batch_events = pg_log.read("batch")
    sequential_events = pg_log.read("sequential")
    assert [(event.kind, event.payload) for event in batch_events] == [
        (event.kind, event.payload) for event in sequential_events
    ]

    batch_meta = pg_log.run_meta("batch")
    sequential_meta = pg_log.run_meta("sequential")
    assert batch_meta == sequential_meta
    assert batch_meta is not None
    assert batch_meta.created_at == 1.0
    assert batch_meta.deleted
    assert batch_meta.config is not None and json.loads(batch_meta.config) == config

    totals = {row[0]: row[1:] for row in pg_log.run_totals()}
    assert totals["batch"] == totals["sequential"]
    assert totals["batch"][:7] == (1.0, 2, 2, 2, 1, 1, 1)
    assert totals["batch"][7] is True
    assert totals["batch"][8] is not None and json.loads(totals["batch"][8]) == config

    states = {}
    for run_id in ("batch", "sequential"):
        state = pg_log._conn.execute("SELECT state FROM runs WHERE id = %s", (run_id,)).fetchone()
        assert state is not None
        states[run_id] = state[0]
    assert states == {"batch": "stopped", "sequential": "stopped"}
    assert pg_log.posts("batch") == pg_log.posts("sequential")
    assert [post.id for post, _ in pg_log.posts("batch")] == [2]
    assert pg_log.thread_votes("batch") == pg_log.thread_votes("sequential") == {2: 4}


def test_append_many_failure_rolls_back_and_preserves_known_runs(
    pg_log: PostgresEventLog,
) -> None:
    rows: list[tuple[str, float, str, int | None, dict[str, object]]] = [
        ("bad-batch", 1.0, "run.created", None, {"config": {}}),
        ("bad-batch", 2.0, "post", 1, {"id": 1}),
    ]

    with pytest.raises(KeyError):
        pg_log.append_many(rows)

    assert "bad-batch" not in pg_log._known_runs
    assert pg_log.count("bad-batch") == 0
    assert "bad-batch" not in pg_log.run_ids()


def test_migrate_sqlite_to_pg_is_idempotent(pg_url: str, tmp_path: Path) -> None:
    sqlite_path = tmp_path / "events.db"
    source = EventLog(sqlite_path)
    source.append("m", 1.0, "run.created", None, {"config": {"width": 16}})
    source.append("m", 2.0, "pixel", 1, {"x": 1, "y": 2, "color": 3})
    source.append(
        "m",
        3.0,
        "post",
        1,
        {
            "id": 1,
            "thread_id": 1,
            "parent_id": None,
            "author_id": 1,
            "title": "before clear",
            "body": "discarded",
            "created_at": 3.0,
        },
    )
    source.append("m", 4.0, "board.vote", 1, {"thread_id": 1, "votes": 1})
    source.append("m", 5.0, "god.clear_board", None, {})
    source.append(
        "m",
        6.0,
        "post",
        2,
        {
            "id": 2,
            "thread_id": 2,
            "parent_id": None,
            "author_id": 2,
            "title": "after clear",
            "body": "kept",
            "created_at": 6.0,
            "society": 1,
        },
    )
    source.append("m", 7.0, "board.vote", 2, {"thread_id": 2, "votes": 1})
    source.append("m", 8.0, "board.vote", 2, {"thread_id": 2, "votes": 3})
    source.close()

    assert migrate(sqlite_path, pg_url) == 8
    assert migrate(sqlite_path, pg_url) == 8

    target = PostgresEventLog(pg_url)
    try:
        assert [e.id for e in target.read("m")] == list(range(1, 9))
        assert target.count("m") == 8
        [row] = target.run_totals()
        assert row[:9] == ("m", 1.0, 1, 2, 0, 0, 0, 0, False)
        assert row[9] is not None and json.loads(row[9]) == {"width": 16}
        assert [post.id for post, _ in target.posts("m")] == [2]
        assert target.thread_votes("m") == {2: 3}
        assert target.append("m", 9.0, "pixel", 1, {}) == 9
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
        conn.execute(
            "DROP TABLE IF EXISTS posts, thread_votes, events, runs, schema_migrations CASCADE"
        )
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
        conn.execute(
            "DROP TABLE IF EXISTS posts, thread_votes, events, runs, schema_migrations CASCADE"
        )

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
        ).fetchall() == [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]


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
            log.append(
                "c",
                float(i),
                "post",
                2,
                {
                    "id": i + 1,
                    "thread_id": i + 1,
                    "parent_id": None,
                    "author_id": 2,
                    "title": str(i),
                    "body": str(i),
                    "created_at": float(i),
                },
            )
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


def _wait_for(check: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def test_buffered_log_reconnects_after_losing_its_connections(pg_url: str) -> None:
    log = BufferedPostgresEventLog(pg_url, flush_interval=0.05)
    try:
        log.append("d", 0.0, "run.state", None, {"state": "running"})
        _wait_for(lambda: log.count("d") == 1)
        with psycopg.connect(pg_url, autocommit=True) as admin:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid()"
            )
        for i in range(1, 5):
            log.append("d", float(i), "pixel", 1, {"x": i, "y": 0, "color": 1})
        log.append("d", 5.0, "run.state", None, {"state": "interrupted", "was": "running"})
        _wait_for(lambda: log.count("d") == 6)
        assert [e.ts for e in log.read("d")] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert "d" in log.interrupted_runs()
    finally:
        log.close()
