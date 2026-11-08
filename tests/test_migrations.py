import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from worldwarpixel.sim.archive import archived_summaries
from worldwarpixel.sim.board import Post
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.migrations import (
    MIGRATIONS,
    SQLITE_MIGRATIONS_TABLE,
    migrate_sqlite,
)
from worldwarpixel.sim.run import RunConfig


def test_sqlite_migrations_are_recorded_once(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    conn = sqlite3.connect(path)
    assert migrate_sqlite(conn) == [1, 2, 3, 4]
    assert conn.execute(
        "SELECT version, name FROM schema_migrations ORDER BY version"
    ).fetchall() == [
        (1, "events"),
        (2, "runs"),
        (3, "posts"),
        (4, "events_run_canvas"),
    ]
    conn.close()

    log = EventLog(path)
    assert migrate_sqlite(log._conn) == []
    assert log.run_totals() == []
    log.close()


def test_legacy_sqlite_events_are_backfilled(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    config = RunConfig(width=32, height=16, agents=2, policy="random").model_dump()
    events: list[tuple[str, float, str, int | None, dict[str, object]]] = [
        ("listed", 1.0, "run.created", None, {"config": config}),
        ("listed", 2.0, "pixel", 1, {"x": 1, "y": 2, "color": 3}),
        (
            "listed",
            3.0,
            "post",
            1,
            {
                "id": 1,
                "thread_id": 1,
                "parent_id": None,
                "author_id": 1,
                "title": "hello",
                "body": "hello",
                "created_at": 3.0,
            },
        ),
        ("listed", 4.0, "agent.thought", 1, {"thought": "Tool call rejected: invalid"}),
        ("listed", 5.0, "agent.thought", 1, {"thought": "Painting a sun"}),
        ("listed", 6.0, "narrator.update", None, {"kind": "update", "text": "News"}),
        ("listed", 7.0, "narrator.update", None, {"kind": "answer", "text": "Answer"}),
        ("listed", 8.0, "narrator.update", None, {"text": "Legacy update"}),
        ("listed", 9.0, "agent.status", 1, {"status": "idle"}),
        ("deleted", 1.0, "run.created", None, {"config": config}),
        ("deleted", 2.0, "run.deleted", None, {}),
        ("configless", 1.0, "run.created", None, {"agents": []}),
        ("no-created", 1.0, "agent.status", 1, {"status": "idle"}),
    ]
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,
            agent_id INTEGER,
            payload TEXT NOT NULL
        );
        CREATE INDEX events_run ON events (run_id, id);
        """
    )
    conn.executemany(
        "INSERT INTO events (run_id, ts, kind, agent_id, payload) VALUES (?, ?, ?, ?, ?)",
        [
            (run_id, ts, kind, agent_id, json.dumps(payload, separators=(",", ":")))
            for run_id, ts, kind, agent_id, payload in events
        ],
    )
    conn.commit()
    conn.close()

    log = EventLog(path)
    [summary] = archived_summaries(log, set())
    assert summary["id"] == "listed"
    assert summary["created_at"] == 1.0
    assert summary["stats"] == {"pixels": 1, "posts": 1, "turns": 2, "errors": 1}
    assert summary["narrator"] == {"updates": 2, "answers": 1, "failures": 0}
    assert log.run_ids() == ["configless", "deleted", "listed", "no-created"]
    log.close()


def test_posts_and_votes_backfill_from_migration_two_database(tmp_path: Path) -> None:
    path = tmp_path / "version-2.db"
    config = {"width": 16, "height": 16, "agents": 2}
    events = [
        ("r", 1.0, "run.created", None, {"config": config}),
        (
            "r",
            2.0,
            "post",
            1,
            {
                "id": 1,
                "thread_id": 1,
                "parent_id": None,
                "author_id": 1,
                "title": "before",
                "body": "discarded",
                "created_at": 2.0,
            },
        ),
        ("r", 3.0, "board.vote", 1, {"thread_id": 1, "votes": 1}),
        ("r", 4.0, "board.vote", 1, {"thread_id": 1, "votes": 2}),
        ("r", 5.0, "god.clear_board", None, {}),
        (
            "r",
            6.0,
            "post",
            2,
            {
                "id": 2,
                "thread_id": 2,
                "parent_id": None,
                "author_id": 2,
                "title": "after",
                "body": "kept",
                "created_at": 6.0,
                "society": 1,
            },
        ),
        ("r", 7.0, "board.vote", 2, {"thread_id": 2, "votes": 1}),
        ("r", 8.0, "board.vote", 2, {"thread_id": 2, "votes": -1}),
        ("r", 9.0, "board.vote", 2, {"thread_id": 3, "votes": 3}),
    ]
    conn = sqlite3.connect(path)
    conn.execute(SQLITE_MIGRATIONS_TABLE)
    for statement in MIGRATIONS[0].sqlite_statements:
        conn.execute(statement)
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        (1, "events", 0.0),
    )
    conn.executemany(
        "INSERT INTO events (run_id, ts, kind, agent_id, payload) VALUES (?, ?, ?, ?, ?)",
        [
            (run_id, ts, kind, agent_id, json.dumps(payload, separators=(",", ":")))
            for run_id, ts, kind, agent_id, payload in events
        ],
    )
    for statement in MIGRATIONS[1].sqlite_statements:
        conn.execute(statement)
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        (2, "runs", 0.0),
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(path)
    assert migrate_sqlite(conn) == [3, 4]
    conn.close()

    log = EventLog(path)
    assert log.posts("r") == [(Post(2, 2, 2, None, "after", "kept", 6.0), 1)]
    assert log.thread_votes("r") == {2: -1, 3: 3}
    log.close()


def test_migration_cli_accepts_a_sqlite_path(tmp_path: Path) -> None:
    path = tmp_path / "cli.db"
    result = subprocess.run(
        [sys.executable, "-m", "worldwarpixel.sim.migrations", "--sqlite", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "Applied migrations: 1, 2, 3, 4"
