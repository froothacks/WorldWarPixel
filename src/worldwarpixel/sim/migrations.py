"""Versioned schema migrations for the SQLite and Postgres event stores."""

import argparse
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

POSTGRES_MIGRATION_LOCK_ID = 7_790_184_232_151_091
CANVAS_KINDS = ("pixel", "god.clear_canvas")
CANVAS_KINDS_SQL = ", ".join(f"'{kind}'" for kind in CANVAS_KINDS)


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sqlite_statements: tuple[str, ...]
    postgres_statements: tuple[str, ...]
    postgres_concurrent_statements: tuple[str, ...] = ()
    postgres_concurrent_indexes: tuple[str, ...] = ()


def postgres_run_rollup_sql(*, filter_run_ids: bool = False) -> str:
    """Return the per-run event rollup SELECT, optionally limited to run ids in ``%s``."""
    where = "WHERE run_id = ANY(%s)" if filter_run_ids else ""
    return f"""
        SELECT run_id, MIN(ts),
            COUNT(*) FILTER (WHERE kind = 'pixel'),
            COUNT(*) FILTER (WHERE kind = 'post'),
            COUNT(*) FILTER (WHERE kind = 'agent.thought'),
            COUNT(*) FILTER (WHERE kind = 'agent.thought'
                AND substr(payload::jsonb ->> 'thought', 1, 18) = 'Tool call rejected'),
            COUNT(*) FILTER (WHERE kind = 'narrator.update'
                AND (payload::jsonb ->> 'kind') IS DISTINCT FROM 'answer'),
            COUNT(*) FILTER (WHERE kind = 'narrator.update'
                AND payload::jsonb ->> 'kind' = 'answer'),
            COALESCE(bool_or(kind = 'run.deleted'), false),
            MAX(CASE WHEN kind = 'run.created'
                THEN (payload::jsonb -> 'config')::text END)
        FROM events
        {where}
        GROUP BY run_id
    """


def postgres_posts_backfill_sql(*, filter_run_ids: bool = False) -> str:
    """Return the posts backfill, optionally limited to run ids in ``%s``."""
    cleared_filter = " AND run_id = ANY(%s)" if filter_run_ids else ""
    posts_filter = " AND e.run_id = ANY(%s)" if filter_run_ids else ""
    return f"""
        WITH cleared AS (
            SELECT run_id, MAX(id) AS id FROM events
            WHERE kind = 'god.clear_board'{cleared_filter}
            GROUP BY run_id
        )
        INSERT INTO posts (
            run_id, id, thread_id, parent_id, author_id, title, body, created_at, society
        )
        SELECT e.run_id, (e.p ->> 'id')::bigint, (e.p ->> 'thread_id')::bigint,
            (e.p ->> 'parent_id')::bigint, (e.p ->> 'author_id')::integer, e.p ->> 'title',
            e.p ->> 'body', (e.p ->> 'created_at')::double precision, (e.p ->> 'society')::integer
        FROM (SELECT run_id, id, payload::jsonb AS p FROM events WHERE kind = 'post') AS e
        LEFT JOIN cleared AS c ON c.run_id = e.run_id
        WHERE e.id > COALESCE(c.id, 0){posts_filter}
        ON CONFLICT (run_id, id) DO NOTHING
    """


def postgres_votes_backfill_sql(*, filter_run_ids: bool = False) -> str:
    """Return the latest vote backfill, optionally limited to run ids in ``%s``."""
    cleared_filter = " AND run_id = ANY(%s)" if filter_run_ids else ""
    votes_filter = " AND e.run_id = ANY(%s)" if filter_run_ids else ""
    return f"""
        WITH cleared AS (
            SELECT run_id, MAX(id) AS id FROM events
            WHERE kind = 'god.clear_board'{cleared_filter}
            GROUP BY run_id
        )
        INSERT INTO thread_votes (run_id, thread_id, votes)
        SELECT run_id, thread_id, votes FROM (
            SELECT DISTINCT ON (e.run_id, e.thread_id) e.run_id, e.thread_id, e.votes
            FROM (
                SELECT run_id, id, (payload::jsonb ->> 'thread_id')::bigint AS thread_id,
                    (payload::jsonb ->> 'votes')::integer AS votes
                FROM events WHERE kind = 'board.vote'
            ) AS e
            LEFT JOIN cleared AS c ON c.run_id = e.run_id
            WHERE e.id > COALESCE(c.id, 0){votes_filter}
            ORDER BY e.run_id, e.thread_id, e.id DESC
        ) AS latest
        ON CONFLICT (run_id, thread_id) DO NOTHING
    """


MIGRATIONS = (
    Migration(
        version=1,
        name="events",
        sqlite_statements=(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                agent_id INTEGER,
                payload TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS events_run ON events (run_id, id)",
        ),
        postgres_statements=(
            """
            CREATE TABLE IF NOT EXISTS events (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                run_id TEXT NOT NULL,
                ts DOUBLE PRECISION NOT NULL,
                kind TEXT NOT NULL,
                agent_id INTEGER,
                payload TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS events_run ON events (run_id, id)",
        ),
    ),
    Migration(
        version=2,
        name="runs",
        sqlite_statements=(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                config TEXT,
                pixels INTEGER NOT NULL DEFAULT 0,
                posts INTEGER NOT NULL DEFAULT 0,
                turns INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                updates INTEGER NOT NULL DEFAULT 0,
                answers INTEGER NOT NULL DEFAULT 0,
                deleted INTEGER NOT NULL DEFAULT 0
            )
            """,
            """
            INSERT INTO runs (
                id, created_at, config, pixels, posts, turns, errors, updates, answers, deleted
            )
            SELECT run_id, MIN(ts),
                MAX(CASE WHEN kind = 'run.created' THEN json_extract(payload, '$.config') END),
                SUM(kind = 'pixel'),
                SUM(kind = 'post'),
                SUM(kind = 'agent.thought'),
                SUM(kind = 'agent.thought'
                    AND substr(json_extract(payload, '$.thought'), 1, 18) = 'Tool call rejected'),
                SUM(kind = 'narrator.update'
                    AND json_extract(payload, '$.kind') IS NOT 'answer'),
                SUM(kind = 'narrator.update'
                    AND json_extract(payload, '$.kind') = 'answer'),
                MAX(kind = 'run.deleted')
            FROM events
            GROUP BY run_id
            """,
        ),
        postgres_statements=(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at DOUBLE PRECISION NOT NULL,
                config TEXT,
                pixels BIGINT NOT NULL DEFAULT 0,
                posts BIGINT NOT NULL DEFAULT 0,
                turns BIGINT NOT NULL DEFAULT 0,
                errors BIGINT NOT NULL DEFAULT 0,
                updates BIGINT NOT NULL DEFAULT 0,
                answers BIGINT NOT NULL DEFAULT 0,
                deleted BOOLEAN NOT NULL DEFAULT false
            )
            """,
            f"""
            INSERT INTO runs (
                id, created_at, pixels, posts, turns, errors, updates, answers, deleted, config
            )
            {postgres_run_rollup_sql()}
            """,
            """
            ALTER TABLE events
            ADD CONSTRAINT events_run_fk
            FOREIGN KEY (run_id) REFERENCES runs (id)
            ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE
            """,
        ),
    ),
    Migration(
        version=3,
        name="run_state",
        sqlite_statements=("ALTER TABLE runs ADD COLUMN state TEXT",),
        postgres_statements=("ALTER TABLE runs ADD COLUMN IF NOT EXISTS state TEXT",),
    ),
    Migration(
        version=4,
        name="posts",
        sqlite_statements=(
            """
            CREATE TABLE IF NOT EXISTS posts (
                run_id TEXT NOT NULL,
                id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL,
                parent_id INTEGER,
                author_id INTEGER NOT NULL,
                title TEXT,
                body TEXT NOT NULL,
                created_at REAL NOT NULL,
                society INTEGER,
                PRIMARY KEY (run_id, id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS posts_thread ON posts (run_id, thread_id, id)",
            """
            CREATE TABLE IF NOT EXISTS thread_votes (
                run_id TEXT NOT NULL,
                thread_id INTEGER NOT NULL,
                votes INTEGER NOT NULL,
                PRIMARY KEY (run_id, thread_id)
            )
            """,
            """
            WITH cleared AS (
                SELECT run_id, MAX(id) AS id FROM events
                WHERE kind = 'god.clear_board'
                GROUP BY run_id
            )
            INSERT OR IGNORE INTO posts (
                run_id, id, thread_id, parent_id, author_id, title, body, created_at, society
            )
            SELECT e.run_id, json_extract(e.payload, '$.id'),
                json_extract(e.payload, '$.thread_id'),
                json_extract(e.payload, '$.parent_id'), json_extract(e.payload, '$.author_id'),
                json_extract(e.payload, '$.title'), json_extract(e.payload, '$.body'),
                json_extract(e.payload, '$.created_at'), json_extract(e.payload, '$.society')
            FROM events AS e
            LEFT JOIN cleared AS c ON c.run_id = e.run_id
            WHERE e.kind = 'post' AND e.id > COALESCE(c.id, 0)
            """,
            """
            WITH cleared AS (
                SELECT run_id, MAX(id) AS id FROM events
                WHERE kind = 'god.clear_board'
                GROUP BY run_id
            )
            INSERT OR IGNORE INTO thread_votes (run_id, thread_id, votes)
            SELECT run_id, thread_id, votes FROM (
                SELECT e.run_id, json_extract(e.payload, '$.thread_id') AS thread_id,
                    json_extract(e.payload, '$.votes') AS votes,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.run_id, json_extract(e.payload, '$.thread_id')
                        ORDER BY e.id DESC
                    ) AS rn
                FROM events AS e
                LEFT JOIN cleared AS c ON c.run_id = e.run_id
                WHERE e.kind = 'board.vote' AND e.id > COALESCE(c.id, 0)
            ) WHERE rn = 1
            """,
        ),
        postgres_statements=(
            """
            CREATE TABLE IF NOT EXISTS posts (
                run_id TEXT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
                id BIGINT NOT NULL,
                thread_id BIGINT NOT NULL,
                parent_id BIGINT,
                author_id INTEGER NOT NULL,
                title TEXT,
                body TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                society INTEGER,
                PRIMARY KEY (run_id, id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS posts_thread ON posts (run_id, thread_id, id)",
            """
            CREATE TABLE IF NOT EXISTS thread_votes (
                run_id TEXT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
                thread_id BIGINT NOT NULL,
                votes INTEGER NOT NULL,
                PRIMARY KEY (run_id, thread_id)
            )
            """,
            postgres_posts_backfill_sql(),
            postgres_votes_backfill_sql(),
        ),
    ),
    Migration(
        version=5,
        name="events_run_canvas",
        sqlite_statements=(
            f"CREATE INDEX IF NOT EXISTS events_run_canvas ON events (run_id, id) "
            f"WHERE kind IN ({CANVAS_KINDS_SQL})",
        ),
        postgres_statements=(),
        postgres_concurrent_statements=(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS events_run_canvas ON events (run_id, id) "
            f"WHERE kind IN ({CANVAS_KINDS_SQL})",
        ),
        postgres_concurrent_indexes=("events_run_canvas",),
    ),
    Migration(
        version=6,
        name="run_last_ts",
        sqlite_statements=(
            "ALTER TABLE runs ADD COLUMN last_ts REAL",
            """
            UPDATE runs SET last_ts = (
                SELECT ts FROM events WHERE events.run_id = runs.id ORDER BY id DESC LIMIT 1
            )
            """,
        ),
        postgres_statements=(
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS last_ts DOUBLE PRECISION",
            """
            UPDATE runs SET last_ts = (
                SELECT ts FROM events WHERE events.run_id = runs.id ORDER BY id DESC LIMIT 1
            )
            """,
        ),
    ),
)

SQLITE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at REAL NOT NULL
)
"""
POSTGRES_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at DOUBLE PRECISION NOT NULL
)
"""


def migrate_sqlite(conn: sqlite3.Connection) -> list[int]:
    """Apply pending SQLite migrations under a write lock and return their versions."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(SQLITE_MIGRATIONS_TABLE)
        applied = {int(row[0]) for row in conn.execute("SELECT version FROM schema_migrations")}
        newly_applied: list[int] = []
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            for statement in migration.sqlite_statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (migration.version, migration.name, time.time()),
            )
            newly_applied.append(migration.version)
        conn.commit()
        return newly_applied
    except Exception:
        conn.rollback()
        raise


def migrate_postgres(conn: psycopg.Connection[tuple[Any, ...]]) -> list[int]:
    """Apply pending Postgres migrations without blocking event writes.

    Ordinary DDL stays in one advisory-locked transaction. Index builds that would take a
    long lock on ``events`` run as ``CONCURRENTLY`` statements outside that transaction, so
    live runs keep appending while the index is built.
    """
    newly_applied: list[int] = []
    applied: set[int] = set()
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (POSTGRES_MIGRATION_LOCK_ID,))
        conn.execute(POSTGRES_MIGRATIONS_TABLE)
        applied = {int(row[0]) for row in conn.execute("SELECT version FROM schema_migrations")}
        for migration in MIGRATIONS:
            if migration.version in applied or not migration.postgres_statements:
                continue
            for statement in migration.postgres_statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (%s, %s, %s)",
                (migration.version, migration.name, time.time()),
            )
            applied.add(migration.version)
            newly_applied.append(migration.version)
    for migration in MIGRATIONS:
        if migration.version in applied or not migration.postgres_concurrent_statements:
            continue
        deadline = time.monotonic() + 300
        while True:
            locked = conn.execute(
                "SELECT pg_try_advisory_lock(%s)", (POSTGRES_MIGRATION_LOCK_ID,)
            ).fetchone()
            if locked is not None and bool(locked[0]):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Postgres migration lock")
            time.sleep(0.1)
        try:
            applied_now = {
                int(row[0]) for row in conn.execute("SELECT version FROM schema_migrations")
            }
            if migration.version in applied_now:
                applied.add(migration.version)
                continue
            for index_name in migration.postgres_concurrent_indexes:
                valid = conn.execute(
                    "SELECT i.indisvalid FROM pg_index AS i"
                    " JOIN pg_class AS c ON c.oid = i.indexrelid WHERE c.relname = %s",
                    (index_name,),
                ).fetchone()
                if valid is not None and not bool(valid[0]):
                    conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name}")
            for statement in migration.postgres_concurrent_statements:
                conn.execute(statement)
            with conn.transaction():
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (%s, %s, %s)"
                    " ON CONFLICT (version) DO NOTHING",
                    (migration.version, migration.name, time.time()),
                )
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (POSTGRES_MIGRATION_LOCK_ID,))
        applied.add(migration.version)
        newly_applied.append(migration.version)
    return newly_applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--database-url", metavar="URL")
    backend.add_argument("--sqlite", type=Path, metavar="PATH")
    args = parser.parse_args(argv)

    if args.sqlite is not None:
        with sqlite3.connect(args.sqlite) as conn:
            applied = migrate_sqlite(conn)
    else:
        database_url = args.database_url or os.environ.get("DATABASE_URL")
        if not database_url:
            parser.error("set DATABASE_URL or pass --database-url or --sqlite")
        with psycopg.connect(database_url, autocommit=True) as conn:
            applied = migrate_postgres(conn)

    if applied:
        print(f"Applied migrations: {', '.join(map(str, applied))}")
    else:
        print("up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
