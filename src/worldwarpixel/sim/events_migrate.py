"""Copy a SQLite event log into Postgres, preserving event ids.

Usage:
    DATABASE_URL=postgresql://user:pass@host/db python -m worldwarpixel.sim.events_migrate events.db
    python -m worldwarpixel.sim.events_migrate events.db --database-url postgresql://...

The copy is idempotent: rows whose SQLite id already exists in Postgres with identical content
are skipped, touched run summaries are recomputed from Postgres events, and the identity sequence
is moved past the largest migrated id so new appends do not collide. A Postgres row with the same
id but different content raises ValueError instead of silently dropping the SQLite event.
"""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Any

import psycopg

from worldwarpixel.sim.events_pg import PostgresEventLog
from worldwarpixel.sim.migrations import postgres_run_rollup_sql

INSERT_SQL = """
INSERT INTO events (id, run_id, ts, kind, agent_id, payload)
OVERRIDING SYSTEM VALUE
VALUES (%s, %s, %s, %s, %s, %s)
"""

SELECT_SQL = """
SELECT id, run_id, ts, kind, agent_id, payload FROM events WHERE id = ANY(%s)
"""


def _reset_identity(conn: psycopg.Connection[tuple[Any, ...]]) -> None:
    row = conn.execute("SELECT MAX(id) FROM events").fetchone()
    if row is None or row[0] is None:
        return
    sequence = conn.execute("SELECT pg_get_serial_sequence('events', 'id')").fetchone()
    if sequence is not None and sequence[0] is not None:
        conn.execute("SELECT setval(%s::regclass, %s)", (str(sequence[0]), int(row[0])))


def migrate(sqlite_path: str | Path, database_url: str, *, batch_size: int = 1000) -> int:
    """Copy every row of ``sqlite_path`` into Postgres in id order; return rows processed."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    PostgresEventLog(database_url).close()
    source = sqlite3.connect(str(sqlite_path))
    target = psycopg.connect(database_url)
    try:
        with target.transaction():
            target.execute("SET CONSTRAINTS events_run_fk DEFERRED")
            cursor = source.execute(
                "SELECT id, run_id, ts, kind, agent_id, payload FROM events ORDER BY id"
            )
            total = 0
            touched_run_ids: set[str] = set()
            while rows := cursor.fetchmany(batch_size):
                ids = [int(row[0]) for row in rows]
                existing = {
                    int(row[0]): row for row in target.execute(SELECT_SQL, (ids,)).fetchall()
                }
                for row in rows:
                    touched_run_ids.add(str(row[1]))
                    current = existing.get(int(row[0]))
                    if current is None:
                        target.execute(INSERT_SQL, row)
                    elif tuple(current) != tuple(row):
                        raise ValueError(
                            f"event id {row[0]} already exists in Postgres with different content"
                        )
                total += len(rows)
            if touched_run_ids:
                target.execute(
                    f"""
                    INSERT INTO runs (
                        id, created_at, pixels, posts, turns, errors, updates, answers, deleted,
                        config
                    )
                    {postgres_run_rollup_sql(filter_run_ids=True)}
                    ON CONFLICT (id) DO UPDATE SET
                        created_at = EXCLUDED.created_at,
                        config = EXCLUDED.config,
                        pixels = EXCLUDED.pixels,
                        posts = EXCLUDED.posts,
                        turns = EXCLUDED.turns,
                        errors = EXCLUDED.errors,
                        updates = EXCLUDED.updates,
                        answers = EXCLUDED.answers,
                        deleted = EXCLUDED.deleted
                    """,
                    (list(touched_run_ids),),
                )
            _reset_identity(target)
        return total
    finally:
        target.close()
        source.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="path to the SQLite events.db to migrate")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres DSN (defaults to DATABASE_URL)",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("DATABASE_URL is not set; pass --database-url")
    count = migrate(args.sqlite, args.database_url, batch_size=args.batch_size)
    print(f"copied {count} events from {args.sqlite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
