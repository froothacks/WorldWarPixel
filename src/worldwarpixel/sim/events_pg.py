from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import psycopg

from worldwarpixel.sim.events import Event, EventStore

SCHEMA_STATEMENTS = (
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
)


class PostgresEventLog(EventStore):
    """Append-only Postgres log of everything that happens in a run, for replay and audit."""

    def __init__(self, database_url: str, *, initialize_schema: bool = True) -> None:
        self.database_url = database_url
        self._conn: psycopg.Connection[tuple[Any, ...]] = psycopg.connect(
            database_url, autocommit=True
        )
        self._batch_depth = 0
        if initialize_schema:
            try:
                self._initialize_schema()
            except Exception:
                self._conn.close()
                raise

    def _initialize_schema(self) -> None:
        for statement in SCHEMA_STATEMENTS:
            self._conn.execute(statement)
        self._conn.commit()

    def append(
        self, run_id: str, ts: float, kind: str, agent_id: int | None, payload: Mapping[str, object]
    ) -> int:
        try:
            row = self._conn.execute(
                "INSERT INTO events (run_id, ts, kind, agent_id, payload)"
                " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (run_id, ts, kind, agent_id, json.dumps(payload, separators=(",", ":"))),
            ).fetchone()
            if row is None:
                raise RuntimeError("event insert did not return an id")
        except Exception:
            if self._batch_depth == 0:
                self._conn.rollback()
            raise
        if self._batch_depth == 0:
            self._conn.commit()
        return int(row[0])

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Run the block in one transaction, committing when it ends without an error."""
        self._batch_depth += 1
        try:
            with self._conn.transaction():
                yield
        finally:
            self._batch_depth -= 1

    def read(self, run_id: str, after_id: int = 0, limit: int = 1000) -> list[Event]:
        rows = self._conn.execute(
            "SELECT id, run_id, ts, kind, agent_id, payload FROM events"
            " WHERE run_id = %s AND id > %s ORDER BY id LIMIT %s",
            (run_id, after_id, limit),
        ).fetchall()
        return [
            Event(
                id=int(r[0]),
                run_id=str(r[1]),
                ts=float(r[2]),
                kind=str(r[3]),
                agent_id=None if r[4] is None else int(r[4]),
                payload=json.loads(str(r[5])),
            )
            for r in rows
        ]

    def count(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = %s", (run_id,)
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def run_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT run_id FROM events ORDER BY run_id").fetchall()
        return [str(r[0]) for r in rows]

    def run_totals(
        self, rejected_prefix: str
    ) -> list[tuple[str, float, int, int, int, int, int, int, bool, str | None]]:
        """Per run: id, first ts, counts of pixels/posts/turns/rejected turns/narrator updates/
        narrator answers, deleted?, config JSON (if created). Rejected turns are thoughts
        starting with ``rejected_prefix``."""
        rows = self._conn.execute(
            "SELECT run_id, MIN(ts),"
            " COUNT(*) FILTER (WHERE kind = 'pixel'),"
            " COUNT(*) FILTER (WHERE kind = 'post'),"
            " COUNT(*) FILTER (WHERE kind = 'agent.thought'),"
            " COUNT(*) FILTER (WHERE kind = 'agent.thought'"
            " AND substr(payload::jsonb ->> 'thought', 1, %s) = %s),"
            " COUNT(*) FILTER (WHERE kind = 'narrator.update'"
            " AND (payload::jsonb ->> 'kind') IS DISTINCT FROM 'answer'),"
            " COUNT(*) FILTER (WHERE kind = 'narrator.update'"
            " AND payload::jsonb ->> 'kind' = 'answer'),"
            " COALESCE(bool_or(kind = 'run.deleted'), false),"
            " MAX(CASE WHEN kind = 'run.created' THEN (payload::jsonb -> 'config')::text END)"
            " FROM events GROUP BY run_id ORDER BY run_id",
            (len(rejected_prefix), rejected_prefix),
        ).fetchall()
        return [
            (
                str(r[0]),
                float(r[1]),
                int(r[2]),
                int(r[3]),
                int(r[4]),
                int(r[5]),
                int(r[6]),
                int(r[7]),
                bool(r[8]),
                None if r[9] is None else str(r[9]),
            )
            for r in rows
        ]

    def reader(self) -> PostgresEventLog:
        """A log on a new connection for use from another thread; ``close`` it when done."""
        return PostgresEventLog(self.database_url, initialize_schema=False)

    def close(self) -> None:
        self._conn.close()
