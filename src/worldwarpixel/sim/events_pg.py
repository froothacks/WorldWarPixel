from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import psycopg

from worldwarpixel.sim.events import SLOW_READ_SECONDS, Event, EventStore, run_deltas
from worldwarpixel.sim.migrations import migrate_postgres

logger = logging.getLogger(__name__)


class PostgresEventLog(EventStore):
    """Append-only Postgres log of everything that happens in a run, for replay and audit."""

    def __init__(self, database_url: str, *, initialize_schema: bool = True) -> None:
        self.database_url = database_url
        self._conn: psycopg.Connection[tuple[Any, ...]] = psycopg.connect(
            database_url, autocommit=True
        )
        self._batch_depth = 0
        try:
            if initialize_schema:
                self._initialize_schema()
            self._known_runs = {
                str(row[0]) for row in self._conn.execute("SELECT id FROM runs").fetchall()
            }
        except Exception:
            self._conn.close()
            raise

    def _initialize_schema(self) -> None:
        migrate_postgres(self._conn)

    def append(
        self, run_id: str, ts: float, kind: str, agent_id: int | None, payload: Mapping[str, object]
    ) -> int:
        delta = run_deltas(kind, payload)
        serialized = json.dumps(payload, separators=(",", ":"))

        def write() -> int:
            if run_id not in self._known_runs or delta.has_changes:
                self._conn.execute(
                    """
                    INSERT INTO runs (
                        id, created_at, config, pixels, posts, turns, errors, updates, answers,
                        deleted
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        created_at = LEAST(runs.created_at, EXCLUDED.created_at),
                        config = COALESCE(runs.config, EXCLUDED.config),
                        pixels = runs.pixels + EXCLUDED.pixels,
                        posts = runs.posts + EXCLUDED.posts,
                        turns = runs.turns + EXCLUDED.turns,
                        errors = runs.errors + EXCLUDED.errors,
                        updates = runs.updates + EXCLUDED.updates,
                        answers = runs.answers + EXCLUDED.answers,
                        deleted = runs.deleted OR EXCLUDED.deleted
                    """,
                    (
                        run_id,
                        ts,
                        delta.config,
                        delta.pixels,
                        delta.posts,
                        delta.turns,
                        delta.errors,
                        delta.updates,
                        delta.answers,
                        delta.deleted,
                    ),
                )
            row = self._conn.execute(
                "INSERT INTO events (run_id, ts, kind, agent_id, payload)"
                " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (run_id, ts, kind, agent_id, serialized),
            ).fetchone()
            if row is None:
                raise RuntimeError("event insert did not return an id")
            return int(row[0])

        try:
            if self._batch_depth == 0:
                with self._conn.transaction():
                    event_id = write()
            else:
                event_id = write()
        except Exception:
            if self._batch_depth == 0:
                self._conn.rollback()
            raise
        self._known_runs.add(run_id)
        return event_id

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Run the block in one transaction, committing when it ends without an error."""
        known_runs = self._known_runs.copy()
        self._batch_depth += 1
        try:
            with self._conn.transaction():
                yield
        except Exception:
            self._known_runs = known_runs
            raise
        finally:
            self._batch_depth -= 1

    def read(self, run_id: str, after_id: int = 0, limit: int = 1000) -> list[Event]:
        started = time.perf_counter()
        rows = self._conn.execute(
            "SELECT id, run_id, ts, kind, agent_id, payload FROM events"
            " WHERE run_id = %s AND id > %s ORDER BY run_id, id LIMIT %s",
            (run_id, after_id, limit),
        ).fetchall()
        events = [
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
        elapsed = time.perf_counter() - started
        if elapsed >= SLOW_READ_SECONDS:
            logger.warning(
                "slow event read run=%s after=%d limit=%d rows=%d in %.2fs",
                run_id,
                after_id,
                limit,
                len(events),
                elapsed,
            )
        return events

    def count(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = %s", (run_id,)
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def run_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT id FROM runs ORDER BY id").fetchall()
        return [str(r[0]) for r in rows]

    def run_totals(self) -> list[tuple[str, float, int, int, int, int, int, int, bool, str | None]]:
        """Return maintained per-run counters, ordered by run id."""
        rows = self._conn.execute(
            "SELECT id, created_at, pixels, posts, turns, errors, updates, answers, deleted, config"
            " FROM runs ORDER BY id"
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
