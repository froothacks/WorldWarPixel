from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from worldwarpixel.sim.migrations import migrate_sqlite

if TYPE_CHECKING:
    from worldwarpixel.config import Settings

REJECTED = "Tool call rejected"
SLOW_READ_SECONDS = 1.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Event:
    id: int
    run_id: str
    ts: float
    kind: str
    agent_id: int | None
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class RunDelta:
    pixels: int = 0
    posts: int = 0
    turns: int = 0
    errors: int = 0
    updates: int = 0
    answers: int = 0
    deleted: bool = False
    config: str | None = None

    @property
    def has_changes(self) -> bool:
        return (
            any((self.pixels, self.posts, self.turns, self.errors, self.updates, self.answers))
            or self.deleted
            or self.config is not None
        )


def run_deltas(kind: str, payload: Mapping[str, object]) -> RunDelta:
    """Return the per-run summary changes represented by one event."""
    if kind == "pixel":
        return RunDelta(pixels=1)
    if kind == "post":
        return RunDelta(posts=1)
    if kind == "agent.thought":
        thought = str(payload.get("thought", ""))
        return RunDelta(turns=1, errors=int(thought.startswith(REJECTED)))
    if kind == "narrator.update":
        if payload.get("kind") == "answer":
            return RunDelta(answers=1)
        return RunDelta(updates=1)
    if kind == "run.deleted":
        return RunDelta(deleted=True)
    if kind == "run.created" and "config" in payload:
        return RunDelta(config=json.dumps(payload["config"], separators=(",", ":")))
    return RunDelta()


class EventStore(Protocol):
    """The append-only event log interface used by runs, replay and rendering."""

    def append(
        self, run_id: str, ts: float, kind: str, agent_id: int | None, payload: Mapping[str, object]
    ) -> int: ...

    def batch(self) -> AbstractContextManager[None]: ...

    def read(self, run_id: str, after_id: int = 0, limit: int = 1000) -> list[Event]: ...

    def count(self, run_id: str) -> int: ...

    def run_ids(self) -> list[str]: ...

    def run_totals(
        self,
    ) -> list[tuple[str, float, int, int, int, int, int, int, bool, str | None]]: ...

    def reader(self) -> EventStore: ...

    def close(self) -> None: ...


class EventLog:
    """Append-only SQLite log of everything that happens in a run, for replay and audit."""

    _batching = False

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        migrate_sqlite(self._conn)
        self._known_runs = {str(row[0]) for row in self._conn.execute("SELECT id FROM runs")}

    def append(
        self, run_id: str, ts: float, kind: str, agent_id: int | None, payload: Mapping[str, object]
    ) -> int:
        delta = run_deltas(kind, payload)
        serialized = json.dumps(payload, separators=(",", ":"))
        if run_id not in self._known_runs or delta.has_changes:
            self._conn.execute(
                """
                INSERT INTO runs (
                    id, created_at, config, pixels, posts, turns, errors, updates, answers, deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    created_at = MIN(runs.created_at, excluded.created_at),
                    config = COALESCE(runs.config, excluded.config),
                    pixels = runs.pixels + excluded.pixels,
                    posts = runs.posts + excluded.posts,
                    turns = runs.turns + excluded.turns,
                    errors = runs.errors + excluded.errors,
                    updates = runs.updates + excluded.updates,
                    answers = runs.answers + excluded.answers,
                    deleted = MAX(runs.deleted, excluded.deleted)
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
                    int(delta.deleted),
                ),
            )
        cur = self._conn.execute(
            "INSERT INTO events (run_id, ts, kind, agent_id, payload) VALUES (?, ?, ?, ?, ?)",
            (run_id, ts, kind, agent_id, serialized),
        )
        if not self._batching:
            self._conn.commit()
        self._known_runs.add(run_id)
        return int(cur.lastrowid or 0)

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Defer the commit until the block ends, for bulk writes like a god-mode brush stroke."""
        self._batching = True
        try:
            yield
        finally:
            self._batching = False
            self._conn.commit()

    def read(self, run_id: str, after_id: int = 0, limit: int = 1000) -> list[Event]:
        started = time.perf_counter()
        rows = self._conn.execute(
            "SELECT id, run_id, ts, kind, agent_id, payload FROM events"
            " WHERE run_id = ? AND id > ? ORDER BY run_id, id LIMIT ?",
            (run_id, after_id, limit),
        ).fetchall()
        events = [Event(r[0], r[1], r[2], r[3], r[4], json.loads(r[5])) for r in rows]
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
            "SELECT COUNT(*) FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row[0])

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
                str(row[0]),
                float(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                int(row[5]),
                int(row[6]),
                int(row[7]),
                bool(row[8]),
                None if row[9] is None else str(row[9]),
            )
            for row in rows
        ]

    def reader(self) -> EventLog:
        """A log on the same database for use from another thread; ``close`` it when done.
        In-memory logs have no second connection, so the shared one is returned."""
        return self if self.path == ":memory:" else EventLog(self.path)

    def close(self) -> None:
        self._conn.close()


def open_event_log(settings: Settings) -> EventStore:
    """Open the configured event store: Postgres when DATABASE_URL is set, else SQLite."""
    if settings.database_url:
        from worldwarpixel.sim.events_pg import PostgresEventLog

        return PostgresEventLog(settings.database_url)
    return EventLog(settings.event_log_path)
