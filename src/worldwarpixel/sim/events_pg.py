from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from typing import Any, TypeVar, cast

import psycopg
from psycopg import errors as pg_errors

from worldwarpixel.sim.board import Post
from worldwarpixel.sim.events import (
    SLOW_READ_SECONDS,
    Event,
    EventStore,
    RunDelta,
    RunMeta,
    run_deltas,
)
from worldwarpixel.sim.migrations import CANVAS_KINDS_SQL, migrate_postgres

logger = logging.getLogger(__name__)

T = TypeVar("T")

RECONNECT_MAX_DELAY = 30.0
"""Longest pause between attempts to reach Postgres again after the writer lost its connection."""

_Row = tuple[str, float, str, int | None, Mapping[str, object]]
"""One queued event: run id, timestamp, kind, agent id and JSON-able payload."""


class PostgresEventLog(EventStore):
    """Append-only Postgres log of everything that happens in a run, for replay and audit."""

    def __init__(
        self,
        database_url: str,
        *,
        initialize_schema: bool = True,
        lock_timeout_ms: int | None = None,
        statement_timeout_ms: int | None = None,
    ) -> None:
        self.database_url = database_url
        # Server-side prepared statements can switch to generic plans that scan events_pkey
        # for run-scoped reads.
        self._conn: psycopg.Connection[tuple[Any, ...]] = psycopg.connect(
            database_url,
            autocommit=True,
            prepare_threshold=None,
        )
        if lock_timeout_ms is not None:
            self._conn.execute(
                "SELECT set_config('lock_timeout', %s, false)", (str(lock_timeout_ms),)
            )
        if statement_timeout_ms is not None:
            self._conn.execute(
                "SELECT set_config('statement_timeout', %s, false)", (str(statement_timeout_ms),)
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

    def _upsert_run(self, run_id: str, created_at: float, last_ts: float, delta: RunDelta) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (
                id, created_at, config, pixels, posts, turns, errors, updates, answers,
                deleted, state, last_ts
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                created_at = LEAST(runs.created_at, EXCLUDED.created_at),
                last_ts = GREATEST(runs.last_ts, EXCLUDED.last_ts),
                config = COALESCE(runs.config, EXCLUDED.config),
                pixels = runs.pixels + EXCLUDED.pixels,
                posts = runs.posts + EXCLUDED.posts,
                turns = runs.turns + EXCLUDED.turns,
                errors = runs.errors + EXCLUDED.errors,
                updates = runs.updates + EXCLUDED.updates,
                answers = runs.answers + EXCLUDED.answers,
                deleted = runs.deleted OR EXCLUDED.deleted,
                state = COALESCE(EXCLUDED.state, runs.state)
            """,
            (
                run_id,
                created_at,
                delta.config,
                delta.pixels,
                delta.posts,
                delta.turns,
                delta.errors,
                delta.updates,
                delta.answers,
                delta.deleted,
                delta.state,
                last_ts,
            ),
        )

    def _update_projection(self, run_id: str, kind: str, payload: Mapping[str, object]) -> None:
        if kind == "post":
            self._conn.execute(
                """
                INSERT INTO posts (
                    run_id, id, thread_id, parent_id, author_id, title, body, created_at,
                    society
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, id) DO NOTHING
                """,
                (
                    run_id,
                    cast(int, payload["id"]),
                    cast(int, payload["thread_id"]),
                    None if payload.get("parent_id") is None else cast(int, payload["parent_id"]),
                    cast(int, payload["author_id"]),
                    None if payload.get("title") is None else str(payload["title"]),
                    str(payload["body"]),
                    cast(float, payload["created_at"]),
                    None if payload.get("society") is None else cast(int, payload["society"]),
                ),
            )
        elif kind == "board.vote":
            self._conn.execute(
                """
                INSERT INTO thread_votes (run_id, thread_id, votes) VALUES (%s, %s, %s)
                ON CONFLICT (run_id, thread_id) DO UPDATE SET votes = EXCLUDED.votes
                """,
                (run_id, cast(int, payload["thread_id"]), cast(int, payload["votes"])),
            )
        elif kind == "god.clear_board":
            self._conn.execute("DELETE FROM posts WHERE run_id = %s", (run_id,))
            self._conn.execute("DELETE FROM thread_votes WHERE run_id = %s", (run_id,))

    def append(
        self, run_id: str, ts: float, kind: str, agent_id: int | None, payload: Mapping[str, object]
    ) -> int:
        delta = run_deltas(kind, payload)
        serialized = json.dumps(payload, separators=(",", ":"))

        def write() -> int:
            if run_id not in self._known_runs or delta.has_changes:
                self._upsert_run(run_id, ts, ts, delta)
            row = self._conn.execute(
                "INSERT INTO events (run_id, ts, kind, agent_id, payload)"
                " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (run_id, ts, kind, agent_id, serialized),
            ).fetchone()
            if row is None:
                raise RuntimeError("event insert did not return an id")
            self._update_projection(run_id, kind, payload)
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

    def append_many(self, rows: Sequence[_Row]) -> None:
        if not rows:
            return

        deltas: dict[str, RunDelta] = {}
        created_at: dict[str, float] = {}
        last_ts: dict[str, float] = {}
        for run_id, ts, kind, _, payload in rows:
            delta = run_deltas(kind, payload)
            previous = deltas.get(run_id, RunDelta())
            deltas[run_id] = RunDelta(
                pixels=previous.pixels + delta.pixels,
                posts=previous.posts + delta.posts,
                turns=previous.turns + delta.turns,
                errors=previous.errors + delta.errors,
                updates=previous.updates + delta.updates,
                answers=previous.answers + delta.answers,
                deleted=previous.deleted or delta.deleted,
                config=previous.config if previous.config is not None else delta.config,
                state=delta.state if delta.state is not None else previous.state,
            )
            if run_id in created_at:
                created_at[run_id] = min(created_at[run_id], ts)
                last_ts[run_id] = max(last_ts[run_id], ts)
            else:
                created_at[run_id] = ts
                last_ts[run_id] = ts

        event_rows = [
            (run_id, ts, kind, agent_id, json.dumps(payload, separators=(",", ":")))
            for run_id, ts, kind, agent_id, payload in rows
        ]

        with self._conn.transaction(), self._conn.pipeline():
            for run_id, delta in deltas.items():
                if run_id not in self._known_runs or delta.has_changes:
                    self._upsert_run(run_id, created_at[run_id], last_ts[run_id], delta)

            with self._conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO events (run_id, ts, kind, agent_id, payload)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    event_rows,
                )

            for run_id, _, kind, _, payload in rows:
                self._update_projection(run_id, kind, payload)

        self._known_runs.update(deltas)

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

    def read(
        self, run_id: str, after_id: int = 0, limit: int = 1000, *, canvas_only: bool = False
    ) -> list[Event]:
        started = time.perf_counter()
        kind_filter = f" AND kind IN ({CANVAS_KINDS_SQL})" if canvas_only else ""
        rows = self._conn.execute(
            "SELECT id, run_id, ts, kind, agent_id, payload FROM events"
            f" WHERE run_id = %s AND id > %s{kind_filter} ORDER BY run_id, id LIMIT %s",
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

    def canvas_span(self, run_id: str) -> tuple[float, float, int] | None:
        first = self._conn.execute(
            "SELECT id, ts FROM events"
            f" WHERE run_id = %s AND kind IN ({CANVAS_KINDS_SQL}) ORDER BY id ASC LIMIT 1",
            (run_id,),
        ).fetchone()
        if first is None:
            return None
        last = self._conn.execute(
            "SELECT id, ts FROM events"
            f" WHERE run_id = %s AND kind IN ({CANVAS_KINDS_SQL}) ORDER BY id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if last is None:
            return None
        return float(first[1]), float(last[1]), int(last[0])

    def canvas_events(
        self, run_id: str, after_id: int, limit: int
    ) -> list[tuple[int, float, str, int | None, int | None, int | None]]:
        started = time.perf_counter()
        rows = self._conn.execute(
            "SELECT id, ts, kind, (payload::jsonb ->> 'x')::int,"
            " (payload::jsonb ->> 'y')::int, (payload::jsonb ->> 'color')::int"
            " FROM events"
            f" WHERE run_id = %s AND id > %s AND kind IN ({CANVAS_KINDS_SQL})"
            " ORDER BY run_id, id LIMIT %s",
            (run_id, after_id, limit),
        ).fetchall()
        elapsed = time.perf_counter() - started
        if elapsed >= SLOW_READ_SECONDS:
            logger.warning(
                "slow event read run=%s after=%d limit=%d rows=%d in %.2fs",
                run_id,
                after_id,
                limit,
                len(rows),
                elapsed,
            )
        return [
            (
                int(row[0]),
                float(row[1]),
                str(row[2]),
                None if row[3] is None else int(row[3]),
                None if row[4] is None else int(row[4]),
                None if row[5] is None else int(row[5]),
            )
            for row in rows
        ]

    def count(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = %s", (run_id,)
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def run_meta(self, run_id: str) -> RunMeta | None:
        row = self._conn.execute(
            "SELECT created_at, config, deleted FROM runs WHERE id = %s", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunMeta(
            created_at=float(row[0]),
            config=None if row[1] is None else str(row[1]),
            deleted=bool(row[2]),
        )

    def posts(self, run_id: str, thread_id: int | None = None) -> list[tuple[Post, int | None]]:
        query = (
            "SELECT id, author_id, thread_id, parent_id, title, body, created_at, society"
            " FROM posts WHERE run_id = %s"
        )
        params: tuple[object, ...] = (run_id,)
        if thread_id is not None:
            query += " AND thread_id = %s"
            params += (thread_id,)
        rows = self._conn.execute(query + " ORDER BY id", params).fetchall()
        return [
            (
                Post(
                    id=int(row[0]),
                    author_id=int(row[1]),
                    thread_id=int(row[2]),
                    parent_id=None if row[3] is None else int(row[3]),
                    title=None if row[4] is None else str(row[4]),
                    body=str(row[5]),
                    created_at=float(row[6]),
                ),
                None if row[7] is None else int(row[7]),
            )
            for row in rows
        ]

    def thread_votes(self, run_id: str) -> dict[int, int]:
        rows = self._conn.execute(
            "SELECT thread_id, votes FROM thread_votes WHERE run_id = %s", (run_id,)
        ).fetchall()
        return {int(row[0]): int(row[1]) for row in rows}

    def run_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT id FROM runs ORDER BY id").fetchall()
        return [str(r[0]) for r in rows]

    def run_totals(
        self,
    ) -> list[tuple[str, float, int, int, int, int, int, int, bool, str | None, float | None]]:
        """Return maintained per-run counters, ordered by run id."""
        rows = self._conn.execute(
            "SELECT id, created_at, pixels, posts, turns, errors, updates, answers, deleted,"
            " config, last_ts"
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
                None if r[10] is None else float(r[10]),
            )
            for r in rows
        ]

    def interrupted_runs(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT id FROM runs WHERE state = 'interrupted' AND NOT deleted ORDER BY id"
        ).fetchall()
        return [str(r[0]) for r in rows]

    def reader(self) -> PostgresEventLog:
        """A log on a new connection for use from another thread; ``close`` it when done."""
        return PostgresEventLog(self.database_url, initialize_schema=False)

    def close(self) -> None:
        self._conn.close()


class BufferedPostgresEventLog(EventStore):
    """Postgres event log that keeps appends off the caller's thread.

    ``append`` only enqueues; a single writer thread drains the queue and inserts each flush in
    one transaction, so a burst of pixel writes never blocks the asyncio loop on a DB round-trip.
    Reads use a separate connection and only see flushed (committed) events.
    """

    def __init__(
        self,
        database_url: str,
        *,
        flush_interval: float = 0.1,
        batch_size: int = 2000,
        initialize_schema: bool = True,
    ) -> None:
        if flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.database_url = database_url
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self._reader = PostgresEventLog(database_url, initialize_schema=initialize_schema)
        self._reader_lock = threading.Lock()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop = object()
        self._closed = False
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._writer, name="event-log-writer", daemon=True)
        self._thread.start()

    def append(
        self, run_id: str, ts: float, kind: str, agent_id: int | None, payload: Mapping[str, object]
    ) -> int:
        self._queue.put((run_id, ts, kind, agent_id, payload))
        return 0

    def batch(self) -> AbstractContextManager[None]:
        # The writer already groups each flush into one transaction.
        return nullcontext()

    def read(
        self, run_id: str, after_id: int = 0, limit: int = 1000, *, canvas_only: bool = False
    ) -> list[Event]:
        return self._read(
            lambda r: r.read(run_id, after_id=after_id, limit=limit, canvas_only=canvas_only)
        )

    def canvas_span(self, run_id: str) -> tuple[float, float, int] | None:
        return self._read(lambda r: r.canvas_span(run_id))

    def canvas_events(
        self, run_id: str, after_id: int, limit: int
    ) -> list[tuple[int, float, str, int | None, int | None, int | None]]:
        return self._read(lambda r: r.canvas_events(run_id, after_id, limit))

    def count(self, run_id: str) -> int:
        return self._read(lambda r: r.count(run_id))

    def run_meta(self, run_id: str) -> RunMeta | None:
        return self._read(lambda r: r.run_meta(run_id))

    def posts(self, run_id: str, thread_id: int | None = None) -> list[tuple[Post, int | None]]:
        return self._read(lambda r: r.posts(run_id, thread_id))

    def thread_votes(self, run_id: str) -> dict[int, int]:
        return self._read(lambda r: r.thread_votes(run_id))

    def run_ids(self) -> list[str]:
        return self._read(lambda r: r.run_ids())

    def run_totals(
        self,
    ) -> list[tuple[str, float, int, int, int, int, int, int, bool, str | None, float | None]]:
        return self._read(lambda r: r.run_totals())

    def interrupted_runs(self) -> list[str]:
        return self._read(lambda r: r.interrupted_runs())

    def reader(self) -> PostgresEventLog:
        """An unbuffered log on a new connection, for readers in other threads."""
        return PostgresEventLog(self.database_url, initialize_schema=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closing.set()
        self._queue.put(self._stop)
        self._thread.join(timeout=10)
        self._reader.close()

    def _read(self, query: Callable[[PostgresEventLog], T]) -> T:
        reader = self._reader
        try:
            return query(reader)
        except psycopg.OperationalError:
            if not _connection_lost(reader):
                raise
            logger.warning("event log reader lost its connection; reconnecting", exc_info=True)
            with self._reader_lock:
                if self._reader is reader:
                    with suppress(Exception):
                        reader.close()
                    self._reader = PostgresEventLog(self.database_url, initialize_schema=False)
            return query(self._reader)

    def _connect_writer(self) -> PostgresEventLog:
        return PostgresEventLog(
            self.database_url,
            initialize_schema=False,
            lock_timeout_ms=30_000,
            statement_timeout_ms=300_000,
        )

    def _writer(self) -> None:
        log: PostgresEventLog | None = None
        try:
            while True:
                batch: list[_Row] = []
                deadline = time.monotonic() + self.flush_interval
                stop = False
                while True:
                    wait = deadline - time.monotonic()
                    if batch and wait <= 0:
                        break
                    try:
                        item = self._queue.get(timeout=wait if batch else self.flush_interval)
                    except queue.Empty:
                        break
                    if item is self._stop:
                        stop = True
                        break
                    batch.append(item)
                    if len(batch) >= self.batch_size:
                        break
                if batch:
                    log = self._flush(log, batch)
                if stop:
                    break
        finally:
            if log is not None:
                log.close()

    def _flush(self, log: PostgresEventLog | None, batch: list[_Row]) -> PostgresEventLog | None:
        """Insert ``batch`` in one transaction and return the writer's (possibly new) connection.

        Lock/statement timeouts are retried a few times. A lost connection (e.g. a Cloud SQL
        restart) is reopened and the batch retried until it lands, holding later events in the
        queue meanwhile, so a database blip doesn't silently drop the rest of every live run.
        """
        timeouts = 0
        lost = 0
        while True:
            try:
                if log is None:
                    log = self._connect_writer()
                log.append_many(batch)
                if lost:
                    logger.warning("event log writer reconnected after %d attempts", lost)
                return log
            except (pg_errors.LockNotAvailable, pg_errors.QueryCanceled):
                timeouts += 1
                if timeouts == 5:
                    logger.exception(
                        "event log flush failed after retries; dropped %d events", len(batch)
                    )
                    return log
                logger.warning("event log flush hit a lock/statement timeout; retrying")
                time.sleep(min(2 ** (timeouts - 1), 8))
            except Exception:
                if log is not None and not _connection_lost(log):
                    logger.exception("event log flush failed; dropped %d events", len(batch))
                    return log
                lost += 1
                if self._closed and lost > 3:
                    logger.exception(
                        "event log unreachable while closing; dropped %d events", len(batch)
                    )
                    return log
                logger.warning(
                    "event log writer lost its connection (attempt %d); %d events waiting",
                    lost,
                    len(batch) + self._queue.qsize(),
                    exc_info=lost == 1,
                )
                if log is not None:
                    with suppress(Exception):
                        log.close()
                    log = None
                # Wait on the closing event so shutdown cuts a long backoff short.
                self._closing.wait(min(2.0 ** (lost - 1), RECONNECT_MAX_DELAY))
                if self._closed:
                    time.sleep(0.5)


def _connection_lost(log: PostgresEventLog) -> bool:
    conn = log._conn
    return bool(conn.closed or conn.broken)
