import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    agent_id INTEGER,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run ON events (run_id, id);
"""


@dataclass(frozen=True, slots=True)
class Event:
    id: int
    run_id: str
    ts: float
    kind: str
    agent_id: int | None
    payload: dict[str, object]


class EventLog:
    """Append-only SQLite log of everything that happens in a run, for replay and audit."""

    _batching = False

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)

    def append(
        self, run_id: str, ts: float, kind: str, agent_id: int | None, payload: Mapping[str, object]
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO events (run_id, ts, kind, agent_id, payload) VALUES (?, ?, ?, ?, ?)",
            (run_id, ts, kind, agent_id, json.dumps(payload, separators=(",", ":"))),
        )
        if not self._batching:
            self._conn.commit()
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
        rows = self._conn.execute(
            "SELECT id, run_id, ts, kind, agent_id, payload FROM events"
            " WHERE run_id = ? AND id > ? ORDER BY id LIMIT ?",
            (run_id, after_id, limit),
        ).fetchall()
        return [Event(r[0], r[1], r[2], r[3], r[4], json.loads(r[5])) for r in rows]

    def count(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row[0])

    def run_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT run_id FROM events ORDER BY run_id").fetchall()
        return [str(r[0]) for r in rows]

    def run_totals(
        self, rejected_prefix: str
    ) -> list[tuple[str, float, int, int, int, int, int, int, bool, str | None]]:
        """Per run: id, first ts, counts of pixels/posts/turns/rejected turns/narrator updates/
        narrator answers, deleted?, config JSON (if created). Rejected turns are thoughts
        starting with ``rejected_prefix``."""
        narrated = "kind = 'narrator.update' AND json_extract(payload, '$.kind')"
        rows = self._conn.execute(
            "SELECT run_id, MIN(ts), SUM(kind = 'pixel'), SUM(kind = 'post'),"
            " SUM(kind = 'agent.thought'),"
            " SUM(kind = 'agent.thought' AND substr(json_extract(payload, '$.thought'), 1, ?) = ?),"
            f" SUM({narrated} IS NOT 'answer'), SUM({narrated} = 'answer'),"
            " MAX(kind = 'run.deleted'),"
            " MAX(CASE WHEN kind = 'run.created' THEN json_extract(payload, '$.config') END)"
            " FROM events GROUP BY run_id ORDER BY run_id",
            (len(rejected_prefix), rejected_prefix),
        ).fetchall()
        return [(*r[:8], bool(r[8]), r[9]) for r in rows]

    def reader(self) -> "EventLog":
        """A log on the same database for use from another thread; ``close`` it when done.
        In-memory logs have no second connection, so the shared one is returned."""
        return self if self.path == ":memory:" else EventLog(self.path)

    def close(self) -> None:
        self._conn.close()
