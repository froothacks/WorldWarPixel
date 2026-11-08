import importlib.util
import sqlite3
from pathlib import Path
from typing import Any

import pytest

SERVE_PATH = Path(__file__).resolve().parents[1] / "deploy" / "serve.py"


def _load_serve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, database_url: str | None
) -> tuple[Any, Path, Path]:
    live = tmp_path / "live" / "events.db"
    backup = tmp_path / "backup" / "events.db"
    monkeypatch.setenv("EVENT_LOG_PATH", str(live))
    monkeypatch.setenv("EVENT_LOG_BACKUP", str(backup))
    monkeypatch.setenv("BACKUP_INTERVAL", "60")
    if database_url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", database_url)

    spec = importlib.util.spec_from_file_location("serve", SERVE_PATH)
    assert spec is not None and spec.loader is not None
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    return serve, live, backup


def _create_db(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE records (value TEXT)")
        connection.execute("INSERT INTO records VALUES (?)", (value,))


def _read_values(path: Path) -> list[str]:
    with sqlite3.connect(path) as connection:
        return [row[0] for row in connection.execute("SELECT value FROM records")]


def test_restore_skips_sqlite_when_database_url_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    serve, live, backup = _load_serve(monkeypatch, tmp_path, "postgresql://example")
    _create_db(backup, "backup")

    serve.restore()

    assert not live.exists()
    assert capsys.readouterr().out == (
        "DATABASE_URL set: event log is in Postgres, skipping SQLite restore and backups\n"
    )


def test_backup_skips_sqlite_when_database_url_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    serve, live, backup = _load_serve(monkeypatch, tmp_path, "postgresql://example")
    _create_db(live, "live")

    serve.backup()

    assert live.exists()
    assert not backup.exists()


def test_restore_and_backup_copy_sqlite_when_database_url_is_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    serve, live, backup = _load_serve(monkeypatch, tmp_path, None)
    _create_db(backup, "backup")

    serve.restore()

    assert _read_values(live) == ["backup"]
    with sqlite3.connect(live) as connection:
        connection.execute("INSERT INTO records VALUES (?)", ("live",))

    serve.backup()

    assert _read_values(backup) == ["backup", "live"]
