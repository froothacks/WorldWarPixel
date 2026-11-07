"""Sandbox entrypoint: restore the event log from the Volume, run the app, back the log up.

SQLite lives on the sandbox's local disk (WAL needs real file locking, which a network Volume
does not promise). A consistent copy is written to the Volume every ``BACKUP_INTERVAL`` seconds,
on SIGTERM and on ``python serve.py backup``, then committed with ``sync`` (Volumes v2).
"""

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

LIVE = Path(os.environ.get("EVENT_LOG_PATH", "/root/state/events.db"))
BACKUP = Path(os.environ.get("EVENT_LOG_BACKUP", "/data/events.db"))
INTERVAL = float(os.environ.get("BACKUP_INTERVAL", "300"))
PORT = os.environ.get("PORT", "8000")


def copy_db(src: Path, dst: Path) -> None:
    """Online SQLite backup (includes un-checkpointed WAL pages), swapped in atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    source, target = sqlite3.connect(src), sqlite3.connect(tmp)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    os.replace(tmp, dst)


def backup() -> None:
    if not LIVE.exists():
        return
    started = time.monotonic()
    copy_db(LIVE, BACKUP)
    subprocess.run(["sync", str(BACKUP.parent)], check=False)
    size = BACKUP.stat().st_size / 1e6
    print(f"backed up {size:.1f} MB to {BACKUP} in {time.monotonic() - started:.1f}s", flush=True)


def serve() -> int:
    if BACKUP.exists() and not LIVE.exists():
        copy_db(BACKUP, LIVE)
        print(f"restored event log from {BACKUP}", flush=True)
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "EVENT_LOG_PATH": str(LIVE)}
    cmd = [sys.executable, "-m", "uvicorn", "worldwarpixel.app:app", "--host", "0.0.0.0"]
    server = subprocess.Popen([*cmd, "--port", PORT], env=env)
    stopping = False

    def stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    next_backup = time.monotonic() + INTERVAL
    while not stopping and server.poll() is None:
        time.sleep(1)
        if time.monotonic() >= next_backup:
            next_backup += INTERVAL
            try:
                backup()
            except Exception as exc:
                print(f"backup failed: {exc!r}", flush=True)
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()
    backup()
    return server.returncode or 0


if __name__ == "__main__":
    if sys.argv[1:] == ["backup"]:
        backup()
    else:
        sys.exit(serve())
