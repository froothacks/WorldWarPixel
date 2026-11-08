"""Restore and back up the event log for the Modal app and legacy Sandbox.

SQLite lives on the container's local disk (WAL needs real file locking, which a network Volume
does not promise). The Modal ASGI container calls ``restore()`` before app startup and backs up
periodically and on exit. ``python serve.py`` remains the legacy server entrypoint, and
``python serve.py backup`` is used to retire its Sandbox. Backups are committed with ``sync``
(Volumes v2). When ``DATABASE_URL`` is set, the event log is in Postgres and this module skips
SQLite restore and backup.
"""

import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import FrameType

LIVE = Path(os.environ.get("EVENT_LOG_PATH", "/root/state/events.db"))
BACKUP = Path(os.environ.get("EVENT_LOG_BACKUP", "/data/events.db"))
POSTGRES = bool(os.environ.get("DATABASE_URL"))
INTERVAL = float(os.environ.get("BACKUP_INTERVAL", "300"))
GRACE = float(os.environ.get("SHUTDOWN_GRACE", "10"))
PORT = os.environ.get("PORT", "8000")
# Cloudflare Tunnel token (Zero Trust → Networks → Tunnels). When set, cloudflared dials out
# to Cloudflare and serves this app on the tunnel's public hostnames (e.g. worldwarpixel.com),
# so the custom domain works without Modal's paid custom-domain feature.
TUNNEL_TOKEN = os.environ.get("CLOUDFLARED_TUNNEL_TOKEN", "")


def _is_local(path: Path) -> bool:
    """True when ``path`` sits on local disk that SQLite can lock, not FUSE/network mounts."""
    try:
        parent = str(path.parent.resolve())
        mounts = []
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) > 2:
                    mounts.append((parts[1].replace("\\040", " "), parts[2]))
        _, fstype = max(
            ((m, t) for m, t in mounts if parent == m or parent.startswith(m + "/")),
            key=lambda mt: len(mt[0]),
            default=("/", ""),
        )
        return not fstype.startswith(("fuse", "nfs", "cifs", "9p"))
    except OSError:
        return True


def copy_db(src: Path, dst: Path) -> None:
    """Online SQLite backup (includes un-checkpointed WAL pages), swapped in atomically.

    SQLite works on local disk only: ``dst`` may sit on a network mount (Modal Volume, Cloud
    Storage FUSE) and ``src`` may too on restore, and neither supports the locking/random writes
    SQLite needs. Copy the database to local disk first when it is not already there.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.gettempdir())
    local_src = src
    if src.parent != workdir and not _is_local(src):
        staged = workdir / f"wwp-src-{os.getpid()}.db"
        shutil.copyfile(src, staged)
        local_src = staged
    local = workdir / f"{src.name}.{os.getpid()}.backup"
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
    local.unlink(missing_ok=True)
    source, target = sqlite3.connect(local_src), sqlite3.connect(local)
    try:
        source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        source.backup(target)
    finally:
        target.close()
        source.close()
    try:
        shutil.copyfile(local, tmp)
        os.replace(tmp, dst)
    finally:
        local.unlink(missing_ok=True)
        if local_src is not src:
            local_src.unlink(missing_ok=True)


def backup() -> None:
    if POSTGRES:
        return
    if not LIVE.exists():
        return
    started = time.monotonic()
    copy_db(LIVE, BACKUP)
    subprocess.run(["sync", str(BACKUP.parent)], check=False)
    size = BACKUP.stat().st_size / 1e6
    print(f"backed up {size:.1f} MB to {BACKUP} in {time.monotonic() - started:.1f}s", flush=True)


def start_tunnel() -> subprocess.Popen[bytes] | None:
    if not TUNNEL_TOKEN:
        return None
    # Remotely-managed tunnel: hostnames and DNS live in the Cloudflare dashboard; the token
    # is the only thing the container needs. --no-autoupdate keeps the image reproducible.
    return subprocess.Popen(
        ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", TUNNEL_TOKEN]
    )


def restore() -> None:
    if POSTGRES:
        print(
            "DATABASE_URL set: event log is in Postgres, skipping SQLite restore and backups",
            flush=True,
        )
        return
    if BACKUP.exists() and not LIVE.exists():
        copy_db(BACKUP, LIVE)
        print(f"restored event log from {BACKUP}", flush=True)
    LIVE.parent.mkdir(parents=True, exist_ok=True)


def serve() -> int:
    restore()
    env = {**os.environ, "EVENT_LOG_PATH": str(LIVE)}
    cmd = [sys.executable, "-m", "uvicorn", "worldwarpixel.app:app", "--host", "0.0.0.0"]
    server = subprocess.Popen([*cmd, "--port", PORT], env=env)
    tunnel = start_tunnel()
    if tunnel is not None:
        print("cloudflared tunnel started", flush=True)
    stopping = False

    def stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    next_backup = None if POSTGRES else time.monotonic() + INTERVAL
    while not stopping and server.poll() is None:
        time.sleep(1)
        if tunnel is not None and tunnel.poll() is not None:
            print("cloudflared exited; restarting", flush=True)
            tunnel = start_tunnel()
        if next_backup is not None and time.monotonic() >= next_backup:
            next_backup += INTERVAL
            try:
                backup()
            except Exception as exc:
                print(f"backup failed: {exc!r}", flush=True)
    if server.poll() is None:
        # Snapshot first: Cloud Run / Modal give only seconds after SIGTERM, and the online backup
        # is consistent while the server still runs. A second pass catches events from shutdown.
        try:
            backup()
        except Exception as exc:
            print(f"pre-shutdown backup failed: {exc!r}", flush=True)
        server.terminate()
        try:
            server.wait(GRACE)
        except subprocess.TimeoutExpired:
            server.kill()
    if tunnel is not None and tunnel.poll() is None:
        tunnel.terminate()
        try:
            tunnel.wait(10)
        except subprocess.TimeoutExpired:
            tunnel.kill()
    try:
        backup()
    except Exception as exc:
        print(f"final backup failed: {exc!r}", flush=True)
        if server.returncode == 0 and not POSTGRES:
            return 1
    return server.returncode or 0


if __name__ == "__main__":
    if sys.argv[1:] == ["backup"]:
        backup()
    else:
        sys.exit(serve())
