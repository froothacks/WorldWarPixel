"""Restore and back up the event log for the Modal app and legacy Sandbox.

SQLite lives on the container's local disk (WAL needs real file locking, which a network Volume
does not promise). The Modal ASGI container calls ``restore()`` before app startup and backs up
periodically and on exit. ``python serve.py`` remains the legacy server entrypoint, and
``python serve.py backup`` is used to retire its Sandbox. Backups are committed with ``sync``
(Volumes v2).
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
# Cloudflare Tunnel token (Zero Trust → Networks → Tunnels). When set, cloudflared dials out
# to Cloudflare and serves this app on the tunnel's public hostnames (e.g. worldwarpixel.com),
# so the custom domain works without Modal's paid custom-domain feature.
TUNNEL_TOKEN = os.environ.get("CLOUDFLARED_TUNNEL_TOKEN", "")


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


def start_tunnel() -> subprocess.Popen[bytes] | None:
    if not TUNNEL_TOKEN:
        return None
    # Remotely-managed tunnel: hostnames and DNS live in the Cloudflare dashboard; the token
    # is the only thing the container needs. --no-autoupdate keeps the image reproducible.
    return subprocess.Popen(
        ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", TUNNEL_TOKEN]
    )


def restore() -> None:
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
    next_backup = time.monotonic() + INTERVAL
    while not stopping and server.poll() is None:
        time.sleep(1)
        if tunnel is not None and tunnel.poll() is not None:
            print("cloudflared exited; restarting", flush=True)
            tunnel = start_tunnel()
        if time.monotonic() >= next_backup:
            next_backup += INTERVAL
            try:
                backup()
            except Exception as exc:
                print(f"backup failed: {exc!r}", flush=True)
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
    if tunnel is not None and tunnel.poll() is None:
        tunnel.terminate()
        try:
            tunnel.wait(10)
        except subprocess.TimeoutExpired:
            tunnel.kill()
    backup()
    return server.returncode or 0


if __name__ == "__main__":
    if sys.argv[1:] == ["backup"]:
        backup()
    else:
        sys.exit(serve())
