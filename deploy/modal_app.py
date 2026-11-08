"""Deploy WorldWarPixel through an in-process ASGI app in one always-on Modal container.

The FastAPI app serves directly at a stable ``*.modal.run`` URL, with no 24-hour Sandbox lifetime
cap. A single WebSocket connection is limited to 24 hours. SQLite is restored from and backed up
to the ``worldwarpixel-data`` Volume every 60 seconds and on exit. Live runs are held in memory, so
a deploy replaces the container and live runs reopen as stopped runs. Deploys may briefly overlap
old and new containers, so up to 60 seconds of events from the old container can be lost.

    uv run --with modal==1.6.3 modal deploy deploy/modal_app.py
    uv run --with modal==1.6.3 python deploy/modal_app.py retire  # retire legacy Sandbox once
"""

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import modal

APP_NAME = "worldwarpixel"
SANDBOX_NAME = "worldwarpixel-web"
VOLUME_NAME = "worldwarpixel-data"
SECRET_NAME = "worldwarpixel"
PORT = 8000
CPU = 1.0
MEMORY_MB = 2048
NODE_VERSION = "22.12.0"
ROOT = Path(__file__).resolve().parents[1]


def build_image() -> modal.Image:
    node = f"node-v{NODE_VERSION}-linux-x64"
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_sync(str(ROOT), extra_options="--no-dev --extra voice")
        # Society screens' voice clips are transcribed in-process; bake the Whisper weights in.
        .run_commands(
            'python -c "from faster_whisper import WhisperModel;'
            " WhisperModel('base', device='cpu', compute_type='int8')\""
        )
        .apt_install("curl", "ca-certificates", "xz-utils")
        .run_commands(
            f"curl -fsSL https://nodejs.org/dist/v{NODE_VERSION}/{node}.tar.xz"
            " | tar -xJ -C /usr/local --strip-components=1"
        )
        # Cloudflare Tunnel client; the container dials out, so no inbound ports or Modal
        # custom domains (a paid feature) are needed to serve worldwarpixel.com.
        .run_commands(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64 -o /usr/local/bin/cloudflared"
            " && chmod +x /usr/local/bin/cloudflared"
        )
        .add_local_dir(
            ROOT / "frontend", "/app/frontend", copy=True, ignore=["node_modules", "dist"]
        )
        .run_commands("cd /app/frontend && npm ci && npm run build")
        .add_local_dir(ROOT / "src", "/app/src", copy=True, ignore=["**/__pycache__"])
        .add_local_file(ROOT / "deploy" / "serve.py", "/app/deploy/serve.py", copy=True)
        .env(
            {
                "PYTHONPATH": "/app/src",
                "EVENT_LOG_PATH": "/root/state/events.db",
                "EVENT_LOG_BACKUP": "/data/events.db",
                "BACKUP_INTERVAL": "60",
            }
        )
    )


def current() -> modal.Sandbox | None:
    try:
        sb = modal.Sandbox.from_name(APP_NAME, SANDBOX_NAME)
    except modal.exception.NotFoundError:
        return None
    return sb if sb.poll() is None else None


def backup(sb: modal.Sandbox) -> None:
    p = sb.exec("python", "/app/deploy/serve.py", "backup", timeout=600)
    code = p.wait()
    print(p.stdout.read(), p.stderr.read(), end="")
    if code != 0:
        raise RuntimeError(f"backup exited with {code}; sandbox left running")


app = modal.App(APP_NAME)


@app.cls(
    image=build_image(),
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={"/data": modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)},
    cpu=CPU,
    memory=MEMORY_MB,
    nonpreemptible=True,
    min_containers=1,
    max_containers=1,
    timeout=24 * 3600,
)
@modal.concurrent(max_inputs=1000)
class Web:
    _tunnel: subprocess.Popen[bytes] | None = None

    @modal.enter()
    def start(self) -> None:
        sys.path.insert(0, "/app/deploy")
        import serve

        serve.restore()
        self._serve = serve
        self._stop = threading.Event()
        self._tunnel = self._start_tunnel()

        def loop() -> None:
            while not self._stop.wait(serve.INTERVAL):
                if self._tunnel is not None and self._tunnel.poll() is not None:
                    print("cloudflared exited; restarting", flush=True)
                    self._tunnel = self._start_tunnel()
                elif self._tunnel is None and serve.TUNNEL_TOKEN:
                    # The initial launch failed (e.g. a transient error); keep retrying.
                    self._tunnel = self._start_tunnel()
                try:
                    serve.backup()
                except Exception as exc:
                    print(f"backup failed: {exc!r}", flush=True)

        self._backups = threading.Thread(target=loop, daemon=True)
        self._backups.start()

    def _start_tunnel(self) -> subprocess.Popen[bytes] | None:
        # A tunnel failure must never take down the app or the backup loop with it.
        try:
            tunnel = self._serve.start_tunnel()
        except Exception as exc:
            print(f"cloudflared failed to start: {exc!r}", flush=True)
            return None
        if tunnel is not None:
            print("cloudflared tunnel started", flush=True)
        return tunnel

    @modal.asgi_app(label="worldwarpixel-home")
    def web(self) -> Any:
        from worldwarpixel.app import app as web_app

        return web_app

    @modal.exit()
    def stop(self) -> None:
        self._stop.set()
        self._backups.join()  # never overlap the final backup with a periodic one
        if self._tunnel is not None and self._tunnel.poll() is None:
            self._tunnel.terminate()
            try:
                self._tunnel.wait(10)
            except subprocess.TimeoutExpired:
                self._tunnel.kill()
        self._serve.backup()


def main(cmd: str) -> None:
    if cmd != "retire":
        raise SystemExit(f"unknown command {cmd!r}; use retire")
    sb = current()
    if sb is None:
        print("no sandbox running")
        return
    backup(sb)
    sb.terminate(wait=True)
    print("stopped")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
