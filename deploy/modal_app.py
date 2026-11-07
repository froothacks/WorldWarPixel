"""Host WorldWarPixel on Modal: the app runs in a long-lived CPU Sandbox.

Sandboxes are capped at 24h and get a new tunnel URL each time, so this module also deploys an
hourly ``keep_alive`` job (restarts the sandbox if it died, replaces it before the cap) and a
``home`` web endpoint with a stable URL that redirects to whichever sandbox is current. The event
log is SQLite on the sandbox disk, backed up to a Volume (see ``serve.py``). Live runs are held
in memory, so they end whenever the sandbox is replaced; their event history survives.

    uv run --with modal python deploy/modal_app.py launch   # build + publish image, (re)start
    uv run --with modal modal deploy deploy/modal_app.py    # keeper job + stable redirect URL
    uv run --with modal python deploy/modal_app.py status | restart | backup | stop
"""

import sys
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "worldwarpixel"
IMAGE_NAME = "worldwarpixel-sandbox"
SANDBOX_NAME = "worldwarpixel-web"
VOLUME_NAME = "worldwarpixel-data"
SECRET_NAME = "worldwarpixel"
PORT = 8000
CPU = 1.0
MEMORY_MB = 2048
LIFETIME = 24 * 3600
REPLACE_AFTER = 23 * 3600
NODE_VERSION = "22.12.0"
ROOT = Path(__file__).resolve().parents[1]


def build_image() -> modal.Image:
    node = f"node-v{NODE_VERSION}-linux-x64"
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_sync(str(ROOT), extra_options="--no-dev")
        .apt_install("curl", "ca-certificates", "xz-utils")
        .run_commands(
            f"curl -fsSL https://nodejs.org/dist/v{NODE_VERSION}/{node}.tar.xz"
            " | tar -xJ -C /usr/local --strip-components=1"
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
            }
        )
    )


def current() -> modal.Sandbox | None:
    try:
        sb = modal.Sandbox.from_name(APP_NAME, SANDBOX_NAME)
    except modal.exception.NotFoundError:
        return None
    return sb if sb.poll() is None else None


def url(sb: modal.Sandbox) -> str:
    return str(sb.tunnels()[PORT].url)


def start() -> modal.Sandbox:
    sb = modal.Sandbox.create(
        "python",
        "/app/deploy/serve.py",
        app=modal.App.lookup(APP_NAME, create_if_missing=True),
        name=SANDBOX_NAME,
        image=modal.Image.from_name(IMAGE_NAME),
        secrets=[modal.Secret.from_name(SECRET_NAME)],
        volumes={"/data": modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)},
        encrypted_ports=[PORT],
        readiness_probe=modal.Probe.with_tcp(PORT),
        timeout=LIFETIME,
        cpu=CPU,
        memory=MEMORY_MB,
        workdir="/app",
    )
    sb.set_tags({"started": str(int(time.time()))})
    sb.wait_until_ready()
    return sb


def backup(sb: modal.Sandbox) -> None:
    p = sb.exec("python", "/app/deploy/serve.py", "backup", timeout=600)
    code = p.wait()
    print(p.stdout.read(), p.stderr.read(), end="")
    if code != 0:
        raise RuntimeError(f"backup exited with {code}; sandbox left running")


def stop(sb: modal.Sandbox) -> None:
    backup(sb)
    sb.terminate(wait=True)


def age(sb: modal.Sandbox) -> float:
    return time.time() - int(sb.get_tags().get("started", "0"))


def ensure(force: bool = False) -> modal.Sandbox:
    """The running sandbox, replacing it when forced, nearly at the 24h cap, or gone."""
    sb = current()
    if sb is not None and not force and age(sb) < REPLACE_AFTER:
        return sb
    if sb is not None:
        stop(sb)
    return start()


app = modal.App(APP_NAME)
control_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("fastapi[standard]")


@app.function(image=control_image, schedule=modal.Period(hours=1), timeout=900)
def keep_alive() -> None:
    print(url(ensure()))


@app.function(image=control_image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def home() -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse, RedirectResponse, Response

    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    cache: dict[str, Any] = {"url": None, "at": 0.0}

    @web.get("/{path:path}")
    def redirect(path: str, request: Request) -> Response:
        if cache["url"] is None or time.time() - cache["at"] > 30:
            sb = current()
            cache["url"], cache["at"] = (url(sb) if sb else None), time.time()
        if cache["url"] is None:
            return PlainTextResponse("WorldWarPixel is restarting, try again in a minute.", 503)
        query = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(f"{cache['url']}/{path}{query}", status_code=307)

    return web


def main(cmd: str) -> None:
    sb: modal.Sandbox | None
    if cmd == "launch":
        with modal.enable_output():
            build_image().build(modal.App.lookup(APP_NAME, create_if_missing=True)).publish(
                IMAGE_NAME
            )
        sb = ensure(force=True)
    elif cmd == "restart":
        sb = ensure(force=True)
    elif cmd in ("status", "backup", "stop"):
        sb = current()
        if sb is None:
            print("no sandbox running")
            return
        if cmd == "backup":
            backup(sb)
        elif cmd == "stop":
            stop(sb)
            print("stopped")
            return
    else:
        raise SystemExit(f"unknown command {cmd!r}; use launch|restart|status|backup|stop")
    print(f"sandbox {sb.object_id}, up {age(sb) / 3600:.1f}h: {url(sb)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "status")
