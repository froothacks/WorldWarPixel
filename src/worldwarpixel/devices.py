"""Society screens (ESP32 touchscreens or the browser emulator).

Each device is bound to one society by its bearer token (``DEVICE_TOKENS``). It receives that
society's live state and can send a short voice clip; the clip is transcribed and condensed into
the society's unattributed murmur. With no tokens configured the endpoints are open and the
society comes from ``?society=`` (local development only).
"""

import asyncio
import contextlib
import hmac
import json
import logging
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from worldwarpixel.api import RunManager
from worldwarpixel.config import get_settings
from worldwarpixel.sim.canvas import PALETTE
from worldwarpixel.sim.run import Run
from worldwarpixel.voice import (
    MAX_SECONDS,
    RATE,
    Murmurer,
    Transcriber,
    TranscriptionUnavailable,
    wav_to_pcm,
)

log = logging.getLogger(__name__)

STATE_INTERVAL = 2.0
MAX_THUMB = 320


@dataclass(frozen=True, slots=True)
class Device:
    id: str
    society: int


def parse_tokens(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in raw.split(","):
        token, _, society = item.strip().rpartition(":")
        if token and society.isdigit():
            out[token] = int(society)
    return out


def authenticate(token: str | None, society: int | None) -> Device | None:
    tokens = parse_tokens(get_settings().device_tokens.get_secret_value())
    if not tokens:
        s = society or 0
        return Device(f"dev-{s}", s)
    for i, (known, s) in enumerate(tokens.items()):
        if token and hmac.compare_digest(token.encode(), known.encode()):
            return Device(f"screen-{i + 1}", s)
    return None


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def require_device(
    authorization: Annotated[str | None, Header()] = None,
    token: Annotated[str | None, Query()] = None,
    society: Annotated[int | None, Query(ge=0, le=5)] = None,
) -> Device:
    device = authenticate(_bearer(authorization) or token, society)
    if device is None:
        raise HTTPException(401, "invalid device token")
    return device


DeviceDep = Annotated[Device, Depends(require_device)]


def current_run(manager: RunManager) -> Run | None:
    """The newest live run with societies (screens follow whatever is on show)."""
    live = [r for r in manager.active() if r.societies is not None]
    return max(live, key=lambda r: r.started_at or 0.0, default=None)


def _run_for(request: Request, device: Device) -> Run:
    run = current_run(request.app.state.runs)
    if run is None or run.societies is None:
        raise HTTPException(409, "no running world with societies")
    if device.society >= run.societies.count:
        raise HTTPException(409, f"this world has no society {device.society}")
    return run


def device_state(run: Run, device: Device, voice: dict[str, Any]) -> dict[str, Any]:
    societies = run.society_summary()
    mine = societies[device.society]
    return {
        "type": "state",
        "run": run.id,
        "society": {k: mine[k] for k in ("id", "name", "color", "members", "joined", "left")},
        "societies": [{k: s[k] for k in ("id", "name", "color", "members")} for s in societies],
        "threads": [
            {"title": d["title"], "author": d["author"], "replies": d["replies"]}
            for d in run.society_threads(device.society, limit=3)
        ],
        "murmur": mine["murmur"],
        "voice": voice,
    }


def canvas_thumb(run: Run, w: int, h: int) -> bytes:
    """Nearest-neighbour downsample: row-major palette indices, one byte per pixel."""
    c = run.canvas
    src = c.pixels
    xs = [x * c.width // w for x in range(w)]
    out = bytearray(w * h)
    for y in range(h):
        row = (y * c.height // h) * c.width
        out[y * w : (y + 1) * w] = bytes(src[row + x] for x in xs)
    return bytes(out)


async def speak(request_app: Any, run: Run, society: int, transcript: str) -> dict[str, Any]:
    transcript = " ".join(transcript.split())
    if not transcript:
        raise ValueError("nothing was heard")
    murmurer: Murmurer = request_app.state.murmurer
    murmur = await murmurer(transcript)
    return run.set_murmur(society, murmur, transcript)


async def transcribe(request_app: Any, pcm: bytes, rate: int) -> str:
    transcriber: Transcriber = request_app.state.transcriber
    return await transcriber(pcm, rate)


router = APIRouter(prefix="/devices", tags=["devices"])


@router.get("/me")
async def device_me(request: Request, device: DeviceDep) -> dict[str, Any]:
    run = current_run(request.app.state.runs)
    out: dict[str, Any] = {"device": device.id, "society": device.society, "palette": PALETTE}
    if run is not None and run.societies is not None and device.society < run.societies.count:
        out["state"] = device_state(run, device, {"status": "idle"})
    return out


@router.get("/canvas")
async def device_canvas(
    request: Request,
    device: DeviceDep,
    w: Annotated[int, Query(ge=1, le=MAX_THUMB)] = 160,
    h: Annotated[int, Query(ge=1, le=MAX_THUMB)] = 100,
) -> Response:
    run = _run_for(request, device)
    return Response(canvas_thumb(run, w, h), media_type="application/octet-stream")


class Say(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@router.post("/say")
async def device_say(request: Request, body: Say, device: DeviceDep) -> dict[str, Any]:
    """Typed instead of spoken: same path as a voice clip minus transcription."""
    run = _run_for(request, device)
    try:
        return await speak(request.app, run, device.society, body.text)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/voice")
async def device_voice(request: Request, device: DeviceDep) -> dict[str, Any]:
    """Body: a 16-bit WAV, or raw 16 kHz mono s16le PCM (Content-Type audio/L16)."""
    run = _run_for(request, device)
    data = await request.body()
    try:
        if data[:4] == b"RIFF":
            pcm, rate = wav_to_pcm(data)
        else:
            pcm, rate = data, RATE
        if len(pcm) > MAX_SECONDS * rate * 2:
            raise ValueError(f"clip longer than {MAX_SECONDS} s")
        transcript = await transcribe(request.app, pcm, rate)
        return await speak(request.app, run, device.society, transcript)
    except TranscriptionUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.websocket("/ws")
async def device_socket(websocket: WebSocket) -> None:
    q = websocket.query_params
    society = q.get("society")
    device = authenticate(
        _bearer(websocket.headers.get("authorization")) or q.get("token"),
        int(society) if society and society.isdigit() else None,
    )
    if device is None:
        await websocket.close(code=4401, reason="invalid device token")
        return
    await websocket.accept()
    await websocket.send_json(
        {"type": "hello", "device": device.id, "society": device.society, "palette": PALETTE}
    )
    voice: dict[str, Any] = {"status": "idle", "transcript": None, "error": None}
    clip: bytearray | None = None
    rate = RATE
    seen = 0
    app = websocket.app
    busy: asyncio.Task[None] | None = None

    async def push() -> None:
        nonlocal seen
        run = current_run(app.state.runs)
        if run is None or run.societies is None or device.society >= run.societies.count:
            await websocket.send_json({"type": "state", "run": None, "voice": voice})
            return
        migs = list(run.societies.migrations)
        for m in migs[seen:] if seen <= len(migs) else migs:
            if device.society in (m.src, m.dst):
                await websocket.send_json(
                    {
                        "type": "migration",
                        "agent": run.agent_name(m.agent_id),
                        "from": m.src,
                        "to": m.dst,
                    }
                )
        seen = len(migs)
        await websocket.send_json(device_state(run, device, voice))

    async def finish(pcm: bytes) -> None:
        voice.update(status="transcribing", error=None)
        try:
            await push()
            run = current_run(app.state.runs)
            if run is None:
                raise ValueError("no running world with societies")
            transcript = await transcribe(app, pcm, rate)
            voice.update(transcript=transcript)
            await speak(app, run, device.society, transcript)
            voice.update(status="done")
        except (TranscriptionUnavailable, ValueError) as e:
            voice.update(status="error", error=str(e))
        except Exception as e:
            log.exception("voice clip failed")
            voice.update(status="error", error=type(e).__name__)
        with contextlib.suppress(Exception):
            await push()

    async def ticker() -> None:
        while True:
            await push()
            await asyncio.sleep(STATE_INTERVAL)

    tick = asyncio.create_task(ticker())
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if clip is not None:
                    clip.extend(msg["bytes"])
                    if len(clip) > MAX_SECONDS * rate * 2:
                        clip = None
                        voice.update(status="error", error=f"clip longer than {MAX_SECONDS} s")
                continue
            try:
                data = json.loads(msg.get("text") or "{}")
            except ValueError:
                continue
            kind = data.get("type")
            if kind == "ping":
                await websocket.send_json({"type": "pong"})
            elif kind == "voice.start":
                rate = int(data.get("rate", RATE))
                clip = bytearray()
                voice.update(status="listening", error=None)
            elif kind == "voice.cancel":
                clip = None
                voice.update(status="idle")
            elif kind == "voice.end" and clip is not None:
                pcm, clip = bytes(clip), None
                if busy is None or busy.done():
                    busy = asyncio.create_task(finish(pcm))
    except WebSocketDisconnect:
        pass
    finally:
        tick.cancel()
        if busy is not None:
            busy.cancel()
