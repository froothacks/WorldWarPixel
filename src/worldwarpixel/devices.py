"""Society screens (ESP32 touchscreens or the browser emulator).

Any bearer token in ``DEVICE_TOKENS`` admits a device; the society it maps to is only the default.
A screen picks which society it shows and speaks to with ``?society=`` (HTTP and WebSocket) or a
``society.select`` WebSocket message, and which run with ``?run=`` or ``run.select`` (by default it
follows the newest live run with societies; ``GET /devices/runs`` lists the choices). It receives
that society's live state and can send a short voice clip; the clip is transcribed and condensed
into the society's unattributed murmur. In interview mode the screen instead picks one agent
(``agents.list`` / ``interview.select``, or a clip with ``"purpose": "pick"`` naming them) and
asks it questions (``"purpose": "interview"``); the agent's answer comes back as
``interview.reply`` followed by binary 16 kHz PCM frames and ``interview.audio_end``. With no
tokens configured the endpoints are open (local development only).
"""

import array
import asyncio
import contextlib
import hmac
import json
import logging
import os
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from worldwarpixel import interview
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
AUDIO_CHUNK = 4096
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
    raw = get_settings().device_tokens.get_secret_value()
    tokens = parse_tokens(raw)
    if not tokens:
        if raw.strip():
            log.error("DEVICE_TOKENS is set but has no valid token:society entries")
            return None
        s = society or 0
        return Device(f"dev-{s}", s)
    for i, (known, s) in enumerate(tokens.items()):
        if token and hmac.compare_digest(token.encode(), known.encode()):
            return Device(f"screen-{i + 1}", s if society is None else society)
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


def current_run(manager: RunManager, run_id: str | None = None) -> Run | None:
    """The newest live run with societies (screens follow whatever is on show), or `run_id`."""
    live = [r for r in manager.active() if r.societies is not None and run_id in (None, r.id)]
    return max(live, key=lambda r: r.started_at or 0.0, default=None)


def run_choices(manager: RunManager) -> list[dict[str, Any]]:
    """Live runs a screen can join, newest first."""
    live = [r for r in manager.active() if r.societies is not None]
    live.sort(key=lambda r: r.started_at or 0.0, reverse=True)
    return [
        {
            "id": r.id,
            "started_at": r.started_at,
            "agents": len(r.agents),
            "model": r.config.model,
            "societies": [s["name"] for s in r.society_summary()],
        }
        for r in live
    ]


def _run_for(request: Request, device: Device) -> Run:
    run = current_run(request.app.state.runs, request.query_params.get("run"))
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


def clip_stats(pcm: bytes, rate: int) -> str:
    """One log line that tells silence, clipping and wrong sample formats apart."""
    s = array.array("h", pcm[: len(pcm) // 2 * 2])
    if not s:
        return "empty clip"
    mean = sum(s) / len(s)
    rms = (sum((x - mean) ** 2 for x in s) / len(s)) ** 0.5
    peak = max(abs(x) for x in s)
    return f"{len(s) / rate:.1f}s @ {rate} Hz, rms {rms:.0f}, peak {peak}, dc {mean:.0f}"


def dump_clip(device: Device, pcm: bytes, rate: int) -> None:
    folder = get_settings().voice_dump_dir
    if not folder:
        return
    path = Path(folder) / f"{device.id}-{time.strftime('%H%M%S')}-{time.time_ns() % 10**9}.wav"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f, wave.open(f, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


router = APIRouter(prefix="/devices", tags=["devices"])
MAX_BODY = MAX_SECONDS * 48_000 * 2 * 2 + 4096
"""Largest upload worth reading: 20 s of 48 kHz stereo 16-bit WAV."""


@router.get("/me")
async def device_me(request: Request, device: DeviceDep) -> dict[str, Any]:
    run = current_run(request.app.state.runs, request.query_params.get("run"))
    out: dict[str, Any] = {"device": device.id, "society": device.society, "palette": PALETTE}
    if run is not None and run.societies is not None and device.society < run.societies.count:
        out["state"] = device_state(run, device, {"status": "idle"})
    return out


@router.get("/runs")
async def device_runs(request: Request, device: DeviceDep) -> dict[str, Any]:
    runs = run_choices(request.app.state.runs)
    return {"runs": runs, "newest": runs[0]["id"] if runs else None}


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
    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        raise HTTPException(413, f"clip longer than {MAX_SECONDS} s")
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > MAX_BODY:
            raise HTTPException(413, f"clip longer than {MAX_SECONDS} s")
    data = bytes(buf)
    try:
        if data[:4] == b"RIFF":
            pcm, rate = wav_to_pcm(data)
        else:
            pcm, rate = data, RATE
        if len(pcm) > MAX_SECONDS * rate * 2:
            raise ValueError(f"clip longer than {MAX_SECONDS} s")
        log.info("voice upload from %s: %s", device.id, clip_stats(pcm, rate))
        with contextlib.suppress(OSError):
            dump_clip(device, pcm, rate)
        transcript = await transcribe(request.app, pcm, rate)
        return await speak(request.app, run, device.society, transcript)
    except TranscriptionUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.websocket("/ws")
async def device_socket(websocket: WebSocket) -> None:
    q = websocket.query_params
    run_id = q.get("run")
    society = q.get("society")
    device = authenticate(
        _bearer(websocket.headers.get("authorization")) or q.get("token"),
        int(society) if society and society.isdigit() else None,
    )
    if device is None:
        await websocket.close(code=4401, reason="invalid device token")
        return
    me = device
    await websocket.accept()
    app = websocket.app
    runs = run_choices(app.state.runs)
    await websocket.send_json(
        {
            "type": "hello",
            "device": me.id,
            "society": me.society,
            "palette": PALETTE,
            "run": run_id,
            "runs": runs,
        }
    )
    listed = runs
    voice: dict[str, Any] = {"status": "idle", "transcript": None, "error": None}
    clip: bytearray | None = None
    rate = RATE
    seen = 0
    busy: asyncio.Task[None] | None = None
    clip_run: Run | None = None
    clip_society = me.society
    clip_purpose = "murmur"
    clip_agent: Any = None
    histories: dict[tuple[str, int], list[tuple[str, str]]] = {}

    async def choose(run: Run, agent_id: Any, heard: str | None = None) -> None:
        agent = run.by_id.get(agent_id) if isinstance(agent_id, int) else None
        if agent is None:
            await websocket.send_json(
                {"type": "interview.error", "error": f"no agent {agent_id!r}", "heard": heard}
            )
            return
        await websocket.send_json(
            {
                "type": "interview.agent",
                "run": run.id,
                "agent": interview.agent_brief(run, agent),
                "heard": interview.ascii_text(heard) if heard else heard,
            }
        )

    async def converse(run: Run, agent_id: Any, question: str) -> None:
        agent = run.by_id.get(agent_id) if isinstance(agent_id, int) else None
        if agent is None:
            raise interview.InterviewError("pick someone to talk to first")
        await websocket.send_json(
            {"type": "interview.heard", "agent": agent.id, "text": interview.ascii_text(question)}
        )
        history = histories.setdefault((run.id, agent.id), [])
        reply = await interview.answer(app.state.interview_chat, run, agent, history, question)
        try:
            pcm = await interview.voice(run, agent, reply)
        except Exception as e:
            log.warning("interview voice failed: %s", e)
            pcm = None
        await websocket.send_json(
            {
                "type": "interview.reply",
                "agent": agent.id,
                "name": interview.ascii_text(agent.name),
                "text": interview.ascii_text(reply),
                "audio": pcm is not None,
                "rate": interview.RATE,
                "bytes": len(pcm) if pcm else 0,
            }
        )
        if pcm:
            for i in range(0, len(pcm), AUDIO_CHUNK):
                await websocket.send_bytes(pcm[i : i + AUDIO_CHUNK])
            await websocket.send_json({"type": "interview.audio_end", "agent": agent.id})

    async def push() -> None:
        nonlocal seen, listed
        runs = run_choices(app.state.runs)
        if runs != listed:
            listed = runs
            await websocket.send_json({"type": "runs", "runs": runs})
        run = current_run(app.state.runs, run_id)
        if run is None or run.societies is None or me.society >= run.societies.count:
            await websocket.send_json({"type": "state", "run": None, "voice": voice})
            return
        migs = list(run.societies.migrations)
        new = min(run.societies.moves - seen, len(migs))
        for m in migs[len(migs) - new :] if new > 0 else []:
            if me.society in (m.src, m.dst):
                await websocket.send_json(
                    {
                        "type": "migration",
                        "agent": run.agent_name(m.agent_id),
                        "from": m.src,
                        "to": m.dst,
                    }
                )
        seen = run.societies.moves
        await websocket.send_json(device_state(run, me, voice))

    async def finish(
        pcm: bytes, run: Run | None, society: int, purpose: str, agent_id: Any
    ) -> None:
        voice.update(status="transcribing", error=None, transcript=None)
        log.info(
            "voice clip (%s) from %s for society %d: %s",
            purpose,
            me.id,
            society,
            clip_stats(pcm, rate),
        )
        with contextlib.suppress(OSError):
            dump_clip(me, pcm, rate)
        try:
            await push()
            if purpose in ("pick", "interview"):
                if run is None:
                    raise ValueError("no running world")
                transcript = " ".join((await transcribe(app, pcm, rate)).split())
                voice.update(transcript=transcript)
                if not transcript:
                    raise ValueError("nothing was heard")
                if purpose == "pick":
                    picked = interview.pick_agent(run, transcript)
                    if picked is None:
                        raise ValueError(f"nobody here matches '{transcript}'")
                    await choose(run, picked.id, transcript)
                else:
                    await converse(run, agent_id, transcript)
            else:
                if run is None or run.societies is None or society >= run.societies.count:
                    raise ValueError("no running world with that society")
                transcript = await transcribe(app, pcm, rate)
                voice.update(transcript=transcript)
                await speak(app, run, society, transcript)
            voice.update(status="done")
        except (TranscriptionUnavailable, ValueError, interview.InterviewError) as e:
            log.warning("voice clip from %s failed: %s", me.id, e)
            voice.update(status="error", error=str(e))
            if purpose != "murmur":
                with contextlib.suppress(Exception):
                    await websocket.send_json({"type": "interview.error", "error": str(e)})
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
                clip_run = current_run(app.state.runs, run_id)
                clip_society = me.society
                purpose = data.get("purpose")
                clip_purpose = purpose if purpose in ("pick", "interview") else "murmur"
                clip_agent = data.get("agent")
                voice.update(status="listening", error=None)
            elif kind == "agents.list":
                run = current_run(app.state.runs, run_id)
                await websocket.send_json(
                    {
                        "type": "agents",
                        "run": run.id if run else None,
                        "agents": interview.agent_list(run) if run else [],
                    }
                )
            elif kind == "interview.select":
                run = current_run(app.state.runs, run_id)
                if run is None:
                    await websocket.send_json(
                        {"type": "interview.error", "error": "no running world"}
                    )
                else:
                    await choose(run, data.get("agent"))
            elif kind == "run.select":
                rid = data.get("run")
                chosen = current_run(app.state.runs, rid) if isinstance(rid, str) else None
                if rid is not None and chosen is None:
                    await websocket.send_json({"type": "error", "error": f"no live run {rid!r}"})
                    continue
                run_id = rid
                chosen = chosen or current_run(app.state.runs)
                if chosen and chosen.societies and me.society >= chosen.societies.count:
                    me = replace(me, society=0)
                await websocket.send_json(
                    {
                        "type": "run.selected",
                        "run": chosen.id if chosen else None,
                        "pinned": rid is not None,
                        "society": me.society,
                    }
                )
                await push()
            elif kind == "society.select":
                n = data.get("society")
                run = current_run(app.state.runs, run_id)
                count = run.societies.count if run and run.societies else 6
                if isinstance(n, int) and not isinstance(n, bool) and 0 <= n < count:
                    me = replace(me, society=n)
                    await websocket.send_json({"type": "society.selected", "society": n})
                    await push()
                else:
                    await websocket.send_json({"type": "error", "error": f"no society {n!r}"})
            elif kind == "voice.cancel":
                clip = None
                voice.update(status="idle")
            elif kind == "voice.end" and clip is not None:
                pcm, clip = bytes(clip), None
                if busy is None or busy.done():
                    busy = asyncio.create_task(
                        finish(pcm, clip_run, clip_society, clip_purpose, clip_agent)
                    )
    except WebSocketDisconnect:
        pass
    finally:
        tick.cancel()
        if busy is not None:
            busy.cancel()
