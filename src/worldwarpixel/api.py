import asyncio
import contextlib
import secrets
import time
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from worldwarpixel.config import get_settings
from worldwarpixel.personas.roster import roster_summaries
from worldwarpixel.sim.agents import Policy, RandomWalkPolicy
from worldwarpixel.sim.archive import archived_summaries, restore_run
from worldwarpixel.sim.board import BoardError
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.hub import CLOSE, RESYNC, snapshot_frame
from worldwarpixel.sim.narrator import NarratorModel, NarratorUnavailable
from worldwarpixel.sim.run import MIN_PIXEL_INTERVAL, Run, RunConfig
from worldwarpixel.sim.speech import SpeechSynth, wav
from worldwarpixel.sim.timelapse import build_timelapse

MAX_RESTORED = 8
"""Finished runs rebuilt from the log that are kept in memory at once (oldest dropped first)."""


class RunManager:
    """Holds the in-memory runs of this process; the event log persists what they did."""

    def __init__(
        self,
        log: EventLog,
        max_active_runs: int = 4,
        make_policy: Callable[[RunConfig], Policy] | None = None,
        narrator_model: NarratorModel | None = None,
        make_narrator: Callable[[RunConfig], NarratorModel | None] | None = None,
        speech: SpeechSynth | None = None,
    ) -> None:
        self.log = log
        self.max_active_runs = max_active_runs
        self.make_policy = make_policy or random_policy
        self.narrator_model = narrator_model
        self.make_narrator = make_narrator
        self.speech = speech
        self.runs: dict[str, Run] = {}
        self._restored: list[str] = []
        self._loading: dict[str, asyncio.Task[Run | None]] = {}

    def active(self) -> list[Run]:
        return [r for r in self.runs.values() if r.state != "stopped"]

    async def create(self, config: RunConfig) -> Run:
        if len(self.active()) >= self.max_active_runs:
            raise HTTPException(409, f"{self.max_active_runs} runs already active; stop one first")
        run_id = secrets.token_hex(4)
        policy = self.make_policy(config)
        narrator = (
            self.make_narrator(config) if self.make_narrator is not None else self.narrator_model
        )
        run = Run(run_id, config, self.log, policy, narrator_model=narrator, speech=self.speech)
        self.runs[run_id] = run
        await run.start()
        return run

    def get(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        return run

    async def fetch(self, run_id: str) -> Run:
        """A run in memory, or a finished one rebuilt from the event log (e.g. pre-restart)."""
        run = self.runs.get(run_id)
        if run is not None:
            return run
        task = self._loading.get(run_id)
        if task is None:
            task = asyncio.create_task(
                asyncio.to_thread(restore_run, self.log, run_id, self.speech)
            )
            self._loading[run_id] = task
            task.add_done_callback(lambda _: self._loading.pop(run_id, None))
        restored = await task
        if restored is None:
            raise HTTPException(404, f"run {run_id} not found")
        if run_id not in self.runs:
            self.runs[run_id] = restored
            self._restored.append(run_id)
            while len(self._restored) > MAX_RESTORED:
                old = self.runs.pop(self._restored.pop(0), None)
                if old is not None:
                    old.hub.close_all()
                    old.speech.hub.close_all()
        return self.runs[run_id]

    async def summaries(self) -> list[dict[str, Any]]:
        """Runs in memory plus the finished runs only the event log still knows about."""
        live = [r.summary() for r in self.runs.values()]
        archived = await asyncio.to_thread(archived_summaries, self.log, set(self.runs))
        return live + archived

    async def delete(self, run_id: str) -> None:
        run = await self.fetch(run_id)
        self.runs.pop(run_id, None)
        if run_id in self._restored:
            self._restored.remove(run_id)
        try:
            await run.stop()
        finally:
            run.hub.close_all()
            run.speech.hub.close_all()
            self.log.append(run_id, time.time(), "run.deleted", None, {})

    async def shutdown(self) -> None:
        await asyncio.gather(*(r.stop() for r in self.runs.values()))
        for run in self.runs.values():
            run.speech.hub.close_all()
        self.log.close()


def random_policy(config: RunConfig) -> Policy:
    return RandomWalkPolicy(seed=config.seed if config.seed is not None else 0)


def get_manager(request: Request) -> RunManager:
    manager: RunManager = request.app.state.runs
    return manager


ManagerDep = Annotated[RunManager, Depends(get_manager)]

router = APIRouter(prefix="/runs", tags=["runs"])
rosters_router = APIRouter(prefix="/rosters", tags=["runs"])


@rosters_router.get("")
async def list_rosters() -> list[dict[str, Any]]:
    """Bundled attendee rosters a run can use instead of the population-seeded world."""
    return roster_summaries()


@router.post("", status_code=201)
async def create_run(config: RunConfig, manager: ManagerDep) -> dict[str, Any]:
    run = await manager.create(config)
    return run.summary()


@router.get("")
async def list_runs(manager: ManagerDep) -> list[dict[str, Any]]:
    return await manager.summaries()


@router.get("/{run_id}")
async def get_run(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    return (await manager.fetch(run_id)).summary()


@router.post("/{run_id}/pause")
async def pause_run(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    run = await manager.fetch(run_id)
    run.pause()
    return run.summary()


@router.post("/{run_id}/resume")
async def resume_run(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    run = await manager.fetch(run_id)
    run.resume()
    return run.summary()


@router.post("/{run_id}/stop")
async def stop_run(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    run = await manager.fetch(run_id)
    await run.stop()
    return run.summary()


def require_god(x_god_token: Annotated[str | None, Header()] = None) -> None:
    """God endpoints are open unless GOD_TOKEN is configured; then the header must match."""
    expected = get_settings().god_token.get_secret_value()
    if expected and not secrets.compare_digest(x_god_token or "", expected):
        raise HTTPException(403, "god mode requires a valid X-God-Token")


GodDep = Annotated[None, Depends(require_god)]


class ThreadVote(BaseModel):
    voter: str = Field(min_length=1, max_length=64, description="Stable per-browser viewer id")
    vote: int = Field(ge=-1, le=1, description="+1 up or -1 down (each call adds), 0 withdraws")


class GodPixels(BaseModel):
    pixels: list[tuple[int, int, int]] = Field(max_length=50_000)
    erase: bool = False


class GodAnnouncement(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=600)


class GodReply(BaseModel):
    thread_id: int = Field(ge=0)
    body: str = Field(min_length=1, max_length=600)


class GodPace(BaseModel):
    cooldown: float | None = Field(default=None, ge=0.1, le=600)
    pixels_per_turn: int | None = Field(default=None, ge=1, le=1024)
    pixel_interval: float | None = Field(default=None, ge=MIN_PIXEL_INTERVAL, le=5)


@router.post("/{run_id}/god/pixels")
async def god_pixels(
    run_id: str, body: GodPixels, manager: ManagerDep, _god: GodDep
) -> dict[str, Any]:
    run = await manager.fetch(run_id)
    try:
        return {"placed": run.god_paint(body.pixels, body.erase)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/{run_id}/god/clear-canvas")
async def god_clear_canvas(run_id: str, manager: ManagerDep, _god: GodDep) -> dict[str, Any]:
    run = await manager.fetch(run_id)
    run.god_clear_canvas()
    return run.summary()


@router.post("/{run_id}/god/clear-board")
async def god_clear_board(run_id: str, manager: ManagerDep, _god: GodDep) -> dict[str, Any]:
    run = await manager.fetch(run_id)
    run.god_clear_board()
    return run.summary()


@router.post("/{run_id}/god/reply", status_code=201)
async def god_reply(
    run_id: str, body: GodReply, manager: ManagerDep, _god: GodDep
) -> dict[str, Any]:
    run = manager.get(run_id)
    try:
        return run.god_reply(body.thread_id, body.body)
    except BoardError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/{run_id}/god/announce", status_code=201)
async def god_announce(
    run_id: str, body: GodAnnouncement, manager: ManagerDep, _god: GodDep
) -> dict[str, Any]:
    run = await manager.fetch(run_id)
    try:
        return run.god_announce(body.title, body.body)
    except BoardError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/{run_id}/god/pace")
async def god_pace(run_id: str, body: GodPace, manager: ManagerDep, _god: GodDep) -> dict[str, Any]:
    run = await manager.fetch(run_id)
    run.god_set_pace(body.cooldown, body.pixels_per_turn, body.pixel_interval)
    return run.summary()


class GodModel(BaseModel):
    model: str = Field(min_length=1, max_length=64)


@router.post("/{run_id}/god/model")
async def god_model(
    run_id: str, body: GodModel, manager: ManagerDep, _god: GodDep
) -> dict[str, Any]:
    """Switch a running LLM run to another model preset; agents use it from their next turn."""
    run = await manager.fetch(run_id)
    if run.config.policy not in ("llm", "llm-guided"):
        raise HTTPException(409, "model switching only applies to LLM runs")
    new_config = run.config.model_copy(update={"model": body.model})
    policy = manager.make_policy(new_config)  # raises 400/422 for unknown or unavailable presets
    narrator = (
        manager.make_narrator(new_config)
        if manager.make_narrator is not None
        else manager.narrator_model
    )
    run.swap_model(policy, narrator, body.model)
    return run.summary()


@router.delete("/{run_id}", status_code=204)
async def delete_run(run_id: str, manager: ManagerDep) -> None:
    """Stop the run (if needed) and drop it from memory; its event log is kept."""
    await manager.delete(run_id)


@router.get("/{run_id}/agents")
async def list_agents(run_id: str, manager: ManagerDep) -> list[dict[str, Any]]:
    return (await manager.fetch(run_id)).agent_list()


@router.get("/{run_id}/agents/{agent_id}")
async def get_agent(run_id: str, agent_id: int, manager: ManagerDep) -> dict[str, Any]:
    detail = (await manager.fetch(run_id)).agent_detail(agent_id)
    if detail is None:
        raise HTTPException(404, f"agent {agent_id} not found")
    return detail


@router.get("/{run_id}/threads")
async def list_threads(run_id: str, manager: ManagerDep) -> list[dict[str, Any]]:
    run = await manager.fetch(run_id)
    return run.board.listing(limit=100, name_of=run.agent_name)


@router.get("/{run_id}/threads/{thread_id}")
async def get_thread(run_id: str, thread_id: int, manager: ManagerDep) -> list[dict[str, Any]]:
    posts = (await manager.fetch(run_id)).board.thread(thread_id)
    if not posts:
        raise HTTPException(404, f"thread {thread_id} not found")
    return [p.to_dict() for p in posts]


@router.post("/{run_id}/threads/{thread_id}/vote")
async def vote_thread(
    run_id: str, thread_id: int, body: ThreadVote, manager: ManagerDep
) -> dict[str, Any]:
    try:
        votes = manager.get(run_id).vote_thread(thread_id, body.voter, body.vote)
    except BoardError as e:
        raise HTTPException(404, str(e)) from e
    return {"thread_id": thread_id, "votes": votes}


@router.get("/{run_id}/world")
async def get_world(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    """Country nodes plus talk (replies, @mentions) and raid (overpainting) edges."""
    return (await manager.fetch(run_id)).world.snapshot()


@router.get("/{run_id}/canvas")
async def get_canvas(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    canvas = (await manager.fetch(run_id)).canvas
    rows = canvas.view(0, 0, canvas.width, canvas.height)
    return {"width": canvas.width, "height": canvas.height, "rows": rows}


@router.get("/{run_id}/events")
async def get_events(
    run_id: str,
    manager: ManagerDep,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 1000,
) -> list[dict[str, Any]]:
    await manager.fetch(run_id)
    events = manager.log.read(run_id, after_id=after, limit=limit)
    return [
        {"id": e.id, "ts": e.ts, "kind": e.kind, "agent_id": e.agent_id, "payload": e.payload}
        for e in events
    ]


@router.get("/{run_id}/timelapse")
async def get_timelapse(
    run_id: str,
    manager: ManagerDep,
    frames: Annotated[int, Query(ge=1, le=600)] = 120,
) -> Response:
    """Binary keyframe deltas of the whole run so far (see sim/timelapse.py for the layout)."""
    run = await manager.fetch(run_id)
    data = await asyncio.to_thread(
        build_timelapse, manager.log, run_id, run.canvas.width, run.canvas.height, frames
    )
    return Response(content=data, media_type="application/octet-stream")


class NarratorQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class SpeechRequest(BaseModel):
    text: str
    agent_id: int | None = None


@router.get("/{run_id}/narrator")
async def get_narrator(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    """The narrator's update history (newest last) plus its interval and stats."""
    return (await manager.fetch(run_id)).narrator.state()


@router.post("/{run_id}/narrator/ask")
async def ask_narrator(run_id: str, body: NarratorQuestion, manager: ManagerDep) -> dict[str, Any]:
    """Answer a viewer's question from the current digest; also streamed as `narrator.update`."""
    run = await manager.fetch(run_id)
    if run.state == "stopped":
        raise HTTPException(409, "run is stopped")
    try:
        return await run.narrator.ask(body.question)
    except NarratorUnavailable as e:
        raise HTTPException(503, f"narrator unavailable: {e}") from e


@router.post("/{run_id}/speak", status_code=202)
async def speak(
    run_id: str, body: SpeechRequest, manager: ManagerDep, _god: GodDep
) -> dict[str, bool]:
    run = await manager.fetch(run_id)
    if run.state == "stopped":
        raise HTTPException(409, "run is not running")
    try:
        run.speech.say(body.text, body.agent_id)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"queued": True}


@router.get("/{run_id}/speech")
async def get_speech(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    return (await manager.fetch(run_id)).speech.state()


@router.get("/{run_id}/speech/lines/{line_id}/audio")
async def speech_audio(
    run_id: str,
    line_id: int,
    manager: ManagerDep,
    format: Annotated[str, Query(pattern="^(wav|pcm)$")] = "wav",
) -> Response:
    speech = (await manager.fetch(run_id)).speech
    pcm = speech.audio(line_id)
    if pcm is None:
        raise HTTPException(404, f"no audio for line {line_id}")
    rate = speech.sample_rate
    if format == "pcm":
        return Response(
            pcm, media_type="application/octet-stream", headers={"X-Sample-Rate": str(rate)}
        )
    return Response(wav(pcm, rate), media_type="audio/wav")


@router.websocket("/{run_id}/speech/ws")
async def speech_socket(websocket: WebSocket, run_id: str) -> None:
    manager: RunManager = websocket.app.state.runs
    try:
        run = await manager.fetch(run_id)
    except HTTPException:
        await websocket.close(code=4004, reason="run not found")
        return
    await websocket.accept()
    run.speech.device_connected()
    queue = run.speech.hub.subscribe()

    async def send_snapshot() -> None:
        await websocket.send_json({"type": "speech.snapshot", **run.speech.state()})

    async def sender() -> None:
        await send_snapshot()
        while True:
            msg = await queue.get()
            if msg == CLOSE:
                await websocket.close(code=1001, reason="run deleted")
                return
            if msg == RESYNC:
                await send_snapshot()
            elif isinstance(msg, str):
                await websocket.send_text(msg)

    async def receiver() -> None:
        while True:
            await websocket.receive_text()

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            with contextlib.suppress(WebSocketDisconnect):
                task.result()
    finally:
        run.speech.hub.unsubscribe(queue)
        run.speech.device_disconnected()


@router.websocket("/{run_id}/ws")
async def run_socket(websocket: WebSocket, run_id: str) -> None:
    manager: RunManager = websocket.app.state.runs
    try:
        run = await manager.fetch(run_id)
    except HTTPException:
        await websocket.close(code=4004, reason="run not found")
        return
    await websocket.accept()
    queue = run.hub.subscribe()

    async def send_snapshot() -> None:
        await websocket.send_json(run.snapshot())
        await websocket.send_bytes(snapshot_frame(run.canvas))

    async def sender() -> None:
        await send_snapshot()
        while True:
            msg = await queue.get()
            if msg == CLOSE:
                await websocket.close(code=1001, reason="run deleted")
                return
            if msg == RESYNC:
                await send_snapshot()
            elif isinstance(msg, bytes):
                await websocket.send_bytes(msg)
            else:
                await websocket.send_text(msg)

    async def receiver() -> None:
        # Client messages are ignored for now; this exists to notice disconnects promptly.
        while True:
            await websocket.receive_text()

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            with contextlib.suppress(WebSocketDisconnect):
                t.result()
    finally:
        run.hub.unsubscribe(queue)
