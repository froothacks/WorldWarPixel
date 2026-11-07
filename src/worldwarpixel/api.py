import asyncio
import contextlib
import secrets
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
from worldwarpixel.sim.agents import Policy, RandomWalkPolicy
from worldwarpixel.sim.board import BoardError
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.hub import CLOSE, RESYNC, snapshot_frame
from worldwarpixel.sim.narrator import NarratorModel, NarratorUnavailable
from worldwarpixel.sim.run import Run, RunConfig
from worldwarpixel.sim.stage import REACTIONS, SpeechSynth, StageBusy, wav
from worldwarpixel.sim.timelapse import build_timelapse


class RunManager:
    """Holds the in-memory runs of this process; the event log persists what they did."""

    def __init__(
        self,
        log: EventLog,
        max_active_runs: int = 4,
        make_policy: Callable[[RunConfig], Policy] | None = None,
        narrator_model: NarratorModel | None = None,
        speech: SpeechSynth | None = None,
        stage_model: NarratorModel | None = None,
    ) -> None:
        self.log = log
        self.speech = speech
        self.stage_model = stage_model
        self.max_active_runs = max_active_runs
        self.make_policy = make_policy or random_policy
        self.narrator_model = narrator_model
        self.runs: dict[str, Run] = {}

    def active(self) -> list[Run]:
        return [r for r in self.runs.values() if r.state != "stopped"]

    async def create(self, config: RunConfig) -> Run:
        if len(self.active()) >= self.max_active_runs:
            raise HTTPException(409, f"{self.max_active_runs} runs already active; stop one first")
        run_id = secrets.token_hex(4)
        policy = self.make_policy(config)
        run = Run(
            run_id,
            config,
            self.log,
            policy,
            narrator_model=self.narrator_model,
            speech=self.speech,
            stage_model=self.stage_model,
        )
        self.runs[run_id] = run
        await run.start()
        return run

    def get(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        return run

    async def delete(self, run_id: str) -> None:
        run = self.runs.pop(run_id, None)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        try:
            await run.stop()
        finally:
            run.hub.close_all()
            run.stage.hub.close_all()

    async def shutdown(self) -> None:
        await asyncio.gather(*(r.stop() for r in self.runs.values()))
        self.log.close()


def random_policy(config: RunConfig) -> Policy:
    return RandomWalkPolicy(seed=config.seed if config.seed is not None else 0)


def get_manager(request: Request) -> RunManager:
    manager: RunManager = request.app.state.runs
    return manager


ManagerDep = Annotated[RunManager, Depends(get_manager)]

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", status_code=201)
async def create_run(config: RunConfig, manager: ManagerDep) -> dict[str, Any]:
    run = await manager.create(config)
    return run.summary()


@router.get("")
async def list_runs(manager: ManagerDep) -> list[dict[str, Any]]:
    return [r.summary() for r in manager.runs.values()]


@router.get("/{run_id}")
async def get_run(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    return manager.get(run_id).summary()


@router.post("/{run_id}/pause")
async def pause_run(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    run = manager.get(run_id)
    run.pause()
    return run.summary()


@router.post("/{run_id}/resume")
async def resume_run(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    run = manager.get(run_id)
    run.resume()
    return run.summary()


@router.post("/{run_id}/stop")
async def stop_run(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    run = manager.get(run_id)
    await run.stop()
    return run.summary()


def require_god(x_god_token: Annotated[str | None, Header()] = None) -> None:
    """God endpoints are open unless GOD_TOKEN is configured; then the header must match."""
    expected = get_settings().god_token.get_secret_value()
    if expected and not secrets.compare_digest(x_god_token or "", expected):
        raise HTTPException(403, "god mode requires a valid X-God-Token")


GodDep = Annotated[None, Depends(require_god)]


class GodPixels(BaseModel):
    pixels: list[tuple[int, int, int]] = Field(max_length=50_000)
    erase: bool = False


class GodAnnouncement(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=600)


class GodPace(BaseModel):
    cooldown: float | None = Field(default=None, ge=0.1, le=600)
    pixels_per_turn: int | None = Field(default=None, ge=1, le=1024)
    pixel_interval: float | None = Field(default=None, ge=0, le=5)


@router.post("/{run_id}/god/pixels")
async def god_pixels(
    run_id: str, body: GodPixels, manager: ManagerDep, _god: GodDep
) -> dict[str, Any]:
    run = manager.get(run_id)
    try:
        return {"placed": run.god_paint(body.pixels, body.erase)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/{run_id}/god/clear-canvas")
async def god_clear_canvas(run_id: str, manager: ManagerDep, _god: GodDep) -> dict[str, Any]:
    run = manager.get(run_id)
    run.god_clear_canvas()
    return run.summary()


@router.post("/{run_id}/god/clear-board")
async def god_clear_board(run_id: str, manager: ManagerDep, _god: GodDep) -> dict[str, Any]:
    run = manager.get(run_id)
    run.god_clear_board()
    return run.summary()


@router.post("/{run_id}/god/announce", status_code=201)
async def god_announce(
    run_id: str, body: GodAnnouncement, manager: ManagerDep, _god: GodDep
) -> dict[str, Any]:
    run = manager.get(run_id)
    try:
        return run.god_announce(body.title, body.body)
    except BoardError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/{run_id}/god/pace")
async def god_pace(run_id: str, body: GodPace, manager: ManagerDep, _god: GodDep) -> dict[str, Any]:
    run = manager.get(run_id)
    run.god_set_pace(body.cooldown, body.pixels_per_turn, body.pixel_interval)
    return run.summary()


@router.delete("/{run_id}", status_code=204)
async def delete_run(run_id: str, manager: ManagerDep) -> None:
    """Stop the run (if needed) and drop it from memory; its event log is kept."""
    await manager.delete(run_id)


@router.get("/{run_id}/agents")
async def list_agents(run_id: str, manager: ManagerDep) -> list[dict[str, Any]]:
    return manager.get(run_id).agent_list()


@router.get("/{run_id}/agents/{agent_id}")
async def get_agent(run_id: str, agent_id: int, manager: ManagerDep) -> dict[str, Any]:
    detail = manager.get(run_id).agent_detail(agent_id)
    if detail is None:
        raise HTTPException(404, f"agent {agent_id} not found")
    return detail


@router.get("/{run_id}/threads")
async def list_threads(run_id: str, manager: ManagerDep) -> list[dict[str, Any]]:
    run = manager.get(run_id)
    return run.board.listing(limit=100, name_of=run.agent_name)


@router.get("/{run_id}/threads/{thread_id}")
async def get_thread(run_id: str, thread_id: int, manager: ManagerDep) -> list[dict[str, Any]]:
    posts = manager.get(run_id).board.thread(thread_id)
    if not posts:
        raise HTTPException(404, f"thread {thread_id} not found")
    return [p.to_dict() for p in posts]


@router.get("/{run_id}/world")
async def get_world(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    """Country nodes plus talk (replies, @mentions) and raid (overpainting) edges."""
    return manager.get(run_id).world.snapshot()


@router.get("/{run_id}/canvas")
async def get_canvas(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    canvas = manager.get(run_id).canvas
    rows = canvas.view(0, 0, canvas.width, canvas.height)
    return {"width": canvas.width, "height": canvas.height, "rows": rows}


@router.get("/{run_id}/events")
async def get_events(
    run_id: str,
    manager: ManagerDep,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 1000,
) -> list[dict[str, Any]]:
    manager.get(run_id)
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
    run = manager.get(run_id)
    data = await asyncio.to_thread(
        build_timelapse, manager.log, run_id, run.canvas.width, run.canvas.height, frames
    )
    return Response(content=data, media_type="application/octet-stream")


class NarratorQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@router.get("/{run_id}/narrator")
async def get_narrator(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    """The narrator's update history (newest last) plus its interval and stats."""
    return manager.get(run_id).narrator.state()


@router.post("/{run_id}/narrator/ask")
async def ask_narrator(run_id: str, body: NarratorQuestion, manager: ManagerDep) -> dict[str, Any]:
    """Answer a viewer's question from the current digest; also streamed as `narrator.update`."""
    run = manager.get(run_id)
    if run.state == "stopped":
        raise HTTPException(409, "run is stopped")
    try:
        return await run.narrator.ask(body.question)
    except NarratorUnavailable as e:
        raise HTTPException(503, f"narrator unavailable: {e}") from e


class StageStart(BaseModel):
    a: int | None = Field(default=None, description="First debater (agent id); picked if omitted")
    b: int | None = Field(default=None, description="Second debater (agent id)")
    topic: str | None = Field(default=None, max_length=200)
    auto: bool | None = Field(default=None, description="Keep starting debates after this one")


class StageAuto(BaseModel):
    auto: bool


class StageVote(BaseModel):
    voter: str = Field(min_length=1, max_length=64, description="Stable per-viewer id")
    agent_id: int


class StageReaction(BaseModel):
    reaction: str = Field(description=f"One of: {', '.join(REACTIONS)}")
    agent_id: int | None = None


class StageNarration(BaseModel):
    text: str = Field(min_length=1, max_length=280)
    name: str | None = Field(default=None, max_length=40)
    announce: bool = Field(default=False, description="Also post it to every agent (god mode)")


def _live(run_id: str, manager: RunManager) -> Run:
    run = manager.get(run_id)
    if run.state == "stopped":
        raise HTTPException(409, "run is stopped")
    return run


@router.get("/{run_id}/stage")
async def get_stage(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    """The live debate: phase, debaters, votes and the spoken transcript."""
    return manager.get(run_id).stage.state()


@router.post("/{run_id}/stage/start")
async def start_debate(
    run_id: str, body: StageStart, manager: ManagerDep, _god: GodDep
) -> dict[str, Any]:
    """Put two agents on stage (the fiercest canvas rivals unless `a` and `b` are given)."""
    stage = _live(run_id, manager).stage
    try:
        if body.auto is not None:
            stage.auto = body.auto
        return stage.start_debate(body.a, body.b, body.topic)
    except StageBusy as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/{run_id}/stage/stop")
async def stop_debate(run_id: str, manager: ManagerDep, _god: GodDep) -> dict[str, Any]:
    """End the current debate without a verdict and turn auto mode off."""
    stage = manager.get(run_id).stage
    stage.auto = False
    stage.end_debate()
    return stage.state(transcript=False)


@router.post("/{run_id}/stage/auto")
async def stage_auto(
    run_id: str, body: StageAuto, manager: ManagerDep, _god: GodDep
) -> dict[str, Any]:
    stage = manager.get(run_id).stage
    stage.set_auto(body.auto)
    return stage.state(transcript=False)


@router.post("/{run_id}/stage/vote")
async def stage_vote(run_id: str, body: StageVote, manager: ManagerDep) -> dict[str, Any]:
    """One vote per `voter`; voting again changes it. Open during the debate and the vote."""
    stage = _live(run_id, manager).stage
    try:
        return {"votes": stage.vote(body.voter, body.agent_id)}
    except StageBusy as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/{run_id}/stage/react", status_code=204)
async def stage_react(run_id: str, body: StageReaction, manager: ManagerDep) -> None:
    """Cheer, boo, laugh or gasp; debaters hear the crowd in their next line."""
    try:
        _live(run_id, manager).stage.react(body.reaction, body.agent_id)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/{run_id}/stage/narrate", status_code=202)
async def stage_narrate(
    run_id: str,
    body: StageNarration,
    manager: ManagerDep,
    x_god_token: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """A live narrator line: spoken on stage next, and the debaters respond to it."""
    run = _live(run_id, manager)
    if body.announce:
        require_god(x_god_token)
    try:
        run.stage.narrate(body.text, body.name)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if body.announce:
        with contextlib.suppress(BoardError):
            run.god_announce(f"Live narrator: {body.name or 'Narrator'}"[:120], body.text)
    return {"queued": True}


@router.get("/{run_id}/stage/lines/{line_id}/audio")
async def stage_audio(
    run_id: str,
    line_id: int,
    manager: ManagerDep,
    format: Annotated[str, Query(pattern="^(wav|pcm)$")] = "wav",
) -> Response:
    """A spoken line's audio: WAV for browsers, raw s16le mono PCM for podium devices."""
    stage = manager.get(run_id).stage
    pcm = stage.audio(line_id)
    if pcm is None:
        raise HTTPException(404, f"no audio for line {line_id}")
    if format == "pcm":
        rate = stage.sample_rate
        return Response(
            pcm, media_type="application/octet-stream", headers={"X-Sample-Rate": str(rate)}
        )
    return Response(wav(pcm, stage.sample_rate), media_type="audio/wav")


@router.websocket("/{run_id}/stage/ws")
async def stage_socket(websocket: WebSocket, run_id: str) -> None:
    """Stage-only stream for podium screens: `stage.snapshot`, then `stage.*` events (JSON)."""
    manager: RunManager = websocket.app.state.runs
    run = manager.runs.get(run_id)
    if run is None:
        await websocket.close(code=4004, reason="run not found")
        return
    await websocket.accept()
    queue = run.stage.hub.subscribe()
    transcript = websocket.query_params.get("transcript", "1") not in ("0", "false")

    async def sender() -> None:
        await websocket.send_json({"type": "stage.snapshot", **run.stage.state(transcript)})
        while True:
            msg = await queue.get()
            if msg == CLOSE:
                await websocket.close(code=1001, reason="run deleted")
                return
            if msg == RESYNC:
                await websocket.send_json({"type": "stage.snapshot", **run.stage.state(transcript)})
            elif isinstance(msg, str):
                await websocket.send_text(msg)

    async def receiver() -> None:
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
        run.stage.hub.unsubscribe(queue)


@router.websocket("/{run_id}/ws")
async def run_socket(websocket: WebSocket, run_id: str) -> None:
    manager: RunManager = websocket.app.state.runs
    run = manager.runs.get(run_id)
    if run is None:
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
