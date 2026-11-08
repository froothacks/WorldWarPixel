import asyncio
import contextlib
import json
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
from worldwarpixel.sim.stage import REACTIONS, SpeechSynth, StageBusy, wav
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
        speech: SpeechSynth | None = None,
        stage_model: NarratorModel | None = None,
        make_narrator: Callable[[RunConfig], NarratorModel | None] | None = None,
    ) -> None:
        self.log = log
        self.speech = speech
        self.stage_model = stage_model
        self.max_active_runs = max_active_runs
        self.make_policy = make_policy or random_policy
        self.narrator_model = narrator_model
        self.make_narrator = make_narrator
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
        run = Run(
            run_id,
            config,
            self.log,
            policy,
            narrator_model=narrator,
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

    async def fetch(self, run_id: str) -> Run:
        """A run in memory, or a finished one rebuilt from the event log (e.g. pre-restart)."""
        run = self.runs.get(run_id)
        if run is not None:
            return run
        task = self._loading.get(run_id)
        if task is None:
            task = asyncio.create_task(asyncio.to_thread(restore_run, self.log, run_id))
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
            run.stage.hub.close_all()
            self.log.append(run_id, time.time(), "run.deleted", None, {})

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


@router.get("/{run_id}/stage/frame")
async def stage_frame(
    run_id: str,
    manager: ManagerDep,
    w: int = Query(default=320, ge=16, le=800),
    h: int = Query(default=240, ge=16, le=800),
    podium: int = Query(default=0, ge=0, le=1),
) -> Response:
    """A nearest-neighbour RGB565 preview of the active debate region."""
    frame = manager.get(run_id).stage.frame(w, h)
    return Response(
        frame,
        media_type="application/octet-stream",
        headers={"X-Width": str(w), "X-Height": str(h)},
    )


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
    podium_raw = websocket.query_params.get("podium")
    podium: int | None = None
    if podium_raw is not None:
        try:
            podium = int(podium_raw)
        except ValueError:
            await websocket.close(code=1008, reason="podium must be 0 or 1")
            return
        if podium not in (0, 1):
            await websocket.close(code=1008, reason="podium must be 0 or 1")
            return
        run.stage.device_connected(podium)
    narrator = websocket.query_params.get("narrator", "0") in ("1", "true")
    queue = run.stage.hub.subscribe()
    transcript = websocket.query_params.get("transcript", "1") not in ("0", "false")

    def includes_line(line: dict[str, Any]) -> bool:
        return line.get("podium") == podium or (
            narrator and line.get("kind") in ("narrator", "verdict")
        )

    def snapshot() -> dict[str, Any]:
        state = run.stage.state(transcript)
        if podium is not None and "transcript" in state:
            state["transcript"] = [line for line in state["transcript"] if includes_line(line)]
        return {"type": "stage.snapshot", **state}

    async def sender() -> None:
        await websocket.send_json(snapshot())
        while True:
            msg = await queue.get()
            if msg == CLOSE:
                await websocket.close(code=1001, reason="run deleted")
                return
            if msg == RESYNC:
                await websocket.send_json(snapshot())
            elif isinstance(msg, str):
                if podium is None:
                    await websocket.send_text(msg)
                    continue
                event = json.loads(msg)
                event_type = event.get("type")
                if event_type in ("stage.state", "stage.votes", "stage.verdict") or (
                    event_type == "stage.line" and includes_line(event)
                ):
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
        if podium is not None:
            run.stage.device_disconnected(podium)


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
