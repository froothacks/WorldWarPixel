import asyncio
import contextlib
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from worldwarpixel.personas.roster import roster_summaries
from worldwarpixel.sim.agents import Policy, RandomWalkPolicy
from worldwarpixel.sim.archive import archived_board, archived_summaries, restore_run
from worldwarpixel.sim.board import Board, BoardError
from worldwarpixel.sim.events import EventStore, RunMeta
from worldwarpixel.sim.hub import CLOSE, RESYNC, snapshot_frame
from worldwarpixel.sim.narrator import NarratorModel, NarratorUnavailable
from worldwarpixel.sim.run import MIN_PIXEL_INTERVAL, Run, RunConfig
from worldwarpixel.sim.speech import SpeechSynth, wav
from worldwarpixel.sim.timelapse import build_timelapse
from worldwarpixel.sim.video import VideoError
from worldwarpixel.videos import VideoStore

RESUME_WINDOW = 600.0
"""Seconds a new server keeps looking for runs to resume: in a rolling deploy the old server
only shuts down (and marks its runs interrupted) after the new one is up."""
RESUME_POLL = 2.0
MAX_RESTORED = 8
"""Finished runs rebuilt from the log that are kept in memory at once (oldest dropped first)."""

logger = logging.getLogger(__name__)


@contextmanager
def reading(log: EventStore) -> Iterator[EventStore]:
    """A separate connection for long reads, so they don't hold up the shared one."""
    reader = log.reader()
    try:
        yield reader
    finally:
        if reader is not log:
            reader.close()


def _logged_run_meta(log: EventStore, run_id: str) -> RunMeta:
    meta = log.run_meta(run_id)
    if meta is None or meta.deleted or meta.config is None:
        raise HTTPException(404, f"run {run_id} not found")
    return meta


class RunManager:
    """Holds the in-memory runs of this process; the event log persists what they did."""

    def __init__(
        self,
        log: EventStore,
        max_active_runs: int = 20,
        make_policy: Callable[[RunConfig], Policy] | None = None,
        narrator_model: NarratorModel | None = None,
        make_narrator: Callable[[RunConfig], NarratorModel | None] | None = None,
        videos: VideoStore | None = None,
        auto_video: bool = False,
        speech: SpeechSynth | None = None,
        name_societies: Callable[[RunConfig], Awaitable[list[str] | None]] | None = None,
    ) -> None:
        self.log = log
        self.name_societies = name_societies
        self.videos = videos
        self.auto_video = auto_video and videos is not None
        self.max_active_runs = max_active_runs
        self.make_policy = make_policy or random_policy
        self.narrator_model = narrator_model
        self.make_narrator = make_narrator
        self.speech = speech
        self.runs: dict[str, Run] = {}
        self._restored: list[str] = []
        self._loading: dict[str, asyncio.Task[Run | None]] = {}
        self._resumer: asyncio.Task[None] | None = None

    def active(self) -> list[Run]:
        return [r for r in self.runs.values() if r.state != "stopped"]

    async def create(self, config: RunConfig) -> Run:
        if len(self.active()) >= self.max_active_runs:
            raise HTTPException(409, f"{self.max_active_runs} runs already active; stop one first")
        run_id = secrets.token_hex(4)
        policy = self.make_policy(config)
        if config.societies and not config.society_names and self.name_societies is not None:
            names = await self.name_societies(config)
            if names:
                config = config.model_copy(update={"society_names": names})
            if len(self.active()) >= self.max_active_runs:  # another launch finished meanwhile
                raise HTTPException(
                    409, f"{self.max_active_runs} runs already active; stop one first"
                )
        narrator = (
            self.make_narrator(config) if self.make_narrator is not None else self.narrator_model
        )
        run = Run(run_id, config, self.log, policy, narrator_model=narrator, speech=self.speech)
        self.runs[run_id] = run
        await run.start()
        if self.auto_video and self.videos is not None:
            self.videos.start_autosave(self.log, run_id)
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

            def restore() -> Run | None:
                started = time.perf_counter()
                with reading(self.log) as reader:
                    restored = restore_run(self.log, run_id, self.speech, source=reader)
                elapsed = time.perf_counter() - started
                if restored is None:
                    logger.info("run %s not found in log", run_id)
                else:
                    logger.info(
                        "restored run %s pixels=%d in %.1fs",
                        run_id,
                        restored.stats["pixels"],
                        elapsed,
                    )
                return restored

            task = asyncio.create_task(asyncio.to_thread(restore))
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

    def start_resuming(self, window: float = RESUME_WINDOW, poll: float = RESUME_POLL) -> None:
        """Keep resuming interrupted runs in the background for ``window`` seconds."""
        self._resumer = asyncio.create_task(self._resume_loop(window, poll))

    async def _resume_loop(self, window: float, poll: float) -> None:
        deadline = time.monotonic() + window
        while True:
            try:
                await self.resume_interrupted()
            except Exception:
                logger.exception("resuming interrupted runs failed")
            if time.monotonic() >= deadline:
                return
            await asyncio.sleep(poll)

    async def resume_interrupted(self) -> list[Run]:
        """Rebuild runs a previous server interrupted from the log and set them going again."""

        def find() -> list[str]:
            with reading(self.log) as reader:
                return reader.interrupted_runs()

        resumed = []
        for run_id in await asyncio.to_thread(find):
            if run_id in self.runs and run_id not in self._restored:
                continue
            run = await self._resume(run_id)
            if run is not None:
                resumed.append(run)
        return resumed

    async def _resume(self, run_id: str) -> Run | None:
        def restore() -> Run | None:
            with reading(self.log) as reader:
                return restore_run(self.log, run_id, self.speech, source=reader)

        run = await asyncio.to_thread(restore)
        if run is None or (run_id in self.runs and run_id not in self._restored):
            return None
        try:
            if len(self.active()) >= self.max_active_runs:
                raise RuntimeError(f"{self.max_active_runs} runs already active")
            run.policy = self.make_policy(run.config)
            run.narrator.set_model(
                self.make_narrator(run.config)
                if self.make_narrator is not None
                else self.narrator_model
            )
        except Exception as e:
            logger.warning("not resuming run %s: %r", run_id, e)
            self.log.append(run_id, time.time(), "run.state", None, {"state": "stopped"})
            return None
        old = self.runs.get(run_id)
        if old is not None and not old.interrupted:  # stopped as an archive while we rebuilt it
            return None
        old = self.runs.pop(run_id, None)
        if old is not None:  # opened as a stopped archive meanwhile; viewers reconnect to this one
            self._restored.remove(run_id)
            old.hub.close_all()
            old.speech.hub.close_all()
        self.runs[run_id] = run
        await run.start(resumed=True)
        if run.resume_paused:
            run.pause()
        if self.auto_video and self.videos is not None:
            self.videos.start_autosave(self.log, run_id)
        logger.info("resumed run %s pixels=%d", run_id, run.stats["pixels"])
        return run

    async def summaries(self) -> list[dict[str, Any]]:
        """Runs in memory plus the finished runs only the event log still knows about."""
        live = [r.summary() for r in self.runs.values()]
        archived = await asyncio.to_thread(archived_summaries, self.log, set(self.runs))
        return live + archived

    async def delete(self, run_id: str) -> None:
        run = await self.fetch(run_id)
        if self.videos is not None:
            self.videos.stop_autosave(run_id)
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
        if self._resumer is not None:
            self._resumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._resumer
        if self.videos is not None:
            self.videos.stop_all_autosaves()
        await asyncio.gather(*(r.interrupt() for r in self.runs.values()))
        for run in self.runs.values():
            run.speech.hub.close_all()
        if self.videos is not None:
            await self.videos.drain()
        self.log.close()


def random_policy(config: RunConfig) -> Policy:
    return RandomWalkPolicy(seed=config.seed if config.seed is not None else 0)


def get_manager(request: Request) -> RunManager:
    manager: RunManager = request.app.state.runs
    return manager


ManagerDep = Annotated[RunManager, Depends(get_manager)]

router = APIRouter(prefix="/runs", tags=["runs"])
rosters_router = APIRouter(prefix="/rosters", tags=["runs"])
videos_router = APIRouter(prefix="/videos", tags=["runs"])


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
    was_live = run.state != "stopped"
    if was_live and manager.videos is not None:
        manager.videos.stop_autosave(run_id)
    await run.stop()
    if was_live and manager.auto_video and manager.videos is not None:
        manager.videos.schedule(manager.log, run_id)
    return run.summary()


def require_god() -> None:
    """God endpoints are open to everyone."""


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
    run = manager.runs.get(run_id)
    if run is not None:
        return run.board.listing(limit=100, name_of=run.agent_name)

    def read_board() -> tuple[Board, Callable[[int], str]] | None:
        with reading(manager.log) as reader:
            _logged_run_meta(reader, run_id)
            return archived_board(reader, run_id)

    archived = await asyncio.to_thread(read_board)
    if archived is None:
        run = await manager.fetch(run_id)
        return run.board.listing(limit=100, name_of=run.agent_name)
    board, name_of = archived
    return board.listing(limit=100, name_of=name_of)


@router.get("/{run_id}/threads/{thread_id}")
async def get_thread(run_id: str, thread_id: int, manager: ManagerDep) -> list[dict[str, Any]]:
    run = manager.runs.get(run_id)
    if run is not None:
        posts = run.board.thread(thread_id)
    else:

        def read_board() -> tuple[Board, Callable[[int], str]] | None:
            with reading(manager.log) as reader:
                _logged_run_meta(reader, run_id)
                return archived_board(reader, run_id, thread_id)

        archived = await asyncio.to_thread(read_board)
        if archived is None:
            run = await manager.fetch(run_id)
            posts = run.board.thread(thread_id)
        else:
            board, _ = archived
            posts = board.thread(thread_id)
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
    def read_events() -> list[dict[str, Any]]:
        with reading(manager.log) as reader:
            if run_id not in manager.runs:
                _logged_run_meta(reader, run_id)
            events = reader.read(run_id, after_id=after, limit=limit)
            return [
                {
                    "id": e.id,
                    "ts": e.ts,
                    "kind": e.kind,
                    "agent_id": e.agent_id,
                    "payload": e.payload,
                }
                for e in events
            ]

    return await asyncio.to_thread(read_events)


@router.get("/{run_id}/timelapse")
async def get_timelapse(
    run_id: str,
    manager: ManagerDep,
    frames: Annotated[int, Query(ge=1, le=600)] = 120,
) -> Response:
    """Binary keyframe deltas of the whole run so far (see sim/timelapse.py for the layout)."""
    run = manager.runs.get(run_id)

    def render() -> bytes:
        with reading(manager.log) as reader:
            if run is None:
                meta = _logged_run_meta(reader, run_id)
                assert meta.config is not None
                config = RunConfig.model_validate(json.loads(meta.config))
                width, height = config.width, config.height
            else:
                width, height = run.canvas.width, run.canvas.height
            return build_timelapse(reader, run_id, width, height, frames)

    data = await asyncio.to_thread(render)
    return Response(content=data, media_type="application/octet-stream")


def _videos(manager: RunManager) -> VideoStore:
    if manager.videos is None:
        raise HTTPException(503, "video rendering is not configured on this server")
    return manager.videos


@router.post("/{run_id}/video", status_code=201)
async def render_run_video(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    """Render (or re-render) the run's timelapse video from its event log so far."""
    await manager.fetch(run_id)
    try:
        return await _videos(manager).render(manager.log, run_id)
    except VideoError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/{run_id}/video")
async def get_run_video(run_id: str, manager: ManagerDep) -> Response:
    """The rendered mp4 (see ``POST`` to render); 202 while a render is in progress."""
    store = _videos(manager)
    meta = store.meta(run_id)
    if meta is None:
        if store.rendering(run_id):
            return Response(status_code=202, content="rendering", media_type="text/plain")
        raise HTTPException(404, "no video rendered for this run yet; POST to render one")
    return FileResponse(
        store.video_path(run_id), media_type="video/mp4", filename=f"worldwarpixel-{run_id}.mp4"
    )


@router.get("/{run_id}/video/meta")
async def get_run_video_meta(run_id: str, manager: ManagerDep) -> dict[str, Any]:
    store = _videos(manager)
    meta = store.meta(run_id)
    if meta is None:
        return {"run_id": run_id, "rendered": False, "rendering": store.rendering(run_id)}
    return {**meta, "rendered": True, "rendering": False}


@videos_router.get("")
async def list_videos(manager: ManagerDep) -> list[dict[str, Any]]:
    """Every rendered run video with its metadata sidecar, newest first."""
    return _videos(manager).list()


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
