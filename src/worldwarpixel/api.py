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
from worldwarpixel.sim.timelapse import build_timelapse


class RunManager:
    """Holds the in-memory runs of this process; the event log persists what they did."""

    def __init__(
        self,
        log: EventLog,
        max_active_runs: int = 4,
        make_policy: Callable[[RunConfig], Policy] | None = None,
        narrator_model: NarratorModel | None = None,
    ) -> None:
        self.log = log
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
        run = Run(run_id, config, self.log, policy, narrator_model=self.narrator_model)
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


class ThreadVote(BaseModel):
    delta: int = Field(ge=-2, le=2, description="+1/-1 to vote, ±2 to flip an earlier vote")


class GodPixels(BaseModel):
    pixels: list[tuple[int, int, int]] = Field(max_length=50_000)
    erase: bool = False


class GodAnnouncement(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=600)


class GodPace(BaseModel):
    cooldown: float | None = Field(default=None, ge=0.5, le=600)
    pixels_per_turn: int | None = Field(default=None, ge=1, le=1024)


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
    run.god_set_pace(body.cooldown, body.pixels_per_turn)
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
    return run.board.digest(limit=100, name_of=run.agent_name)


@router.get("/{run_id}/threads/{thread_id}")
async def get_thread(run_id: str, thread_id: int, manager: ManagerDep) -> list[dict[str, Any]]:
    posts = manager.get(run_id).board.thread(thread_id)
    if not posts:
        raise HTTPException(404, f"thread {thread_id} not found")
    return [p.to_dict() for p in posts]


@router.post("/{run_id}/threads/{thread_id}/vote")
async def vote_thread(
    run_id: str, thread_id: int, body: ThreadVote, manager: ManagerDep
) -> dict[str, Any]:
    try:
        votes = manager.get(run_id).vote_thread(thread_id, body.delta)
    except BoardError as e:
        raise HTTPException(404, str(e)) from e
    return {"thread_id": thread_id, "votes": votes}


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
