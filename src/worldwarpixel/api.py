import asyncio
import contextlib
import secrets
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

from worldwarpixel.sim.agents import RandomWalkPolicy
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.hub import CLOSE, RESYNC, snapshot_frame
from worldwarpixel.sim.run import Run, RunConfig


class RunManager:
    """Holds the in-memory runs of this process; the event log persists what they did."""

    def __init__(self, log: EventLog, max_active_runs: int = 4) -> None:
        self.log = log
        self.max_active_runs = max_active_runs
        self.runs: dict[str, Run] = {}

    def active(self) -> list[Run]:
        return [r for r in self.runs.values() if r.state != "stopped"]

    async def create(self, config: RunConfig) -> Run:
        if len(self.active()) >= self.max_active_runs:
            raise HTTPException(409, f"{self.max_active_runs} runs already active; stop one first")
        run_id = secrets.token_hex(4)
        policy = RandomWalkPolicy(seed=config.seed if config.seed is not None else 0)
        run = Run(run_id, config, self.log, policy)
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
