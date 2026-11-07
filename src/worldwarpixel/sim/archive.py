"""Finished runs rebuilt from the event log, so runs from an earlier server process (a restart,
a redeploy) can still be listed, opened and replayed. Restored runs are always stopped."""

import json
from typing import Any, cast

from worldwarpixel.sim.agents import AgentSpec, RandomWalkPolicy
from worldwarpixel.sim.board import Post
from worldwarpixel.sim.canvas import PixelWrite
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.run import Run, RunConfig
from worldwarpixel.sim.timelapse import iter_events

REJECTED = "Tool call rejected"


def restore_run(log: EventLog, run_id: str) -> Run | None:
    """Replay a run's events into a stopped ``Run``; None if it is unknown or was deleted."""
    events = iter_events(log, run_id)
    first = next(iter(events), None)
    if first is None or first.kind != "run.created":
        return None
    created = cast(dict[str, Any], first.payload)
    config = RunConfig.model_validate(created["config"])
    # Runs logged before the roster was recorded fall back to re-seeding (exact only with a seed).
    roster = [AgentSpec(**a) for a in created.get("agents", [])]
    run = Run(run_id, config, log, RandomWalkPolicy(seed=0), agents=roster or None)
    run.created_at = run.started_at = first.ts
    run.state = "stopped"
    stats, narrator = run.stats, run.narrator
    for e in events:
        p = cast(dict[str, Any], e.payload)
        if e.kind == "pixel":
            x, y = int(p["x"]), int(p["y"])
            if run.canvas.in_bounds(x, y):
                owner = int(p.get("owner", e.agent_id or 0))
                run.canvas.write(PixelWrite(x, y, int(p["color"]), owner))
            stats["pixels"] += 1
        elif e.kind == "god.clear_canvas":
            run.canvas.clear()
        elif e.kind == "post":
            run.board.restore(Post(**p))
            stats["posts"] += 1
        elif e.kind == "god.clear_board":
            run.board.clear()
        elif e.kind == "agent.thought":
            stats["turns"] += 1
            stats["errors"] += str(p.get("thought", "")).startswith(REJECTED)
            if e.agent_id in run.thoughts:
                run.thoughts[e.agent_id].append(p)
        elif e.kind == "narrator.update":
            narrator.history.append(p)
            narrator.stats["answers" if p.get("kind") == "answer" else "updates"] += 1
        elif e.kind == "god.pace":
            if p.get("cooldown") is not None:
                config.cooldown = float(p["cooldown"])
                for agent in run.agents:
                    agent.cooldown = config.cooldown
            if p.get("pixels_per_turn") is not None:
                config.pixels_per_turn = int(p["pixels_per_turn"])
        elif e.kind == "run.deleted":
            return None
    return run


def archived_summaries(log: EventLog, skip: set[str]) -> list[dict[str, Any]]:
    """``Run.summary``-shaped rows for logged runs not in ``skip``, without replaying them."""
    rows = []
    for run_id, first_ts, pixels, posts, turns, deleted, config in log.run_totals():
        if run_id in skip or deleted or config is None:
            continue
        cfg = RunConfig.model_validate(json.loads(config)).model_dump()
        rows.append(
            {
                "id": run_id,
                "state": "stopped",
                "config": cfg,
                "created_at": first_ts,
                "started_at": first_ts,
                "stats": {"pixels": pixels, "posts": posts, "turns": turns, "errors": 0},
                "agents": cfg["agents"],
                "threads": 0,
                "clients": 0,
                "llm": None,
                "narrator": {"updates": 0, "answers": 0, "failures": 0},
            }
        )
    return rows
