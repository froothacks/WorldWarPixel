"""Finished runs rebuilt from the event log, so runs from an earlier server process (a restart,
a redeploy) can still be listed, opened and replayed. Restored runs are always stopped."""

import json
from typing import Any, cast

from worldwarpixel.sim.agents import AgentSpec, RandomWalkPolicy
from worldwarpixel.sim.board import Post
from worldwarpixel.sim.canvas import PixelWrite
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.run import MIN_PIXEL_INTERVAL, Run, RunConfig
from worldwarpixel.sim.societies import Migration
from worldwarpixel.sim.timelapse import iter_events

REJECTED = "Tool call rejected"


def _stored_config(raw: dict[str, Any]) -> RunConfig:
    data = raw.copy()
    if "pixel_interval" in data:
        data["pixel_interval"] = max(data["pixel_interval"], MIN_PIXEL_INTERVAL)
    return RunConfig.model_validate(data)


def restore_run(log: EventLog, run_id: str) -> Run | None:
    """Replay a run's events into a stopped ``Run``; None if it is unknown or was deleted."""
    events = iter_events(log, run_id)
    first = next(iter(events), None)
    if first is None or first.kind != "run.created":
        return None
    created = cast(dict[str, Any], first.payload)
    config = _stored_config(cast(dict[str, Any], created["config"]))
    # Runs logged before the roster was recorded fall back to re-seeding (exact only with a seed).
    roster = [AgentSpec(**a) for a in created.get("agents", [])]
    run = Run(run_id, config, log, RandomWalkPolicy(seed=0), agents=roster or None)
    run.created_at = run.started_at = first.ts
    run.state = "stopped"
    stats, narrator, socs = run.stats, run.narrator, run.societies
    for e in events:
        p = cast(dict[str, Any], e.payload)
        if e.kind == "pixel":
            x, y = int(p["x"]), int(p["y"])
            if run.canvas.in_bounds(x, y):
                prev = int(p.get("prev_owner", run.canvas.owner(x, y)))
                owner = int(p.get("owner", e.agent_id or 0))
                run.canvas.write(PixelWrite(x, y, int(p["color"]), owner))
                run.world.observe_pixel(e.agent_id or 0, prev, x, y)
            stats["pixels"] += 1
        elif e.kind == "god.clear_canvas":
            run.canvas.clear()
        elif e.kind == "post":
            society = p.get("society")
            post = Post(**{k: v for k, v in p.items() if k != "society"})
            run.board.restore(post, None if society is None else int(society))
            run.world.observe_post(p)
            stats["posts"] += 1
        elif e.kind == "god.clear_board":
            run.board.clear()
        elif e.kind == "agent.thought":
            stats["turns"] += 1
            stats["errors"] += str(p.get("thought", "")).startswith(REJECTED)
            if e.agent_id in run.thoughts:
                run.thoughts[e.agent_id].append(p)
        elif e.kind == "society.join" and socs and e.agent_id in socs.members:
            socs.members[e.agent_id] = int(p["to"])
            socs.migrations.append(Migration(e.ts, e.agent_id, int(p["from"]), int(p["to"])))
            socs.moves += 1
        elif e.kind == "society.renamed" and socs:
            socs.rename(int(p["society"]), str(p["name"]))
        elif e.kind == "society.murmur" and socs:
            socs.set_murmur(int(p["society"]), str(p["murmur"]), str(p["transcript"]), e.ts)
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
            if p.get("pixel_interval") is not None:
                config.pixel_interval = float(p["pixel_interval"])
        elif e.kind == "run.deleted":
            return None
    return run


def archived_summaries(log: EventLog, skip: set[str]) -> list[dict[str, Any]]:
    """``Run.summary``-shaped rows for logged runs not in ``skip``, without replaying them."""
    rows = []
    for row in log.run_totals(REJECTED):
        run_id, first_ts, pixels, posts, turns, errors, updates, answers, deleted, config = row
        if run_id in skip or deleted or config is None:
            continue
        cfg = _stored_config(json.loads(config)).model_dump()
        rows.append(
            {
                "id": run_id,
                "state": "stopped",
                "config": cfg,
                "created_at": first_ts,
                "started_at": first_ts,
                "stats": {"pixels": pixels, "posts": posts, "turns": turns, "errors": errors},
                "agents": cfg["agents"],
                "threads": 0,
                "clients": 0,
                "llm": None,
                "narrator": {"updates": updates, "answers": answers, "failures": 0},
            }
        )
    return rows
