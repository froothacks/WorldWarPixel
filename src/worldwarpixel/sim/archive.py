"""Finished runs rebuilt from the event log, so runs from an earlier server process (a restart,
a redeploy) can still be listed, opened and replayed. Restored runs are always stopped."""

import json
from collections.abc import Callable
from typing import Any, cast

from worldwarpixel.sim.agents import AgentSpec, RandomWalkPolicy
from worldwarpixel.sim.board import Board, Post
from worldwarpixel.sim.canvas import PixelWrite
from worldwarpixel.sim.dispatch import DISPATCH_ID, DISPATCH_NAME
from worldwarpixel.sim.events import REJECTED, EventStore
from worldwarpixel.sim.run import GOD_ID, GOD_NAME, Run, RunConfig
from worldwarpixel.sim.societies import SOCIETY_NAMES, Migration
from worldwarpixel.sim.speech import SpeechSynth
from worldwarpixel.sim.timelapse import iter_events


def restore_run(
    log: EventStore,
    run_id: str,
    speech: SpeechSynth | None = None,
    *,
    source: EventStore | None = None,
) -> Run | None:
    """Replay a run's events into a stopped ``Run``; read from source if supplied."""
    events = iter_events(source or log, run_id)
    first = next(iter(events), None)
    if first is None or first.kind != "run.created":
        return None
    created = cast(dict[str, Any], first.payload)
    config = RunConfig.model_validate(created["config"])
    if config.societies and config.society_names is None:  # logged before names were generated
        config.society_names = list(SOCIETY_NAMES[: config.societies])
    # Runs logged before the roster was recorded fall back to re-seeding (exact only with a seed).
    roster = [AgentSpec(**a) for a in created.get("agents", [])]
    run = Run(run_id, config, log, RandomWalkPolicy(seed=0), agents=roster or None, speech=speech)
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
        elif e.kind == "board.vote":
            if "voter" in p:
                run.board.vote(int(p["thread_id"]), str(p["voter"]), int(p["value"]))
            else:
                run.board.set_votes(int(p["thread_id"]), int(p["votes"]))
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
            narrator._next_id = max(narrator._next_id, int(p.get("id", 0)) + 1)
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
        elif e.kind == "god.model_changed":
            config.model = str(p["model"])
        elif e.kind == "run.state":
            run.interrupted = p.get("state") == "interrupted"
            run.resume_paused = run.interrupted and p.get("was") == "paused"
        elif e.kind == "run.deleted":
            return None
    return run


def archived_board(
    log: EventStore, run_id: str, thread_id: int | None = None
) -> tuple[Board, Callable[[int], str]] | None:
    first = log.read(run_id, 0, 1)
    if not first or first[0].kind != "run.created":
        return None
    created = cast(dict[str, Any], first[0].payload)
    roster = cast(list[dict[str, Any]], created.get("agents") or [])
    if not roster:
        return None
    names = {int(agent["id"]): str(agent["name"]) for agent in roster}

    def name_of(agent_id: int) -> str:
        if agent_id == DISPATCH_ID:
            return DISPATCH_NAME
        if agent_id == GOD_ID:
            return GOD_NAME
        return names.get(agent_id, f"agent {agent_id}")

    board = Board()
    for post, society in log.posts(run_id, thread_id):
        board.restore(post, society)
    for tid, votes in log.thread_votes(run_id).items():
        board.set_votes(tid, votes)
    return board, name_of


def archived_summaries(log: EventStore, skip: set[str]) -> list[dict[str, Any]]:
    """``Run.summary``-shaped rows for logged runs not in ``skip``, without replaying them."""
    rows = []
    for row in log.run_totals():
        run_id, first_ts, pixels, posts, turns, errors, updates, answers, deleted, config = row
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
                "stats": {"pixels": pixels, "posts": posts, "turns": turns, "errors": errors},
                "agents": cfg["agents"],
                "threads": 0,
                "clients": 0,
                "llm": None,
                "narrator": {"updates": updates, "answers": answers, "failures": 0},
            }
        )
    return rows
