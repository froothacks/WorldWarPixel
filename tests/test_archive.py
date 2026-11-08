import asyncio
import logging
import struct
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.agents import AgentSpec, RandomWalkPolicy
from worldwarpixel.sim.archive import archived_summaries, restore_run
from worldwarpixel.sim.board import Board, Post
from worldwarpixel.sim.dispatch import DISPATCH_ID, DISPATCH_NAME
from worldwarpixel.sim.events import EventLog, EventStore
from worldwarpixel.sim.hub import FRAME_SNAPSHOT
from worldwarpixel.sim.run import Run, RunConfig

SMALL = {"width": 32, "height": 16, "agents": 3, "cooldown": 0.01, "jitter": 0, "policy": "random"}


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Each ``TestClient(app)`` block is one server process sharing an on-disk event log."""
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.db"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_runs_survive_a_restart_as_stopped_runs(server: None) -> None:
    with TestClient(app) as client:
        rid = client.post("/runs", json=SMALL).json()["id"]
        deadline = time.time() + 3
        while client.get(f"/runs/{rid}").json()["stats"]["posts"] == 0 and time.time() < deadline:
            time.sleep(0.02)
        client.app.state.runs.get(rid).god_announce("Hear ye", "Restarting soon")  # type: ignore[attr-defined]
        client.post(f"/runs/{rid}/god/pixels", json={"pixels": [[1, 1, 5]]})
        before = client.post(f"/runs/{rid}/stop").json()
        canvas = client.get(f"/runs/{rid}/canvas").json()
        agents = client.get(f"/runs/{rid}/agents").json()
        threads = client.get(f"/runs/{rid}/threads").json()
        world = client.get(f"/runs/{rid}/world").json()
        thoughts = client.get(f"/runs/{rid}/agents/1").json()["thoughts"]

    with TestClient(app) as client:
        listed = client.get("/runs").json()
        assert [r["id"] for r in listed] == [rid]
        assert listed[0]["state"] == "stopped" and listed[0]["agents"] == 3
        assert listed[0]["stats"]["pixels"] == before["stats"]["pixels"] > 0
        run = client.get(f"/runs/{rid}").json()
        assert run["state"] == "stopped" and run["stats"] == before["stats"]
        assert client.get(f"/runs/{rid}/canvas").json() == canvas
        restored_agents = client.get(f"/runs/{rid}/agents").json()
        assert [a["name"] for a in restored_agents] == [a["name"] for a in agents]
        assert client.get(f"/runs/{rid}/threads").json() == threads
        assert client.get(f"/runs/{rid}/world").json() == world
        assert client.get(f"/runs/{rid}/agents/1").json()["thoughts"] == thoughts
        assert client.get(f"/runs/{rid}/events", params={"limit": 1}).json()[0]["kind"] == (
            "run.created"
        )
        assert client.get(f"/runs/{rid}/timelapse", params={"frames": 4}).status_code == 200
        with client.websocket_connect(f"/runs/{rid}/ws") as ws:
            assert ws.receive_json()["run"]["state"] == "stopped"
            assert ws.receive_bytes()[0] == FRAME_SNAPSHOT
        assert client.delete(f"/runs/{rid}").status_code == 204
        assert client.get("/runs").json() == []
        assert client.get(f"/runs/{rid}").status_code == 404

    with TestClient(app) as client:
        assert client.get("/runs").json() == []
        assert client.get(f"/runs/{rid}").status_code == 404
        assert client.delete(f"/runs/{rid}").status_code == 404


def test_unknown_and_legacy_runs() -> None:
    log = EventLog()
    assert restore_run(log, "nope") is None
    # Logged before the roster was recorded: seeded agents are rebuilt from the config.
    config = RunConfig(width=16, height=16, agents=2, seed=3, policy="random")
    log.append("old", 1.0, "run.created", None, {"config": config.model_dump()})
    log.append("old", 2.0, "pixel", 1, {"x": 2, "y": 3, "color": 4})
    log.append("old", 3.0, "agent.thought", 1, {"thought": "Tool call rejected: bad json"})
    log.append("old", 4.0, "agent.thought", 2, {"thought": "Painting a sun"})
    log.append("old", 4.5, "agent.thought", 2, {"thought": "tool call rejected? no"})
    log.append("old", 5.0, "narrator.update", None, {"kind": "update", "text": "Hi"})
    log.append("old", 6.0, "narrator.update", None, {"kind": "answer", "text": "Yes"})
    run = restore_run(log, "old")
    assert run is not None and run.state == "stopped" and len(run.agents) == 2
    [listed] = archived_summaries(log, set())
    assert listed["stats"] == run.stats and listed["narrator"] == run.narrator.stats
    assert run.canvas.get(2, 3) == 4 and run.canvas.owner(2, 3) == 1
    assert run.stats["pixels"] == 1 and run.created_at == 1.0
    # Legacy pixel events had no prev_owner; the replayed canvas supplies it.
    log.append("old", 7.0, "pixel", 2, {"x": 2, "y": 3, "color": 5})
    raided = restore_run(log, "old")
    assert raided is not None and raided.world.snapshot() != run.world.snapshot()


def test_restore_run_reads_from_source_but_keeps_original_log() -> None:
    log = EventLog()
    source = EventLog()
    config = RunConfig(width=16, height=16, agents=2, seed=3, policy="random")
    source.append("source", 1.0, "run.created", None, {"config": config.model_dump()})
    source.append("source", 2.0, "pixel", 1, {"x": 2, "y": 3, "color": 4})

    run = restore_run(log, "source", source=source)

    assert run is not None and run.log is log
    assert run.created_at == 1.0 and run.stats["pixels"] == 1
    assert run.canvas.get(2, 3) == 4


def test_archived_api_reads_use_and_close_readers(
    server: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with TestClient(app) as client:
        manager = client.app.state.runs  # type: ignore[attr-defined]
        run_id = "archived"
        config = RunConfig(width=16, height=16, agents=2, seed=3, policy="random")
        manager.log.append(run_id, 1.0, "run.created", None, {"config": config.model_dump()})
        manager.log.append(run_id, 2.0, "pixel", 1, {"x": 2, "y": 3, "color": 4})

        readers = []
        closed = []
        original_reader = manager.log.reader

        def tracked_reader() -> EventStore:
            reader = cast(EventStore, original_reader())
            readers.append(reader)
            close = reader.close

            def tracked_close() -> None:
                closed.append(reader)
                close()

            monkeypatch.setattr(reader, "close", tracked_close)
            return reader

        monkeypatch.setattr(manager.log, "reader", tracked_reader)

        with caplog.at_level(logging.INFO, logger="worldwarpixel.api"):
            events = client.get(f"/runs/{run_id}/events")
        assert events.status_code == 200
        assert [event["kind"] for event in events.json()] == ["run.created", "pixel"]
        assert events.json()[1]["payload"] == {"x": 2, "y": 3, "color": 4}
        assert len(readers) == 1 and closed == readers
        assert not any(
            "restored run archived pixels=1 in" in record.getMessage() for record in caplog.records
        )

        timelapse = client.get(f"/runs/{run_id}/timelapse", params={"frames": 3})
        assert timelapse.status_code == 200
        assert timelapse.headers["content-type"] == "application/octet-stream"
        assert struct.unpack_from("<HHH", timelapse.content, 0) == (3, 16, 16)
        assert len(readers) == 2 and closed == readers


def test_archived_board_and_event_reads_do_not_replay(
    server: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(app) as client:
        manager = client.app.state.runs  # type: ignore[attr-defined]
        run_id = "archived-board"
        roster = [
            AgentSpec(1, "Alice Example", "A", 1, 0, 0, 1.0),
            AgentSpec(2, "Bob Example", "B", 2, 1, 0, 1.0),
        ]
        config = RunConfig(
            width=16,
            height=16,
            agents=2,
            policy="random",
            societies=2,
            society_names=["First", "Second"],
        )
        manager.log.append(
            run_id,
            1.0,
            "run.created",
            None,
            {"config": config.model_dump(), "agents": [agent.to_dict() for agent in roster]},
        )
        manager.log.append(run_id, 2.0, "pixel", 1, {"x": 1, "y": 1, "color": 5, "owner": 1})
        run = Run(run_id, config, manager.log, RandomWalkPolicy(seed=1), agents=roster)
        root = run.board.post_thread(1, "First thread", "Root body", 3.0, society=1)
        run._on_write(1, "post", {**root.to_dict(), "society": 1})
        reply = run.board.reply(2, root.id, "Reply body", 4.0)
        run._on_write(2, "post", reply.to_dict())
        society_thread = run.board.post_thread(2, "Second thread", "Another body", 5.0, society=2)
        run._on_write(2, "post", {**society_thread.to_dict(), "society": 2})
        run.vote_thread(root.id, "viewer", 1)
        run.agent_vote(2, root.id, True)
        run.vote_thread(society_thread.id, "viewer", -1)
        run.state = "stopped"
        vote_events = [event for event in manager.log.read(run_id) if event.kind == "board.vote"]
        assert [
            (
                event.agent_id,
                event.payload["thread_id"],
                event.payload["value"],
                event.payload["votes"],
            )
            for event in vote_events
        ] == [
            (None, root.id, 1, 1),
            (2, root.id, 1, 2),
            (None, society_thread.id, -1, -1),
        ]
        manager.runs[run_id] = run

        before_threads = client.get(f"/runs/{run_id}/threads")
        before_thread = client.get(f"/runs/{run_id}/threads/{root.id}")
        before_events = client.get(f"/runs/{run_id}/events")
        before_timelapse = client.get(f"/runs/{run_id}/timelapse", params={"frames": 3})
        assert before_threads.status_code == before_thread.status_code == 200
        assert before_events.status_code == before_timelapse.status_code == 200

        manager.runs.pop(run_id)
        monkeypatch.setattr(
            "worldwarpixel.api.restore_run",
            lambda *args, **kwargs: pytest.fail("archived read replayed the run"),
        )
        assert client.get(f"/runs/{run_id}/threads").json() == before_threads.json()
        assert client.get(f"/runs/{run_id}/threads/{root.id}").json() == before_thread.json()
        assert client.get(f"/runs/{run_id}/events").json() == before_events.json()
        after_timelapse = client.get(f"/runs/{run_id}/timelapse", params={"frames": 3})
        assert after_timelapse.status_code == 200
        assert after_timelapse.content == before_timelapse.content
        assert manager.log.thread_votes(run_id) == {root.id: 2, society_thread.id: -1}


def test_archived_endpoints_reject_deleted_and_unknown_runs(server: None) -> None:
    with TestClient(app) as client:
        manager = client.app.state.runs  # type: ignore[attr-defined]
        config = RunConfig(width=16, height=16, agents=1, policy="random")
        manager.log.append("deleted", 1.0, "run.created", None, {"config": config.model_dump()})
        manager.log.append("deleted", 2.0, "run.deleted", None, {})
        manager.log.append("configless", 1.0, "run.created", None, {"agents": []})

        for run_id in ("deleted", "configless", "unknown"):
            assert client.get(f"/runs/{run_id}/threads").status_code == 404
            assert client.get(f"/runs/{run_id}/threads/1").status_code == 404
            assert client.get(f"/runs/{run_id}/events").status_code == 404
            assert client.get(f"/runs/{run_id}/timelapse").status_code == 404


def test_legacy_archived_board_falls_back_to_restore(
    server: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(app) as client:
        manager = client.app.state.runs  # type: ignore[attr-defined]
        config = RunConfig(width=16, height=16, agents=1, seed=1, policy="random")
        manager.log.append(
            "legacy-board", 1.0, "run.created", None, {"config": config.model_dump()}
        )
        post = Post(1, 65535, 1, None, "legacy", "saved", 2.0)
        manager.log.append("legacy-board", 2.0, "post", 65535, post.to_dict())
        calls = []

        def track_restore(*args: object, **kwargs: object) -> Run | None:
            calls.append(args)
            return restore_run(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("worldwarpixel.api.restore_run", track_restore)
        response = client.get("/runs/legacy-board/threads")

        assert response.status_code == 200
        assert response.json()[0]["title"] == "legacy"
        assert len(calls) == 1


def test_board_restore_keeps_ids_and_skips_orphans() -> None:
    board = Board()
    board.restore(Post(5, 1, 5, None, "t", "root", 1.0))
    board.restore(Post(6, 2, 5, 5, None, "reply", 2.0))
    board.restore(Post(7, 2, 99, None, None, "orphan", 3.0))
    assert [p.id for p in board.thread(5)] == [5, 6] and board.get(7) is None
    assert board.post_thread(1, "next", "body", 4.0).id == 8


def test_restored_runs_keep_god_pace_changes() -> None:
    log = EventLog()
    config = RunConfig(width=16, height=16, agents=2, seed=3, cooldown=10, policy="random")
    log.append("p", 1.0, "run.created", None, {"config": config.model_dump()})
    log.append("p", 2.0, "god.pace", 65535, {"cooldown": 2.5, "pixels_per_turn": None})
    log.append(
        "p", 3.0, "god.pace", 65535, {"cooldown": None, "pixels_per_turn": 7, "pixel_interval": 0.5}
    )
    run = restore_run(log, "p")
    assert run is not None
    assert (run.config.cooldown, run.config.pixels_per_turn, run.config.pixel_interval) == (
        2.5,
        7,
        0.5,
    )
    assert {a.cooldown for a in run.agents} == {2.5}


def test_restore_run_restores_board_vote_scores() -> None:
    log = EventLog()
    config = RunConfig(width=16, height=16, agents=1, seed=3, policy="random")
    log.append("votes", 1.0, "run.created", None, {"config": config.model_dump()})
    post = Post(1, 1, 1, None, "thread", "body", 2.0)
    log.append("votes", 2.0, "post", 1, post.to_dict())
    log.append("votes", 3.0, "board.vote", 1, {"thread_id": 1, "votes": 4})

    run = restore_run(log, "votes")

    assert run is not None and run.board.votes(1) == 4


def test_dispatch_posts_restore_with_the_world_news_author() -> None:
    log = EventLog()
    config = RunConfig(width=16, height=16, agents=1, seed=3, policy="random")
    agent = AgentSpec(1, "Priya Raman", "India", 2, 0, 0, 1.0)
    log.append(
        "dispatch",
        1.0,
        "run.created",
        None,
        {"config": config.model_dump(), "agents": [agent.to_dict()]},
    )
    post = Post(1, DISPATCH_ID, 1, None, "⚔️ India vs Japan", "India is holding.", 2.0)
    log.append("dispatch", 2.0, "post", DISPATCH_ID, post.to_dict())

    run = restore_run(log, "dispatch")

    assert run is not None
    assert run.agent_name(DISPATCH_ID) == DISPATCH_NAME
    assert run.board.threads() == [post]
    assert run.world.snapshot()["edges"] == []
    run.narrator.observe_write(DISPATCH_ID, "post", post.to_dict())


def test_runs_archived_without_a_model_still_load() -> None:
    config = RunConfig.model_validate({**RunConfig().model_dump(), "model": None})
    assert config.model == RunConfig().model


def test_runs_interrupted_by_a_restart_resume(
    server: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESUME_RUNS", "true")
    get_settings.cache_clear()
    with TestClient(app) as client:
        live = client.post("/runs", json=SMALL).json()["id"]
        paused = client.post("/runs", json=SMALL).json()["id"]
        stopped = client.post("/runs", json=SMALL).json()["id"]
        time.sleep(0.2)
        client.post(f"/runs/{paused}/pause")
        client.post(f"/runs/{stopped}/stop")
        before = client.get(f"/runs/{live}").json()["stats"]["pixels"]

    with TestClient(app) as client:
        deadline = time.time() + 5
        while client.get(f"/runs/{live}").json()["state"] != "running" and time.time() < deadline:
            time.sleep(0.05)
        assert client.get(f"/runs/{live}").json()["state"] == "running"
        assert client.get(f"/runs/{paused}").json()["state"] == "paused"
        assert client.get(f"/runs/{stopped}").json()["state"] == "stopped"
        deadline = time.time() + 3
        while client.get(f"/runs/{live}").json()["stats"]["pixels"] <= before:
            assert time.time() < deadline
            time.sleep(0.05)
        kinds = [e["kind"] for e in client.get(f"/runs/{live}/events").json()]
        assert kinds.count("run.created") == 1
        client.delete(f"/runs/{paused}")

    with TestClient(app) as client:  # resumed again after the second restart, deleted one not
        deadline = time.time() + 5
        while client.get(f"/runs/{live}").json()["state"] != "running" and time.time() < deadline:
            time.sleep(0.05)
        assert client.get(f"/runs/{live}").json()["state"] == "running"
        assert client.get(f"/runs/{paused}").status_code == 404
        client.post(f"/runs/{live}/stop")


def test_restored_interrupted_runs_keep_model_and_narrator_ids() -> None:
    log = EventLog()
    config = RunConfig(width=32, height=16, agents=2, seed=3, policy="random")
    log.append("m", 1.0, "run.created", None, {"config": config.model_dump()})
    log.append("m", 2.0, "god.model_changed", 65535, {"model": "other"})
    log.append("m", 3.0, "narrator.update", None, {"id": 4, "kind": "update", "headline": "h"})
    log.append("m", 4.0, "run.state", None, {"state": "interrupted", "was": "running"})
    run = restore_run(log, "m")
    assert run is not None
    assert run.config.model == "other"
    assert run.narrator._next_id == 5
    assert run.interrupted and not run.resume_paused
    assert log.interrupted_runs() == ["m"]
    asyncio.run(run.stop())  # stopped while waiting to resume: it must not come back
    assert not run.interrupted
    assert log.interrupted_runs() == []
