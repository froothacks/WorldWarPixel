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
from worldwarpixel.sim.agents import AgentSpec
from worldwarpixel.sim.archive import archived_summaries, restore_run
from worldwarpixel.sim.board import Board, Post
from worldwarpixel.sim.dispatch import DISPATCH_ID, DISPATCH_NAME
from worldwarpixel.sim.events import EventLog, EventStore
from worldwarpixel.sim.hub import FRAME_SNAPSHOT
from worldwarpixel.sim.run import RunConfig

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
        assert len(readers) == 2 and closed == readers
        assert any(
            "restored run archived pixels=1 in" in record.getMessage() for record in caplog.records
        )

        timelapse = client.get(f"/runs/{run_id}/timelapse", params={"frames": 3})
        assert timelapse.status_code == 200
        assert timelapse.headers["content-type"] == "application/octet-stream"
        assert struct.unpack_from("<HHH", timelapse.content, 0) == (3, 16, 16)
        assert len(readers) == 3 and closed == readers


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
