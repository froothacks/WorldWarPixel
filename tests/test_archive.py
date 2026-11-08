import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.agents import AgentSpec
from worldwarpixel.sim.archive import archived_summaries, restore_run
from worldwarpixel.sim.board import Board, Post
from worldwarpixel.sim.dispatch import DISPATCH_ID, DISPATCH_NAME
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.hub import FRAME_SNAPSHOT
from worldwarpixel.sim.run import MIN_PIXEL_INTERVAL, RunConfig

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


def test_archived_runs_floor_legacy_pixel_intervals(server: None) -> None:
    with TestClient(app) as client:
        log = app.state.runs.log
        run_ids = ("legacy-003", "legacy-zero")
        expected = (0.03, MIN_PIXEL_INTERVAL)  # 0.03 is valid now; 0 is floored
        for run_id, pixel_interval in zip(run_ids, (0.03, 0), strict=True):
            log.append(
                run_id,
                1.0,
                "run.created",
                None,
                {"config": {**SMALL, "pixel_interval": pixel_interval}},
            )
            restored = restore_run(log, run_id)
            assert restored is not None
            assert restored.config.pixel_interval == expected[run_ids.index(run_id)]

        listed = client.get("/runs")
        assert listed.status_code == 200
        summaries = {run["id"]: run for run in listed.json()}
        assert set(run_ids) <= summaries.keys()
        assert all(
            summaries[run_id]["config"]["pixel_interval"] == expected[run_ids.index(run_id)]
            for run_id in run_ids
        )
        for run_id in run_ids:
            opened = client.get(f"/runs/{run_id}")
            assert opened.status_code == 200
            assert opened.json()["config"]["pixel_interval"] == expected[run_ids.index(run_id)]


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
