import asyncio

import pytest
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.agents import RandomWalkPolicy
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.hub import RESYNC
from worldwarpixel.sim.run import GOD_ID, GOD_NAME, Run, RunConfig

CONFIG = RunConfig(
    width=64, height=32, agents=3, cooldown=0.01, jitter=0, pixels_per_turn=4, seed=1
)


def test_god_paint_erase_and_clear() -> None:
    async def main() -> None:
        run = Run("g", CONFIG, EventLog(), RandomWalkPolicy(seed=1))
        q = run.hub.subscribe()
        assert run.god_paint([(1, 1, 5), (2, 1, 5), (99, 99, 5)]) == 2
        assert run.canvas.get(1, 1) == 5 and run.canvas.owner(1, 1) == GOD_ID
        with pytest.raises(ValueError):
            run.god_paint([(3, 3, 5), (4, 4, 99)])
        assert run.canvas.get(3, 3) == 0  # nothing from the bad batch landed
        assert run.god_paint([(1, 1, 0)], erase=True) == 1
        assert run.canvas.get(1, 1) == 0 and run.canvas.owner(1, 1) == 0
        run.hub.flush()
        assert isinstance(q.get_nowait(), bytes)
        run.god_clear_canvas()
        assert run.canvas.get(2, 1) == 0 and run.canvas.owner(2, 1) == 0
        assert q.get_nowait() == RESYNC
        assert run.agent_name(GOD_ID) == GOD_NAME
        events = run.log.read("g")
        kinds = [e.kind for e in events]
        assert kinds.count("pixel") == 3 and "god.paint" in kinds
        erased = [e for e in events if e.kind == "pixel" and e.payload["color"] == 0]
        assert erased and erased[-1].agent_id == GOD_ID
        assert erased[-1].payload["prev_owner"] == GOD_ID and erased[-1].payload["owner"] == 0

    asyncio.run(main())


def test_god_board_and_pace() -> None:
    async def main() -> None:
        run = Run("g", CONFIG, EventLog(), RandomWalkPolicy(seed=1))
        post = run.god_announce("Hear ye", "The canvas belongs to the brave.")
        assert post["author_id"] == GOD_ID and run.board.threads()
        assert run.board.digest(name_of=run.agent_name)[0]["author"] == GOD_NAME
        board = run.board
        run.god_clear_board()
        assert run.board is board and run.board.threads() == []
        again = run.god_announce("Again", "ids keep growing")
        assert again["thread_id"] > post["thread_id"]
        run.god_set_pace(cooldown=3.0, pixels_per_turn=99)
        assert all(a.cooldown == 3.0 for a in run.agents)
        assert run.config.pixels_per_turn == 99

    asyncio.run(main())


def test_god_endpoints() -> None:
    with TestClient(app) as client:
        run = client.post("/runs", json={**CONFIG.model_dump(), "policy": "random"}).json()
        rid = run["id"]
        r = client.post(f"/runs/{rid}/god/pixels", json={"pixels": [[1, 1, 5], [2, 2, 6]]})
        assert r.status_code == 200 and r.json()["placed"] == 2
        assert (
            client.post(f"/runs/{rid}/god/pixels", json={"pixels": [[1, 1, 42]]}).status_code == 422
        )
        r = client.post(f"/runs/{rid}/god/pixels", json={"pixels": [[1, 1, 0]], "erase": True})
        assert r.json()["placed"] == 1
        r = client.post(f"/runs/{rid}/god/announce", json={"title": "Hi", "body": "Paint hearts."})
        assert r.status_code == 201 and r.json()["author_id"] == GOD_ID
        assert (
            client.post(f"/runs/{rid}/god/announce", json={"title": "", "body": "x"}).status_code
            == 422
        )
        r = client.post(f"/runs/{rid}/god/pace", json={"cooldown": 2.5})
        assert r.status_code == 200 and r.json()["config"]["cooldown"] == 2.5
        assert client.post(f"/runs/{rid}/god/clear-board").status_code == 200
        assert client.post(f"/runs/{rid}/god/clear-canvas").status_code == 200
        assert client.get(f"/runs/{rid}/threads").json() == []
        client.post(f"/runs/{rid}/stop")


def test_god_endpoints_can_be_locked_with_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOD_TOKEN", "s3cret")
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            rid = client.post("/runs", json={**CONFIG.model_dump(), "policy": "random"}).json()[
                "id"
            ]
            assert client.post(f"/runs/{rid}/god/clear-board").status_code == 403
            ok = client.post(f"/runs/{rid}/god/clear-board", headers={"X-God-Token": "s3cret"})
            assert ok.status_code == 200
            client.post(f"/runs/{rid}/stop")
    finally:
        get_settings.cache_clear()


def test_announcements_reach_the_observation_until_they_expire() -> None:
    clock = [1000.0]
    run = Run("g", CONFIG, EventLog(), RandomWalkPolicy(seed=1), clock=lambda: clock[0])
    assert run.recent_announcements() == []
    post = run.god_announce("Draw hearts", "A heart for every flag, please.")
    (a,) = run.recent_announcements()
    assert a["thread_id"] == post["thread_id"] and a["title"] == "Draw hearts"
    clock[0] += 599
    assert len(run.recent_announcements()) == 1
    clock[0] += 2
    assert run.recent_announcements() == []
    run.god_announce("Again", "x")
    run.god_clear_board()
    assert run.recent_announcements() == []
