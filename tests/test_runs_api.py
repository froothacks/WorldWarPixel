import asyncio
import struct
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.hub import FRAME_PIXELS, FRAME_SNAPSHOT

SMALL = {
    "width": 32,
    "height": 16,
    "agents": 3,
    "cooldown": 0.01,
    "jitter": 0,
    "seed": 7,
    "policy": "random",
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EVENT_LOG_PATH", ":memory:")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_run_lifecycle(client: TestClient) -> None:
    created = client.post("/runs", json=SMALL).json()
    run_id = created["id"]
    assert created["state"] == "running" and created["agents"] == 3

    agents = client.get(f"/runs/{run_id}/agents").json()
    assert len(agents) == 3 and {a["id"] for a in agents} == {1, 2, 3}

    deadline = time.time() + 2
    while client.get(f"/runs/{run_id}").json()["stats"]["pixels"] == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert client.get(f"/runs/{run_id}").json()["stats"]["pixels"] > 0

    assert client.post(f"/runs/{run_id}/pause").json()["state"] == "paused"
    assert client.post(f"/runs/{run_id}/resume").json()["state"] == "running"
    canvas = client.get(f"/runs/{run_id}/canvas").json()
    assert len(canvas["rows"]) == 16 and len(canvas["rows"][0]) == 32
    events = client.get(f"/runs/{run_id}/events", params={"limit": 5}).json()
    assert events[0]["kind"] == "run.created"
    assert client.get(f"/runs/{run_id}/events", params={"limit": -1}).status_code == 422
    assert client.get(f"/runs/{run_id}/events", params={"limit": 99999}).status_code == 422
    agent = client.get(f"/runs/{run_id}/agents/1").json()
    assert agent["id"] == 1 and agent["name"] and agent["persona"]
    assert client.get(f"/runs/{run_id}/agents/9").status_code == 404
    assert client.get("/runs/nope").status_code == 404

    assert client.post(f"/runs/{run_id}/stop").json()["state"] == "stopped"
    assert [r["id"] for r in client.get("/runs").json()] == [run_id]
    assert client.delete(f"/runs/{run_id}").status_code == 204
    assert client.get("/runs").json() == []
    assert client.delete(f"/runs/{run_id}").status_code == 404


def test_active_run_cap(client: TestClient) -> None:
    ids = [client.post("/runs", json=SMALL).json()["id"] for _ in range(4)]
    assert client.post("/runs", json=SMALL).status_code == 409
    client.post(f"/runs/{ids[0]}/stop")
    assert client.post("/runs", json=SMALL).status_code == 201
    for r in client.get("/runs").json():
        client.delete(f"/runs/{r['id']}")


def test_websocket_sends_snapshot_then_diffs(client: TestClient) -> None:
    run_id = client.post("/runs", json=SMALL).json()["id"]
    with client.websocket_connect(f"/runs/{run_id}/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        assert len(snapshot["palette"]) == 16 and len(snapshot["agents"]) == 3
        frame = ws.receive_bytes()
        assert frame[0] == FRAME_SNAPSHOT
        assert struct.unpack_from("<HH", frame, 1) == (32, 16)
        assert len(frame) == 5 + 32 * 16 * 3
        owners = struct.unpack_from(f"<{32 * 16}H", frame, 5 + 32 * 16)
        assert set(owners) <= {0, 1, 2, 3}
        deadline = time.time() + 3
        while time.time() < deadline:
            msg = ws.receive()
            if msg.get("bytes") is not None:
                assert msg["bytes"][0] == FRAME_PIXELS
                break
        else:
            pytest.fail("no pixel diff frame received")
    client.post(f"/runs/{run_id}/stop")


def test_websocket_closes_when_client_disconnects_from_stopped_run(client: TestClient) -> None:
    run_id = client.post("/runs", json=SMALL).json()["id"]
    client.post(f"/runs/{run_id}/stop")
    manager = client.app.state.runs  # type: ignore[attr-defined]
    run = manager.get(run_id)
    with client.websocket_connect(f"/runs/{run_id}/ws") as ws:
        ws.receive_json()
        ws.receive_bytes()
        assert run.hub.client_count == 1
    deadline = time.time() + 2
    while run.hub.client_count and time.time() < deadline:
        time.sleep(0.02)
    assert run.hub.client_count == 0


def test_deleting_a_run_closes_its_websockets(client: TestClient) -> None:
    run_id = client.post("/runs", json=SMALL).json()["id"]
    manager = client.app.state.runs  # type: ignore[attr-defined]
    run = manager.get(run_id)
    with client.websocket_connect(f"/runs/{run_id}/ws") as ws:
        ws.receive_json()
        ws.receive_bytes()
        assert client.delete(f"/runs/{run_id}").status_code == 204
        deadline = time.time() + 2
        while time.time() < deadline:
            msg = ws.receive()
            if msg["type"] == "websocket.close":
                assert msg["code"] == 1001
                break
        else:
            pytest.fail("socket was not closed")
    assert run.hub.client_count == 0


def test_spa_fallback_does_not_swallow_api_404s(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from worldwarpixel import app as app_module

    (tmp_path / "index.html").write_text("<html>spa</html>")
    monkeypatch.setattr(app_module, "FRONTEND_DIST", tmp_path)
    assert Path(asyncio.run(app_module.frontend("some/ui/route")).path) == tmp_path / "index.html"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_module.frontend("runs/nope/unknown"))
    assert exc.value.status_code == 404


def test_thread_votes_reorder_the_board(client: TestClient) -> None:
    run_id = client.post("/runs", json={**SMALL, "posts_per_turn": 0}).json()["id"]
    first = client.post(f"/runs/{run_id}/god/announce", json={"title": "A", "body": "a"}).json()
    second = client.post(f"/runs/{run_id}/god/announce", json={"title": "B", "body": "b"}).json()
    order = [t["thread_id"] for t in client.get(f"/runs/{run_id}/threads").json()]
    assert order == [second["id"], first["id"]]
    r = client.post(f"/runs/{run_id}/threads/{first['id']}/vote", json={"delta": 1})
    assert r.status_code == 200 and r.json() == {"thread_id": first["id"], "votes": 1}
    threads = client.get(f"/runs/{run_id}/threads").json()
    assert [t["thread_id"] for t in threads] == [first["id"], second["id"]]
    assert threads[0]["votes"] == 1
    assert client.post(f"/runs/{run_id}/threads/999/vote", json={"delta": 1}).status_code == 404
    assert (
        client.post(f"/runs/{run_id}/threads/{first['id']}/vote", json={"delta": 5}).status_code
        == 422
    )
    client.post(f"/runs/{run_id}/stop")
