import struct
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.hub import FRAME_PIXELS, FRAME_SNAPSHOT

SMALL = {"width": 32, "height": 16, "agents": 3, "cooldown": 0.01, "jitter": 0, "seed": 7}


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
    assert client.get(f"/runs/{run_id}/agents/1").json()["name"] == "Agent 1"
    assert client.get(f"/runs/{run_id}/agents/9").status_code == 404
    assert client.get("/runs/nope").status_code == 404

    assert client.post(f"/runs/{run_id}/stop").json()["state"] == "stopped"
    assert [r["id"] for r in client.get("/runs").json()] == [run_id]


def test_websocket_sends_snapshot_then_diffs(client: TestClient) -> None:
    run_id = client.post("/runs", json=SMALL).json()["id"]
    with client.websocket_connect(f"/runs/{run_id}/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        assert len(snapshot["palette"]) == 16 and len(snapshot["agents"]) == 3
        frame = ws.receive_bytes()
        assert frame[0] == FRAME_SNAPSHOT
        assert struct.unpack_from("<HH", frame, 1) == (32, 16)
        assert len(frame) == 5 + 32 * 16
        deadline = time.time() + 3
        while time.time() < deadline:
            msg = ws.receive()
            if msg.get("bytes") is not None:
                assert msg["bytes"][0] == FRAME_PIXELS
                break
        else:
            pytest.fail("no pixel diff frame received")
    client.post(f"/runs/{run_id}/stop")
