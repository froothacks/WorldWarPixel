import struct

from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.timelapse import build_timelapse


def _frames(data: bytes) -> list[tuple[float, dict[int, int]]]:
    n, _w, _h = struct.unpack_from("<HHH", data, 0)
    off, out = 6, []
    for _ in range(n):
        ts, count = struct.unpack_from("<dI", data, off)
        off += 12
        deltas = {}
        for _ in range(count):
            idx, c = struct.unpack_from("<IB", data, off)
            off += 5
            deltas[idx] = c
        out.append((ts, deltas))
    assert off == len(data)
    return out


def test_timelapse_frames_hold_deltas_and_honour_clears() -> None:
    log = EventLog()
    log.append("r", 0.0, "pixel", 1, {"x": 0, "y": 0, "color": 5})
    log.append("r", 1.0, "pixel", 1, {"x": 1, "y": 0, "color": 6})
    log.append("r", 2.0, "post", 1, {"body": "ignored"})
    log.append("r", 3.0, "pixel", 2, {"x": 0, "y": 0, "color": 7})
    log.append("r", 4.0, "god.clear_canvas", None, {})
    log.append("r", 5.0, "pixel", 2, {"x": 2, "y": 1, "color": 3})
    frames = _frames(build_timelapse(log, "r", 4, 2, frames=5))
    assert len(frames) == 5
    # cutoffs at t=1,2,3,4,5: events at ts<=cutoff land in that frame
    assert frames[0][1] == {0: 5, 1: 6} and frames[1][1] == {}
    assert frames[2][1] == {0: 7}
    assert frames[3][1] == {0: 0, 1: 0}  # the clear blanks what was painted
    assert frames[4][1] == {6: 3} and frames[4][0] == 5.0
    assert _frames(build_timelapse(EventLog(), "none", 4, 2, 10)) == []


def test_timelapse_endpoint() -> None:
    with TestClient(app) as client:
        cfg = {"width": 32, "height": 16, "agents": 2, "cooldown": 0.05, "policy": "random"}
        rid = client.post("/runs", json=cfg).json()["id"]
        client.post(f"/runs/{rid}/god/pixels", json={"pixels": [[1, 1, 5]]})
        r = client.get(f"/runs/{rid}/timelapse", params={"frames": 3})
        assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
        n, w, h = struct.unpack_from("<HHH", r.content, 0)
        assert (w, h) == (32, 16) and 1 <= n <= 3
        client.post(f"/runs/{rid}/stop")
