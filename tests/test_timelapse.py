import struct

import pytest
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.sim.events import Event, EventLog
from worldwarpixel.sim.timelapse import build_timelapse, iter_events


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
    log.append("r", 2.0, "agent.status", 1, {"status": "ignored"})
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


def test_events_appended_after_the_span_was_measured_are_ignored() -> None:
    class GrowingLog(EventLog):
        """Appends a late pixel the first time the second pass reads."""

        reads = 0

        def read(
            self, run_id: str, after_id: int = 0, limit: int = 1000, *, canvas_only: bool = False
        ) -> list[Event]:
            self.reads += 1
            if self.reads == 2:  # first read of the second pass
                EventLog.append(self, run_id, 99.0, "pixel", 1, {"x": 3, "y": 0, "color": 7})
            return super().read(run_id, after_id=after_id, limit=limit, canvas_only=canvas_only)

    log = GrowingLog()
    log.append("r", 1.0, "pixel", 1, {"x": 0, "y": 0, "color": 5})
    log.append("r", 2.0, "pixel", 1, {"x": 1, "y": 0, "color": 6})
    frames = _frames(build_timelapse(log, "r", 4, 1, 2))
    assert all(3 not in deltas for _ts, deltas in frames)


@pytest.mark.parametrize(("event_count", "expected_reads"), [(2, 1), (4, 2)])
def test_iter_events_stops_after_a_short_page(
    monkeypatch: pytest.MonkeyPatch, event_count: int, expected_reads: int
) -> None:
    monkeypatch.setattr("worldwarpixel.sim.timelapse.PAGE", 3)

    class CountingLog(EventLog):
        def __init__(self) -> None:
            super().__init__()
            self.read_calls = 0

        def read(
            self, run_id: str, after_id: int = 0, limit: int = 1000, *, canvas_only: bool = False
        ) -> list[Event]:
            self.read_calls += 1
            return super().read(run_id, after_id=after_id, limit=limit, canvas_only=canvas_only)

    log = CountingLog()
    for index in range(event_count):
        log.append("r", float(index), "agent.status", None, {"index": index})

    events = list(iter_events(log, "r"))

    assert [event.id for event in events] == list(range(1, event_count + 1))
    assert log.read_calls == expected_reads
    log.close()
