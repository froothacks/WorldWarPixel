import struct

import pytest
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.sim.events import Event, EventLog
from worldwarpixel.sim.timelapse import build_timelapse, iter_events, iter_timelapse


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


def _streamed(
    log: EventLog, run_id: str, width: int, height: int, frames: int | None = None
) -> bytes:
    expected = build_timelapse(log, run_id, width, height, frames)
    data = b"".join(iter_timelapse(log, run_id, width, height, frames))
    assert data == expected
    return data


def test_timelapse_frames_hold_deltas_and_honour_clears() -> None:
    log = EventLog()
    log.append("r", 0.0, "pixel", 1, {"x": 0, "y": 0, "color": 5})
    log.append("r", 1.0, "pixel", 1, {"x": 1, "y": 0, "color": 6})
    log.append("r", 2.0, "agent.status", 1, {"status": "ignored"})
    log.append("r", 3.0, "pixel", 2, {"x": 0, "y": 0, "color": 7})
    log.append("r", 4.0, "god.clear_canvas", None, {})
    log.append("r", 5.0, "pixel", 2, {"x": 2, "y": 1, "color": 3})
    frames = _frames(_streamed(log, "r", 4, 2, frames=5))
    assert len(frames) == 5
    # cutoffs at t=1,2,3,4,5: events at ts<=cutoff land in that frame
    assert frames[0][1] == {0: 5, 1: 6} and frames[1][1] == {}
    assert frames[2][1] == {0: 7}
    assert frames[3][1] == {0: 0, 1: 0}  # the clear blanks what was painted
    assert frames[4][1] == {6: 3} and frames[4][0] == 5.0
    assert _frames(_streamed(EventLog(), "none", 4, 2, 10)) == []


def test_timelapse_defaults_to_one_frame_per_second() -> None:
    log = EventLog()
    log.append("r", 0.0, "pixel", 1, {"x": 0, "y": 0, "color": 5})
    log.append("r", 0.5, "pixel", 1, {"x": 1, "y": 0, "color": 6})
    log.append("r", 3.2, "pixel", 1, {"x": 2, "y": 0, "color": 7})

    frames = _frames(_streamed(log, "r", 4, 1))

    assert len(frames) == 4
    assert frames[-1][1] == {2: 7}


def test_timelapse_single_instant_has_one_frame() -> None:
    log = EventLog()
    log.append("r", 3.2, "pixel", 1, {"x": 0, "y": 0, "color": 5})

    frames = _frames(_streamed(log, "r", 1, 1))

    assert len(frames) == 1
    assert frames[0][1] == {0: 5}


def test_timelapse_clamps_explicit_frame_count_to_u16() -> None:
    log = EventLog()
    log.append("r", 0.0, "pixel", 1, {"x": 0, "y": 0, "color": 5})
    log.append("r", 70_000.0, "pixel", 1, {"x": 0, "y": 0, "color": 6})

    data = _streamed(log, "r", 1, 1, frames=100_000)

    assert struct.unpack_from("<H", data, 0)[0] == 65_535


def test_timelapse_endpoint() -> None:
    with TestClient(app) as client:
        cfg = {"width": 32, "height": 16, "agents": 2, "cooldown": 0.05, "policy": "random"}
        rid = client.post("/runs", json=cfg).json()["id"]
        client.post(f"/runs/{rid}/god/pixels", json={"pixels": [[1, 1, 5]]})
        client.post(f"/runs/{rid}/stop")
        manager = client.app.state.runs  # type: ignore[attr-defined]
        expected = build_timelapse(manager.log, rid, 32, 16)
        r = client.get(f"/runs/{rid}/timelapse")
        assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
        n, w, h = struct.unpack_from("<HHH", r.content, 0)
        assert (w, h) == (32, 16) and n >= 1
        assert r.content == expected
        assert client.get("/runs/nope/timelapse").status_code == 404
        assert client.get(f"/runs/{rid}/timelapse", params={"frames": 70_000}).status_code == 422


def test_iter_timelapse_yields_header_first() -> None:
    log = EventLog()
    log.append("r", 1.0, "pixel", 1, {"x": 0, "y": 0, "color": 2})
    stream = iter_timelapse(log, "r", 2, 3, frames=1)

    header = next(stream)

    assert header == struct.pack("<HHH", 1, 2, 3)
    assert header + b"".join(stream) == build_timelapse(log, "r", 2, 3, frames=1)


def test_iter_timelapse_flushes_small_chunks_on_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("worldwarpixel.sim.timelapse.CHUNK_FLUSH_INTERVAL", 0.0)
    log = EventLog()
    for ts in range(5):
        log.append("r", float(ts), "pixel", 1, {"x": 0, "y": 0, "color": 2})

    chunks = list(iter_timelapse(log, "r", 1, 1, frames=4))

    assert chunks[0] == struct.pack("<HHH", 4, 1, 1)
    assert len(chunks) > 2
    assert b"".join(chunks) == build_timelapse(log, "r", 1, 1, frames=4)


def test_events_appended_after_the_span_was_measured_are_ignored() -> None:
    class GrowingLog(EventLog):
        """Appends a late pixel after the canvas span has been measured."""

        appended = False

        def canvas_events(
            self, run_id: str, after_id: int, limit: int
        ) -> list[tuple[int, float, str, int | None, int | None, int | None]]:
            if not self.appended:
                self.appended = True
                EventLog.append(self, run_id, 99.0, "pixel", 1, {"x": 3, "y": 0, "color": 7})
            return super().canvas_events(run_id, after_id, limit)

    log = GrowingLog()
    log.append("r", 1.0, "pixel", 1, {"x": 0, "y": 0, "color": 5})
    log.append("r", 1.5, "god.clear_canvas", None, {})
    log.append("r", 2.0, "pixel", 1, {"x": 1, "y": 0, "color": 6})
    frames = _frames(b"".join(iter_timelapse(log, "r", 4, 1, 2)))
    assert frames[0][1] == {0: 0}
    assert frames[1][1] == {1: 6}
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
