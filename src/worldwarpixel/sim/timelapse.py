"""Rebuild a run's canvas history from the event log as a compact binary of keyframe deltas.

Layout (little-endian): u16 frames, u16 width, u16 height, then per frame
f64 ts, u32 n, n x (u32 pixel_index, u8 colour). Frames are evenly spaced in time between the
first and last canvas events; by default, there is one frame per second of history. Each frame
holds only the pixels that changed since the previous one, so the client accumulates them into a
buffer while scrubbing.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Iterator

from worldwarpixel.sim.events import Event, EventStore

MAX_FRAMES = 0xFFFF
PAGE = 10_000
CHUNK_SIZE = 64 * 1024


def iter_events(log: EventStore, run_id: str, *, canvas_only: bool = False) -> Iterable[Event]:
    after = 0
    while True:
        page = log.read(run_id, after_id=after, limit=PAGE, canvas_only=canvas_only)
        if not page:
            return
        yield from page
        if len(page) < PAGE:
            return
        after = page[-1].id


def iter_timelapse(
    log: EventStore, run_id: str, width: int, height: int, frames: int | None = None
) -> Iterator[bytes]:
    """Yield binary frame deltas progressively, one frame per second of history by default."""
    span = log.canvas_span(run_id)
    if span is None:
        yield struct.pack("<HHH", 0, width, height)
        return
    t0, t1, last_id = span
    if frames is None:
        frames = math.ceil(t1 - t0)
    frames = max(1, min(frames, MAX_FRAMES))
    yield struct.pack("<HHH", frames, width, height)

    step = (t1 - t0) / frames if t1 > t0 else 0.0
    state = bytearray(width * height)
    pending: dict[int, int] = {}
    written = 0
    chunk = bytearray()

    def canvas_events() -> Iterator[tuple[int, float, str, int | None, int | None, int | None]]:
        after_id = 0
        while True:
            page = log.canvas_events(run_id, after_id, PAGE)
            if not page:
                return
            for event in page:
                if event[0] > last_id:
                    return
                yield event
            if len(page) < PAGE:
                return
            after_id = page[-1][0]

    events = iter(canvas_events())
    nxt = next(events, None)

    def flush(ts: float) -> Iterator[bytes]:
        nonlocal written
        frame = bytearray(struct.pack("<dI", ts, len(pending)))
        for idx in sorted(pending):
            frame.extend(struct.pack("<IB", idx, pending[idx]))
        pending.clear()
        written += 1

        view = memoryview(frame)
        offset = 0
        while offset < len(view):
            size = min(CHUNK_SIZE - len(chunk), len(view) - offset)
            chunk.extend(view[offset : offset + size])
            offset += size
            if len(chunk) == CHUNK_SIZE:
                yield bytes(chunk)
                chunk.clear()

    for f in range(frames):
        cutoff = t1 if f == frames - 1 else t0 + step * (f + 1)
        while nxt is not None and (nxt[1] <= cutoff or f == frames - 1):
            e = nxt
            nxt = next(events, None)
            if e[2] == "god.clear_canvas":
                for idx, v in enumerate(state):
                    if v:
                        pending[idx] = 0
                state = bytearray(width * height)
                continue
            x, y, color = e[3], e[4], e[5]
            if x is None or y is None or color is None:
                raise TypeError("pixel event is missing coordinates or color")
            if 0 <= x < width and 0 <= y < height:
                idx = y * width + x
                if state[idx] != color:
                    state[idx] = color
                    pending[idx] = color
        yield from flush(cutoff)
    assert written == frames
    if chunk:
        yield bytes(chunk)


def build_timelapse(
    log: EventStore, run_id: str, width: int, height: int, frames: int | None = None
) -> bytes:
    """Build a complete timelapse; by default, emit one frame per second of history."""
    return b"".join(iter_timelapse(log, run_id, width, height, frames))
