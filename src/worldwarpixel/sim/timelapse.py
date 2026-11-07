"""Rebuild a run's canvas history from the event log as a compact binary of keyframe deltas.

Layout (little-endian): u16 frames, u16 width, u16 height, then per frame
f64 ts, u32 n, n x (u32 pixel_index, u8 colour). Frames are evenly spaced in time between the
first and last pixel event; each frame holds only the pixels that changed since the previous one,
so the client accumulates them into a buffer while scrubbing.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable

from worldwarpixel.sim.events import Event, EventLog

MAX_FRAMES = 600
PAGE = 10_000


def iter_events(log: EventLog, run_id: str) -> Iterable[Event]:
    after = 0
    while True:
        page = log.read(run_id, after_id=after, limit=PAGE)
        if not page:
            return
        yield from page
        after = page[-1].id


def build_timelapse(log: EventLog, run_id: str, width: int, height: int, frames: int) -> bytes:
    frames = max(1, min(frames, MAX_FRAMES))
    events = [e for e in iter_events(log, run_id) if e.kind in ("pixel", "god.clear_canvas")]
    out = bytearray(struct.pack("<HHH", 0, width, height))
    if not events:
        return bytes(out)
    t0, t1 = events[0].ts, events[-1].ts
    step = (t1 - t0) / frames if t1 > t0 else 0.0
    state = bytearray(width * height)
    pending: dict[int, int] = {}
    written = 0
    i = 0

    def flush(ts: float) -> None:
        nonlocal written
        out.extend(struct.pack("<dI", ts, len(pending)))
        for idx in sorted(pending):
            out.extend(struct.pack("<IB", idx, pending[idx]))
        pending.clear()
        written += 1

    for f in range(frames):
        cutoff = t1 if f == frames - 1 else t0 + step * (f + 1)
        while i < len(events) and (events[i].ts <= cutoff or f == frames - 1):
            e = events[i]
            i += 1
            if e.kind == "god.clear_canvas":
                for idx, v in enumerate(state):
                    if v:
                        pending[idx] = 0
                state = bytearray(width * height)
                continue
            p = e.payload
            x, y, c = int(p["x"]), int(p["y"]), int(p["color"])  # type: ignore[call-overload]
            if 0 <= x < width and 0 <= y < height:
                idx = y * width + x
                if state[idx] != c:
                    state[idx] = c
                    pending[idx] = c
                elif idx in pending:
                    pending[idx] = c
        flush(cutoff)
    struct.pack_into("<H", out, 0, written)
    return bytes(out)
