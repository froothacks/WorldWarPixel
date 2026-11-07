"""Rebuild a run's canvas history from the event log as a compact binary of keyframe deltas.

Layout (little-endian): u16 frames, u16 width, u16 height, then per frame
f64 ts, u32 n, n x (u32 pixel_index, u8 colour). Frames are evenly spaced in time between the
first and last pixel event; each frame holds only the pixels that changed since the previous one,
so the client accumulates them into a buffer while scrubbing.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from itertools import takewhile

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


KINDS = ("pixel", "god.clear_canvas")


def _time_span(log: EventLog, run_id: str) -> tuple[float, float, int] | None:
    """(first ts, last ts, last event id) of the relevant events, or None if there are none."""
    t0 = t1 = None
    last_id = 0
    for e in iter_events(log, run_id):
        if e.kind in KINDS:
            t0 = e.ts if t0 is None else t0
            t1, last_id = e.ts, e.id
    return None if t0 is None or t1 is None else (t0, t1, last_id)


def build_timelapse(log: EventLog, run_id: str, width: int, height: int, frames: int) -> bytes:
    """Two passes over the log (span, then deltas) so memory stays O(canvas), not O(events)."""
    frames = max(1, min(frames, MAX_FRAMES))
    out = bytearray(struct.pack("<HHH", 0, width, height))
    span = _time_span(log, run_id)
    if span is None:
        return bytes(out)
    t0, t1, last_id = span
    step = (t1 - t0) / frames if t1 > t0 else 0.0
    state = bytearray(width * height)
    pending: dict[int, int] = {}
    written = 0
    relevant = (e for e in iter_events(log, run_id) if e.kind in KINDS)
    # A live run keeps appending past the span; cut by id so a clock step never truncates it.
    events = takewhile(lambda e: e.id <= last_id, relevant)
    nxt = next(events, None)

    def flush(ts: float) -> None:
        nonlocal written
        out.extend(struct.pack("<dI", ts, len(pending)))
        for idx in sorted(pending):
            out.extend(struct.pack("<IB", idx, pending[idx]))
        pending.clear()
        written += 1

    for f in range(frames):
        cutoff = t1 if f == frames - 1 else t0 + step * (f + 1)
        while nxt is not None and (nxt.ts <= cutoff or f == frames - 1):
            e = nxt
            nxt = next(events, None)
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
        flush(cutoff)
    struct.pack_into("<H", out, 0, written)
    return bytes(out)
