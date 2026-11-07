import asyncio
import contextlib
import json
import struct
from typing import Any

from worldwarpixel.sim.canvas import Canvas

FRAME_SNAPSHOT = 1
FRAME_PIXELS = 2
RESYNC = b"\x00"
"""Queued for a subscriber that fell behind: the socket must resend a full snapshot."""
CLOSE = b"\x00\x01"
"""Queued when the run is deleted: the socket must close."""
PIXEL_STRUCT = struct.Struct("<HHBH")  # x, y, color, agent_id


def snapshot_frame(canvas: Canvas) -> bytes:
    header = struct.pack("<HH", canvas.width, canvas.height)
    return bytes([FRAME_SNAPSHOT]) + header + canvas.snapshot()


class Hub:
    """Fan-out of one run's live events to WebSocket clients.

    Pixel writes are packed into one binary frame every `flush_interval` seconds; everything
    else is sent immediately as JSON text.
    """

    def __init__(self, flush_interval: float = 0.25, queue_size: int = 2000) -> None:
        self.flush_interval = flush_interval
        self._queue_size = queue_size
        self._subs: set[asyncio.Queue[bytes | str]] = set()
        self._pixels = bytearray()
        self._task: asyncio.Task[None] | None = None

    def subscribe(self) -> asyncio.Queue[bytes | str]:
        q: asyncio.Queue[bytes | str] = asyncio.Queue(self._queue_size)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes | str]) -> None:
        self._subs.discard(q)

    @property
    def client_count(self) -> int:
        return len(self._subs)

    def _send(self, msg: bytes | str) -> None:
        for q in self._subs:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                self._mark_stale(q)

    @staticmethod
    def _mark_stale(q: asyncio.Queue[bytes | str]) -> None:
        """Drop the backlog; dropped pixel frames can't be recovered, so force a resync."""
        while not q.empty():
            q.get_nowait()
        q.put_nowait(RESYNC)

    def publish(self, type_: str, **data: Any) -> None:
        self._send(json.dumps({"type": type_, **data}, separators=(",", ":")))

    def push_pixel(self, x: int, y: int, color: int, agent_id: int) -> None:
        self._pixels += PIXEL_STRUCT.pack(x, y, color, agent_id)

    def flush(self) -> None:
        if self._pixels:
            self._send(bytes([FRAME_PIXELS]) + bytes(self._pixels))
            self._pixels.clear()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.flush_interval)
            self.flush()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.flush()

    def close_all(self) -> None:
        """Tell every subscriber to disconnect and forget them."""
        for q in self._subs:
            self._mark_stale(q)
            q.get_nowait()
            q.put_nowait(CLOSE)
        self._subs.clear()
