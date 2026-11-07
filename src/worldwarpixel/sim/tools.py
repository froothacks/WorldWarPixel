from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from worldwarpixel.sim.board import Board, BoardError
from worldwarpixel.sim.canvas import PALETTE, Canvas, PixelWrite

MAX_VIEW_SIDE = 64

EventSink = Callable[[str, dict[str, Any]], None]
"""Receives (kind, payload) for every write an agent makes."""


class ToolError(Exception):
    """Raised for a rejected tool call; the message is meant to be shown to the agent."""


@dataclass(slots=True)
class TurnBudget:
    pixels: int = 8
    posts: int = 1
    views: int = 3
    reads: int = 3

    def to_dict(self) -> dict[str, int]:
        return {
            "pixels": self.pixels,
            "posts": self.posts,
            "views": self.views,
            "reads": self.reads,
        }


class ToolSet:
    """The tools one agent may call during one turn, enforcing that turn's budget."""

    def __init__(
        self,
        canvas: Canvas,
        board: Board,
        agent_id: int,
        budget: TurnBudget,
        sink: EventSink,
        clock: Callable[[], float],
    ) -> None:
        self._canvas = canvas
        self._board = board
        self._agent_id = agent_id
        self.budget = budget
        self._sink = sink
        self._clock = clock

    def _check(self, kind: str, n: int) -> None:
        left = getattr(self.budget, kind)
        if n > left:
            raise ToolError(f"budget exceeded: {n} {kind} requested, {left} left this turn")

    def _spend(self, kind: str, n: int) -> None:
        """Charge the budget; only called once the call has succeeded, so rejects are free."""
        self._check(kind, n)
        setattr(self.budget, kind, getattr(self.budget, kind) - n)

    def _write(self, writes: list[PixelWrite]) -> None:
        for w in writes:
            self._canvas.write(w)
            self._sink("pixel", {"x": w.x, "y": w.y, "color": w.color})

    def place_pixels(self, pixels: Sequence[tuple[int, int, int]]) -> dict[str, Any]:
        if not pixels:
            raise ToolError("no pixels given")
        if self.budget.pixels <= 0:
            raise ToolError("no pixel budget left this turn")
        # Small models often miscount; keep the first N pixels instead of rejecting the turn.
        dropped = max(0, len(pixels) - self.budget.pixels)
        pixels = pixels[: self.budget.pixels]
        writes: list[PixelWrite] = []
        for x, y, color in pixels:
            if not self._canvas.in_bounds(x, y):
                raise ToolError(f"({x}, {y}) is off the canvas")
            if not 0 <= color < len(PALETTE):
                raise ToolError(f"color {color} is not in 0..{len(PALETTE) - 1}")
            writes.append(PixelWrite(x, y, color, self._agent_id))
        self._spend("pixels", len(writes))
        self._write(writes)
        out: dict[str, Any] = {"placed": len(writes), "pixels_left": self.budget.pixels}
        if dropped:
            out["dropped"] = dropped
            out["note"] = f"only {len(writes)} fit the budget; {dropped} pixels were dropped"
        return out

    def fill_rect(self, x: int, y: int, w: int, h: int, color: int) -> dict[str, Any]:
        if w <= 0 or h <= 0:
            raise ToolError("width and height must be positive")
        if not (self._canvas.in_bounds(x, y) and self._canvas.in_bounds(x + w - 1, y + h - 1)):
            raise ToolError("rectangle goes off the canvas")
        self._check("pixels", w * h)
        cells = [(xx, yy, color) for yy in range(y, y + h) for xx in range(x, x + w)]
        return self.place_pixels(cells)

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self._canvas.width, self._canvas.height

    def png(self, max_side: int = 640) -> tuple[bytes, int]:
        """Free full-canvas image used to build the observation."""
        return self._canvas.to_png(max_side)

    def peek(self, x: int, y: int, w: int, h: int) -> list[str]:
        """Free read used to build the observation; does not touch the view budget."""
        return self._canvas.view(x, y, w, h)

    def view_canvas(self, x: int, y: int, w: int, h: int) -> dict[str, Any]:
        if w <= 0 or h <= 0 or w > MAX_VIEW_SIDE or h > MAX_VIEW_SIDE:
            raise ToolError(f"view size must be 1..{MAX_VIEW_SIDE} on each side")
        self._spend("views", 1)
        return {"x": x, "y": y, "rows": self._canvas.view(x, y, w, h)}

    def post_thread(self, title: str, body: str) -> dict[str, Any]:
        self._check("posts", 1)
        try:
            post = self._board.post_thread(self._agent_id, title, body, self._clock())
        except BoardError as e:
            raise ToolError(str(e)) from e
        self._spend("posts", 1)
        self._sink("post", post.to_dict())
        return {"thread_id": post.id, "post_id": post.id}

    def reply(self, thread_id: int, body: str, parent_id: int | None = None) -> dict[str, Any]:
        self._check("posts", 1)
        try:
            post = self._board.reply(self._agent_id, thread_id, body, self._clock(), parent_id)
        except BoardError as e:
            raise ToolError(str(e)) from e
        self._spend("posts", 1)
        self._sink("post", post.to_dict())
        return {"thread_id": thread_id, "post_id": post.id}

    def read_thread(self, thread_id: int) -> dict[str, Any]:
        self._check("reads", 1)
        posts = self._board.thread(thread_id)
        if not posts:
            raise ToolError(f"thread {thread_id} does not exist")
        self._spend("reads", 1)
        return {"thread_id": thread_id, "posts": [p.to_dict() for p in posts]}
