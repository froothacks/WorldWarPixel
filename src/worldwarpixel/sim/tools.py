from collections.abc import Callable, Mapping, Sequence
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


def apply_writes(
    canvas: Canvas, writes: Sequence[PixelWrite], sink: Callable[[str, dict[str, Any]], None]
) -> None:
    """Paint pixels onto the canvas and report each one (with its previous owner) to the sink."""
    for w in writes:
        prev = canvas.owner(w.x, w.y)
        canvas.write(w)
        sink("pixel", {"x": w.x, "y": w.y, "color": w.color, "prev_owner": prev})


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
        writer: Callable[[list[PixelWrite]], None] | None = None,
        pending: Mapping[tuple[int, int], tuple[int, int]] | None = None,
        blocked: Callable[[int, int], bool] | None = None,
    ) -> None:
        self._canvas = canvas
        self._board = board
        self._agent_id = agent_id
        self.budget = budget
        self._sink = sink
        self._clock = clock
        self._writer = writer
        self._pending = pending if pending is not None else {}
        """(x, y) -> (color, owner) of pixels handed to the writer but not yet on the canvas."""
        self._blocked = blocked

    def _get(self, x: int, y: int) -> int:
        hit = self._pending.get((x, y))
        return hit[0] if hit else self._canvas.get(x, y)

    def _owner(self, x: int, y: int) -> int:
        hit = self._pending.get((x, y))
        return hit[1] if hit else self._canvas.owner(x, y)

    def _check(self, kind: str, n: int) -> None:
        left = getattr(self.budget, kind)
        if n > left:
            raise ToolError(f"budget exceeded: {n} {kind} requested, {left} left this turn")

    def _spend(self, kind: str, n: int) -> None:
        """Charge the budget; only called once the call has succeeded, so rejects are free."""
        self._check(kind, n)
        setattr(self.budget, kind, getattr(self.budget, kind) - n)

    def _write(self, writes: list[PixelWrite]) -> None:
        """Hand accepted pixels to the run's writer (which may drip them out) or paint them now."""
        if self._writer is not None:
            self._writer(writes)
        else:
            apply_writes(self._canvas, writes, self._sink)

    def place_pixels(self, pixels: Sequence[tuple[int, int, int]]) -> dict[str, Any]:
        if not pixels:
            raise ToolError("no pixels given")
        writes: list[PixelWrite] = []
        for x, y, color in pixels:
            if not self._canvas.in_bounds(x, y):
                raise ToolError(f"({x}, {y}) is off the canvas")
            if not 0 <= color < len(PALETTE):
                raise ToolError(f"color {color} is not in 0..{len(PALETTE) - 1}")
            writes.append(PixelWrite(x, y, color, self._agent_id))
        blocked: list[PixelWrite] = []
        allowed: list[PixelWrite] = []
        for write in writes:
            if self._blocked is not None and self._blocked(write.x, write.y):
                blocked.append(write)
            else:
                allowed.append(write)
        if self.budget.pixels <= 0 and allowed:
            raise ToolError("no pixel budget left this turn")
        dropped = max(0, len(allowed) - self.budget.pixels)
        allowed = allowed[: self.budget.pixels]
        self._spend("pixels", len(allowed))
        if allowed:
            self._write(allowed)
        out: dict[str, Any] = {"placed": len(allowed), "pixels_left": self.budget.pixels}
        notes: list[str] = []
        if blocked:
            out["blocked"] = len(blocked)
            notes.append(
                f"{len(blocked)} pixels are inside a region held by the debate winner; "
                "they were not placed"
            )
        if dropped:
            out["dropped"] = dropped
            notes.append(f"only {len(allowed)} fit the budget; {dropped} pixels were dropped")
        if notes:
            out["note"] = " ".join(notes)
        return out

    def fill_rect(self, x: int, y: int, w: int, h: int, color: int) -> dict[str, Any]:
        if w <= 0 or h <= 0:
            raise ToolError("width and height must be positive")
        if not (self._canvas.in_bounds(x, y) and self._canvas.in_bounds(x + w - 1, y + h - 1)):
            raise ToolError("rectangle goes off the canvas")
        cells = [(xx, yy, color) for yy in range(y, y + h) for xx in range(x, x + w)]
        allowed = sum(1 for xx, yy, _ in cells if not self._blocked or not self._blocked(xx, yy))
        self._check("pixels", allowed)
        return self.place_pixels(cells)

    def paint_stencil(self, cells: Sequence[tuple[int, int, int]]) -> dict[str, Any]:
        """Paint towards a target picture: only cells that differ, in raster order, up to budget.

        Calling again with the same stencil continues (or repairs) it, so big drawings take
        several turns and a raided drawing can be restored.
        """
        # Visible progress first: cells whose colour changes, then same-coloured cells we merely
        # claim from other agents (or from the blank background).
        wanted = [
            (x, y, c) for x, y, c in cells if self._canvas.in_bounds(x, y) and self._get(x, y) != c
        ] + [
            (x, y, c)
            for x, y, c in cells
            if self._canvas.in_bounds(x, y)
            and self._get(x, y) == c
            and self._owner(x, y) != self._agent_id
        ]
        if not cells:
            raise ToolError("the stencil is empty")
        if not any(self._canvas.in_bounds(x, y) for x, y, _ in cells):
            raise ToolError("the drawing would be entirely off the canvas")
        if not wanted:
            return {"placed": 0, "remaining": 0, "done": True, "note": "already complete"}
        candidates = wanted[: self.budget.pixels] if self.budget.pixels > 0 else wanted
        out = self.place_pixels(candidates)
        blocked = int(out.get("blocked", 0))
        out.pop("dropped", None)
        out.pop("note", None)
        remaining = len(wanted) - int(out["placed"])
        out.update({"remaining": remaining, "done": remaining == 0})
        notes: list[str] = []
        if blocked:
            notes.append(
                f"{blocked} pixels are inside a region held by the debate winner; "
                "they were not placed"
            )
        if remaining:
            notes.append(f"{remaining} px still to paint; call the same tool again next turn")
        if notes:
            out["note"] = " ".join(notes)
        return out

    def record_action(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        """Log a successful tool call as a structured event (for the narrator and replay)."""
        self._sink("action", {"tool": tool, "args": args, "result": result})

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self._canvas.width, self._canvas.height

    def regions(self, limit: int | None = 10) -> list[dict[str, int]]:
        """Cheap summary of the busiest canvas areas used to build the observation."""
        return self._canvas.regions(limit)

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
