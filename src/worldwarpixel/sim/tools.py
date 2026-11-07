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


def apply_writes(
    canvas: Canvas, writes: Sequence[PixelWrite], sink: Callable[[str, dict[str, Any]], None]
) -> None:
    """Paint pixels onto the canvas and report each one (with its previous owner) to the sink."""
    for w in writes:
        prev = canvas.owner(w.x, w.y)
        canvas.write(w)
        sink("pixel", {"x": w.x, "y": w.y, "color": w.color, "prev_owner": prev})


class ToolSet:
    """The tools one agent may call during one turn, enforcing that turn's budget.

    Agents only get pixel-level painting; see the design rule in llm_policy.py.
    """

    def __init__(
        self,
        canvas: Canvas,
        board: Board,
        agent_id: int,
        budget: TurnBudget,
        sink: EventSink,
        clock: Callable[[], float],
        writer: Callable[[list[PixelWrite]], None] | None = None,
        blocked: Callable[[int, int], bool] | None = None,
    ) -> None:
        self._canvas = canvas
        self._board = board
        self._agent_id = agent_id
        self.budget = budget
        self._sink = sink
        self._clock = clock
        self._writer = writer
        self._blocked = blocked

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
        if allowed:
            out["last"] = [allowed[-1].x, allowed[-1].y]
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
            out["note"] = "; ".join(notes)
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

    def _own_words(self, *texts: str) -> None:
        """Reject posts that parrot the organiser's pinned 📢 notices instead of answering them."""
        if any("🗞" in t for t in texts):
            raise ToolError(
                "do not copy the 🗞 news; react to it in your own words with reply(thread_id=...)"
            )
        if any("📢" in t or "ANNOUNCEMENT:" in t for t in texts):
            raise ToolError(
                "do not copy the 📢 notice; answer it in your own words with reply(thread_id=...)"
            )

    def post_thread(self, title: str, body: str) -> dict[str, Any]:
        self._check("posts", 1)
        self._own_words(title, body)
        norm = title.strip().casefold()
        for t in self._board.threads():
            if (t.title or "").casefold() == norm:
                raise ToolError(
                    f"a thread titled {title.strip()!r} already exists (thread {t.id}); "
                    "reply in it instead of starting another"
                )
        try:
            post = self._board.post_thread(self._agent_id, title, body, self._clock())
        except BoardError as e:
            raise ToolError(str(e)) from e
        self._spend("posts", 1)
        self._sink("post", post.to_dict())
        return {"thread_id": post.id, "post_id": post.id}

    def reply(self, thread_id: int, body: str, parent_id: int | None = None) -> dict[str, Any]:
        self._check("posts", 1)
        self._own_words(body)
        normalized = " ".join(body.casefold().split())
        if any(
            " ".join(post.body.casefold().split()) == normalized
            for post in self._board.thread(thread_id)[-20:]
        ):
            raise ToolError("that was already said in this thread; say something new or stay quiet")
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
