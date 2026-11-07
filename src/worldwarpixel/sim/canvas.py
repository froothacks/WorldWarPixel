from array import array
from dataclasses import dataclass

# The 16-colour r/place (2017) palette. A pixel is one palette index, so it fits in a byte
# and renders as a single hex character in agent prompts.
PALETTE: tuple[str, ...] = (
    "#FFFFFF",
    "#E4E4E4",
    "#888888",
    "#222222",
    "#FFA7D1",
    "#E50000",
    "#E59500",
    "#A06A42",
    "#E5D900",
    "#94E044",
    "#02BE01",
    "#00D3DD",
    "#0083C7",
    "#0000EA",
    "#CF6EE4",
    "#820080",
)
MAX_SIDE = 65535


@dataclass(frozen=True, slots=True)
class PixelWrite:
    x: int
    y: int
    color: int
    agent_id: int


class Canvas:
    """Palette-indexed pixel grid plus the id of the agent that last wrote each pixel."""

    def __init__(self, width: int, height: int) -> None:
        if not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE):
            raise ValueError(f"canvas side must be in 1..{MAX_SIDE}")
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height)
        self.owners = array("H", bytes(2 * width * height))

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _index(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            raise ValueError(f"({x}, {y}) is outside the {self.width}x{self.height} canvas")
        return y * self.width + x

    def get(self, x: int, y: int) -> int:
        return self.pixels[self._index(x, y)]

    def owner(self, x: int, y: int) -> int:
        """Agent id that last wrote the pixel, or 0 if nobody has."""
        return self.owners[self._index(x, y)]

    def write(self, w: PixelWrite) -> None:
        if not 0 <= w.color < len(PALETTE):
            raise ValueError(f"color must be in 0..{len(PALETTE) - 1}")
        i = self._index(w.x, w.y)
        self.pixels[i] = w.color
        self.owners[i] = w.agent_id

    def view(self, x: int, y: int, w: int, h: int) -> list[str]:
        """Rows of hex digits for the region, clipped to the canvas."""
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, self.width), min(y + h, self.height)
        rows: list[str] = []
        for yy in range(y0, y1):
            start = yy * self.width
            rows.append("".join(f"{c:x}" for c in self.pixels[start + x0 : start + x1]))
        return rows

    def snapshot(self) -> bytes:
        return bytes(self.pixels)
