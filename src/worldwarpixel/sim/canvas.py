import struct
import sys
import zlib
from array import array
from dataclasses import dataclass

# A classic 16-colour palette. A pixel is one palette index, so it fits in a byte
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
TILE = 64
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
        self._tiles_w = -(-width // TILE)
        tiles = self._tiles_w * -(-height // TILE)
        self._tile_counts = [[0] * len(PALETTE) for _ in range(tiles)]

    def clear(self) -> None:
        """Blank every pixel and forget all owners."""
        self.pixels = bytearray(self.width * self.height)
        self.owners = array("H", bytes(2 * self.width * self.height))
        self._tile_counts = [[0] * len(PALETTE) for _ in self._tile_counts]

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
        old = self.pixels[i]
        if old != w.color:
            counts = self._tile_counts[(w.y // TILE) * self._tiles_w + w.x // TILE]
            counts[old] -= 1
            counts[w.color] += 1
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

    def owners_le(self) -> bytes:
        """Owner ids as little-endian uint16, whatever the host byte order."""
        if sys.byteorder == "little":
            return self.owners.tobytes()
        swapped = array("H", self.owners)
        swapped.byteswap()
        return swapped.tobytes()

    def regions(self, limit: int | None = 10) -> list[dict[str, int]]:
        """Busiest TILE x TILE areas: bounds, painted (non-zero) pixel count and dominant colour.

        limit=None returns every tile, including empty ones (colour 0), busiest first.

        Counts are maintained incrementally in write(), so this is O(tiles), not O(pixels).
        """
        out: list[dict[str, int]] = []
        for t, counts in enumerate(self._tile_counts):
            painted = sum(counts[1:])
            if not painted and limit is not None:
                continue
            x0, y0 = (t % self._tiles_w) * TILE, (t // self._tiles_w) * TILE
            out.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": min(x0 + TILE, self.width) - 1,
                    "y1": min(y0 + TILE, self.height) - 1,
                    "painted": painted,
                    "size": (min(x0 + TILE, self.width) - x0) * (min(y0 + TILE, self.height) - y0),
                    "color": max(range(1, len(PALETTE)), key=counts.__getitem__) if painted else 0,
                }
            )
        out.sort(key=lambda r: r["painted"], reverse=True)
        return out if limit is None else out[:limit]

    def to_png(self, max_side: int = 640) -> tuple[bytes, int]:
        """Palette PNG of the canvas, nearest-neighbour downsampled so no side exceeds max_side.

        Returns (png_bytes, scale) where one PNG pixel covers scale x scale canvas pixels.
        """
        scale = max(1, -(-max(self.width, self.height) // max_side))
        w, h = -(-self.width // scale), -(-self.height // scale)
        raw = bytearray()
        for yy in range(h):
            raw.append(0)
            row = (yy * scale) * self.width
            raw.extend(self.pixels[row + xx * scale] for xx in range(w))
        plte = b"".join(bytes.fromhex(c[1:]) for c in PALETTE)

        def chunk(tag: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
            + chunk(b"PLTE", plte)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b"")
        )
        return png, scale
