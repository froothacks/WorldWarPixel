import struct
import sys
import zlib
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

    def owners_le(self) -> bytes:
        """Owner ids as little-endian uint16, whatever the host byte order."""
        if sys.byteorder == "little":
            return self.owners.tobytes()
        swapped = array("H", self.owners)
        swapped.byteswap()
        return swapped.tobytes()

    def regions(self, tile: int = 64, limit: int = 10) -> list[dict[str, int]]:
        """Busiest tiles: bounds, how many pixels are painted (non-zero) and the dominant colour."""
        out: list[dict[str, int]] = []
        for y0 in range(0, self.height, tile):
            for x0 in range(0, self.width, tile):
                counts = [0] * len(PALETTE)
                for yy in range(y0, min(y0 + tile, self.height)):
                    row = yy * self.width
                    for v in self.pixels[row + x0 : row + min(x0 + tile, self.width)]:
                        counts[v] += 1
                painted = sum(counts[1:])
                if painted:
                    dominant = max(range(1, len(PALETTE)), key=counts.__getitem__)
                    out.append(
                        {
                            "x0": x0,
                            "y0": y0,
                            "x1": min(x0 + tile, self.width) - 1,
                            "y1": min(y0 + tile, self.height) - 1,
                            "painted": painted,
                            "color": dominant,
                        }
                    )
        out.sort(key=lambda r: r["painted"], reverse=True)
        return out[:limit]

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
