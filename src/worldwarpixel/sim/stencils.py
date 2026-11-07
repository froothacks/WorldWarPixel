# ruff: noqa: E501
"""Stencils a tiny model can draw with: a 3x5 font, pixel-art shapes and simplified flags.

Every helper returns absolute (x, y, color) cells; ToolSet.paint_stencil paints them progressively.
"""

from __future__ import annotations

from collections.abc import Sequence

Cell = tuple[int, int, int]

# fmt: off
FONT: dict[str, tuple[str, ...]] = {
    "A": ("###", "#.#", "###", "#.#", "#.#"), "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": ("###", "#..", "#..", "#..", "###"), "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "##.", "#..", "###"), "F": ("###", "#..", "##.", "#..", "#.."),
    "G": ("###", "#..", "#.#", "#.#", "###"), "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"), "J": ("..#", "..#", "..#", "#.#", "###"),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"), "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "###", "#.#", "#.#"), "N": ("##.", "#.#", "#.#", "#.#", "#.#"),
    "O": ("###", "#.#", "#.#", "#.#", "###"), "P": ("###", "#.#", "###", "#..", "#.."),
    "Q": ("###", "#.#", "#.#", "###", "..#"), "R": ("##.", "#.#", "##.", "#.#", "#.#"),
    "S": ("###", "#..", "###", "..#", "###"), "T": ("###", ".#.", ".#.", ".#.", ".#."),
    "U": ("#.#", "#.#", "#.#", "#.#", "###"), "V": ("#.#", "#.#", "#.#", "#.#", ".#."),
    "W": ("#.#", "#.#", "###", "###", "#.#"), "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."), "Z": ("###", "..#", ".#.", "#..", "###"),
    "0": ("###", "#.#", "#.#", "#.#", "###"), "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###", "..#", "###", "#..", "###"), "3": ("###", "..#", "###", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"), "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"), "7": ("###", "..#", "..#", "..#", "..#"),
    "8": ("###", "#.#", "###", "#.#", "###"), "9": ("###", "#.#", "###", "..#", "###"),
    "!": (".#.", ".#.", ".#.", "...", ".#."), "?": ("###", "..#", ".##", "...", ".#."),
    ".": ("...", "...", "...", "...", ".#."), ",": ("...", "...", "...", ".#.", "#.."),
    "-": ("...", "...", "###", "...", "..."), "+": ("...", ".#.", "###", ".#.", "..."),
    "'": (".#.", ".#.", "...", "...", "..."), ":": ("...", ".#.", "...", ".#.", "..."),
    "/": ("..#", "..#", ".#.", "#..", "#.."), "<": ("..#", ".#.", "#..", ".#.", "..#"),
    ">": ("#..", ".#.", "..#", ".#.", "#.."), "=": ("...", "###", "...", "###", "..."),
    "#": ("#.#", "###", "#.#", "###", "#.#"), "@": ("###", "#.#", "###", "#..", "###"),
    "&": (".#.", "#.#", ".#.", "#.#", ".##"), "*": ("#.#", ".#.", "###", ".#.", "#.#"),
    "(": (".#.", "#..", "#..", "#..", ".#."), ")": (".#.", "..#", "..#", "..#", ".#."),
    " ": ("...", "...", "...", "...", "..."),
}

# '#' is the main colour, 'o' the secondary colour (defaults to black).
SHAPES: dict[str, tuple[str, ...]] = {
    "heart": (".##.##.", "#######", "#######", ".#####.", "..###..", "...#..."),
    "star": ("...#...", "..###..", "#######", ".#####.", "..#.#..", ".#...#."),
    "smiley": (".#####.", "#######", "#o###o#", "#######", "#o###o#", "##ooo##", ".#####."),
    "skull": (".#####.", "#######", "#o###o#", "#######", ".##.##.", ".#o#o#.", ".#####."),
    "diamond": ("...#...", "..###..", ".#####.", "#######", ".#####.", "..###..", "...#..."),
    "arrow": ("...#...", "...##..", "#######", "#######", "...##..", "...#..."),
    "cross": ("..##..", "..##..", "######", "######", "..##..", "..##.."),
    "check": (".....#", "....##", "#..##.", "##.#..", ".###..", "..#..."),
    "house": ("...#...", "..###..", ".#####.", "#######", ".ooooo.", ".o#o#o.", ".o#o#o.", ".ooooo."),
    "tree": ("...#...", "..###..", ".#####.", "..###..", ".#####.", "#######", "...o...", "...o..."),
    "flower": ("..###..", ".#####.", "##ooo##", "##ooo##", ".#####.", "..###..", "...o...", "...o..."),
    "cat": ("#.....#", "##...##", "#######", "#o###o#", "#######", ".#####.", ".#.#.#."),
    "fish": ("....##.", "..####.#", ".######.", "#o######", ".######.", "..####.#", "....##."),
    "sun": ("...#...", ".#.#.#.", "..###..", "#######", "..###..", ".#.#.#.", "...#..."),
    "moon": ("..###..", ".##....", "##.....", "##.....", "##.....", ".##....", "..###.."),
    "crown": ("#..#..#", "##.#.##", "#######", "#######", "#o#o#o#", "#######"),
    "rocket": ("...#...", "..###..", "..#o#..", "..###..", ".#####.", "#..#..#", "...o..."),
    "invader": ("..#...#..", "...#.#...", "..#####..", ".##o###o##", "#########", "#.#####.#", "#.#...#.#", "...##.##..."),
    "mushroom": ("..#####..", ".##o###o#.", "#o#######", "#########", "..ooooo..", "..o.o.o..", "..ooooo.."),
    "peace": (".#####.", "#..#..#", "#..#..#", "#.###.#", "##.#.##", "#..#..#", ".#####."),
    "sword": ("......#", ".....##", "....##.", "...##..", "o.##...", ".o#....", "oo.....", "o.o...."),
    "note": ("....##", "...#.#", "...#..", "...#..", "...#..", ".###..", "####..", ".##..."),
    "circle": (".#####.", "#######", "#######", "#######", "#######", "#######", ".#####."),
    "ring": (".#####.", "##...##", "#.....#", "#.....#", "#.....#", "##...##", ".#####."),
    "ghost": (".#####.", "#######", "#o###o#", "#######", "#######", "#######", "#.#.#.#"),
    "bird": ("..##...", ".#o##..", "######.", ".######", "..###..", "...#..."),
}
# fmt: on

# Flag layers in paint order; fractions are of the flag's width/height.
# ("h", colours) / ("v", colours): equal stripes.  ("disc", c, cx, cy, r)
# ("canton", c, w, h)  ("cross", c, thickness)  ("tri", c, width)  ("band", c, y0, y1)
WHITE, LGREY, GREY, BLACK, PINK, RED, ORANGE, BROWN = range(8)
YELLOW, LIME, GREEN, CYAN, BLUE, DBLUE, MAGENTA, PURPLE = range(8, 16)
Layer = tuple[object, ...]
FLAGS: dict[str, list[Layer]] = {
    "india": [("h", [ORANGE, WHITE, GREEN]), ("disc", DBLUE, 0.5, 0.5, 0.12)],
    "china": [("h", [RED]), ("disc", YELLOW, 0.2, 0.3, 0.12)],
    "usa": [("h", [RED, WHITE, RED, WHITE, RED, WHITE, RED]), ("canton", DBLUE, 0.4, 0.55)],
    "indonesia": [("h", [RED, WHITE])],
    "pakistan": [("h", [GREEN]), ("band_v", WHITE, 0.0, 0.25), ("disc", WHITE, 0.62, 0.5, 0.2)],
    "nigeria": [("v", [GREEN, WHITE, GREEN])],
    "brazil": [("h", [GREEN]), ("disc", YELLOW, 0.5, 0.5, 0.36), ("disc", DBLUE, 0.5, 0.5, 0.2)],
    "bangladesh": [("h", [GREEN]), ("disc", RED, 0.45, 0.5, 0.3)],
    "russia": [("h", [WHITE, BLUE, RED])],
    "mexico": [("v", [GREEN, WHITE, RED]), ("disc", BROWN, 0.5, 0.5, 0.12)],
    "ethiopia": [("h", [GREEN, YELLOW, RED]), ("disc", BLUE, 0.5, 0.5, 0.25)],
    "japan": [("h", [WHITE]), ("disc", RED, 0.5, 0.5, 0.3)],
    "philippines": [("h", [BLUE, RED]), ("tri", WHITE, 0.45)],
    "egypt": [("h", [RED, WHITE, BLACK]), ("disc", YELLOW, 0.5, 0.5, 0.1)],
    "vietnam": [("h", [RED]), ("disc", YELLOW, 0.5, 0.5, 0.2)],
    "congo": [("h", [CYAN]), ("diag", RED), ("disc", YELLOW, 0.15, 0.25, 0.1)],
    "iran": [("h", [GREEN, WHITE, RED]), ("disc", RED, 0.5, 0.5, 0.1)],
    "turkey": [("h", [RED]), ("disc", WHITE, 0.4, 0.5, 0.25), ("disc", RED, 0.46, 0.5, 0.2)],
    "germany": [("h", [BLACK, RED, YELLOW])],
    "thailand": [("h", [RED, WHITE, BLUE, BLUE, WHITE, RED])],
    "uk": [("h", [DBLUE]), ("cross", WHITE, 0.3), ("cross", RED, 0.15)],
    "france": [("v", [BLUE, WHITE, RED])],
    "italy": [("v", [GREEN, WHITE, RED])],
    "south_africa": [("h", [RED, GREEN, BLUE]), ("band", GREEN, 0.4, 0.6), ("tri", BLACK, 0.35)],
    "tanzania": [("h", [GREEN]), ("diag", BLACK), ("tri_br", CYAN)],
    "kenya": [("h", [BLACK, RED, RED, GREEN]), ("disc", RED, 0.5, 0.5, 0.15)],
    "south_korea": [("h", [WHITE]), ("disc", RED, 0.5, 0.5, 0.22), ("band", BLUE, 0.5, 0.72)],
    "colombia": [("h", [YELLOW, YELLOW, BLUE, RED])],
    "spain": [("h", [RED, YELLOW, YELLOW, RED])],
    "argentina": [("h", [CYAN, WHITE, CYAN]), ("disc", YELLOW, 0.5, 0.5, 0.1)],
    "ukraine": [("h", [BLUE, YELLOW])],
    "canada": [("v", [RED, WHITE, WHITE, RED]), ("disc", RED, 0.5, 0.5, 0.2)],
    "poland": [("h", [WHITE, RED])],
    "sweden": [("h", [BLUE]), ("cross", YELLOW, 0.2)],
    "norway": [("h", [RED]), ("cross", WHITE, 0.3), ("cross", DBLUE, 0.15)],
    "denmark": [("h", [RED]), ("cross", WHITE, 0.2)],
    "finland": [("h", [WHITE]), ("cross", BLUE, 0.2)],
    "saudi_arabia": [("h", [GREEN]), ("band", WHITE, 0.6, 0.7)],
    "iraq": [("h", [RED, WHITE, BLACK])],
    "morocco": [("h", [RED]), ("disc", GREEN, 0.5, 0.5, 0.2)],
    "peru": [("v", [RED, WHITE, RED])],
    "netherlands": [("h", [RED, WHITE, BLUE])],
    "belgium": [("v", [BLACK, YELLOW, RED])],
    "greece": [("h", [BLUE, WHITE, BLUE, WHITE, BLUE, WHITE, BLUE, WHITE, BLUE])],
    "portugal": [("v", [GREEN, RED, RED]), ("disc", YELLOW, 0.33, 0.5, 0.18)],
    "israel": [
        ("h", [WHITE]),
        ("band", BLUE, 0.1, 0.25),
        ("band", BLUE, 0.75, 0.9),
        ("disc", BLUE, 0.5, 0.5, 0.15),
    ],
    "palestine": [("h", [BLACK, WHITE, GREEN]), ("tri", RED, 0.4)],
    "cuba": [("h", [BLUE, WHITE, BLUE, WHITE, BLUE]), ("tri", RED, 0.4)],
    "venezuela": [("h", [YELLOW, BLUE, RED])],
    "chile": [("h", [WHITE, RED]), ("canton", BLUE, 0.33, 0.5)],
    "malaysia": [("h", [RED, WHITE] * 4), ("canton", DBLUE, 0.5, 0.5)],
    "myanmar": [("h", [YELLOW, GREEN, RED]), ("disc", WHITE, 0.5, 0.5, 0.2)],
    "uganda": [("h", [BLACK, YELLOW, RED, BLACK, YELLOW, RED])],
    "ghana": [("h", [RED, YELLOW, GREEN]), ("disc", BLACK, 0.5, 0.5, 0.1)],
    "algeria": [("v", [GREEN, WHITE]), ("disc", RED, 0.5, 0.5, 0.2)],
    "sudan": [("h", [RED, WHITE, BLACK]), ("tri", GREEN, 0.35)],
    "uzbekistan": [("h", [CYAN, WHITE, GREEN])],
    "ireland": [("v", [GREEN, WHITE, ORANGE])],
    "switzerland": [("h", [RED]), ("cross_c", WHITE, 0.2)],
    "austria": [("h", [RED, WHITE, RED])],
    "hungary": [("h", [RED, WHITE, GREEN])],
    "romania": [("v", [BLUE, YELLOW, RED])],
    "czechia": [("h", [WHITE, RED]), ("tri", BLUE, 0.5)],
    "australia": [("h", [DBLUE]), ("canton", DBLUE, 0.5, 0.5), ("disc", WHITE, 0.75, 0.5, 0.08)],
    "nepal": [("h", [RED]), ("tri", RED, 1.0), ("disc", WHITE, 0.3, 0.5, 0.1)],
    "pride": [("h", [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE])],
    "eu": [("h", [DBLUE]), ("disc", YELLOW, 0.5, 0.5, 0.3), ("disc", DBLUE, 0.5, 0.5, 0.22)],
    "un": [("h", [CYAN]), ("disc", WHITE, 0.5, 0.5, 0.25), ("disc", CYAN, 0.5, 0.5, 0.15)],
    "pirate": [
        ("h", [BLACK]),
        ("disc", WHITE, 0.5, 0.45, 0.2),
        ("disc", BLACK, 0.43, 0.4, 0.06),
        ("disc", BLACK, 0.57, 0.4, 0.06),
    ],
}


def text_cells(text: str, x: int, y: int, color: int, scale: int = 1) -> list[Cell]:
    cells: list[Cell] = []
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT["?"])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "#":
                    cells.extend(_block(cx + gx * scale, y + gy * scale, scale, color))
        cx += 4 * scale
    return cells


def shape_cells(shape: str, x: int, y: int, color: int, color2: int, scale: int = 1) -> list[Cell]:
    rows = SHAPES[shape]
    cells: list[Cell] = []
    for gy, row in enumerate(rows):
        for gx, bit in enumerate(row):
            if bit in "#o":
                c = color if bit == "#" else color2
                cells.extend(_block(x + gx * scale, y + gy * scale, scale, c))
    return cells


def flag_cells(name: str, x: int, y: int, width: int) -> list[Cell]:
    layers = FLAGS[name]
    w, h = width, max(3, round(width * 2 / 3))
    cells: list[Cell] = []
    for py in range(h):
        for px in range(w):
            fx, fy = (px + 0.5) / w, (py + 0.5) / h
            c = _flag_colour(layers, fx, fy, w, h)
            if c is not None:
                cells.append((x + px, y + py, c))
    return cells


def _flag_colour(layers: Sequence[Layer], fx: float, fy: float, w: int, h: int) -> int | None:
    c: int | None = None
    for layer in layers:
        kind = layer[0]
        if kind == "h":
            cols = layer[1]
            assert isinstance(cols, list)
            c = cols[min(int(fy * len(cols)), len(cols) - 1)]
        elif kind == "v":
            cols = layer[1]
            assert isinstance(cols, list)
            c = cols[min(int(fx * len(cols)), len(cols) - 1)]
        elif kind == "disc":
            _, col, cx, cy, r = layer
            if ((fx - cx) * (w / h)) ** 2 + (fy - cy) ** 2 <= r**2:  # type: ignore[operator]
                c = col  # type: ignore[assignment]
        elif kind == "canton":
            _, col, cw, ch = layer
            if fx < cw and fy < ch:  # type: ignore[operator]
                c = col  # type: ignore[assignment]
        elif kind in ("cross", "cross_c"):
            _, col, t = layer
            vx = 0.5 if kind == "cross_c" else 0.36
            if abs(fx - vx) * (w / h) < t / 2 or abs(fy - 0.5) < t / 2:  # type: ignore[operator]
                c = col  # type: ignore[assignment]
        elif kind == "tri":
            _, col, tw = layer
            if fx < tw * (1 - abs(fy - 0.5) * 2):  # type: ignore[operator]
                c = col  # type: ignore[assignment]
        elif kind == "tri_br":
            if fx + fy > 1.3:
                c = layer[1]  # type: ignore[assignment]
        elif kind == "diag":
            if abs(fx + fy - 1) < 0.15:
                c = layer[1]  # type: ignore[assignment]
        elif kind == "band":
            _, col, y0, y1 = layer
            if y0 <= fy < y1:  # type: ignore[operator]
                c = col  # type: ignore[assignment]
        elif kind == "band_v":
            _, col, x0, x1 = layer
            if x0 <= fx < x1:  # type: ignore[operator]
                c = col  # type: ignore[assignment]
    return c


def _block(x: int, y: int, scale: int, color: int) -> list[Cell]:
    return [(x + i, y + j, color) for j in range(scale) for i in range(scale)]
