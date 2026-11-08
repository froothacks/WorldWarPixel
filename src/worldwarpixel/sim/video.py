"""Render a finished (or live) run from its event log into an mp4: the canvas timelapse on the
left and the message board on the right, both advancing on the same clock, with the latest
narrator headline as a caption. Frames are composed with Pillow and piped to ffmpeg as raw RGB.

Two passes over the log keep memory at O(canvas + posts): the first collects the roster, posts,
narrator headlines and the time span; the second streams pixel events while frames are emitted.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from itertools import takewhile
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageFont

from worldwarpixel.sim.canvas import PALETTE
from worldwarpixel.sim.events import EventStore
from worldwarpixel.sim.timelapse import iter_events

FPS = 30
MAX_FRAMES = 1800
CANVAS_PX = 720
"""Height (and max width) of the canvas pane; the canvas is upscaled by an integer factor."""
BOARD_W = 480
HEADER_H = 48
CAPTION_H = 44
PAD = 16
HOLD_START_S = 0.5
HOLD_END_S = 2.0
MAX_BOARD_POSTS = 60

BG = (14, 15, 18)
PANEL = (22, 24, 29)
FG = (236, 237, 240)
MUTED = (140, 146, 158)
ACCENT = (90, 140, 255)
RGB_PALETTE = tuple(tuple(int(h[i : i + 2], 16) for i in (1, 3, 5)) for h in PALETTE)
FLAT_PALETTE = [c for rgb in RGB_PALETTE for c in rgb] + [0] * (768 - 3 * len(RGB_PALETTE))


class VideoError(RuntimeError):
    pass


@dataclass(slots=True)
class Timeline:
    """Everything but the pixels, collected in the first pass."""

    config: dict[str, Any]
    agents: dict[int, dict[str, Any]]
    posts: list[dict[str, Any]] = field(default_factory=list)
    board_clears: list[float] = field(default_factory=list)
    headlines: list[tuple[float, str]] = field(default_factory=list)
    t0: float | None = None
    t1: float | None = None
    last_id: int = 0
    pixels: int = 0
    pixels_by_country: dict[str, int] = field(default_factory=dict)


def load_timeline(log: EventStore, run_id: str) -> Timeline | None:
    events = iter_events(log, run_id)
    first = next(iter(events), None)
    if first is None or first.kind != "run.created":
        return None
    created = cast(dict[str, Any], first.payload)
    tl = Timeline(
        config=dict(created.get("config", {})),
        agents={int(a["id"]): a for a in created.get("agents", [])},
    )
    for e in events:
        p = cast(dict[str, Any], e.payload)
        if e.kind in ("pixel", "post", "god.clear_canvas", "god.clear_board"):
            tl.t0 = e.ts if tl.t0 is None else tl.t0
            tl.t1, tl.last_id = e.ts, e.id
        if e.kind == "pixel":
            tl.pixels += 1
            agent = tl.agents.get(e.agent_id or 0)
            if agent:
                country = str(agent.get("country", "?"))
                tl.pixels_by_country[country] = tl.pixels_by_country.get(country, 0) + 1
        elif e.kind == "post":
            tl.posts.append({**p, "ts": e.ts})
        elif e.kind == "god.clear_board":
            tl.board_clears.append(e.ts)
        elif e.kind == "narrator.update" and p.get("kind") != "answer" and p.get("headline"):
            tl.headlines.append((e.ts, str(p["headline"])))
            if tl.t0 is not None:
                tl.t1 = max(tl.t1 or e.ts, e.ts)
        elif e.kind == "run.deleted":
            return None
    return tl


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class FrameComposer:
    """Draws one video frame from the canvas state and the board/narrator state at a time."""

    def __init__(self, tl: Timeline, width: int, height: int) -> None:
        self.tl = tl
        self.cw, self.ch = width, height
        self.scale = max(1, min(CANVAS_PX // max(width, 1), CANVAS_PX // max(height, 1)))
        self.canvas_w, self.canvas_h = width * self.scale, height * self.scale
        self.w = PAD + self.canvas_w + PAD + BOARD_W + PAD
        self.h = HEADER_H + max(self.canvas_h, 360) + CAPTION_H + PAD
        # ffmpeg's yuv420p needs even dimensions.
        self.w += self.w % 2
        self.h += self.h % 2
        self.font = _font(15)
        self.font_small = _font(13)
        self.font_title = _font(16)
        self.font_header = _font(20)
        self.canvas_top = HEADER_H + (self.h - HEADER_H - CAPTION_H - PAD - self.canvas_h) // 2
        self.board_x = PAD + self.canvas_w + PAD
        self.board_top = HEADER_H
        self.board_h = self.h - HEADER_H - CAPTION_H - PAD

    @property
    def size(self) -> tuple[int, int]:
        return self.w, self.h

    def agent_label(self, agent_id: int) -> str:
        a = self.tl.agents.get(agent_id)
        if not a:
            return f"agent {agent_id}"
        country = a.get("country")
        return f"{a.get('name', agent_id)} · {country}" if country else str(a.get("name"))

    def compose(
        self, state: bytes, ts: float, pixels: int, posts: list[dict[str, Any]], headline: str
    ) -> bytes:
        img = Image.new("RGB", self.size, BG)
        draw = ImageDraw.Draw(img)
        # canvas
        canvas = Image.frombytes("P", (self.cw, self.ch), state)
        canvas.putpalette(FLAT_PALETTE)
        canvas = canvas.convert("RGB")
        if self.scale > 1:
            canvas = canvas.resize((self.canvas_w, self.canvas_h), Image.Resampling.NEAREST)
        img.paste(canvas, (PAD, self.canvas_top))
        draw.rectangle(
            (PAD - 1, self.canvas_top - 1, PAD + self.canvas_w, self.canvas_top + self.canvas_h),
            outline=(60, 64, 72),
        )
        # header
        elapsed = _fmt_elapsed(ts - (self.tl.t0 or ts))
        draw.text((PAD, 13), "WorldWarPixel", fill=FG, font=self.font_header)
        stats = (
            f"{self.cw}x{self.ch} · {len(self.tl.agents)} agents · {pixels:,} px"
            f" · {len(posts):,} posts · {elapsed}"
        )
        draw.text((PAD + 170, 17), stats, fill=MUTED, font=self.font)
        # board
        self._draw_board(draw, posts)
        # caption
        if headline:
            y = self.h - CAPTION_H - PAD + 8
            draw.rectangle((PAD, y - 6, self.w - PAD, y + 26), fill=PANEL)
            draw.text((PAD + 10, y), "Narrator", fill=ACCENT, font=self.font_small)
            draw.text((PAD + 80, y - 1), _ellipsis(headline, 95), fill=FG, font=self.font)
        return img.tobytes()

    def _draw_board(self, draw: ImageDraw.ImageDraw, posts: list[dict[str, Any]]) -> None:
        x, y = self.board_x, self.board_top
        draw.rectangle((x, y, x + BOARD_W, y + self.board_h), fill=PANEL)
        draw.text((x + 12, y + 10), "Message board", fill=FG, font=self.font_title)
        draw.text((x + 150, y + 13), f"{len(posts)} posts", fill=MUTED, font=self.font_small)
        y += 40
        bottom = self.board_top + self.board_h - 8
        for p in reversed(posts[-MAX_BOARD_POSTS:]):
            is_reply = p.get("title") is None
            title = str(p.get("title") or "")
            body = str(p.get("body", "")).replace("\n", " ")
            author = self.agent_label(int(p.get("author_id", 0)))
            lines = textwrap.wrap(body, 56)[: 1 if is_reply else 2]
            block_h = 18 + 17 * len(lines) + (20 if title else 0) + 8
            if y + block_h > bottom:
                break
            if title:
                draw.text((x + 12, y), _ellipsis(title, 48), fill=FG, font=self.font_title)
                y += 20
            prefix = "↳ " if is_reply else ""
            draw.text((x + 12, y), f"{prefix}{author}", fill=ACCENT, font=self.font_small)
            y += 18
            for line in lines:
                draw.text((x + 12, y), line, fill=MUTED, font=self.font_small)
                y += 17
            y += 8
            draw.line((x + 12, y - 4, x + BOARD_W - 12, y - 4), fill=(40, 44, 52))


def _ellipsis(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def ffmpeg_available(ffmpeg: str = "ffmpeg") -> bool:
    return shutil.which(ffmpeg) is not None


def render_video(
    log: EventStore,
    run_id: str,
    out: Path,
    *,
    duration: float = 20.0,
    fps: int = FPS,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    """Write ``out`` (mp4) and return its metadata; VideoError if there is nothing to show."""
    tl = load_timeline(log, run_id)
    if tl is None:
        raise VideoError(f"run {run_id} has no event log")
    if tl.t0 is None or tl.t1 is None:
        raise VideoError(f"run {run_id} has no canvas or board activity")
    if not ffmpeg_available(ffmpeg):
        raise VideoError(f"{ffmpeg} not found")
    width, height = int(tl.config["width"]), int(tl.config["height"])
    frames = max(1, min(int(duration * fps), MAX_FRAMES))
    composer = FrameComposer(tl, width, height)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{composer.w}x{composer.h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(tmp),
    ]  # fmt: skip
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    t0, t1 = tl.t0, tl.t1
    step = (t1 - t0) / frames if t1 > t0 else 0.0
    state = bytearray(width * height)
    pixels_so_far = 0
    relevant = (
        e
        for e in iter_events(log, run_id, canvas_only=True)
        if e.kind in ("pixel", "god.clear_canvas")
    )
    events = takewhile(lambda e: e.id <= tl.last_id, relevant)
    nxt = next(events, None)
    posts_sorted = sorted(tl.posts, key=lambda p: p["ts"])
    headlines = tl.headlines
    pi = hi = 0
    visible: list[dict[str, Any]] = []
    clears = list(tl.board_clears)
    headline = ""
    try:
        for f in range(frames):
            cutoff = t1 if f == frames - 1 else t0 + step * (f + 1)
            while nxt is not None and (nxt.ts <= cutoff or f == frames - 1):
                e = nxt
                nxt = next(events, None)
                if e.kind == "god.clear_canvas":
                    state = bytearray(width * height)
                    continue
                p = e.payload
                x, y, c = int(p["x"]), int(p["y"]), int(p["color"])  # type: ignore[call-overload]
                if 0 <= x < width and 0 <= y < height and c < len(PALETTE):
                    state[y * width + x] = c
                    pixels_so_far += 1
            while pi < len(posts_sorted) and posts_sorted[pi]["ts"] <= cutoff:
                visible.append(posts_sorted[pi])
                pi += 1
            while clears and clears[0] <= cutoff:
                cleared_at = clears.pop(0)
                visible = [p for p in visible if p["ts"] > cleared_at]
            while hi < len(headlines) and headlines[hi][0] <= cutoff:
                headline = headlines[hi][1]
                hi += 1
            frame = composer.compose(bytes(state), cutoff, pixels_so_far, visible, headline)
            repeat = 1
            if f == 0:
                repeat += int(HOLD_START_S * fps)
            if f == frames - 1:
                repeat += int(HOLD_END_S * fps)
            for _ in range(repeat):
                proc.stdin.write(frame)
        proc.stdin.close()
        code = proc.wait()
    except BrokenPipeError as e:
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise VideoError(f"ffmpeg stopped early: {_stderr(proc)}") from e
    except BaseException:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    if code != 0:
        tmp.unlink(missing_ok=True)
        raise VideoError(f"ffmpeg exited with {code}: {_stderr(proc)}")
    tmp.replace(out)
    total_frames = frames + int(HOLD_START_S * fps) + int(HOLD_END_S * fps)
    countries = sorted(tl.pixels_by_country.items(), key=lambda kv: -kv[1])[:10]
    return {
        "run_id": run_id,
        "file": out.name,
        "bytes": out.stat().st_size,
        "width": composer.w,
        "height": composer.h,
        "fps": fps,
        "frames": total_frames,
        "duration_s": round(total_frames / fps, 2),
        "run_seconds": round(t1 - t0, 1),
        "started_at": t0,
        "ended_at": t1,
        "rendered_at": time.time(),
        "render_seconds": round(time.monotonic() - started, 1),
        "config": tl.config,
        "agents": len(tl.agents),
        "pixels": tl.pixels,
        "posts": len(tl.posts),
        "threads": sum(1 for p in tl.posts if p.get("title")),
        "top_countries": [{"country": c, "pixels": n} for c, n in countries],
        "narrator_headlines": [h for _, h in tl.headlines][-20:],
    }


def _stderr(proc: subprocess.Popen[bytes]) -> str:
    return (proc.stderr.read().decode(errors="replace").strip() if proc.stderr else "")[-500:]
