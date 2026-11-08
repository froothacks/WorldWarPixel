#!/usr/bin/env python3
"""Build a hype sizzle reel from WorldWarPixel timelapse videos.

Pulls ``{run_id}.mp4`` + ``{run_id}.json`` sidecars from the GCS timelapse bucket (or a local
dir), scores runs from the sidecar metadata, picks highlight windows by frame-to-frame change,
then renders 1080p shots with ffmpeg (speed ramps, punch-in zooms, burned overlays, title cards)
and a synthesized 128 BPM music loop. No video-gen model is required; DeepSeek/GLM are text-only
and are used, when credentials exist, only to pick the reel's theme/order.

    uv run python scripts/sizzle_reel.py --top 3 --clips-per-run 2 \
        --out /tmp/wwp-sizzle/sizzle-reel.mp4

GCS source uses ``gcloud storage cp``, so run it with gcloud credentials that can read the
bucket (e.g. CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE or `gcloud auth application-default login`).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import random
import re
import subprocess
import sys
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

DEFAULT_SOURCE = "gs://world-war-pixel-timelapses-514101/timelapses/"
DEFAULT_WORKDIR = "/tmp/wwp-sizzle"
DEFAULT_OUT = "/tmp/wwp-sizzle/sizzle-reel.mp4"
FPS = 30
WIDTH, HEIGHT = 1920, 1080
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


@dataclass(slots=True)
class RunPick:
    run_id: str
    meta: dict[str, Any]
    video: Path
    score: float
    rank: int
    windows: list[tuple[float, float, float]] = field(default_factory=list)
    peak_frame: Path | None = None


def sh(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-1200:]}"
        )
    return proc


def gcs_parts(source: str) -> tuple[str, str]:
    src = source.replace("gcs://", "gs://", 1)
    if not src.startswith("gs://"):
        raise ValueError(f"unsupported source {source!r}; use gs://bucket/prefix or a local dir")
    rest = src[len("gs://") :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def fetch(source: str, workdir: Path) -> Path:
    src_dir = workdir / "src"
    if source.startswith("gs://") or source.startswith("gcs://"):
        bucket, prefix = gcs_parts(source)
        src_dir.mkdir(parents=True, exist_ok=True)
        for stale in src_dir.glob("*"):
            if stale.is_file():
                stale.unlink()
        base = f"gs://{bucket}/{prefix}/" if prefix else f"gs://{bucket}/"
        for pattern in ("*.json", "*.mp4"):
            sh(["gcloud", "storage", "cp", f"{base}{pattern}", str(src_dir) + "/"])
        return src_dir
    path = Path(source).expanduser()
    if not path.is_dir():
        raise ValueError(f"source dir does not exist: {path}")
    return path


def _num(meta: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def dead_air(meta: dict[str, Any]) -> int:
    headlines = [str(h) for h in meta.get("narrator_headlines", [])]
    return sum(1 for h in headlines if "0 turns" in h and "0 pixels" in h)


def score_run(meta: dict[str, Any]) -> float:
    pixels = _num(meta, "pixels")
    posts = _num(meta, "posts")
    threads = _num(meta, "threads")
    agents = _num(meta, "agents")
    run_seconds = max(_num(meta, "run_seconds", 60.0), 60.0)
    countries = len(meta.get("top_countries", []) or [])
    headlines = [str(h) for h in meta.get("narrator_headlines", [])]
    action = sum(
        1 for h in headlines if re.search(r"wipe|raid|flatten|turns|mural war|front", h, re.I)
    )
    density = pixels / run_seconds
    dead = dead_air(meta)
    return (
        math.log1p(pixels) * 22.0
        + math.log1p(posts) * 16.0
        + math.log1p(threads) * 10.0
        + math.log1p(agents) * 4.0
        + min(density / 250.0, 40.0)
        + min(countries, 10) * 3.0
        + min(action, 8) * 2.0
        - dead * 18.0
    )


def load_runs(src_dir: Path) -> list[RunPick]:
    runs: list[RunPick] = []
    for sidecar in sorted(src_dir.glob("*.json")):
        try:
            meta = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            continue
        run_id = str(meta.get("run_id") or sidecar.stem)
        video = src_dir / f"{run_id}.mp4"
        if video.name.endswith(".tmp.mp4") or not video.exists():
            continue
        runs.append(RunPick(run_id=run_id, meta=meta, video=video, score=score_run(meta), rank=0))
    return runs


def print_score_table(runs: list[RunPick]) -> None:
    rows = sorted(runs, key=lambda r: r.score, reverse=True)
    print(f"{'run':10} {'score':>8} {'px':>9} {'posts':>6} {'thr':>4} {'agt':>4} {'sec':>7}  model")
    for r in rows[:12]:
        m = r.meta
        cfg = m.get("config", {}) if isinstance(m.get("config"), dict) else {}
        print(
            f"{r.run_id:10} {r.score:8.1f} {_num(m, 'pixels'):9.0f} {_num(m, 'posts'):6.0f} "
            f"{_num(m, 'threads'):4.0f} {_num(m, 'agents'):4.0f} {_num(m, 'run_seconds'):7.0f}  "
            f"{cfg.get('model', '?')}"
        )


async def llm_theme(runs: list[RunPick], model_id: str) -> dict[str, Any] | None:
    """Optional text-model pass: pick a title/tagline and an order for the top runs."""
    try:
        from worldwarpixel.config import get_settings
        from worldwarpixel.models import available_endpoints, find_model
        from worldwarpixel.routing import endpoint_client
    except Exception:
        return None
    spec = find_model(model_id)
    if spec is None:
        return None
    settings = get_settings()
    endpoints = available_endpoints(settings, spec)
    if not endpoints:
        return None
    endpoint = endpoints[0]
    client = endpoint_client(endpoint, settings.secret, timeout=60.0)
    summary = [
        {
            "run_id": r.run_id,
            "score": round(r.score, 1),
            "pixels": int(_num(r.meta, "pixels")),
            "posts": int(_num(r.meta, "posts")),
            "threads": int(_num(r.meta, "threads")),
            "agents": int(_num(r.meta, "agents")),
            "model": (r.meta.get("config") or {}).get("model"),
            "top_countries": [c.get("country") for c in (r.meta.get("top_countries") or [])[:3]],
            "headlines": [str(h) for h in (r.meta.get("narrator_headlines") or [])[-3:]],
        }
        for r in runs[:8]
    ]
    prompt = (
        "You are editing a 30s hype reel for WorldWarPixel, a pixel-art war simulation. "
        "Choose the most cinematic order and write a punchy title + one-line tagline. "
        'Return ONLY JSON: {"title": str, "tagline": str, "order": [run_id, ...]}.\n'
        + json.dumps(summary, ensure_ascii=False)
    )
    try:
        resp = await client.chat.completions.create(
            model=endpoint.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = resp.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except Exception as exc:
        print(f"LLM theme failed ({exc}); using deterministic order", file=sys.stderr)
        return None


def pil_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = list(FONT_CANDIDATES)
    if bold:
        names.insert(0, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def cover(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert("RGB")
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    nw, nh = math.ceil(iw * scale), math.ceil(ih * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines[:3]


def make_text_card(path: Path, title: str, subtitle: str) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), (10, 11, 15))
    draw = ImageDraw.Draw(img)
    # subtle accent bars
    draw.rectangle((0, 0, WIDTH, 10), fill=(90, 140, 255))
    draw.rectangle((0, HEIGHT - 10, WIDTH, HEIGHT), fill=(90, 140, 255))
    title_font = pil_font(96, bold=True)
    sub_font = pil_font(40)
    small_font = pil_font(28)
    tw = title_font.getlength(title)
    draw.text(
        ((WIDTH - tw) / 2, HEIGHT * 0.40),
        title,
        font=title_font,
        fill=(245, 246, 250),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    sw = sub_font.getlength(subtitle)
    draw.text(
        ((WIDTH - sw) / 2, HEIGHT * 0.40 + 130),
        subtitle,
        font=sub_font,
        fill=(180, 186, 198),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    mark = "WORLDWARPPIXEL"
    mw = small_font.getlength(mark)
    draw.text(((WIDTH - mw) / 2, HEIGHT * 0.40 + 210), mark, font=small_font, fill=(90, 140, 255))
    img.save(path)


def make_peak_card(peak: Path, path: Path, title: str, subtitle: str) -> None:
    base = cover(Image.open(peak), WIDTH, HEIGHT)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0, 120))
    base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(base)
    title_font = pil_font(72, bold=True)
    sub_font = pil_font(34)
    for i, line in enumerate(_wrap(title, title_font, WIDTH - 160)):
        tw = title_font.getlength(line)
        draw.text(
            ((WIDTH - tw) / 2, 250 + i * 86),
            line,
            font=title_font,
            fill=(245, 246, 250),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    for i, line in enumerate(_wrap(subtitle, sub_font, WIDTH - 200)):
        sw = sub_font.getlength(line)
        draw.text(
            ((WIDTH - sw) / 2, 520 + i * 46),
            line,
            font=sub_font,
            fill=(190, 196, 208),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    base.save(path)


def _font_path() -> str | None:
    for name in FONT_CANDIDATES:
        if Path(name).exists():
            return name
    return None


def _font_arg() -> str:
    path = _font_path()
    return f"fontfile={path}" if path else "font=DejaVuSans"


def _ellipsis(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def shot_overlays(run: RunPick) -> tuple[str, str]:
    cfg = run.meta.get("config") if isinstance(run.meta.get("config"), dict) else {}
    model = cfg.get("model", "?")
    countries = [
        str(c.get("country")) for c in (run.meta.get("top_countries") or [])[:3] if c.get("country")
    ]
    pixels = int(_num(run.meta, "pixels"))
    posts = int(_num(run.meta, "posts"))
    title = f"{run.run_id} · {model} · {pixels:,} px · {posts:,} posts"
    headlines = [str(h) for h in (run.meta.get("narrator_headlines") or []) if h]
    caption = (
        headlines[-1]
        if headlines
        else ", ".join(countries) or f"{run.meta.get('agents', '?')} agents"
    )
    return _ellipsis(title, 110), _ellipsis(caption, 110)


def render_shot(
    video: Path,
    start: float,
    dur: float,
    out: Path,
    *,
    kind: str,
    run: RunPick,
    speed: float,
    tmp: Path,
) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    title_txt = tmp / "title.txt"
    caption_txt = tmp / "caption.txt"
    title, caption = shot_overlays(run)
    title_txt.write_text(title)
    caption_txt.write_text(caption)
    font = _font_arg()
    if kind == "zoom":
        visual = (
            f"setpts=PTS/{speed},fps={FPS},"
            "crop=iw*0.72:ih*0.72:iw*0.14:ih*0.14,"
            "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
            "setsar=1,format=yuv420p"
        )
    else:
        visual = (
            f"setpts=PTS/{speed},fps={FPS},"
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0f0f12,setsar=1,format=yuv420p"
        )
    vf = (
        visual + f",drawtext={font}:textfile={title_txt}:x=28:y=28:fontsize=30:fontcolor=white:"
        "box=1:boxcolor=black@0.45:boxborderw=10"
        + f",drawtext={font}:textfile={caption_txt}:x=28:y=h-th-28:fontsize=26:fontcolor=white:"
        "box=1:boxcolor=black@0.45:boxborderw=8"
    )
    sh(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(video),
            "-vf",
            vf,
            "-an",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ]
    )


def render_card_video(card: Path, out: Path, seconds: float = 2.6) -> None:
    fade_out = max(0.2, seconds - 0.4)
    vf = f"fps={FPS},format=yuv420p,fade=t=in:st=0:d=0.35,fade=t=out:st={fade_out:.2f}:d=0.35"
    sh(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-t",
            f"{seconds:.2f}",
            "-i",
            str(card),
            "-vf",
            vf,
            "-an",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ]
    )


def _noise(t: float) -> float:
    x = math.sin(t * 12.9898) * 43758.5453
    return (x - math.floor(x)) * 2.0 - 1.0


def synth_music(path: Path, seconds: float = 7.5, bpm: float = 128.0, sr: int = 22050) -> None:
    """Tiny deterministic 4-bar loop: kick/snare/hat/bass/arp. Looped by ffmpeg."""
    n = int(seconds * sr)
    mix = array("f", [0.0]) * n
    beat = 60.0 / bpm
    bar = beat * 4
    roots = [55.0, 43.65, 65.41, 49.0]  # A1 F1 C2 G1
    arp_notes = [220.0, 261.63, 329.63, 440.0, 523.25, 659.25]
    step16 = beat / 4.0
    for i in range(n):
        t = i / sr
        bar_i = int(t // bar) % len(roots)
        beat_i = int(t // beat)
        phase = t % beat
        v = 0.0
        # kick on every beat
        if phase < 0.13:
            env = math.exp(-phase * 28.0)
            freq = 42.0 + 95.0 * math.exp(-phase * 22.0)
            v += math.sin(2 * math.pi * freq * t) * env * 0.85
        # snare on beats 2 and 4
        if beat_i % 4 in (1, 3) and phase < 0.16:
            env = math.exp(-phase * 22.0)
            v += (_noise(t) * 0.55 + math.sin(2 * math.pi * 185.0 * t) * 0.25) * env * 0.5
        # hats on 8ths
        hphase = t % (beat / 2)
        if hphase < 0.035:
            v += _noise(t + 0.13) * math.exp(-hphase * 90.0) * 0.16
        # bass on beats 1 and 3 of each bar
        if beat_i % 2 == 0 and phase < 0.24:
            env = math.exp(-phase * 7.0)
            saw = 2.0 * (t * roots[bar_i] % 1.0) - 1.0
            v += saw * env * 0.34
        # arp 16ths, quiet
        aphase = t % step16
        if aphase < 0.06:
            step = int(t // step16)
            freq = arp_notes[step % len(arp_notes)] * (2.0 if bar_i == 2 else 1.0)
            sq = 1.0 if math.sin(2 * math.pi * freq * t) >= 0 else -1.0
            v += sq * math.exp(-aphase * 60.0) * 0.07
        # soft clip + makeup
        v = math.tanh(v * 1.15) * 0.82
        mix[i] = v
    pcm = array("h", (max(-32768, min(32767, int(v * 32767))) for v in mix))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def concat_videos(paths: list[Path], out: Path) -> None:
    list_file = out.with_suffix(".concat.txt")
    list_file.write_text("".join(f"file '{p}'\n" for p in paths))
    proc = sh(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out),
        ],
        check=False,
    )
    if proc.returncode != 0:
        sh(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-vf",
                f"fps={FPS},format=yuv420p",
                "-r",
                str(FPS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out),
            ]
        )


def mux_with_music(video: Path, music: Path | None, out: Path) -> None:
    if music is None:
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-c", "copy", str(out)])
        return
    sh(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-stream_loop",
            "-1",
            "-i",
            str(music),
            "-shortest",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out),
        ]
    )


def probe_duration(path: Path) -> float | None:
    proc = sh(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(path),
        ],
        check=False,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def frames_dir_for(workdir: Path, run_id: str) -> Path:
    d = workdir / "frames" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def analyze_windows(run: RunPick, workdir: Path, *, count: int, win: int) -> None:
    duration = max(_num(run.meta, "duration_s", 22.5), 1.0)
    frames_dir = frames_dir_for(workdir, run.run_id)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    sh(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(run.video),
            "-vf",
            "fps=1,scale=160:-1",
            str(frames_dir / "f_%04d.png"),
        ]
    )
    frames = sorted(frames_dir.glob("f_*.png"))
    if len(frames) < 2:
        run.windows = [(0.0, duration, 0.0)]
        run.peak_frame = frames[0] if frames else None
        return
    imgs = [Image.open(p).convert("L") for p in frames]
    diffs = [0.0]
    for a, b in itertools.pairwise(imgs):
        diffs.append(ImageStat.Stat(ImageChops.difference(a, b)).mean[0])
    n = len(diffs)
    win = max(2, min(win, n))
    scored = [(sum(diffs[s : s + win]), s) for s in range(0, n - win + 1)]
    scored.sort(reverse=True)
    chosen: list[tuple[float, int]] = []
    for score, start in scored:
        if all(abs(start - prev) >= win for _, prev in chosen):
            chosen.append((score, start))
        if len(chosen) >= count:
            break
    chosen.sort(key=lambda x: x[1])
    windows: list[tuple[float, float, float]] = []
    best_score, best_start = -1.0, 0
    for score, start in chosen:
        windows.append((float(start), min(duration, float(start + win)), float(score)))
        if score > best_score:
            best_score, best_start = score, start
    run.windows = windows
    mid = min(best_start + win // 2, len(frames) - 1)
    run.peak_frame = frames[mid]


def subtitle_for(run: RunPick) -> str:
    cfg = run.meta.get("config") if isinstance(run.meta.get("config"), dict) else {}
    countries = [
        str(c.get("country")) for c in (run.meta.get("top_countries") or [])[:3] if c.get("country")
    ]
    bits = [
        f"{int(_num(run.meta, 'pixels')):,} px",
        f"{int(_num(run.meta, 'posts')):,} posts",
        f"{int(_num(run.meta, 'threads')):,} threads",
        f"{int(_num(run.meta, 'agents'))} agents",
        str(cfg.get("model", "?")),
    ]
    if countries:
        bits.append(" · ".join(countries))
    return " · ".join(bits)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", default=DEFAULT_SOURCE, help="gs://bucket/prefix or local dir")
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--top", type=int, default=3, help="number of runs to include")
    p.add_argument("--clips-per-run", type=int, default=2)
    p.add_argument("--window", type=int, default=5, help="highlight window length in seconds")
    p.add_argument("--speed", type=float, default=1.5, help="ffmpeg setpts divisor (>1 = faster)")
    p.add_argument("--card-seconds", type=float, default=2.6)
    p.add_argument("--music", default="", help="optional music file; default synthesizes a loop")
    p.add_argument("--no-music", action="store_true")
    p.add_argument(
        "--llm-model", default="", help="optional model id for theme/order (needs creds)"
    )
    p.add_argument(
        "--upload", default="", help="optional gs://bucket/prefix to upload reel+manifest"
    )
    p.add_argument("--seed", type=int, default=7)
    return p


async def _amain(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    workdir = Path(args.workdir).expanduser()
    out = Path(args.out).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"fetching {args.source} -> {workdir / 'src'}")
    src_dir = fetch(args.source, workdir)
    runs = load_runs(src_dir)
    if not runs:
        print("no rendered timelapses found", file=sys.stderr)
        return 1
    print_score_table(runs)

    picked = sorted(runs, key=lambda r: r.score, reverse=True)[: max(1, args.top)]
    for i, r in enumerate(picked, 1):
        r.rank = i

    theme: dict[str, Any] | None = None
    if args.llm_model:
        theme = await llm_theme(picked, args.llm_model)
        if theme:
            print(f"LLM theme: {theme.get('title')!r} — {theme.get('tagline')!r}")
            order = theme.get("order")
            if isinstance(order, list):
                by_id = {r.run_id: r for r in picked}
                ordered_ids: list[str] = []
                for x in order:
                    rid = str(x)
                    if rid in by_id and rid not in ordered_ids:
                        ordered_ids.append(rid)
                ordered_ids.extend(r.run_id for r in picked if r.run_id not in ordered_ids)
                picked = [by_id[rid] for rid in ordered_ids[: args.top]]
                for i, r in enumerate(picked, 1):
                    r.rank = i

    print(f"analyzing {len(picked)} runs x {args.clips_per_run} clips")
    for r in picked:
        analyze_windows(r, workdir, count=args.clips_per_run, win=args.window)
        peak = r.peak_frame.name if r.peak_frame else "-"
        print(f"  {r.run_id}: {len(r.windows)} windows, peak={peak}")

    shots_dir = workdir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    for old in shots_dir.glob("*.mp4"):
        old.unlink()
    shot_paths: list[Path] = []

    intro_title = str((theme or {}).get("title") or "WORLDWARPPIXEL")
    intro_sub = str((theme or {}).get("tagline") or "top runs · pixel wars · narrator chaos")
    intro_card = workdir / "intro.png"
    make_text_card(intro_card, intro_title, intro_sub)
    intro_mp4 = shots_dir / "000_intro.mp4"
    render_card_video(intro_card, intro_mp4, args.card_seconds)
    shot_paths.append(intro_mp4)

    for r in picked:
        subtitle = subtitle_for(r)
        card_png = workdir / f"card_{r.run_id}.png"
        card_mp4 = shots_dir / f"{len(shot_paths):03d}_card_{r.run_id}.mp4"
        if r.peak_frame is not None:
            make_peak_card(r.peak_frame, card_png, f"#{r.rank}  {r.run_id}", subtitle)
        else:
            make_text_card(card_png, f"#{r.rank}  {r.run_id}", subtitle)
        render_card_video(card_png, card_mp4, args.card_seconds)
        shot_paths.append(card_mp4)
        for ci, (start, end, _score) in enumerate(r.windows):
            dur = max(1.0, end - start)
            kind = "zoom" if ci % 2 == 1 else "normal"
            shot = shots_dir / f"{len(shot_paths):03d}_{r.run_id}_{ci}_{kind}.mp4"
            render_shot(
                r.video,
                start,
                dur,
                shot,
                kind=kind,
                run=r,
                speed=args.speed,
                tmp=workdir / "shot_tmp" / r.run_id / str(ci),
            )
            shot_paths.append(shot)

    concat = workdir / "concat.mp4"
    print(f"concatenating {len(shot_paths)} shots")
    concat_videos(shot_paths, concat)

    music: Path | None = None
    if not args.no_music:
        music = Path(args.music).expanduser() if args.music else workdir / "music.wav"
        if not args.music:
            synth_music(music)
        print(f"muxing music: {music}")
    mux_with_music(concat, music, out)
    duration = probe_duration(out)

    manifest = {
        "output": str(out),
        "duration_s": duration,
        "source": args.source,
        "theme": theme,
        "runs": [
            {
                "rank": r.rank,
                "run_id": r.run_id,
                "score": round(r.score, 2),
                "windows": [[round(a, 2), round(b, 2), round(s, 2)] for a, b, s in r.windows],
                "peak_frame": str(r.peak_frame) if r.peak_frame else None,
                "meta": {
                    "pixels": _num(r.meta, "pixels"),
                    "posts": _num(r.meta, "posts"),
                    "threads": _num(r.meta, "threads"),
                    "agents": _num(r.meta, "agents"),
                    "run_seconds": _num(r.meta, "run_seconds"),
                    "model": (r.meta.get("config") or {}).get("model"),
                },
            }
            for r in picked
        ],
        "shots": [str(p) for p in shot_paths],
    }
    manifest_path = out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} ({duration and round(duration, 2)}s) + {manifest_path}")

    if args.upload:
        prefix = args.upload if args.upload.endswith("/") else args.upload + "/"
        sh(["gcloud", "storage", "cp", str(out), str(manifest_path), prefix])
        print(f"uploaded to {prefix}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
