"""Narrator: periodic high-level commentary on what the agents are painting and saying."""

import asyncio
import base64
import contextlib
import json
import logging
import math
import re
import struct
import time
import zlib
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from worldwarpixel.sim.canvas import PALETTE, Canvas

if TYPE_CHECKING:
    from worldwarpixel.sim.run import Run

log = logging.getLogger(__name__)

UpdateKind = Literal["update", "answer"]

MAX_HISTORY = 50
MAX_IMAGE_SIDE = 128
MAX_NEW_POSTS = 200
GRID = 4
"""The canvas is split into GRID x GRID regions for the change statistics."""
TOP_N = 5
PROMPT_POSTS = 20
BODY_CHARS = 200

SYSTEM_PROMPT = (
    "You are the narrator of WorldWarPixel, an r/place-style world where many AI agents from "
    "different countries paint one shared pixel canvas and talk on a public message board. "
    "You watch everything and brief the human audience: what the conversation is about, who is "
    "painting what and where, alliances, conflicts and trends. Be concrete and vivid but short; "
    "name agents, countries and threads from the digest, never invent any.\n\n"
    "Reply with JSON only, in this shape:\n"
    '{"headline": "one line", "bullets": ["2 to 4 short bullets"], '
    '"highlights": [{"type": "thread", "id": 4, "label": "short label"}, '
    '{"type": "agent", "id": 12, "label": "short label"}, '
    '{"type": "region", "x": 0, "y": 0, "w": 64, "h": 64, "label": "short label"}]}\n'
    "Highlights are links for the viewer: only use thread ids, agent ids and regions that appear "
    "in the digest."
)


class NarratorModel(Protocol):
    """The one call the narrator needs from an LLM; tests swap in a fake."""

    async def complete(self, messages: list[ChatCompletionMessageParam]) -> str: ...


class OpenAINarratorModel:
    def __init__(self, client: AsyncOpenAI, model: str, timeout: float = 60.0) -> None:
        self._client = client
        self.model = model
        self._timeout = timeout

    async def complete(self, messages: list[ChatCompletionMessageParam]) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=700,
            temperature=0.4,
            timeout=self._timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content or ""


class NarratorUnavailable(Exception):
    """The narrator could not get an answer from its model."""


@dataclass(slots=True)
class Window:
    """Everything that happened since the previous update."""

    started_at: float
    turns: Counter[int] = field(default_factory=Counter)
    pixels: Counter[int] = field(default_factory=Counter)
    posts: Counter[int] = field(default_factory=Counter)
    thread_posts: Counter[int] = field(default_factory=Counter)
    regions: Counter[tuple[int, int]] = field(default_factory=Counter)
    overwrites: Counter[tuple[int, int]] = field(default_factory=Counter)
    claimed_blank: int = 0
    new_posts: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=MAX_NEW_POSTS))


def canvas_png(canvas: Canvas, max_side: int = MAX_IMAGE_SIDE) -> bytes:
    """The canvas as an RGB PNG, point-sampled down so neither side exceeds `max_side`."""
    step = max(1, math.ceil(max(canvas.width, canvas.height) / max_side))
    rgb = [bytes.fromhex(c[1:]) for c in PALETTE]
    raw = bytearray()
    for y in range(0, canvas.height, step):
        raw.append(0)  # PNG filter type: none
        row = y * canvas.width
        for x in range(0, canvas.width, step):
            raw += rgb[canvas.pixels[row + x]]
    out_w = math.ceil(canvas.width / step)
    out_h = math.ceil(canvas.height / step)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", out_w, out_h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(text[start : end + 1])
    return None


def parse_reply(text: str) -> tuple[str, list[str], list[Any]]:
    """(headline, bullets, raw highlights) from the model's reply; tolerates plain text."""
    data = _extract_json(text)
    if isinstance(data, dict):
        headline = str(data.get("headline") or "").strip()
        raw_bullets = data.get("bullets")
        bullets = (
            [str(b).strip() for b in raw_bullets if str(b).strip()]
            if isinstance(raw_bullets, list)
            else []
        )
        highlights = data.get("highlights")
        if headline:
            return headline, bullets[:4], highlights if isinstance(highlights, list) else []
    salvaged = _salvage_json(text)
    if salvaged is not None:
        return salvaged
    lines = [ln.strip(" -*•\t") for ln in text.splitlines() if ln.strip(" -*•\t")]
    if not lines:
        return "(no response)", [], []
    return lines[0], lines[1:5], []


_JSON_STRING = r'"((?:[^"\\]|\\.)*)"'


def _salvage_json(text: str) -> tuple[str, list[str], list[Any]] | None:
    """Pull headline/bullets out of JSON the model truncated or mangled."""
    m = re.search(r'"headline"\s*:\s*' + _JSON_STRING, text)
    if not m:
        return None
    headline = _unescape(m.group(1))
    bullets: list[str] = []
    bm = re.search(r'"bullets"\s*:\s*\[(.*?)(?:\]|$)', text, re.S)
    if bm:
        for raw in re.findall(_JSON_STRING, bm.group(1)):
            bullet = _unescape(raw).strip()
            if bullet and bullet not in bullets:
                bullets.append(bullet)
    return headline, bullets[:4], []


def _unescape(raw: str) -> str:
    try:
        return str(json.loads(f'"{raw}"'))
    except json.JSONDecodeError:
        return raw


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class Narrator:
    """Watches one run and every `interval` seconds asks a model for a digest of what happened."""

    def __init__(
        self,
        run: "Run",
        model: NarratorModel | None,
        running: asyncio.Event,
        interval: float = 60.0,
        include_image: bool = True,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._run = run
        self._model = model
        self._running = running
        self.interval = interval
        self.include_image = include_image
        self._clock = clock
        self._sleep = sleep
        self.history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.stats = {"updates": 0, "answers": 0, "failures": 0}
        self.next_update_at: float | None = None
        self._window = Window(clock())
        self._next_id = 1
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    # -- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._window = Window(self._clock())
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.next_update_at = None

    async def _loop(self) -> None:
        while True:
            self.next_update_at = self._clock() + self.interval
            await self._sleep(self.interval)
            await self._running.wait()
            try:
                await self.tick()
            except Exception:  # keep narrating even if one update blows up
                log.exception("narrator tick failed")

    # -- observation -----------------------------------------------------------------------

    def observe_turn(self, agent_id: int) -> None:
        self._window.turns[agent_id] += 1

    def observe_write(self, agent_id: int, kind: str, payload: dict[str, Any]) -> None:
        w = self._window
        if kind == "pixel":
            w.pixels[agent_id] += 1
            canvas = self._run.canvas
            rx = int(payload["x"]) * GRID // canvas.width
            ry = int(payload["y"]) * GRID // canvas.height
            w.regions[(rx, ry)] += 1
            prev = int(payload.get("prev_owner") or 0)
            if prev == 0:
                w.claimed_blank += 1
            elif prev != agent_id:
                w.overwrites[(agent_id, prev)] += 1
        elif kind == "post":
            w.posts[agent_id] += 1
            w.thread_posts[int(payload["thread_id"])] += 1
            w.new_posts.append(payload)

    # -- digest ----------------------------------------------------------------------------

    def _region_box(self, rx: int, ry: int) -> dict[str, int]:
        canvas = self._run.canvas
        cw, ch = math.ceil(canvas.width / GRID), math.ceil(canvas.height / GRID)
        x, y = rx * cw, ry * ch
        return {"x": x, "y": y, "w": min(cw, canvas.width - x), "h": min(ch, canvas.height - y)}

    def build_digest(self) -> dict[str, Any]:
        """Structured summary of the window; also what gets rendered into the prompt."""
        run, w = self._run, self._window
        by_id = run.by_id
        now = self._clock()

        countries: Counter[str] = Counter()
        for aid, n in w.pixels.items():
            if aid in by_id:
                countries[by_id[aid].country] += n
        active = Counter({aid: w.pixels[aid] + 10 * w.posts[aid] for aid in w.turns})
        top_agents = [
            {
                "id": aid,
                "name": by_id[aid].name,
                "country": by_id[aid].country,
                "turns": w.turns[aid],
                "pixels": w.pixels[aid],
                "posts": w.posts[aid],
            }
            for aid, _ in active.most_common(TOP_N)
            if aid in by_id
        ]
        threads = []
        for tid, n in w.thread_posts.most_common(TOP_N):
            posts = run.board.thread(tid)
            if posts:
                threads.append(
                    {
                        "thread_id": tid,
                        "title": posts[0].title,
                        "author": run.agent_name(posts[0].author_id),
                        "new_posts": n,
                        "replies": len(posts) - 1,
                    }
                )
        new_posts = []
        for p in list(w.new_posts)[-PROMPT_POSTS:]:
            root = run.board.get(int(p["thread_id"]))
            body = str(p["body"])
            new_posts.append(
                {
                    "thread_id": p["thread_id"],
                    "title": root.title if root else None,
                    "author": run.agent_name(int(p["author_id"])),
                    "body": body[:BODY_CHARS] + ("…" if len(body) > BODY_CHARS else ""),
                }
            )
        regions = [
            {**self._region_box(rx, ry), "changes": n}
            for (rx, ry), n in w.regions.most_common(TOP_N)
        ]
        overwrites = [
            {
                "by": aid,
                "by_name": run.agent_name(aid),
                "over": victim,
                "over_name": run.agent_name(victim),
                "pixels": n,
            }
            for (aid, victim), n in w.overwrites.most_common(TOP_N)
        ]
        return {
            "run_id": run.id,
            "state": run.state,
            "canvas": {"width": run.canvas.width, "height": run.canvas.height},
            "agents_total": len(run.agents),
            "countries_total": len({a.country for a in run.agents}),
            "threads_total": len(run.board.threads()),
            "totals": dict(run.stats),
            "window_seconds": round(now - w.started_at, 1),
            "turns": sum(w.turns.values()),
            "pixels": sum(w.pixels.values()),
            "posts": sum(w.posts.values()),
            "claimed_blank": w.claimed_blank,
            "top_agents": top_agents,
            "countries": [{"country": c, "pixels": n} for c, n in countries.most_common(TOP_N)],
            "active_threads": threads,
            "new_posts": new_posts,
            "regions": regions,
            "overwrites": overwrites,
            "previous_headlines": [u["headline"] for u in list(self.history)[-3:]],
        }

    @staticmethod
    def render_digest(d: dict[str, Any]) -> str:
        c, t = d["canvas"], d["totals"]
        lines = [
            f"Run {d['run_id']} ({d['state']}): {c['width']}x{c['height']} canvas, "
            f"{d['agents_total']} agents from {d['countries_total']} countries, "
            f"{d['threads_total']} threads. Totals: {t['turns']} turns, {t['pixels']} pixels, "
            f"{t['posts']} posts.",
            f"Last {d['window_seconds']}s: {d['turns']} turns, {d['pixels']} pixels "
            f"({d['claimed_blank']} on blank canvas), {d['posts']} posts.",
        ]
        if d["top_agents"]:
            lines.append("Most active agents:")
            lines += [
                f"- agent {a['id']} {a['name']} ({a['country']}): {a['pixels']} px, "
                f"{a['posts']} posts, {a['turns']} turns"
                for a in d["top_agents"]
            ]
        if d["countries"]:
            lines.append(
                "Pixels by country: "
                + ", ".join(f"{x['country']} {x['pixels']}" for x in d["countries"])
            )
        if d["regions"]:
            lines.append("Canvas regions with most change (x, y, w, h):")
            lines += [
                f"- region ({r['x']}, {r['y']}, {r['w']}, {r['h']}): {r['changes']} px"
                for r in d["regions"]
            ]
        if d["overwrites"]:
            lines.append("Overwrites:")
            lines += [
                f"- {o['by_name']} (agent {o['by']}) painted over {o['over_name']} "
                f"(agent {o['over']}): {o['pixels']} px"
                for o in d["overwrites"]
            ]
        if d["active_threads"]:
            lines.append("Most active threads:")
            lines += [
                f'- thread {x["thread_id"]} "{x["title"]}" by {x["author"]}: '
                f"{x['new_posts']} new posts, {x['replies']} replies total"
                for x in d["active_threads"]
            ]
        if d["new_posts"]:
            lines.append("New posts (oldest first):")
            lines += [
                f"- [thread {p['thread_id']}] {p['author']}: {p['body']}" for p in d["new_posts"]
            ]
        else:
            lines.append("No new posts.")
        if d["previous_headlines"]:
            lines.append("Your previous headlines: " + " | ".join(d["previous_headlines"]))
        return "\n".join(lines)

    # -- composing updates -----------------------------------------------------------------

    def _messages(self, digest: dict[str, Any], question: str | None) -> list[Any]:
        text = self.render_digest(digest)
        if question:
            text += (
                f"\n\nA viewer asks: {question}\nAnswer them from the digest, in the same JSON "
                "shape: headline is the direct answer in one line, bullets add detail."
            )
        else:
            text += "\n\nWrite the next update."
        content: Any = text
        if self.include_image:
            png = base64.b64encode(canvas_png(self._run.canvas)).decode()
            content = [
                {"type": "text", "text": text + "\nThe image is the current canvas, scaled down."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
            ]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def clean_highlights(self, raw: list[Any]) -> list[dict[str, Any]]:
        """Keep only highlights that point at real threads, agents and on-canvas regions."""
        run = self._run
        out: list[dict[str, Any]] = []
        for h in raw[:8]:
            if not isinstance(h, dict):
                continue
            label = str(h.get("label") or "")[:80]
            match h.get("type"):
                case "thread":
                    tid = _int(h.get("id"))
                    posts = run.board.thread(tid) if tid is not None else []
                    if not posts:
                        continue
                    out.append({"type": "thread", "id": tid, "label": label or posts[0].title})
                case "agent":
                    aid = _int(h.get("id"))
                    agent = run.by_id.get(aid) if aid is not None else None
                    if agent is None:
                        continue
                    label = label or f"{agent.name} ({agent.country})"
                    out.append({"type": "agent", "id": aid, "label": label})
                case "region":
                    box = [_int(h.get(k)) for k in ("x", "y", "w", "h")]
                    if any(v is None for v in box):
                        continue
                    x, y, w, hh = (v or 0 for v in box)
                    x, y = max(0, x), max(0, y)
                    w, hh = min(w, run.canvas.width - x), min(hh, run.canvas.height - y)
                    if w <= 0 or hh <= 0:
                        continue
                    label = label or f"({x}, {y}) {w}x{hh}"
                    out.append({"type": "region", "x": x, "y": y, "w": w, "h": hh, "label": label})
        return out

    def fallback_update(self, digest: dict[str, Any]) -> tuple[str, list[str], list[Any]]:
        """Stats-only update used when the model is missing or fails."""
        headline = (
            f"{digest['turns']} turns, {digest['pixels']} pixels and {digest['posts']} posts "
            f"in the last {round(digest['window_seconds'])}s"
        )
        bullets: list[str] = []
        highlights: list[Any] = []
        if digest["top_agents"]:
            a = digest["top_agents"][0]
            bullets.append(f"Most active: {a['name']} ({a['country']}), {a['pixels']} px.")
            highlights.append({"type": "agent", "id": a["id"]})
        if digest["regions"]:
            r = digest["regions"][0]
            bullets.append(f"Hottest region: ({r['x']}, {r['y']}), {r['changes']} px changed.")
            highlights.append({"type": "region", **{k: r[k] for k in ("x", "y", "w", "h")}})
        if digest["active_threads"]:
            t = digest["active_threads"][0]
            bullets.append(f'Busiest thread: "{t["title"]}", {t["new_posts"]} new posts.')
            highlights.append({"type": "thread", "id": t["thread_id"]})
        if digest["overwrites"]:
            o = digest["overwrites"][0]
            bullets.append(f"{o['by_name']} painted over {o['over_name']} ({o['pixels']} px).")
        return headline, bullets, highlights

    async def _compose(
        self, digest: dict[str, Any], kind: UpdateKind, question: str | None
    ) -> dict[str, Any]:
        source = "model"
        error: str | None = None
        if self._model is None:
            source, error = "fallback", "narrator model not configured"
            headline, bullets, raw = self.fallback_update(digest)
        else:
            try:
                reply = await self._model.complete(
                    cast(list[ChatCompletionMessageParam], self._messages(digest, question))
                )
                headline, bullets, raw = parse_reply(reply)
            except Exception as e:  # network / API errors are not fatal for the run
                self.stats["failures"] += 1
                log.warning("narrator model call failed: %s", e)
                if kind == "answer":
                    raise NarratorUnavailable(f"{type(e).__name__}: {e}") from e
                source, error = "fallback", f"model call failed: {type(e).__name__}"
                headline, bullets, raw = self.fallback_update(digest)
        update = {
            "id": self._next_id,
            "ts": self._clock(),
            "kind": kind,
            "question": question,
            "headline": headline,
            "bullets": bullets,
            "highlights": self.clean_highlights(raw),
            "source": source,
            "error": error,
            "window": {
                "seconds": digest["window_seconds"],
                "turns": digest["turns"],
                "pixels": digest["pixels"],
                "posts": digest["posts"],
            },
        }
        self._next_id += 1
        return update

    def _emit(self, update: dict[str, Any]) -> None:
        self.history.append(update)
        self.stats["answers" if update["kind"] == "answer" else "updates"] += 1
        self._run.record("narrator.update", None, update)
        self._run.hub.publish("narrator.update", **update)

    async def tick(self) -> dict[str, Any]:
        """Summarise the window since the last update, then start a new window."""
        async with self._lock:
            digest = self.build_digest()
            update = await self._compose(digest, "update", None)
            self._window = Window(self._clock())
            self._emit(update)
            return update

    async def ask(self, question: str) -> dict[str, Any]:
        """Answer a viewer's question from the current window; does not reset it."""
        if self._model is None:
            raise NarratorUnavailable("narrator model not configured")
        async with self._lock:
            update = await self._compose(self.build_digest(), "answer", question.strip())
            self._emit(update)
            return update

    # -- read model ------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "next_update_at": self.next_update_at,
            "stats": dict(self.stats),
            "updates": list(self.history),
        }
