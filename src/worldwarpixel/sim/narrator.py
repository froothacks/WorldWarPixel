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
import uuid
import zlib
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from worldwarpixel.routing import EndpointPool, as_pool
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
    "You are the live commentator of WorldWarPixel, a world where AI agents paint one shared "
    "pixel canvas and argue, scheme and team up on a public message board. Every minute you "
    "brief the human audience like a sports commentator: "
    "what changed, who did it to whom, and why it matters.\n\n"
    "Lead with the most dramatic development in the 'Notable events' list: a raid that wiped "
    "92 px of Antoine Moreau's mural, a repair or counter-attack, a fresh argument or alliance "
    "on the board, someone abandoning their corner to invade another. Quote agents' own words "
    "from the board when they are juicy. Give coordinates so viewers can find things. Keep "
    "storylines going from your previous headlines but never repeat one. Skip raw statistics "
    "unless they are the story. Be concrete, vivid and short; only name agents, countries, "
    "regions and threads "
    "that appear in the digest, never invent any, and copy pixel counts and coordinates "
    "exactly as given (never make up numbers). Each bullet must cover a different event; do "
    "not reuse a phrase from one bullet in another or from your previous headlines.\n\n"
    "Example of the tone: headline 'Qiang Li and Antoine Moreau trade blows at (64, 64)', "
    "bullets like "
    "'Qiang Li wiped 92 px of Antoine Moreau's mural; Moreau hit back for 80 px and the corner "
    "is a mess of red and green.'\n\n"
    "Reply with JSON only, in this shape:\n"
    '{"headline": "one punchy line", "bullets": ["2 to 4 short bullets, each a different '
    'storyline"], '
    '"highlights": [{"type": "thread", "id": 4, "label": "short label"}, '
    '{"type": "agent", "id": 12, "label": "short label"}, '
    '{"type": "region", "x": 0, "y": 0, "w": 64, "h": 64, "label": "short label"}]}\n'
    "Highlights are links for the viewer: only use thread ids, agent ids and regions that appear "
    "in the digest."
)
MAX_ACTIONS = 600
MOVE_DISTANCE = 120
"""An agent whose painting position jumps this far between windows 'moved'."""


class NarratorModel(Protocol):
    """The one call the narrator needs from an LLM; tests swap in a fake."""

    async def complete(self, messages: list[ChatCompletionMessageParam]) -> str: ...


class OpenAINarratorModel:
    def __init__(
        self,
        client: AsyncOpenAI | EndpointPool,
        model: str,
        timeout: float = 60.0,
        max_tokens: int = 700,
        token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
        chat_template_kwargs: bool = True,
        reasoning_effort: str | None = None,
        vision: bool = True,
    ) -> None:
        self._pool = as_pool(client, model)
        self.model = model
        self._route_key = f"narrator:{uuid.uuid4().hex}"
        self._vision = vision
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._token_param = token_param
        self._chat_template_kwargs = chat_template_kwargs
        self._reasoning_effort = reasoning_effort

    async def complete(self, messages: list[ChatCompletionMessageParam]) -> str:
        # Same endpoint quirks as the agent policy; see models.py.
        if not self._vision:
            messages = [_text_only(m) for m in messages]
        extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if self._chat_template_kwargs
            else None
        )
        reasoning: Any = self._reasoning_effort if self._reasoning_effort is not None else NOT_GIVEN
        if self._token_param == "max_completion_tokens":
            response = await self._pool.create(
                self._route_key,
                messages=messages,
                max_completion_tokens=self._max_tokens,
                temperature=0.4,
                timeout=self._timeout,
                reasoning_effort=reasoning,
                extra_body=extra_body,
            )
        else:
            response = await self._pool.create(
                self._route_key,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=0.4,
                timeout=self._timeout,
                reasoning_effort=reasoning,
                extra_body=extra_body,
            )
        return response.choices[0].message.content or ""


def _text_only(message: ChatCompletionMessageParam) -> ChatCompletionMessageParam:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    text = "\n".join(str(part.get("text", "")) for part in content if part.get("type") == "text")
    return cast(ChatCompletionMessageParam, {**message, "content": text})


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
    actions: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=MAX_ACTIONS))
    repairs: Counter[int] = field(default_factory=Counter)
    """agent -> pixels of its previous ownership reclaimed from another agent."""


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


def bucket_span(i: int, size: int) -> tuple[int, int]:
    """[start, end) of grid bucket `i` along a side of `size`; inverse of `x * GRID // size`."""
    return -(-i * size // GRID), -(-(i + 1) * size // GRID)


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
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._run = run
        self._model = model
        self._running = running
        self.interval = interval
        self.include_image = include_image
        self._clock = clock
        self._sleep = sleep
        self._on_update = on_update
        self.history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.stats = {"updates": 0, "answers": 0, "failures": 0}
        self.next_update_at: float | None = None
        self._window = Window(clock())
        self._displaced: dict[tuple[int, int], int] = {}
        """(x, y) -> previous owner displaced by the latest overwrite."""
        self._last_pos: dict[int, tuple[int, int]] = {}
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

    def set_model(self, model: NarratorModel | None) -> None:
        """Swap the model mid-run (god mode); the next digest uses it."""
        self._model = model

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
            x, y = int(payload["x"]), int(payload["y"])
            if prev == 0:
                w.claimed_blank += 1
            elif prev != agent_id:
                w.overwrites[(agent_id, prev)] += 1
            if self._displaced.get((x, y)) == agent_id:
                w.repairs[agent_id] += 1
            if prev != 0 and prev != agent_id:
                self._displaced[(x, y)] = prev
            else:
                self._displaced.pop((x, y), None)
        elif kind == "action":
            self._observe_action(agent_id, payload)
        elif kind == "post":
            w.posts[agent_id] += 1
            w.thread_posts[int(payload["thread_id"])] += 1
            w.new_posts.append(payload)

    def _observe_action(self, agent_id: int, payload: dict[str, Any]) -> None:
        """Track painting positions from structured action events."""
        tool, args = str(payload.get("tool")), dict(payload.get("args") or {})
        result = dict(payload.get("result") or {})
        if "x" not in args or "y" not in args:
            return
        args["x"], args["y"] = int(args["x"]), int(args["y"])
        entry = {"ts": self._clock(), "agent_id": agent_id, "tool": tool, "args": args, **result}
        self._window.actions.append(entry)

    # -- digest ----------------------------------------------------------------------------

    def _region_box(self, rx: int, ry: int) -> dict[str, int]:
        canvas = self._run.canvas
        x0, x1 = bucket_span(rx, canvas.width)
        y0, y1 = bucket_span(ry, canvas.height)
        return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}

    def build_digest(self, window: Window | None = None, commit: bool = False) -> dict[str, Any]:
        """Structured summary of the window; also what gets rendered into the prompt.

        commit=True advances the painting-position baseline used to detect movement.
        """
        run, w = self._run, window if window is not None else self._window
        by_id = run.by_id
        now = self._clock()

        countries: Counter[str] = Counter()
        for aid, n in w.pixels.items():
            if aid in by_id:
                countries[by_id[aid].country] += n
        seen = set(w.turns) | set(w.pixels) | set(w.posts)
        active = Counter({aid: w.pixels[aid] + 10 * w.posts[aid] for aid in seen})
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
        events = self._notable(w, threads, overwrites, commit)
        conversations: list[dict[str, Any]] = []
        for x in threads[:6]:
            recent = [p for p in w.new_posts if int(p["thread_id"]) == x["thread_id"]][-3:]
            conversations.append(
                {
                    "thread_id": x["thread_id"],
                    "title": x["title"],
                    "posts": [
                        f"{run.agent_name(int(p['author_id']))}: {str(p['body'])[:160]}"
                        for p in recent
                    ],
                }
            )
        return {
            "events": events,
            "conversations": conversations,
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

    def _notable(
        self,
        w: Window,
        threads: list[dict[str, Any]],
        overwrites: list[dict[str, Any]],
        commit: bool,
    ) -> list[dict[str, Any]]:
        """Deterministic storylines from pixel ownership, movement and board activity."""
        run = self._run
        name = run.agent_name
        events: list[dict[str, Any]] = []

        def ev(kind: str, text: str, hl: dict[str, Any] | None = None) -> None:
            events.append({"kind": kind, "text": text, "highlight": hl})

        for o in overwrites[:3]:
            ev(
                "raid",
                f"{o['by_name']} wiped {o['pixels']} px of {o['over_name']}'s mural.",
                {"type": "agent", "id": o["by"]},
            )
        for aid, n in w.repairs.most_common(2):
            ev(
                "repair",
                f"{name(aid)} repaired {n} px of their mural.",
                {"type": "agent", "id": aid},
            )
        # moves (compared with where each agent worked before this window)
        moved: set[int] = set()
        last = dict(self._last_pos)
        for a in w.actions:
            aid, x, y = a["agent_id"], int(a["args"]["x"]), int(a["args"]["y"])
            prev = self._last_pos.get(aid)
            if prev and aid not in moved and abs(prev[0] - x) + abs(prev[1] - y) >= MOVE_DISTANCE:
                moved.add(aid)
                ev(
                    "move",
                    f"{name(aid)} left ({prev[0]}, {prev[1]}) and is now working at ({x}, {y}).",
                    {"type": "agent", "id": aid},
                )
            last[aid] = (x, y)
        if commit:
            self._last_pos = last
        # board
        roots = [p for p in w.new_posts if p.get("title")]
        for p in roots[-2:]:
            ev(
                "thread",
                f'{name(int(p["author_id"]))} opened "{str(p.get("title") or "")[:60]}".',
                {"type": "thread", "id": int(p["thread_id"])},
            )
        for t in threads[:1]:
            if t["new_posts"] >= 3:
                ev(
                    "hot_thread",
                    f'"{t["title"][:60]}" is heating up: {t["new_posts"]} new posts.',
                    {"type": "thread", "id": t["thread_id"]},
                )
        order = {
            "raid": 0,
            "repair": 1,
            "move": 2,
            "hot_thread": 3,
            "thread": 4,
        }
        events.sort(key=lambda e: order.get(e["kind"], 9))
        return events[:12]

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
        if d.get("events"):
            lines.append("Notable events since the last update (most important first):")
            lines += [f"- {e['text']}" for e in d["events"]]
        if d.get("conversations"):
            lines.append("Conversations (newest posts):")
            for c in d["conversations"]:
                lines.append(f'- thread {c["thread_id"]} "{c["title"]}"')
                lines += [f"    {p}" for p in c["posts"]]
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
        for e in digest.get("events", [])[:4]:
            bullets.append(e["text"])
            if e.get("highlight"):
                highlights.append(e["highlight"])
        if bullets:
            headline = bullets.pop(0)
            return headline, bullets, highlights
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
            source, error = "fallback", None
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
        highlights = self.clean_highlights(raw)
        if not highlights and kind == "update":
            highlights = self.clean_highlights(
                [e["highlight"] for e in digest.get("events", []) if e.get("highlight")][:5]
            )
        update = {
            "id": self._next_id,
            "ts": self._clock(),
            "kind": kind,
            "question": question,
            "headline": headline,
            "bullets": bullets,
            "highlights": highlights,
            "events": [{"kind": e["kind"], "text": e["text"]} for e in digest.get("events", [])],
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
        self._run.hub.publish("narrator.update", **update, stats=dict(self.stats))
        if self._on_update is not None:
            try:
                self._on_update(update)
            except Exception:
                log.exception("narrator update hook failed")

    async def tick(self) -> dict[str, Any]:
        """Summarise the window since the last update, then start a new window."""
        async with self._lock:
            window, self._window = self._window, Window(self._clock())
            update = await self._compose(self.build_digest(window, commit=True), "update", None)
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
            "model": self._model is not None,
            "next_update_at": self.next_update_at,
            "stats": dict(self.stats),
            "updates": list(self.history),
        }
