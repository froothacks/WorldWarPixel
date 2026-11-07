import base64
import json
import logging
import random
import re
import time
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from worldwarpixel.sim.agents import AgentSpec, Observation
from worldwarpixel.sim.canvas import PALETTE
from worldwarpixel.sim.tools import ToolError, ToolSet

log = logging.getLogger(__name__)

COLOR_NAMES = (
    "white",
    "light grey",
    "grey",
    "black",
    "pink",
    "red",
    "orange",
    "brown",
    "yellow",
    "lime",
    "green",
    "cyan",
    "blue",
    "dark blue",
    "magenta",
    "purple",
)

# Design rule: agents draw only with place_pixels. Do NOT add tools that draw flags, text,
# shapes, rectangles or other stencils; agents compose bigger pictures from pixels by
# looking at the canvas.
TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "place_pixels",
            "description": "Paint up to your remaining pixel budget. Coordinates are 0-based.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "integer"},
                                "y": {"type": "integer"},
                                "color": {"type": "integer", "minimum": 0, "maximum": 15},
                            },
                            "required": ["x", "y", "color"],
                        },
                    }
                },
                "required": ["pixels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_thread",
            "description": "Start a new public thread on the message board.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                "required": ["title", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reply",
            "description": "Reply in an existing thread. Mention people with @Name.",
            "parameters": {
                "type": "object",
                "properties": {"thread_id": {"type": "integer"}, "body": {"type": "string"}},
                "required": ["thread_id", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_canvas",
            "description": "Look at a canvas region (max 64x64) as rows of hex colour digits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "w": {"type": "integer", "minimum": 1, "maximum": 64},
                    "h": {"type": "integer", "minimum": 1, "maximum": 64},
                },
                "required": ["x", "y", "w", "h"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_thread",
            "description": "Read a whole thread, including replies.",
            "parameters": {
                "type": "object",
                "properties": {"thread_id": {"type": "integer"}},
                "required": ["thread_id"],
            },
        },
    },
]

READ_TOOLS = {"view_canvas", "read_thread"}
MAX_ROUNDS = 2


def palette_legend() -> str:
    return ", ".join(f"{i}={name}" for i, name in enumerate(COLOR_NAMES))


def system_prompt(agent: AgentSpec, width: int, height: int, pixels: int, posts: int) -> str:
    return (
        f"You are {agent.persona}\n\n"
        f"You have joined a global r/place-style event: a {width}x{height} shared pixel canvas "
        f"(x grows right, y grows down, 0-based) and a public message board, open to people from "
        f"every country. Nobody is in charge and nothing is pre-planned. Each turn you may paint "
        f"up to {pixels} pixels and make {posts} post or reply. Colours: {palette_legend()}. "
        f"You started near ({agent.home_x}, {agent.home_y}) but may paint anywhere.\n\n"
        "In events like this people have done all sorts of things: drawn pixel-art mascots, "
        "animals, anime or game characters, memes, hearts, rainbows, religious symbols, "
        "portraits of heroes or musicians; painted a city skyline, a landmark, a sports club "
        "crest or a university logo; written slogans, jokes, names, protest messages, "
        "lyrics, QR codes or a tiny webcomic; built roads, borders, flowers and landscapes "
        "connecting their region to neighbours. Groups have formed around a shared idea and "
        "recruited on the board, negotiated land swaps and peace treaties, merged rival murals "
        "into one, formed alliances to defend a mural against raids, or launched raids to "
        "overwrite someone else's work, spread a single colour like a void, troll a neighbour, "
        "camouflage graffiti into a bigger picture, or ambush an unfinished drawing. Others "
        "simply repaired damage, kept a corner tidy, or wandered the board making friends and "
        "enemies.\n\n"
        "Do whatever your persona would genuinely do here: pick what matters to you, decide who "
        "you side with or against, and pursue it over many turns. Each turn you see the whole "
        "canvas as an image plus a short list of the busiest areas with their coordinates.\n\n"
        "Practical notes:\n"
        f"- Use your whole pixel budget every turn ({pixels} pixels). place_pixels is your only "
        "brush: plan a picture as rows of coordinates and paint part of it each turn, continuing "
        "next turn where you left off (that also repairs damage). Look at the canvas image to line "
        "new pixels up with what is already there.\n"
        "- Flags are the most overdone, least memorable thing on a canvas like this. Avoid flags "
        "and make something original instead: a creature, a scene, a symbol, a joke or a slogan "
        "that says something about you. To show where you are from, draw what your place is known "
        "for (a landmark, food, animal, team or hero) rather than its flag.\n"
        "- Posts marked 🗞 are World News: automatic reports of real wars, alliances, "
        "territory changes and raids on this canvas. If one concerns your "
        "country, react to it in that thread: defend, retaliate, negotiate or celebrate.\n"
        "- Only post when it matters to others: a claim with coordinates, a call for help, "
        "a threat, a deal, or an answer to someone. No greetings or status chatter. Prefer "
        "replying in an existing thread to starting a new one; address people by @Name. "
        "Never reuse another thread's title.\n"
        "- Your thought is one or two short sentences in first person about what you see and "
        "intend; do not describe colour numbers, the image format or the tools."
    )


def region_view(
    agent: AgentSpec, tiles: list[dict[str, int]], rng: random.Random | None = None
) -> str:
    """Busiest areas plus a few quiet ones (randomised per agent so agents spread out)."""
    rng = rng or random.Random(agent.id)
    busy = [t for t in tiles if t["painted"]][:6]
    quiet = [t for t in tiles if t["painted"] * 20 < t["size"]]
    quiet = rng.sample(quiet, min(3, len(quiet)))
    if not busy:
        return (
            f"The canvas is still blank. You started near ({agent.home_x}, {agent.home_y}); "
            "a good spot to begin is anywhere around there."
        )
    lines = [
        f"- x {r['x0']}-{r['x1']}, y {r['y0']}-{r['y1']}: {r['painted']} px painted, "
        f"mostly {COLOR_NAMES[r['color']]}"
        for r in busy
    ]
    text = (
        f"You last painted near ({agent.home_x}, {agent.home_y}). Busiest areas right now "
        "(the image shows what is actually drawn there):\n" + "\n".join(lines)
    )
    if quiet:
        text += "\nStill mostly empty: " + "; ".join(
            f"x {r['x0']}-{r['x1']}, y {r['y0']}-{r['y1']}" for r in quiet
        )
        q = quiet[0]
        sx = rng.randint(q["x0"], max(q["x0"], q["x1"] - 24))
        sy = rng.randint(q["y0"], max(q["y0"], q["y1"] - 16))
        text += f"\nA free spot nobody is using yet: ({sx}, {sy})."
    return text


def pick_threads(
    board: list[dict[str, Any]], rng: random.Random, k: int = 6
) -> list[dict[str, Any]]:
    """Keep important threads first, then mix a couple of recent ones with random choices."""
    if k <= 0:
        return []
    tagged = [entry for entry in board if entry.get("tag")][:k]
    untagged = [entry for entry in board if not entry.get("tag")]
    count = min(2, max(0, k - len(tagged)), len(untagged))
    picked = [*tagged, *untagged[:count]]
    remaining = untagged[count:]
    picked.extend(rng.sample(remaining, min(len(remaining), k - len(picked))))
    return picked


def user_prompt(
    agent: AgentSpec,
    obs: Observation,
    view: str,
    rng: random.Random | None = None,
) -> str:
    threads = pick_threads(obs.board, rng or random.Random(agent.id))
    prefixes = {"announcement": "", "dispatch": "[WORLD NEWS] ", "mention": "[about you] "}
    board = (
        "\n".join(
            f"{prefixes.get(str(d.get('tag')), '')}- thread {d['thread_id']} "
            f'"{d["title"]}" by {d["author"]} ({d["replies"]} replies)'
            + (
                f"; latest {d['latest']['author']}: {str(d['latest']['body'])[:160]}"
                if d["latest"]
                else ""
            )
            for d in threads
        )
        or "- (no threads yet; you could start one)"
    )
    budget = (
        f"Turn for {agent.name}. Budget: {obs.budget.pixels} pixels, "
        "no posts this turn (you posted recently).\n"
        if obs.budget.posts == 0
        else f"Turn for {agent.name}. Budget: {obs.budget.pixels} pixels, "
        f"{obs.budget.posts} post.\n"
    )
    return (
        budget
        + f"{view}\n\nSome current threads:\n{board}\n\n"
        + "One or two sentences of thought, then call place_pixels (not a flag) on whatever "
        + "you are pursuing. If a thread above matters to you, answer it with "
        + "reply(thread_id=..., body=...) rather than starting a new one."
    )


def _sample(samples: list[float], value: float, keep: int = 2000) -> None:
    samples.append(value)
    if len(samples) > keep:
        del samples[: len(samples) - keep]


def latency_summary(samples: list[float]) -> dict[str, float] | None:
    """n / mean / p50 / p95 / max in seconds over the recent samples."""
    if not samples:
        return None
    xs = sorted(samples)

    def pick(q: float) -> float:
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 3),
        "p50": round(pick(0.5), 3),
        "p95": round(pick(0.95), 3),
        "max": round(xs[-1], 3),
    }


class LLMPolicy:
    """One LLM call per turn (plus a follow-up after a read tool); thoughts come from the text."""

    def __init__(
        self, client: AsyncOpenAI, model: str, timeout: float = 45.0, vision: bool = True
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout
        self._vision = vision
        self._rng = random.Random()
        self.stats: dict[str, Any] = {
            "calls": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "no_tools": 0,
            "failures": 0,
        }
        self._latency: dict[str, list[float]] = {"model": [], "turn": []}

    def describe(self, agent: AgentSpec, width: int, height: int, pixels: int, posts: int) -> str:
        return system_prompt(agent, width, height, pixels, posts)

    def public_stats(self) -> dict[str, Any]:
        """Counters plus latency percentiles of the recent model calls and whole turns."""
        return {
            **self.stats,
            "latency": {k: latency_summary(v) for k, v in self._latency.items()},
        }

    async def act(self, agent: AgentSpec, tools: ToolSet, obs: Observation) -> str:
        turn_t0 = time.monotonic()
        text = user_prompt(
            agent,
            obs,
            region_view(agent, tools.regions(limit=None), self._rng),
            self._rng,
        )
        content: list[ChatCompletionContentPartParam] = [{"type": "text", "text": text}]
        if self._vision:
            png, scale = tools.png()
            note = (
                f"Image: the whole {obs.canvas_width}x{obs.canvas_height} canvas"
                + (f", downscaled {scale}x (multiply image coords by {scale})" if scale > 1 else "")
                + "."
            )
            content.insert(0, {"type": "text", "text": note})
            content.insert(
                1,
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
                },
            )
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": system_prompt(
                    agent, obs.canvas_width, obs.canvas_height, obs.budget.pixels, obs.budget.posts
                ),
            },
            {"role": "user", "content": content},
        ]
        thoughts: list[str] = []
        actions: list[str] = []
        for round_no in range(MAX_ROUNDS):
            self.stats["calls"] += 1
            call_t0 = time.monotonic()
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=self._tools_for(obs),
                    tool_choice="auto",
                    max_tokens=700,
                    temperature=1.0,
                    timeout=self._timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as e:  # network / API errors are per-turn failures, not crashes
                _sample(self._latency["model"], time.monotonic() - call_t0)
                self.stats["failures"] += 1
                self.stats["last_error"] = str(e)[:200]
                log.warning("agent %s LLM call failed: %s", agent.id, e)
                thoughts.append(f"(model call failed: {type(e).__name__}: {str(e)[:140]})")
                break
            _sample(self._latency["model"], time.monotonic() - call_t0)
            msg = response.choices[0].message
            if msg.content:
                thoughts.append(msg.content.strip())
            calls = msg.tool_calls or []
            if not calls:
                if round_no == 0:
                    self.stats["no_tools"] += 1
                break
            messages.append(
                cast(
                    ChatCompletionMessageParam,
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [c.model_dump() for c in calls],
                    },
                )
            )
            used_read = False
            for call in calls:
                if call.type != "function":
                    continue
                self.stats["tool_calls"] += 1
                name = _canonical_tool(call.function.name)
                result = self._execute(agent, tools, name, call.function.arguments, actions)
                used_read |= name in READ_TOOLS
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if not used_read:
                break
        thought = " ".join(t for t in thoughts if t)
        if actions:
            thought = f"{thought} [{'; '.join(actions)}]" if thought else "; ".join(actions)
        _sample(self._latency["turn"], time.monotonic() - turn_t0)
        return thought or "(no response)"

    def _execute(
        self, agent: AgentSpec, tools: ToolSet, name: str, raw_args: str, actions: list[str]
    ) -> str:
        try:
            args: dict[str, Any] = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            self.stats["tool_errors"] += 1
            return "error: arguments were not valid JSON"
        try:
            result = self._dispatch(agent, tools, name, args)
        except ToolError as e:
            self.stats["tool_errors"] += 1
            self._note_error(f"{name}: {e}")
            actions.append(f"{name} rejected: {e}")
            return f"error: {e}"
        except (TypeError, ValueError, KeyError) as e:
            self._note_error(f"{name}: bad arguments {raw_args[:120]!r}: {e!r}")
            self.stats["tool_errors"] += 1
            return f"error: bad arguments ({e})"
        actions.append(_summarise(name, args, result))
        _track_position(agent, name, args, result)
        if name not in ("view_canvas", "read_thread"):
            tools.record_action(name, _public_args(name, args), result)
        return json.dumps(result)

    def _tools_for(self, obs: Observation) -> list[ChatCompletionToolParam]:
        """Once the board is busy, usually hide post_thread so agents reply instead of piling on."""
        if obs.budget.posts == 0:
            return [t for t in TOOLS if t["function"]["name"] not in ("post_thread", "reply")]
        if len(obs.board) >= 6 and self._rng.random() < 0.75:
            return [t for t in TOOLS if t["function"]["name"] != "post_thread"]
        return TOOLS

    def _note_error(self, text: str) -> None:
        """Keep the last few tool errors so run summaries show what the model gets wrong."""
        recent = self.stats.setdefault("recent_errors", [])
        recent.append(text[:200])
        del recent[:-8]

    def _dispatch(
        self, agent: AgentSpec, tools: ToolSet, name: str, a: dict[str, Any]
    ) -> dict[str, Any]:
        # The small model is sloppy with numbers and lengths: clamp instead of rejecting.
        w, h = tools.canvas_size

        def colour(v: Any) -> int:
            try:
                c = int(v)
            except (TypeError, ValueError):
                return agent.color
            return c if 0 <= c < len(PALETTE) else agent.color

        # Design rule: agents draw only with place_pixels. Do NOT add tools that draw flags, text,
        # shapes, rectangles or other stencils; agents compose bigger pictures from pixels by
        # looking at the canvas.
        match name:
            case "place_pixels":
                if isinstance(a["pixels"], str):  # the model often double-encodes the list
                    a["pixels"] = _parse_pixel_list(a["pixels"])
                a["pixels"] = [
                    {"x": _clamp(p["x"], w), "y": _clamp(p["y"], h), "color": colour(p["color"])}
                    for p in map(_pixel_fields, a["pixels"])
                ]
                return tools.place_pixels([(p["x"], p["y"], p["color"]) for p in a["pixels"]])
            case "post_thread":
                return tools.post_thread(str(a["title"])[:120], str(a["body"])[:600])
            case "reply":
                return tools.reply(int(a["thread_id"]), str(a["body"])[:600])
            case "view_canvas":
                return tools.view_canvas(int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"]))
            case "read_thread":
                return tools.read_thread(int(a["thread_id"]))
        raise ToolError(f"unknown tool {name}")


_FIELD_RE = {k: re.compile(rf"(?<![A-Za-z_]){k}\\?\"?\s*:\s*(-?\d+)") for k in ("x", "y", "color")}


def _parse_pixel_list(text: str) -> list[dict[str, int]]:
    """Decode a stringified pixel list, salvaging truncated or sloppy JSON object by object."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except ValueError:
        pass
    found: list[dict[str, int]] = []
    for obj in re.findall(r"\{[^{}]*\}", text):  # only complete {...} objects, any key order
        fields = {k: rx.search(obj) for k, rx in _FIELD_RE.items()}
        if all(fields.values()):
            found.append({k: int(m.group(1)) for k, m in fields.items() if m})
    if not found:
        raise ToolError("pixels must be a list of {x, y, color}")
    return found


def _pixel_fields(p: Any) -> dict[str, Any]:
    """Accept {x, y, color}, {x, y} (colour filled in later) or the [x, y, color] shorthand."""
    if isinstance(p, (list, tuple)) and len(p) in (2, 3):
        p = dict(zip(("x", "y", "color"), p, strict=False))
    return {"x": p["x"], "y": p["y"], "color": p.get("color", "default")}


_TOOL_HINTS = (
    ("view_thread", "read_thread"),
    ("read", "read_thread"),
    ("pixel", "place_pixels"),
    ("paint", "place_pixels"),
    ("canvas", "view_canvas"),
    ("image", "view_canvas"),
    ("see", "view_canvas"),
    ("look", "view_canvas"),
    ("view", "view_canvas"),
    ("reply", "reply"),
    ("thread", "post_thread"),
    ("post", "post_thread"),
)


def _canonical_tool(name: str) -> str:
    """Map invented tool names (put_pixels, draw_image, see_canvas) onto the real ones."""
    known = {t["function"]["name"] for t in TOOLS}
    if name in known:
        return name
    low = name.lower()
    if (low.startswith("draw_") and "image" not in low) or low.startswith("fill_"):
        return name
    for hint, real in _TOOL_HINTS:
        if hint in low:
            return real
    return name


def _clamp(v: Any, size: int) -> int:
    return min(max(int(v), 0), size - 1)


def _track_position(
    agent: AgentSpec, name: str, args: dict[str, Any], result: dict[str, Any]
) -> None:
    """Remember where the agent last painted so the next crop follows it."""
    placed = int(result.get("placed", 0))
    if name == "place_pixels" and args.get("pixels") and placed:
        last = args["pixels"][min(placed, len(args["pixels"])) - 1]
        agent.home_x, agent.home_y = int(last["x"]), int(last["y"])


def _public_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Compact, JSON-safe view of a tool call's arguments for the event log."""
    if name == "place_pixels":
        px = args.get("pixels") or []
        return {"count": len(px), "x": px[0]["x"], "y": px[0]["y"]} if px else {"count": 0}
    out: dict[str, Any] = {}
    if name in ("post_thread", "reply"):
        out["thread_id"] = args.get("thread_id")
        out["title"] = str(args.get("title", ""))[:80] or None
    return out


def _summarise(name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    match name:
        case "place_pixels":
            px = args.get("pixels") or []
            first = f" near ({px[0]['x']}, {px[0]['y']})" if px else ""
            return f"placed {result.get('placed', 0)} px{first}"
        case "post_thread":
            return f"posted thread {result.get('thread_id')}"
        case "reply":
            return f"replied in thread {args.get('thread_id')}"
        case "view_canvas":
            return "looked at the canvas"
        case "read_thread":
            return f"read thread {args.get('thread_id')}"
    return name
