import base64
import json
import logging
import re
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
            "name": "fill_rect",
            "description": "Fill a w by h rectangle with one colour. Costs w*h pixels of budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "w": {"type": "integer", "minimum": 1},
                    "h": {"type": "integer", "minimum": 1},
                    "color": {"type": "integer", "minimum": 0, "maximum": 15},
                },
                "required": ["x", "y", "w", "h", "color"],
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
VIEW = 32


def palette_legend() -> str:
    return ", ".join(f"{i}={name}" for i, name in enumerate(COLOR_NAMES))


def system_prompt(agent: AgentSpec, width: int, height: int, pixels: int, posts: int) -> str:
    return (
        f"You are {agent.persona}\n\n"
        f"You have joined a global r/place-style event: a {width}x{height} shared pixel canvas "
        f"(x grows right, y grows down, 0-based) and a public message board, open to people from "
        f"every country. Nobody is in charge and nothing is pre-planned. Each turn you may paint "
        f"up to {pixels} pixels and make {posts} post or reply. Colours: {palette_legend()}. "
        f"Your usual colour is {agent.color} ({COLOR_NAMES[agent.color]}). "
        f"You started near ({agent.home_x}, {agent.home_y}) but may paint anywhere.\n\n"
        "In events like this people have done all sorts of things: painted their national or "
        "regional flag, a city skyline, a sports club crest or a university logo; drawn pixel-art "
        "mascots, animals, anime or game characters, memes, hearts, rainbows, religious symbols, "
        "portraits of heroes or musicians; written slogans, jokes, names, protest messages, "
        "lyrics, QR codes or a tiny webcomic; built roads, borders, flowers and landscapes "
        "connecting their region to neighbours. Groups have formed around a shared idea and "
        "recruited on the board, negotiated land swaps and peace treaties, merged rival flags "
        "into one, formed alliances to defend a mural against raids, or launched raids to "
        "overwrite someone else's work, spread a single colour like a void, troll a neighbour, "
        "camouflage graffiti into a bigger picture, or ambush an unfinished drawing. Others "
        "simply repaired damage, kept a corner tidy, or wandered the board making friends and "
        "enemies.\n\n"
        "Do whatever your persona would genuinely do here: pick what matters to you, decide who "
        "you side with or against, and pursue it over many turns. You see the whole canvas as an "
        "image plus a zoomed text crop around where you last painted.\n\n"
        "Practical notes:\n"
        f"- Use your whole pixel budget every turn ({pixels} pixels), usually one place_pixels "
        "call with adjacent pixels or one fill_rect, so your work grows into something "
        "recognisable rather than scattered dots.\n"
        "- Give coordinates when you coordinate with others. Prefer replying in an existing "
        "thread to starting a new one; address people by @Name.\n"
        "- Your thought is one or two short sentences in first person about what you see and "
        "intend; do not describe the hex grid or the tools."
    )


def crop_origin(obs: Observation, agent: AgentSpec) -> tuple[int, int]:
    x0 = max(0, min(agent.home_x - VIEW // 2, obs.canvas_width - VIEW))
    y0 = max(0, min(agent.home_y - VIEW // 2, obs.canvas_height - VIEW))
    return x0, y0


def local_view(obs: Observation, agent: AgentSpec, rows: list[str]) -> str:
    x0, y0 = crop_origin(obs, agent)
    header = (
        f"Zoomed {VIEW}x{VIEW} crop around where you last painted, top-left ({x0}, {y0}), "
        "one hex digit per pixel (row = y, column = x):"
    )
    return header + "\n" + "\n".join(rows)


def user_prompt(agent: AgentSpec, obs: Observation, view: str) -> str:
    board = (
        "\n".join(
            f'- thread {d["thread_id"]} "{d["title"]}" by {d["author"]} ({d["replies"]} replies)'
            + (f"; latest {d['latest']['author']}: {d['latest']['body']}" if d["latest"] else "")
            for d in obs.board
        )
        or "- (no threads yet; you could start one)"
    )
    return (
        f"Turn for {agent.name}. Budget: {obs.budget.pixels} pixels, {obs.budget.posts} post.\n"
        f"{view}\n\nMost active threads:\n{board}\n\n"
        "One or two sentences of thought, then call place_pixels (or fill_rect) with your full "
        "budget on whatever you are pursuing. If a thread above matters to you, answer it with "
        "reply(thread_id=..., body=...) rather than starting a new one."
    )


class LLMPolicy:
    """One LLM call per turn (plus a follow-up after a read tool); thoughts come from the text."""

    def __init__(
        self, client: AsyncOpenAI, model: str, timeout: float = 45.0, vision: bool = True
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout
        self._vision = vision
        self.stats: dict[str, Any] = {
            "calls": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "no_tools": 0,
            "failures": 0,
        }

    def describe(self, agent: AgentSpec, width: int, height: int, pixels: int, posts: int) -> str:
        return system_prompt(agent, width, height, pixels, posts)

    async def act(self, agent: AgentSpec, tools: ToolSet, obs: Observation) -> str:
        x0, y0 = crop_origin(obs, agent)
        rows = tools.peek(x0, y0, VIEW, VIEW)
        text = user_prompt(agent, obs, local_view(obs, agent, rows))
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
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=700,
                    temperature=0.8,
                    timeout=self._timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as e:  # network / API errors are per-turn failures, not crashes
                self.stats["failures"] += 1
                log.warning("agent %s LLM call failed: %s", agent.id, e)
                thoughts.append(f"(model call failed: {type(e).__name__})")
                break
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
                name = call.function.name
                result = self._execute(agent, tools, name, call.function.arguments, actions)
                used_read |= name in READ_TOOLS
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if not used_read:
                break
        thought = " ".join(t for t in thoughts if t)
        if actions:
            thought = f"{thought} [{'; '.join(actions)}]" if thought else "; ".join(actions)
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
        return json.dumps(result)

    def _note_error(self, text: str) -> None:
        """Keep the last few tool errors so run summaries show what the model gets wrong."""
        recent = self.stats.setdefault("recent_errors", [])
        recent.append(text[:200])
        del recent[:-8]

    @staticmethod
    def _dispatch(agent: AgentSpec, tools: ToolSet, name: str, a: dict[str, Any]) -> dict[str, Any]:
        # The small model is sloppy with numbers and lengths: clamp instead of rejecting.
        w, h = tools.canvas_size

        def colour(v: Any) -> int:
            try:
                c = int(v)
            except (TypeError, ValueError):
                return agent.color
            return c if 0 <= c < len(PALETTE) else agent.color

        match name:
            case "place_pixels":
                if isinstance(a["pixels"], str):  # the model often double-encodes the list
                    a["pixels"] = _parse_pixel_list(a["pixels"])
                a["pixels"] = [
                    {"x": _clamp(p["x"], w), "y": _clamp(p["y"], h), "color": colour(p["color"])}
                    for p in map(_pixel_fields, a["pixels"])
                ]
                return tools.place_pixels([(p["x"], p["y"], p["color"]) for p in a["pixels"]])
            case "fill_rect":
                a["x"], a["y"] = _clamp(a["x"], w), _clamp(a["y"], h)
                a["w"], a["h"] = min(int(a["w"]), w - a["x"]), min(int(a["h"]), h - a["y"])
                return tools.fill_rect(a["x"], a["y"], a["w"], a["h"], colour(a["color"]))
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
    elif name == "fill_rect":
        agent.home_x, agent.home_y = int(args["x"]), int(args["y"])


def _summarise(name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    match name:
        case "place_pixels":
            px = args.get("pixels") or []
            first = f" near ({px[0]['x']}, {px[0]['y']})" if px else ""
            return f"placed {result.get('placed', 0)} px{first}"
        case "fill_rect":
            return f"filled {args['w']}x{args['h']} at ({args['x']}, {args['y']})"
        case "post_thread":
            return f"posted thread {result.get('thread_id')}"
        case "reply":
            return f"replied in thread {args.get('thread_id')}"
        case "view_canvas":
            return "looked at the canvas"
        case "read_thread":
            return f"read thread {args.get('thread_id')}"
    return name
