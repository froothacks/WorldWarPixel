import json
import logging
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from worldwarpixel.sim.agents import AgentSpec, Observation
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
VIEW = 16


def palette_legend() -> str:
    return ", ".join(f"{i}={name}" for i, name in enumerate(COLOR_NAMES))


def system_prompt(agent: AgentSpec, width: int, height: int, pixels: int, posts: int) -> str:
    return (
        f"You are {agent.persona}\n\n"
        f"You are one of many people from around the world sharing a {width}x{height} pixel canvas "
        f"(x grows right, y grows down, 0-based) and a public message board. Nobody is in charge. "
        f"Each turn you may paint up to {pixels} pixels and make {posts} post or reply. "
        f"Colours: {palette_legend()}. "
        f"Your usual colour is {agent.color} ({COLOR_NAMES[agent.color]}). "
        f"Your home spot is ({agent.home_x}, {agent.home_y}).\n\n"
        "Act like yourself: build something, defend it, team up or argue with others on the board. "
        "Draw simple recognisable shapes a few pixels at a time near your home or your team. "
        "Each turn: write one or two sentences about what you notice and intend (this is shown to "
        "viewers as your thought), then call tools. Keep posts short and in character."
    )


def local_view(obs: Observation, agent: AgentSpec, rows: list[str]) -> str:
    x0 = max(0, min(agent.home_x - VIEW // 2, obs.canvas_width - VIEW))
    y0 = max(0, min(agent.home_y - VIEW // 2, obs.canvas_height - VIEW))
    header = f"Canvas around your home, top-left ({x0}, {y0}), one hex digit per pixel:"
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
        "Say what you're thinking in one or two sentences, then call tools."
    )


class LLMPolicy:
    """One LLM call per turn (plus a follow-up after a read tool); thoughts come from the text."""

    def __init__(self, client: AsyncOpenAI, model: str, timeout: float = 45.0) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout
        self.stats = {"calls": 0, "tool_calls": 0, "tool_errors": 0, "no_tools": 0, "failures": 0}

    def describe(self, agent: AgentSpec, width: int, height: int, pixels: int, posts: int) -> str:
        return system_prompt(agent, width, height, pixels, posts)

    async def act(self, agent: AgentSpec, tools: ToolSet, obs: Observation) -> str:
        x0 = max(0, min(agent.home_x - VIEW // 2, obs.canvas_width - VIEW))
        y0 = max(0, min(agent.home_y - VIEW // 2, obs.canvas_height - VIEW))
        rows = tools.peek(x0, y0, VIEW, VIEW)
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": system_prompt(
                    agent, obs.canvas_width, obs.canvas_height, obs.budget.pixels, obs.budget.posts
                ),
            },
            {"role": "user", "content": user_prompt(agent, obs, local_view(obs, agent, rows))},
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
                    max_tokens=400,
                    temperature=0.8,
                    timeout=self._timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as e:  # network / API errors are per-turn failures, not crashes
                self.stats["failures"] += 1
                log.warning("agent %s LLM call failed: %s", agent.id, e)
                return f"(model call failed: {type(e).__name__})"
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
                result = self._execute(tools, name, call.function.arguments, actions)
                used_read |= name in READ_TOOLS
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if not used_read:
                break
        thought = " ".join(t for t in thoughts if t)
        if actions:
            thought = f"{thought} [{'; '.join(actions)}]" if thought else "; ".join(actions)
        return thought or "(no response)"

    def _execute(self, tools: ToolSet, name: str, raw_args: str, actions: list[str]) -> str:
        try:
            args: dict[str, Any] = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            self.stats["tool_errors"] += 1
            return "error: arguments were not valid JSON"
        try:
            result = self._dispatch(tools, name, args)
        except ToolError as e:
            self.stats["tool_errors"] += 1
            actions.append(f"{name} rejected: {e}")
            return f"error: {e}"
        except (TypeError, ValueError, KeyError) as e:
            self.stats["tool_errors"] += 1
            return f"error: bad arguments ({e})"
        actions.append(_summarise(name, args, result))
        return json.dumps(result)

    @staticmethod
    def _dispatch(tools: ToolSet, name: str, a: dict[str, Any]) -> dict[str, Any]:
        match name:
            case "place_pixels":
                px = [(int(p["x"]), int(p["y"]), int(p["color"])) for p in a["pixels"]]
                return tools.place_pixels(px)
            case "fill_rect":
                return tools.fill_rect(
                    int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"]), int(a["color"])
                )
            case "post_thread":
                return tools.post_thread(str(a["title"]), str(a["body"]))
            case "reply":
                return tools.reply(int(a["thread_id"]), str(a["body"]))
            case "view_canvas":
                return tools.view_canvas(int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"]))
            case "read_thread":
                return tools.read_thread(int(a["thread_id"]))
        raise ToolError(f"unknown tool {name}")


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
