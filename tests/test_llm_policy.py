import asyncio
import json
import random
from typing import Any

from worldwarpixel.sim.agents import AgentSpec, Observation
from worldwarpixel.sim.board import Board
from worldwarpixel.sim.canvas import Canvas, PixelWrite
from worldwarpixel.sim.llm_policy import (
    LLMPolicy,
    _canonical_tool,
    _parse_pixel_list,
    pick_threads,
    system_prompt,
    user_prompt,
)
from worldwarpixel.sim.tools import ToolSet, TurnBudget


class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def model_dump(self) -> dict[str, Any]:
        d = self.__dict__
        fn = d["function"].__dict__
        return {
            "id": d["id"],
            "type": "function",
            "function": {"name": fn["name"], "arguments": fn["arguments"]},
        }


def _response(content: str | None, calls: list[tuple[str, dict[str, Any]]]) -> Any:
    tool_calls = [
        _Obj(id=f"c{i}", type="function", function=_Obj(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(calls)
    ]
    return _Obj(choices=[_Obj(message=_Obj(content=content, tool_calls=tool_calls or None))])


class FakeCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kw: Any) -> Any:
        self.requests.append(kw)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.chat: Any = _Obj(completions=FakeCompletions(responses))


def _setup() -> tuple[AgentSpec, ToolSet, Observation, list[tuple[str, dict[str, Any]]]]:
    canvas, board = Canvas(64, 40), Board()
    events: list[tuple[str, dict[str, Any]]] = []
    agent = AgentSpec(1, "Priya Raman", "India", 5, 10, 10, 1.0, "Priya Raman, 34, from India.")
    tools = ToolSet(
        canvas,
        board,
        1,
        TurnBudget(pixels=4),
        lambda k, p: events.append((k, p)) if k != "action" else None,  # writes only
        lambda: 0.0,
    )
    obs = Observation(1.0, tools.budget, 64, 40, [])
    return agent, tools, obs, events


def test_tool_calls_are_executed_and_summarised() -> None:
    agent, tools, obs, events = _setup()
    client = FakeClient(
        [
            _response(
                "I'll start a small flag.",
                [
                    (
                        "place_pixels",
                        {"pixels": [{"x": 1, "y": 2, "color": 5}, {"x": 2, "y": 2, "color": 5}]},
                    ),
                    ("post_thread", {"title": "Flag", "body": "Join me"}),
                ],
            )
        ]
    )
    policy = LLMPolicy(client, "m")  # type: ignore[arg-type]
    thought = asyncio.run(policy.act(agent, tools, obs))
    assert thought.startswith("I'll start a small flag.")
    assert "placed 2 px near (1, 2)" in thought and "posted thread 1" in thought
    assert [k for k, _ in events] == ["pixel", "pixel", "post"]
    assert (agent.home_x, agent.home_y) == (2, 2)
    assert policy.stats == {
        "calls": 1,
        "tool_calls": 2,
        "tool_errors": 0,
        "no_tools": 0,
        "failures": 0,
    }
    latency = policy.public_stats()["latency"]
    assert latency["model"]["n"] == 1 and latency["turn"]["n"] == 1
    assert 0 <= latency["turn"]["p50"] <= latency["turn"]["max"]
    req = client.chat.completions.requests[0]
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    parts = req["messages"][1]["content"]
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith(
        "data:image/png"
    )


def test_read_tool_triggers_second_round_and_errors_are_reported() -> None:
    agent, tools, obs, events = _setup()
    client = FakeClient(
        [
            _response(None, [("view_canvas", {"x": 0, "y": 0, "w": 8, "h": 8})]),
            _response("Too wide.", [("view_canvas", {"x": 0, "y": 0, "w": 65, "h": 1})]),
        ]
    )
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    thought = asyncio.run(policy.act(agent, tools, obs))
    assert "view size" in thought and events == []
    assert policy.stats["calls"] == 2 and policy.stats["tool_errors"] == 1
    tool_msgs = [m for m in client.chat.completions.requests[1]["messages"] if m["role"] == "tool"]
    assert "rows" in tool_msgs[0]["content"]


def test_no_tool_response_counts_and_prompt_mentions_budget() -> None:
    agent, tools, obs, _ = _setup()
    client = FakeClient([_response("Just thinking.", [])])
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    assert asyncio.run(policy.act(agent, tools, obs)) == "Just thinking."
    assert policy.stats["no_tools"] == 1
    assert "4 pixels" in system_prompt(agent, 64, 40, 4, 1) and "Priya Raman, 34" in system_prompt(
        agent, 64, 40, 4, 1
    )


def test_agent_max_tokens_is_passed_to_the_model() -> None:
    agent, tools, obs, _ = _setup()
    client = FakeClient([_response(None, [])])
    policy = LLMPolicy(client, "m", vision=False, max_tokens=4000)  # type: ignore[arg-type]
    asyncio.run(policy.act(agent, tools, obs))
    assert client.chat.completions.requests[0]["max_tokens"] == 4000


def test_openai_reasoning_model_params_are_passed_to_the_model() -> None:
    agent, tools, obs, _ = _setup()
    client = FakeClient([_response(None, [])])
    policy = LLMPolicy(
        client,  # type: ignore[arg-type]
        "gpt-6-luna",
        vision=False,
        max_tokens=3000,
        token_param="max_completion_tokens",
        chat_template_kwargs=False,
        reasoning_effort="none",
    )
    asyncio.run(policy.act(agent, tools, obs))
    request = client.chat.completions.requests[0]
    assert request["max_completion_tokens"] == 3000
    assert "max_tokens" not in request
    assert request["reasoning_effort"] == "none"
    assert not request.get("extra_body")  # no chat_template_kwargs for api.openai.com


def test_system_prompt_does_not_assign_a_usual_colour() -> None:
    agent, *_ = _setup()
    prompt = system_prompt(agent, 64, 40, 4, 1)
    assert "usual colour" not in prompt.lower()


def test_only_pixels_are_exposed_as_a_drawing_tool() -> None:
    agent, _, obs, _ = _setup()
    policy = LLMPolicy(FakeClient([]), "m", vision=False)  # type: ignore[arg-type]
    names = [tool["function"]["name"] for tool in policy._tools_for(obs)]
    removed = (
        "draw_" + "flag",
        "draw_" + "text",
        "draw_" + "shape",
        "fill" + "_rect",
    )
    assert [name for name in names if name == "place_pixels" or name in removed] == ["place_pixels"]
    prompt = system_prompt(agent, 64, 40, 4, 1)
    assert all(name not in prompt for name in removed)


def test_removed_tool_calls_use_the_unknown_tool_error_path() -> None:
    async def main() -> None:
        agent, tools, obs, _ = _setup()
        removed = (
            "draw_" + "flag",
            "draw_" + "text",
            "draw_" + "shape",
            "fill" + "_rect",
            "draw_" + "flag_on_canvas",
        )
        client = FakeClient([_response(None, [(name, {}) for name in removed])])
        policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
        thought = await policy.act(agent, tools, obs)
        assert all(f"unknown tool {name}" in thought for name in removed)
        assert policy.stats["tool_errors"] == len(removed)

    asyncio.run(main())


def test_follow_up_failure_keeps_completed_actions() -> None:
    agent, tools, obs, events = _setup()

    class Boom(FakeCompletions):
        async def create(self, **kw: Any) -> Any:
            if not self.responses:
                raise RuntimeError("down")
            return await super().create(**kw)

    client = FakeClient([])
    client.chat.completions = Boom(
        [
            _response(
                "Paint then look.",
                [
                    ("place_pixels", {"pixels": [{"x": 3, "y": 3, "color": 2}]}),
                    ("view_canvas", {"x": 0, "y": 0, "w": 4, "h": 4}),
                ],
            )
        ]
    )
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    thought = asyncio.run(policy.act(agent, tools, obs))
    assert "Paint then look." in thought and "placed 1 px near (3, 3)" in thought
    assert "model call failed: RuntimeError" in thought
    assert len(events) == 1 and policy.stats["failures"] == 1


def test_out_of_range_arguments_are_clamped() -> None:
    agent, tools, obs, events = _setup()
    client = FakeClient(
        [
            _response(
                "Edge.",
                [
                    ("place_pixels", {"pixels": [{"x": 999, "y": -4, "color": 21}]}),
                    ("post_thread", {"title": "t" * 300, "body": "b" * 5000}),
                ],
            )
        ]
    )
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    thought = asyncio.run(policy.act(agent, tools, obs))
    assert "placed 1 px near (63, 0)" in thought and "posted thread 1" in thought
    assert events[0][1]["color"] == agent.color and policy.stats["tool_errors"] == 0


def test_double_encoded_pixel_list_is_accepted() -> None:
    agent, tools, obs, events = _setup()
    encoded = json.dumps([{"x": 3, "y": 4, "color": 2}, {"x": 4, "y": 4, "color": 2}])
    client = FakeClient([_response("Line.", [("place_pixels", {"pixels": encoded})])])
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    thought = asyncio.run(policy.act(agent, tools, obs))
    assert "placed 2 px" in thought and len(events) == 2 and policy.stats["tool_errors"] == 0


def test_truncated_pixel_string_is_salvaged() -> None:
    agent, tools, obs, events = _setup()
    broken = (
        '[{"x": 3, "y": 4, "color": 2}, {color: 2, "x": 5, "y": 4}, {"x": 6, "y": 9}, {"x": 7, "y'
    )
    client = FakeClient([_response("Line.", [("place_pixels", {"pixels": broken})])])
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    thought = asyncio.run(policy.act(agent, tools, obs))
    assert "placed 2 px" in thought and len(events) == 2 and policy.stats["tool_errors"] == 0


def test_truncated_place_pixels_json_is_salvaged() -> None:
    agent, tools, obs, events = _setup()
    raw = (
        '{"pixels": [{"x":0,"y":1,"color":2},{"x":1,"y":1,"color":2},'
        '{"x":2,"y":1,"color":2},{"x":3,"y":1,"color'
    )
    call = _Obj(id="c0", type="function", function=_Obj(name="place_pixels", arguments=raw))
    resp = _Obj(choices=[_Obj(message=_Obj(content="Line.", tool_calls=[call]))])
    client = FakeClient([resp])
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    thought = asyncio.run(policy.act(agent, tools, obs))
    assert "placed 3 px" in thought and len(events) == 3 and policy.stats["tool_errors"] == 0


def test_malformed_pixels_are_skipped_not_fatal() -> None:
    agent, tools, obs, events = _setup()
    pixels = [
        {"x": 1, "y": 1, "color": 2},
        {"x": 5, "color": 3},  # missing y
        {"x": {"bad": 1}, "y": 2, "color": 3},  # x is an object
        [6, 7, 4],  # shorthand
        {"x": 8, "y": 8},  # missing colour -> agent colour
    ]
    client = FakeClient([_response("Mixed.", [("place_pixels", {"pixels": pixels})])])
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    thought = asyncio.run(policy.act(agent, tools, obs))
    assert "placed 3 px" in thought and len(events) == 3
    assert policy.stats["tool_errors"] == 0


def test_incomplete_pixel_objects_are_skipped() -> None:
    assert _parse_pixel_list(
        '[{"x": 1, "y": 2}, {"x": 9, "y": 9, "color": 3}, {"x": 4, "color'
    ) == [{"x": 9, "y": 9, "color": 3}]


def test_pixel_shorthand_and_missing_colour() -> None:
    agent, tools, obs, events = _setup()
    pixels = [[1, 2, 3], {"x": 4, "y": 5}, {"x": 6, "y": 7, "color": "red"}]
    client = FakeClient([_response("Mixed.", [("place_pixels", {"pixels": pixels})])])
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    asyncio.run(policy.act(agent, tools, obs))
    assert [e[1]["color"] for e in events] == [3, agent.color, agent.color]
    assert policy.stats["tool_errors"] == 0


def test_max_concurrency_limits_in_flight_model_calls() -> None:
    active = 0
    peak = 0

    class Completions:
        async def create(self, **kw: Any) -> Any:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return _response(None, [])

    class Client:
        chat = _Obj(completions=Completions())

    agent, tools, obs, _events = _setup()
    policy = LLMPolicy(Client(), "m", vision=False, max_concurrency=1)  # type: ignore[arg-type]

    async def main() -> None:
        await asyncio.gather(*(policy.act(agent, tools, obs) for _ in range(4)))

    asyncio.run(main())
    assert peak == 1


def test_regions_report_busiest_tiles_with_dominant_colour() -> None:
    canvas = Canvas(128, 64)
    for x in range(10):
        canvas.write(PixelWrite(x=x, y=0, color=5, agent_id=0))
    canvas.write(PixelWrite(x=70, y=10, color=2, agent_id=0))
    canvas.write(PixelWrite(x=0, y=0, color=0, agent_id=0))
    regions = canvas.regions()
    assert [r["x0"] for r in regions] == [0, 64]
    assert regions[0]["painted"] == 9 and regions[0]["color"] == 5
    assert regions[0]["x1"] == 63 and regions[0]["y1"] == 63
    every = canvas.regions(limit=None)
    assert len(every) == 2 and every[-1]["size"] == 64 * 64
    narrow = Canvas(70, 64)
    for x in range(64, 70):
        for y in range(64):
            narrow.write(PixelWrite(x=x, y=y, color=5, agent_id=0))
    edge = next(t for t in narrow.regions(limit=None) if t["x0"] == 64)
    assert edge["size"] == 6 * 64 and edge["painted"] == edge["size"]
    assert not any(t["x0"] == 64 for t in Canvas(70, 64).regions(limit=None) if t["color"] != 0)


def test_pick_threads_mixes_top_and_random_sample() -> None:
    board = [{"thread_id": i} for i in range(10)]
    picked = pick_threads(board, random.Random(1), k=6)
    assert [d["thread_id"] for d in picked[:2]] == [0, 1]
    assert len(picked) == 6 and len({d["thread_id"] for d in picked}) == 6


def test_pick_threads_keeps_tagged_entries_before_recent_and_random_threads() -> None:
    board: list[dict[str, Any]] = [
        {"thread_id": 0, "tag": "dispatch"},
        {"thread_id": 1},
        {"thread_id": 2, "tag": "announcement"},
        {"thread_id": 3},
        {"thread_id": 4, "tag": "mention"},
        {"thread_id": 5},
        {"thread_id": 6},
    ]
    picked = pick_threads(board, random.Random(1), k=5)
    assert [entry["thread_id"] for entry in picked[:3]] == [0, 2, 4]
    assert [entry["thread_id"] for entry in picked[3:5]] == [1, 3]


def test_prompt_marks_world_news_and_hides_post_tools_without_budget() -> None:
    agent, _, obs, _ = _setup()
    obs.board = [
        {
            "thread_id": 7,
            "title": "⚔️ India vs Japan",
            "author": "World News",
            "replies": 0,
            "latest": {"author": "World News", "body": "India and Japan are at war."},
            "tag": "dispatch",
        }
    ]
    obs.budget.posts = 0
    prompt = user_prompt(agent, obs, "canvas")
    assert "[WORLD NEWS] - thread 7" in prompt
    assert "Budget: 4 pixels, no posts this turn (you posted recently)." in prompt
    assert "Hand of God" not in prompt
    assert "Hand of God" not in system_prompt(agent, 64, 40, 4, 1)
    policy = LLMPolicy(FakeClient([]), "m", vision=False)  # type: ignore[arg-type]
    names = [tool["function"]["name"] for tool in policy._tools_for(obs)]
    assert "post_thread" not in names and "reply" not in names


def test_invented_tool_names_map_onto_real_tools() -> None:
    assert _canonical_tool("put_pixels") == "place_pixels"
    assert _canonical_tool("draw_image") == "view_canvas"
    assert _canonical_tool("see_canvas") == "view_canvas"
    assert _canonical_tool("reply") == "reply"
    assert _canonical_tool("view_thread") == "read_thread"
    assert _canonical_tool("teleport") == "teleport"


def test_successful_pixel_calls_emit_action_events() -> None:
    canvas, board = Canvas(64, 40), Board()
    events: list[tuple[str, dict[str, Any]]] = []
    agent = AgentSpec(1, "Priya Raman", "India", 5, 10, 10, 1.0, "Priya.")
    tools = ToolSet(
        canvas, board, 1, TurnBudget(pixels=40), lambda k, p: events.append((k, p)), lambda: 0.0
    )
    obs = Observation(1.0, tools.budget, 64, 40, [])
    client = FakeClient(
        [
            _response(
                "Painting.",
                [("place_pixels", {"pixels": [{"x": 2, "y": 2, "color": 5}]})],
            )
        ]
    )
    policy = LLMPolicy(client, "m", vision=False)  # type: ignore[arg-type]
    asyncio.run(policy.act(agent, tools, obs))
    actions = [p for k, p in events if k == "action"]
    assert len(actions) == 1 and actions[0]["tool"] == "place_pixels"
    assert actions[0]["args"] == {"count": 1, "x": 2, "y": 2}
    assert actions[0]["result"]["placed"] == 1
