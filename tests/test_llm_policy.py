import asyncio
import json
from typing import Any

from worldwarpixel.sim.agents import AgentSpec, Observation
from worldwarpixel.sim.board import Board
from worldwarpixel.sim.canvas import Canvas
from worldwarpixel.sim.llm_policy import LLMPolicy, system_prompt
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
        canvas, board, 1, TurnBudget(pixels=4), lambda k, p: events.append((k, p)), lambda: 0.0
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
