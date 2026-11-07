import asyncio
import json
import struct
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.agents import RandomWalkPolicy
from worldwarpixel.sim.canvas import Canvas, PixelWrite
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.narrator import (
    GRID,
    NarratorUnavailable,
    bucket_span,
    canvas_png,
    parse_reply,
)
from worldwarpixel.sim.run import Run, RunConfig

CONFIG = RunConfig(
    width=64, height=32, agents=5, cooldown=0.01, jitter=0, pixels_per_turn=4, seed=1
)

REPLY = {
    "headline": "Five painters, one corner",
    "bullets": ["Everyone is piling into the top-left.", "Threads are all territorial claims."],
    "highlights": [
        {"type": "agent", "id": 1, "label": "the leader"},
        {"type": "agent", "id": 99},
        {"type": "thread", "id": 1},
        {"type": "thread", "id": 4242},
        {"type": "region", "x": 48, "y": 16, "w": 100, "h": 100},
        {"type": "region", "x": 200, "y": 0, "w": 10, "h": 10},
        "garbage",
    ],
}


class FakeNarratorModel:
    def __init__(self, reply: str | None = None, fail: bool = False) -> None:
        self.reply = reply if reply is not None else json.dumps(REPLY)
        self.fail = fail
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(self, messages: Any) -> str:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("boom")
        return self.reply


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    content = messages[-1]["content"]
    return str(content if isinstance(content, str) else content[0]["text"])


def test_periodic_update_is_logged_published_and_snapshotted() -> None:
    async def main() -> None:
        log = EventLog()
        model = FakeNarratorModel()
        run = Run("n", CONFIG, log, RandomWalkPolicy(seed=1, post_chance=1.0), narrator_model=model)
        run.narrator.interval = 0.15
        queue = run.hub.subscribe()
        await run.start()
        await asyncio.sleep(0.3)
        await run.stop()

        assert model.calls, "narrator never asked the model"
        text = _prompt_text(model.calls[0])
        assert "Most active agents:" in text and "New posts (oldest first):" in text
        assert "Canvas regions with most change" in text
        parts = model.calls[0][-1]["content"]
        assert isinstance(parts, list) and parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

        updates = run.narrator.history
        assert updates and updates[0]["kind"] == "update" and updates[0]["source"] == "model"
        first = updates[0]
        assert first["headline"] == REPLY["headline"] and len(first["bullets"]) == 2
        assert first["window"]["turns"] > 0 and first["window"]["pixels"] > 0
        assert first["highlights"] == [
            {"type": "agent", "id": 1, "label": "the leader"},
            {"type": "thread", "id": 1, "label": run.board.thread(1)[0].title},
            {"type": "region", "x": 48, "y": 16, "w": 16, "h": 16, "label": "(48, 16) 16x16"},
        ]
        logged = [e for e in log.read("n", limit=10_000) if e.kind == "narrator.update"]
        assert [e.payload["id"] for e in logged] == [u["id"] for u in updates]
        published = [
            json.loads(m) for m in _drain(queue) if isinstance(m, str) and '"narrator.update"' in m
        ]
        assert published and published[0]["headline"] == REPLY["headline"]
        snap = run.snapshot()["narrator"]
        assert snap["updates"] == list(updates) and snap["interval"] == 0.15
        assert run.summary()["narrator"]["updates"] == len(updates)
        # The second update only covers what happened after the first.
        if len(model.calls) > 1:
            assert "Your previous headlines:" in _prompt_text(model.calls[1])

    asyncio.run(main())


def _drain(queue: asyncio.Queue[bytes | str]) -> list[bytes | str]:
    out: list[bytes | str] = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def test_digest_tracks_overwrites_and_regions() -> None:
    async def main() -> None:
        model = FakeNarratorModel()
        run = Run("d", CONFIG, EventLog(), RandomWalkPolicy(seed=1), narrator_model=model)
        for i in range(3):
            run._on_write(1, "pixel", {"x": i, "y": 0, "color": 5, "prev_owner": 0})
        for i in range(2):
            run._on_write(2, "pixel", {"x": i, "y": 0, "color": 6, "prev_owner": 1})
        run._on_write(2, "pixel", {"x": 60, "y": 30, "color": 6, "prev_owner": 2})
        digest = run.narrator.build_digest()
        assert digest["pixels"] == 6 and digest["claimed_blank"] == 3
        assert digest["overwrites"] == [
            {
                "by": 2,
                "by_name": run.agent_name(2),
                "over": 1,
                "over_name": run.agent_name(1),
                "pixels": 2,
            }
        ]
        assert digest["regions"][0] == {"x": 0, "y": 0, "w": 16, "h": 8, "changes": 5}
        assert digest["regions"][1] == {"x": 48, "y": 24, "w": 16, "h": 8, "changes": 1}
        text = run.narrator.render_digest(digest)
        assert f"{run.agent_name(2)} (agent 2) painted over {run.agent_name(1)}" in text

    asyncio.run(main())


def test_fallback_when_model_is_missing_or_fails() -> None:
    async def main() -> None:
        run = Run("f", CONFIG, EventLog(), RandomWalkPolicy(seed=1))
        run._on_write(3, "pixel", {"x": 1, "y": 1, "color": 5, "prev_owner": 0})
        run.narrator.observe_turn(3)
        update = await run.narrator.tick()
        assert update["source"] == "fallback" and update["error"] is None
        assert update["headline"].startswith("1 turns, 1 pixels")
        assert update["highlights"][0] == {
            "type": "agent",
            "id": 3,
            "label": f"{run.agent_name(3)} ({run.by_id[3].country})",
        }
        assert update["highlights"][1]["type"] == "region"
        with pytest.raises(NarratorUnavailable):
            await run.narrator.ask("anyone there?")

        failing = FakeNarratorModel(fail=True)
        run2 = Run("g", CONFIG, EventLog(), RandomWalkPolicy(seed=1), narrator_model=failing)
        update = await run2.narrator.tick()
        assert update["source"] == "fallback" and "RuntimeError" in update["error"]
        with pytest.raises(NarratorUnavailable, match="RuntimeError"):
            await run2.narrator.ask("still there?")
        assert run2.narrator.stats == {"updates": 1, "answers": 0, "failures": 2}

    asyncio.run(main())


def test_ask_uses_the_question_and_does_not_reset_the_window() -> None:
    async def main() -> None:
        model = FakeNarratorModel(json.dumps({"headline": "Yes", "bullets": ["Because."]}))
        run = Run("q", CONFIG, EventLog(), RandomWalkPolicy(seed=1), narrator_model=model)
        run._on_write(1, "pixel", {"x": 1, "y": 1, "color": 5, "prev_owner": 0})
        answer = await run.narrator.ask("  Is anyone winning?  ")
        assert answer["kind"] == "answer" and answer["question"] == "Is anyone winning?"
        assert answer["headline"] == "Yes" and answer["bullets"] == ["Because."]
        assert "A viewer asks: Is anyone winning?" in _prompt_text(model.calls[0])
        assert run.narrator.build_digest()["pixels"] == 1
        assert run.narrator.stats["answers"] == 1

    asyncio.run(main())


def test_parse_reply_tolerates_fences_and_plain_text() -> None:
    fenced = '```json\n{"headline": "H", "bullets": ["a", "b"], "highlights": []}\n```'
    assert parse_reply(fenced) == ("H", ["a", "b"], [])
    chatty = 'Sure! {"headline": "H2", "bullets": "not a list"} hope that helps'
    assert parse_reply(chatty) == ("H2", [], [])
    plain = "The board is quiet.\n- Nobody posted.\n- Paint keeps landing.\n"
    bullets = ["Nobody posted.", "Paint keeps landing."]
    assert parse_reply(plain) == ("The board is quiet.", bullets, [])
    assert parse_reply("") == ("(no response)", [], [])


def test_canvas_png_is_downsampled() -> None:
    canvas = Canvas(300, 100)
    canvas.write(PixelWrite(0, 0, 5, 1))
    png = canvas_png(canvas, max_side=128)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert png[12:16] == b"IHDR"
    assert struct.unpack(">II", png[16:24]) == (100, 34)
    assert png.endswith(b"IEND\xaeB`\x82")
    assert struct.unpack(">II", canvas_png(Canvas(64, 32))[16:24]) == (64, 32)


SMALL = {
    "width": 32,
    "height": 16,
    "agents": 3,
    "cooldown": 0.01,
    "jitter": 0,
    "seed": 7,
    "policy": "random",
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EVENT_LOG_PATH", ":memory:")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_narrator_endpoints(client: TestClient) -> None:
    manager = client.app.state.runs  # type: ignore[attr-defined]
    manager.narrator_model = FakeNarratorModel()
    manager.make_policy = lambda _cfg: RandomWalkPolicy(seed=1)
    run_id = client.post("/runs", json={**SMALL, "policy": "llm"}).json()["id"]

    state = client.get(f"/runs/{run_id}/narrator").json()
    assert state["updates"] == [] and state["interval"] == 60
    assert client.post(f"/runs/{run_id}/narrator/ask", json={"question": ""}).status_code == 422

    with client.websocket_connect(f"/runs/{run_id}/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["narrator"]["interval"] == 60
        ws.receive_bytes()
        answer = client.post(f"/runs/{run_id}/narrator/ask", json={"question": "Who leads?"})
        assert answer.status_code == 200
        body = answer.json()
        assert body["kind"] == "answer" and body["headline"] == REPLY["headline"]
        while True:
            msg = ws.receive()
            if msg.get("text") and '"narrator.update"' in msg["text"]:
                assert json.loads(msg["text"])["id"] == body["id"]
                break

    assert [u["id"] for u in client.get(f"/runs/{run_id}/narrator").json()["updates"]] == [
        body["id"]
    ]
    events = client.get(f"/runs/{run_id}/events", params={"limit": 10_000}).json()
    assert any(e["kind"] == "narrator.update" for e in events)

    manager.narrator_model = FakeNarratorModel(fail=True)
    failing = client.post("/runs", json=SMALL).json()["id"]
    res = client.post(f"/runs/{failing}/narrator/ask", json={"question": "hello?"})
    assert res.status_code == 503 and "narrator unavailable" in res.json()["detail"]

    client.post(f"/runs/{run_id}/stop")
    res = client.post(f"/runs/{run_id}/narrator/ask", json={"question": "done?"})
    assert res.status_code == 409
    assert client.get("/runs/nope/narrator").status_code == 404
    for r in client.get("/runs").json():
        client.delete(f"/runs/{r['id']}")


def test_parse_reply_salvages_truncated_json() -> None:
    text = (
        '{\n  "headline": "Noise spreads across the grid",\n  "bullets": [\n'
        '    "Agent 12 painted at (5, 5)",\n    "Agent 12 painted at (5, 5)",\n'
        '    "Agent 3 replied in thread 4",\n    "Agent 7 ov'
    )
    headline, bullets, highlights = parse_reply(text)
    assert headline == "Noise spreads across the grid"
    assert bullets == ["Agent 12 painted at (5, 5)", "Agent 3 replied in thread 4"]
    assert highlights == []


def test_bucket_span_matches_bucket_math_for_odd_sizes() -> None:
    for size in (7, 128, 130, 129, 1000):
        for x in range(size):
            start, end = bucket_span(x * GRID // size, size)
            assert start <= x < end
        assert bucket_span(0, size)[0] == 0 and bucket_span(GRID - 1, size)[1] == size


def test_random_runs_do_not_call_the_narrator_model() -> None:
    config = CONFIG.model_copy(update={"policy": "random"})
    run = Run("r", config, EventLog(), RandomWalkPolicy(seed=1), narrator_model=FakeNarratorModel())
    assert run.narrator._model is None
