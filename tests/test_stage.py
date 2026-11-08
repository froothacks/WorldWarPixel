import asyncio
import json
import struct
import time
from collections.abc import Iterator
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.agents import RandomWalkPolicy
from worldwarpixel.sim.canvas import PALETTE, PixelWrite
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.run import GOD_ID, Run, RunConfig
from worldwarpixel.sim.stage import (
    SAMPLE_RATE,
    StageBusy,
    clean_thought,
    parse_line,
    voice_for,
    wav,
)

CONFIG = RunConfig(
    width=64,
    height=32,
    agents=6,
    cooldown=0.01,
    jitter=0,
    pixels_per_turn=4,
    seed=1,
    stage_lines=4,
    stage_vote_seconds=3,
).model_copy(update={"pixel_interval": 0})


class FakeModel:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(self, messages: Any) -> str:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("boom")
        return json.dumps({"line": f"Line number {len(self.calls)}!", "emotion": "smug"})


class FakeSpeech:
    sample_rate = SAMPLE_RATE

    def __init__(self, seconds: float = 0.02) -> None:
        self.seconds = seconds
        self.calls: list[tuple[str, str, str]] = []

    async def synth(self, text: str, voice: str, instructions: str) -> bytes:
        self.calls.append((text, voice, instructions))
        return b"\x00\x00" * int(self.sample_rate * self.seconds)


def make_run(
    model: Any = None,
    speech: Any = None,
    config: RunConfig = CONFIG,
    clock: Any = time.time,
) -> Run:
    run = Run(
        "s",
        config.model_copy(update={"policy": "llm"}) if model else config,
        EventLog(),
        RandomWalkPolicy(seed=1),
        clock=clock,
        narrator_model=model,
        speech=speech,
    )
    run.narrator.interval = 3600
    run.stage.line_gap = 0.0
    return run


def test_swap_model_updates_stage_fallback_but_keeps_explicit_override() -> None:
    original = FakeModel()
    replacement = FakeModel()
    run = make_run(model=original)
    assert run.stage._model is original

    run.swap_model(RandomWalkPolicy(seed=2), replacement, "deepseek")
    assert run.stage._model is replacement

    override = FakeModel()
    run = Run(
        "override",
        CONFIG.model_copy(update={"policy": "llm"}),
        EventLog(),
        RandomWalkPolicy(seed=1),
        narrator_model=original,
        stage_model=override,
    )
    run.swap_model(RandomWalkPolicy(seed=3), replacement, "deepseek")
    assert run.stage._model is override


async def wait_for(cond: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def test_auto_stage_with_one_agent_waits_before_retrying() -> None:
    async def main() -> None:
        run = make_run(config=CONFIG.model_copy(update={"agents": 1}))
        run.stage.auto = True

        await asyncio.wait_for(run.stage._step(), 3)

        assert run.stage.phase == "idle"
        assert run.stage._next_auto_at > run.stage._clock()

    asyncio.run(main())


def test_old_verdict_does_not_end_a_new_debate() -> None:
    async def main() -> None:
        run = make_run(speech=FakeSpeech(seconds=0.5))
        run.stage.lines_per_debate = 1
        run.stage.vote_seconds = 0.01
        await run.start()
        old_state = run.stage.start_debate(1, 2, "old topic")
        old_thread_id = old_state["thread_id"]

        await wait_for(lambda: run.stage.phase == "voting")
        await wait_for(lambda: any(line["kind"] == "verdict" for line in run.stage.transcript))

        old_debate = run.stage.debate_id
        run.stage.end_debate()
        run.stage.lines_per_debate = 10
        new_state = run.stage.start_debate(3, 4, "new topic")
        new_thread_id = new_state["thread_id"]
        new_debate = run.stage.debate_id
        await asyncio.sleep(0.6)
        phase, debaters, debate_id = (
            run.stage.phase,
            run.stage.debaters,
            run.stage.debate_id,
        )
        await run.stop()

        assert new_debate != old_debate
        assert phase == "debating"
        assert debaters == [3, 4]
        assert debate_id == new_debate
        announcements = run.recent_announcements()
        assert announcements[-1]["thread_id"] == old_thread_id
        assert "old topic" in announcements[-1]["body"]
        assert not any(
            announcement["thread_id"] == new_thread_id
            and announcement["title"] == "Live debate verdict"
            for announcement in announcements
        )

    asyncio.run(main())


def test_end_debate_discards_queued_intro_but_keeps_audience_interjections() -> None:
    run = make_run()
    run.stage._interjections.append({"name": "Audience", "text": "Boo!"})
    run.stage.start_debate(1, 2)

    assert any("Debate #" in item["text"] for item in run.stage._interjections)
    run.stage.end_debate()

    assert not any("Debate #" in item["text"] for item in run.stage._interjections)
    assert list(run.stage._interjections) == [{"name": "Audience", "text": "Boo!"}]
    assert run.stage._interjections.maxlen == 10


def test_reactions_during_playback_reach_a_later_debate_prompt() -> None:
    async def main() -> None:
        model = FakeModel()
        run = make_run(model, FakeSpeech(seconds=0.2))
        await run.start()
        run.stage.start_debate(1, 2)
        await wait_for(lambda: any(line["kind"] == "debate" for line in run.stage.transcript))
        assert run.stage.transcript[0]["kind"] == "narrator"
        assert run.stage.transcript[1]["speaker_id"] == 1

        calls_before_reaction = len(model.calls)
        run.stage.react("boo", 2)
        await wait_for(lambda: len(model.calls) >= calls_before_reaction + 2)
        later_calls = model.calls[calls_before_reaction:]
        await run.stop()

        assert any("boo" in str(call[-1]["content"]) for call in later_calls)

    asyncio.run(main())


def test_debate_alternates_voices_votes_and_announces_the_verdict() -> None:
    async def main() -> None:
        model, speech = FakeModel(), FakeSpeech()
        run = make_run(model, speech)
        run.stage.vote_seconds = 0.2
        queue = run.stage.hub.subscribe()
        await run.start()
        state = run.stage.start_debate(1, 2, "Who owns the top-left?")
        assert state["phase"] == "debating" and [d["id"] for d in state["debaters"]] == [1, 2]
        assert state["thread_id"] is not None
        with pytest.raises(StageBusy):
            run.stage.start_debate()
        await wait_for(lambda: run.stage.phase == "voting")
        run.stage.vote("v1", 2)
        run.stage.vote("v2", 2)
        run.stage.vote("v3", 1)
        run.stage.vote("v3", 2)  # changed their mind: one vote per voter
        assert run.stage.tally() == {1: 0, 2: 3}
        with pytest.raises(ValueError):
            run.stage.vote("v4", 3)
        await wait_for(lambda: run.stage.phase == "idle")
        await run.stop()

        lines = list(run.stage.transcript)
        intro = lines[0]
        assert intro["kind"] == "narrator" and intro["name"] == "Narrator"
        assert intro["text"].startswith(
            f"Debate #1: {run.by_id[1].name} of {run.by_id[1].country} "
            f"versus {run.by_id[2].name} of {run.by_id[2].country}."
        )
        debate = [ln for ln in lines if ln["kind"] == "debate"]
        assert [ln["speaker_id"] for ln in debate] == [1, 2, 1, 2]
        assert [ln["podium"] for ln in debate] == [0, 1, 0, 1]
        assert debate[0]["text"] == "Line number 1!" and debate[0]["emotion"] == "smug"
        assert all(ln["audio"] and run.stage.audio(ln["id"]) for ln in lines)
        verdict = lines[-1]
        assert verdict["kind"] == "verdict" and verdict["podium"] is None
        assert "3 votes to 0" in verdict["text"]
        # Rebuttals see the narrator's introduction and the debate so far.
        assert "Debate #1:" in str(model.calls[0][-1]["content"])
        assert "Line number 1!" in str(model.calls[1][-1]["content"])
        thread = run.board.thread(state["thread_id"])
        assert thread[0].author_id == GOD_ID
        debate_replies = [post for post in thread[1:] if post.body == debate[0]["text"]]
        assert debate_replies and debate_replies[0].author_id == debate[0]["speaker_id"]
        verdict_reply = next(post for post in thread[1:] if verdict["text"] in post.body)
        assert verdict_reply.author_id == GOD_ID
        assert any(
            item["thread_id"] == state["thread_id"] and item["title"] == "Live debate verdict"
            for item in run.recent_announcements()
        )
        assert run.stage.stats["debates"] == 1

        types = []
        while not queue.empty():
            msg = queue.get_nowait()
            assert isinstance(msg, str)
            types.append(json.loads(msg)["type"])
        assert {"stage.state", "stage.line", "stage.votes", "stage.verdict"} <= set(types)
        kinds = {e.kind for e in run.log.read("s", limit=100_000)}
        assert {"stage.debate", "stage.line", "stage.verdict"} <= kinds

    asyncio.run(main())


def test_debaters_only_paint_after_they_speak() -> None:
    async def main() -> None:
        run = make_run(speech=FakeSpeech(seconds=0.4))
        await run.start()
        run.stage.start_debate(1, 2)
        before = {a: len(run.thoughts[a]) for a in (1, 2, 3)}
        await asyncio.sleep(0.5)
        after = {a: len(run.thoughts[a]) for a in (1, 2, 3)}
        run.stage.end_debate()
        await asyncio.sleep(0.1)
        released = len(run.thoughts[1])
        await run.stop()
        assert after[3] - before[3] > 5, "the swarm keeps its own clock"
        assert after[1] - before[1] <= 2 and after[2] - before[2] <= 2
        assert released > after[1], "ending the debate releases the debaters"

    asyncio.run(main())


def test_narrator_interjection_is_spoken_next_and_debaters_hear_it() -> None:
    async def main() -> None:
        model = FakeModel()
        run = make_run(model, FakeSpeech(seconds=0.1))
        await run.start()
        run.stage.start_debate(1, 2)
        await wait_for(lambda: any(ln["kind"] == "debate" for ln in run.stage.transcript))
        run.stage.narrate("  Brazil just  painted over everyone!  ", "Megan")
        run.stage.react("boo", 2)
        await wait_for(
            lambda: any(
                ln["text"] == "Brazil just painted over everyone!" for ln in run.stage.transcript
            )
        )
        await wait_for(
            lambda: (
                run.stage.transcript[-1]["kind"] == "debate"
                and run.stage.transcript[-2]["text"] == "Brazil just painted over everyone!"
            )
        )
        await run.stop()
        narr = next(
            ln for ln in run.stage.transcript if ln["text"] == "Brazil just painted over everyone!"
        )
        assert narr["name"] == "Megan" and narr["text"] == "Brazil just painted over everyone!"
        idx = list(run.stage.transcript).index(narr)
        assert run.stage.transcript[idx - 1]["kind"] == "debate"
        assert any(
            "Megan: Brazil just painted over everyone!" in str(c[-1]["content"])
            for c in model.calls
        )
        with pytest.raises(ValueError):
            run.stage.react("throw tomatoes", None)

    asyncio.run(main())


def test_rivals_are_the_pair_that_overwrote_each_other_most() -> None:
    run = make_run()
    for i in range(5):
        run.narrator.observe_write(3, "pixel", {"x": 10 + i, "y": 12, "prev_owner": 5})
    run.narrator.observe_write(5, "pixel", {"x": 40, "y": 4, "prev_owner": 3})
    run.narrator.observe_write(1, "pixel", {"x": 1, "y": 1, "prev_owner": 2})
    run.narrator.observe_write(1, "pixel", {"x": 1, "y": 1, "prev_owner": GOD_ID})
    run.narrator.observe_write(1, "pixel", {"x": 1, "y": 1, "prev_owner": 0})
    assert run.narrator.rivalry[(3, 5)] == 5
    assert run.narrator.rivalry[(5, 3)] == 1
    a, b, topic, region = run.stage.pick_rivals()
    assert (a, b) == (3, 5)
    assert "(14, 12)" in topic and region == {"x": 0, "y": 0, "w": 48, "h": 32}
    assert run.narrator.rivalry[(3, 5)] == 2


def test_rivalry_region_is_clamped_at_the_canvas_edges() -> None:
    run = make_run()
    run.narrator.rivalry[(1, 2)] = 1
    run.narrator.rivalry_at[(1, 2)] = (63, 31)

    _, _, _, region = run.stage.pick_rivals()

    assert region == {"x": 16, "y": 0, "w": 48, "h": 32}
    assert region["x"] >= 0 and region["y"] >= 0
    assert region["x"] + region["w"] <= run.canvas.width
    assert region["y"] + region["h"] <= run.canvas.height


def test_small_canvas_rivalry_region_renders_debater_corner_pixel() -> None:
    run = make_run(config=CONFIG.model_copy(update={"width": 16, "height": 16}))
    run.narrator.rivalry[(1, 2)] = 1
    run.narrator.rivalry_at[(1, 2)] = (15, 15)

    state = run.stage.start_debate()
    assert state["region"] == {"x": 0, "y": 0, "w": 16, "h": 16}
    run.canvas.write(PixelWrite(15, 15, 5, 1))

    frame = run.stage.frame(16, 16)
    expected_rgb = tuple(int(PALETTE[5][i : i + 2], 16) for i in (1, 3, 5))
    red, green, blue = expected_rgb
    expected = ((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3)
    assert len(frame) == 16 * 16 * 2
    assert struct.unpack_from("<H", frame, (15 * 16 + 15) * 2)[0] == expected
    assert expected != 0


def test_verdict_holds_its_region_against_other_agents_until_expiry() -> None:
    async def main() -> None:
        now = [100.0]
        config = CONFIG.model_copy(update={"stage_hold_seconds": 5})
        run = make_run(
            speech=FakeSpeech(),
            config=config,
            clock=lambda: now[0],
        )

        class FixedPolicy:
            def __init__(self) -> None:
                self.results: dict[str, dict[str, Any]] = {}

            async def act(self, agent: Any, tools: Any, obs: Any) -> str:
                if agent.id == 1:
                    self.results["holder"] = tools.place_pixels([(10, 10, 5)])
                elif agent.id == 2:
                    self.results["blocked"] = tools.place_pixels([(11, 11, 6)])
                    self.results["outside"] = tools.place_pixels([(12, 12, 6), (30, 20, 7)])
                return "painted"

        policy = FixedPolicy()
        run.policy = policy
        run.stage.start_debate(1, 2)
        run.stage.region = {"x": 10, "y": 10, "w": 10, "h": 10}
        run.stage.phase = "voting"
        run.stage.vote("viewer", 1)

        async def no_wait(_: float) -> None:
            return None

        run.stage._sleep = no_wait
        await run.stage._verdict()
        verdict = next(line for line in run.stage.transcript if line["kind"] == "verdict")
        assert "holds that ground for 5s." in verdict["text"]
        assert run.stage.hold_state() == {
            "agent_id": 1,
            "region": {"x": 10, "y": 10, "w": 10, "h": 10},
            "until": 105.0,
            "seconds_left": 5,
        }

        await run._turn(run.by_id[1], 0)
        await run._turn(run.by_id[2], 0)
        assert policy.results["blocked"]["placed"] == 0
        assert policy.results["blocked"]["blocked"] == 1
        assert policy.results["blocked"]["pixels_left"] == config.pixels_per_turn
        assert policy.results["outside"]["blocked"] == 1
        assert policy.results["outside"]["last"] == [30, 20]
        assert policy.results["holder"]["placed"] == 1
        assert "blocked" not in policy.results["holder"]
        assert run.canvas.owner(10, 10) == 1
        assert run.canvas.owner(11, 11) == 0
        assert run.canvas.owner(30, 20) == 2

        run._running.set()
        pending = {(12, 12): (8, 3)}
        queue: asyncio.Queue[PixelWrite] = asyncio.Queue()
        queue.put_nowait(PixelWrite(12, 12, 8, 3))
        painter = asyncio.create_task(run._painter(run.by_id[3], queue, pending))
        await asyncio.sleep(0.01)
        painter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await painter
        assert run.canvas.owner(12, 12) == 0 and (12, 12) not in pending

        now[0] += 6
        assert run.stage.hold_state() is None
        await run._turn(run.by_id[2], 0)
        assert run.canvas.owner(11, 11) == 2

    asyncio.run(main())


def test_fallback_lines_without_model_or_speech() -> None:
    async def main() -> None:
        run = make_run()
        run.thoughts[1].append({"ts": 0, "thought": "Defending my flag. [draw_flag at (3, 4)]"})
        run.stage.start_debate(1, 2)
        line, _ = await run.stage._compose(1)
        assert line["audio"] is False and line["duration"] >= 2.0
        assert "Defending my flag." in line["text"] and "draw_flag" not in line["text"]

    asyncio.run(main())


class EchoModel(FakeModel):
    async def complete(self, messages: Any) -> str:
        self.calls.append(messages)
        return json.dumps({"line": "The canvas is mine and you know it.", "emotion": "smug"})


def test_echoed_lines_are_retried_then_replaced() -> None:
    async def main() -> None:
        model = EchoModel()
        run = make_run(model, FakeSpeech())
        await run.start()
        run.stage.start_debate(1, 2)
        await wait_for(
            lambda: len([line for line in run.stage.transcript if line["kind"] == "debate"]) >= 2
        )
        await run.stop()
        first, second = [line for line in run.stage.transcript if line["kind"] == "debate"][:2]
        assert first["text"] == "The canvas is mine and you know it."
        assert second["text"] != first["text"]
        assert run.stage.stats["repeats"] >= 2 and len(model.calls) >= 3

    asyncio.run(main())


def test_rivals_with_the_same_voice_are_told_apart() -> None:
    run = make_run()
    a, b = run.agents[0], run.agents[1]
    a.persona_card["sex"] = b.persona_card["sex"] = "Male"
    run.stage.debaters = [a.id, b.id]
    if voice_for(a) == voice_for(b):
        assert run.stage._voice(b.id) != run.stage._voice(a.id)
    assert run.stage._voice(a.id) == voice_for(a)


def test_helpers() -> None:
    assert parse_line('```json\n{"line": " Hi  there ", "emotion": "ANGRY"}\n```') == (
        "Hi there",
        "angry",
    )
    assert parse_line('"Just text"') == ("Just text", "defiant")
    assert clean_thought("Going big. [draw_text 'HI']") == "Going big."
    assert clean_thought("(model call failed: Timeout)") == ""
    run = make_run()
    assert all(voice_for(a) == voice_for(a) for a in run.agents)
    data = wav(b"\x01\x00" * 10, 16000)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack("<I", data[24:28])[0] == 16000 and data[44:] == b"\x01\x00" * 10


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EVENT_LOG_PATH", ":memory:")
    monkeypatch.setenv("TTS_API_KEY", "")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_stage_api(client: TestClient) -> None:
    body = {"width": 32, "height": 16, "agents": 3, "cooldown": 0.05, "policy": "random"}
    run_id = client.post("/runs", json=body).json()["id"]
    assert client.get(f"/runs/{run_id}/stage").json()["phase"] == "idle"
    assert (
        client.post(f"/runs/{run_id}/stage/vote", json={"voter": "x", "agent_id": 1}).status_code
        == 409
    )
    assert client.post(f"/runs/{run_id}/stage/start", json={"a": 1, "b": 1}).status_code == 422
    assert client.post(f"/runs/{run_id}/stage/start", json={"a": 1}).status_code == 422

    with client.websocket_connect(f"/runs/{run_id}/stage/ws") as ws:
        snap = ws.receive_json()
        assert snap["type"] == "stage.snapshot" and snap["transcript"] == []
        started = client.post(
            f"/runs/{run_id}/stage/start", json={"a": 1, "b": 2, "topic": "Pineapple on pizza?"}
        ).json()
        assert started["topic"] == "Pineapple on pizza?" and started["speech"] is False
        assert client.post(f"/runs/{run_id}/stage/start", json={}).status_code == 409
        msg = ws.receive_json()
        assert msg["type"] == "stage.state" and msg["phase"] == "debating"
        line = ws.receive_json()
        assert (
            line["type"] == "stage.line"
            and line["speaker_id"] == 0
            and line["kind"] == "narrator"
            and not line["audio"]
        )

    vote = client.post(f"/runs/{run_id}/stage/vote", json={"voter": "x", "agent_id": 2})
    assert vote.json()["votes"] == {"1": 0, "2": 1}
    assert (
        client.post(f"/runs/{run_id}/stage/vote", json={"voter": "x", "agent_id": 3}).status_code
        == 422
    )
    assert client.post(f"/runs/{run_id}/stage/react", json={"reaction": "cheer"}).status_code == 204
    assert client.post(f"/runs/{run_id}/stage/react", json={"reaction": "nope"}).status_code == 422
    narrate = {"text": "Hello crowd", "name": "MC", "announce": True}
    assert client.post(f"/runs/{run_id}/stage/narrate", json=narrate).status_code == 202
    titles = [t["title"] for t in client.get(f"/runs/{run_id}/threads").json()]
    assert "Live narrator: MC" in titles
    assert client.get(f"/runs/{run_id}/stage/lines/1/audio").status_code == 404
    assert client.get(f"/runs/{run_id}/stage/lines/1/audio?format=mp3").status_code == 422
    stopped = client.post(f"/runs/{run_id}/stage/stop").json()
    assert stopped["phase"] == "idle" and stopped["auto"] is False
    assert client.get(f"/runs/{run_id}").json()["stage"]["debate_id"] == 1


def test_stage_frame_endpoint_returns_rgb565_with_debater_highlight(client: TestClient) -> None:
    run_id = client.post(
        "/runs",
        json={"width": 16, "height": 16, "agents": 3, "cooldown": 600, "policy": "random"},
    ).json()["id"]
    run = cast(FastAPI, client.app).state.runs.get(run_id)
    run.stage.start_debate(1, 2)
    run.stage.region = {"x": 0, "y": 0, "w": 2, "h": 1}
    run.canvas.write(PixelWrite(0, 0, 5, 1))
    run.canvas.write(PixelWrite(1, 0, 6, 3))

    response = client.get(f"/runs/{run_id}/stage/frame?w=32&h=16&podium=0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-width"] == "32" and response.headers["x-height"] == "16"
    assert len(response.content) == 32 * 16 * 2
    assert struct.unpack_from("<H", response.content, (8 * 32 + 8) * 2)[0] == 0xE000
    assert struct.unpack_from("<H", response.content, (8 * 32 + 24) * 2)[0] == 0x59C0


def test_stage_websocket_filters_podium_lines_and_tracks_devices(client: TestClient) -> None:
    run_id = client.post(
        "/runs",
        json={"width": 32, "height": 16, "agents": 2, "cooldown": 600, "policy": "random"},
    ).json()["id"]
    run = cast(FastAPI, client.app).state.runs.get(run_id)
    with client.websocket_connect(f"/runs/{run_id}/stage/ws?podium=0&narrator=1") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "stage.snapshot"
        assert snapshot["devices"] == {"0": 1, "1": 0}
        run.stage._publish("stage.line", id=1, podium=1, kind="debate", text="other podium")
        run.stage._publish("stage.line", id=2, podium=0, kind="debate", text="this podium")
        run.stage._publish("stage.line", id=3, podium=None, kind="narrator", text="host")
        run.stage._publish("stage.verdict", debate_id=1, winner_id=1)
        podium_line = ws.receive_json()
        narrator_line = ws.receive_json()
        verdict = ws.receive_json()
        assert podium_line["id"] == 2 and podium_line["type"] == "stage.line"
        assert narrator_line["id"] == 3 and narrator_line["type"] == "stage.line"
        assert verdict["type"] == "stage.verdict"
    assert run.stage.state()["devices"] == {"0": 0, "1": 0}


def test_stage_audio_endpoint() -> None:
    async def main() -> None:
        run = make_run(speech=FakeSpeech())
        await run.start()
        run.stage.start_debate(1, 2)
        await wait_for(lambda: len(run.stage.transcript) >= 1)
        await run.stop()
        line = run.stage.transcript[0]
        pcm = run.stage.audio(line["id"])
        assert pcm is not None and len(pcm) == 2 * int(SAMPLE_RATE * 0.02)
        assert line["sample_rate"] == SAMPLE_RATE and line["duration"] == 0.02

    asyncio.run(main())
