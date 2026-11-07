import asyncio
import json
import struct
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.agents import RandomWalkPolicy
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
    pixel_interval=0,
    seed=1,
    stage_lines=4,
    stage_vote_seconds=3,
)


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


def make_run(model: Any = None, speech: Any = None, config: RunConfig = CONFIG) -> Run:
    run = Run(
        "s",
        config.model_copy(update={"policy": "llm"}) if model else config,
        EventLog(),
        RandomWalkPolicy(seed=1),
        narrator_model=model,
        speech=speech,
    )
    run.narrator.interval = 3600
    run.stage.line_gap = 0.0
    return run


async def wait_for(cond: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def test_debate_alternates_voices_votes_and_announces_the_verdict() -> None:
    async def main() -> None:
        model, speech = FakeModel(), FakeSpeech()
        run = make_run(model, speech)
        run.stage.vote_seconds = 0.2
        queue = run.stage.hub.subscribe()
        await run.start()
        state = run.stage.start_debate(1, 2, "Who owns the top-left?")
        assert state["phase"] == "debating" and [d["id"] for d in state["debaters"]] == [1, 2]
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
        debate = [ln for ln in lines if ln["kind"] == "debate"]
        assert [ln["speaker_id"] for ln in debate] == [1, 2, 1, 2]
        assert [ln["podium"] for ln in debate] == [0, 1, 0, 1]
        assert debate[0]["text"] == "Line number 1!" and debate[0]["emotion"] == "smug"
        assert all(ln["audio"] and run.stage.audio(ln["id"]) for ln in lines)
        verdict = lines[-1]
        assert verdict["kind"] == "verdict" and verdict["podium"] is None
        assert "3 votes to 0" in verdict["text"]
        # Rebuttals see the debate so far; the opening line does not.
        assert "You open the debate." in str(model.calls[0][-1]["content"])
        assert "Line number 1!" in str(model.calls[1][-1]["content"])
        # Every agent hears the verdict through a pinned announcement.
        threads = run.board.listing(limit=10, name_of=run.agent_name)
        assert any(t["author_id"] == GOD_ID and "verdict" in str(t["title"]) for t in threads)
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
        await wait_for(lambda: len(run.stage.transcript) >= 1)
        run.stage.narrate("  Brazil just  painted over everyone!  ", "Megan")
        run.stage.react("boo", 2)
        await wait_for(lambda: any(ln["kind"] == "narrator" for ln in run.stage.transcript))
        await wait_for(lambda: run.stage.transcript[-1]["kind"] == "debate")
        await run.stop()
        narr = next(ln for ln in run.stage.transcript if ln["kind"] == "narrator")
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
        run.stage.observe_write(3, "pixel", {"x": 10 + i, "y": 12, "prev_owner": 5})
    run.stage.observe_write(5, "pixel", {"x": 40, "y": 4, "prev_owner": 3})
    run.stage.observe_write(1, "pixel", {"x": 1, "y": 1, "prev_owner": 2})
    run.stage.observe_write(1, "pixel", {"x": 1, "y": 1, "prev_owner": GOD_ID})
    run.stage.observe_write(1, "pixel", {"x": 1, "y": 1, "prev_owner": 0})
    a, b, topic, region = run.stage.pick_rivals()
    assert (a, b) == (3, 5)
    assert "(14, 12)" in topic and region == {"x": 0, "y": 0, "w": 48, "h": 48}


def test_fallback_lines_without_model_or_speech() -> None:
    async def main() -> None:
        run = make_run()
        run.thoughts[1].append({"ts": 0, "thought": "Defending my flag. [draw_flag at (3, 4)]"})
        await run.start()
        run.stage.start_debate(1, 2)
        await wait_for(lambda: len(run.stage.transcript) >= 1)
        await run.stop()
        line = run.stage.transcript[0]
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
        await wait_for(lambda: len(run.stage.transcript) >= 2)
        await run.stop()
        first, second = run.stage.transcript[0], run.stage.transcript[1]
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
        assert line["type"] == "stage.line" and line["speaker_id"] == 1 and not line["audio"]

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
