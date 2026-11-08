import asyncio
import contextlib
import json
import time
from collections.abc import Iterator
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.agents import RandomWalkPolicy
from worldwarpixel.sim.canvas import PALETTE
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.narrator import Narrator
from worldwarpixel.sim.run import Run, RunConfig
from worldwarpixel.sim.speech import (
    MAX_AUDIO_CLIPS,
    NARRATOR_VOICE,
    SAMPLE_RATE,
    Speech,
    voice_for,
)

PCM = b"\x01\x00" * SAMPLE_RATE


class FakeSpeech:
    sample_rate = SAMPLE_RATE

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str, str]] = []

    async def synth(self, text: str, voice: str, instructions: str) -> bytes:
        self.calls.append((text, voice, instructions))
        if self.fail:
            raise RuntimeError("synthesis failed")
        return PCM


CONFIG = RunConfig(
    width=64,
    height=32,
    agents=3,
    cooldown=600,
    jitter=0,
    pixels_per_turn=4,
    seed=1,
)


def make_run(synth: FakeSpeech | None = None) -> Run:
    return Run("speech-test", CONFIG, EventLog(), RandomWalkPolicy(seed=1), speech=synth)


def emit_line(speech: Speech, text: str, agent_id: int | None = None) -> dict[str, Any]:
    async def main() -> dict[str, Any]:
        speech._running.set()
        queue = speech.hub.subscribe()
        task = asyncio.create_task(speech.run())
        try:
            speech.say(text, agent_id)
            await asyncio.wait_for(speech._queue.join(), 2)
            message = queue.get_nowait()
            assert isinstance(message, str)
            return cast(dict[str, Any], json.loads(message))
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            speech.hub.unsubscribe(queue)

    return asyncio.run(main())


def test_agent_line_uses_agent_identity_voice_and_pcm_duration() -> None:
    synth = FakeSpeech()
    run = make_run(synth)
    agent = run.by_id[1]
    line = emit_line(run.speech, "Hello panel.", agent.id)

    assert line["type"] == "speech.line"
    assert line["speaker_id"] == agent.id
    assert line["name"] == agent.name
    assert line["color_hex"] == PALETTE[agent.color]
    assert line["voice"] == voice_for(agent)
    assert line["audio"] is True and line["duration"] == 1.0
    assert run.speech.audio(line["id"]) == PCM
    assert synth.calls[0][0] == "Hello panel."
    assert synth.calls[0][2] == (
        f"You are {agent.name} from {agent.country}. Speak naturally with an accent from "
        f"{agent.country}."
    )


def test_narrator_line_uses_narrator_voice() -> None:
    synth = FakeSpeech()
    run = make_run(synth)
    line = emit_line(run.speech, "A narrator line.")
    assert line["speaker_id"] == 0 and line["name"] == "Narrator"
    assert line["voice"] == NARRATOR_VOICE
    assert line["color_hex"] is None
    assert synth.calls[0][2] == (
        "You are the lively live commentator of a pixel-war game. Clear and energetic."
    )


def test_say_rejects_empty_oversized_and_unknown_agent_text() -> None:
    speech = make_run().speech
    with pytest.raises(ValueError, match="empty"):
        speech.say("  ")
    with pytest.raises(ValueError, match="280"):
        speech.say("x" * 281)
    with pytest.raises(ValueError, match="unknown agent"):
        speech.say("Hello", 999)


def test_synthesis_failure_still_publishes_a_text_line() -> None:
    run = make_run(FakeSpeech(fail=True))
    line = emit_line(run.speech, "Hello there.")
    assert line["type"] == "speech.line"
    assert line["audio"] is False and line["duration"] == 2.0
    assert run.speech.stats["failures"] == 1


def test_narrate_queues_headline_and_bullets_only_when_enabled() -> None:
    run = make_run()
    speech = Speech(run, None, run._running)
    speech.narrate({"headline": "Fresh paint.", "bullets": ["A repaired flag.", "A new thread."]})
    assert speech._queue.get_nowait() == (
        "Fresh paint. A repaired flag. A new thread.",
        None,
    )
    speech.narrate({"headline": "Failed.", "error": "model unavailable"})
    assert speech._queue.empty()

    disabled = Speech(run, None, run._running, narrate=False)
    disabled.narrate({"headline": "Not spoken.", "bullets": []})
    assert disabled._queue.empty()


def test_narrate_truncates_at_a_word_boundary() -> None:
    speech = make_run().speech
    speech.narrate({"headline": "start", "bullets": ["word " * 100]})
    text, agent_id = speech._queue.get_nowait()
    assert agent_id is None and len(text) <= 280
    assert not text.endswith(" wor")


def test_narrator_emit_calls_the_optional_update_hook() -> None:
    run = make_run()
    updates: list[dict[str, Any]] = []
    narrator = Narrator(run, None, run._running, on_update=updates.append)
    update = {"id": 1, "kind": "update", "headline": "A headline"}
    narrator._emit(update)
    assert updates == [update]


def test_narrator_hook_errors_do_not_escape() -> None:
    run = make_run()

    def fail(_: dict[str, Any]) -> None:
        raise RuntimeError("hook failed")

    narrator = Narrator(run, None, run._running, on_update=fail)
    narrator._emit({"id": 1, "kind": "update"})


def test_only_the_last_audio_clips_are_retained() -> None:
    run = make_run(FakeSpeech())

    async def main() -> None:
        run._running.set()
        task = asyncio.create_task(run.speech.run())
        try:
            for n in range(MAX_AUDIO_CLIPS + 1):
                run.speech.say(f"Line {n}")
                await asyncio.wait_for(run.speech._queue.join(), 2)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(main())
    lines = run.speech.state()["lines"]
    assert len(lines) == MAX_AUDIO_CLIPS
    assert run.speech.audio(1) is None
    assert run.speech.audio(MAX_AUDIO_CLIPS + 1) == PCM


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EVENT_LOG_PATH", ":memory:")
    monkeypatch.setenv("TTS_API_KEY", "")
    monkeypatch.setenv("TTS_BASE_URL", "")
    monkeypatch.delenv("GOD_TOKEN", raising=False)
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_speech_api_websocket_audio(client: TestClient) -> None:
    run_id = client.post(
        "/runs",
        json={"width": 32, "height": 16, "agents": 3, "cooldown": 600, "policy": "random"},
    ).json()["id"]
    manager = cast(FastAPI, client.app).state.runs
    run = manager.get(run_id)
    run.speech._synth = FakeSpeech()

    with client.websocket_connect(f"/runs/{run_id}/speech/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "speech.snapshot"
        assert snapshot["devices"] == 1
        queued = client.post(
            f"/runs/{run_id}/speak", json={"text": "Hello from an agent.", "agent_id": 1}
        )
        assert queued.status_code == 202 and queued.json() == {"queued": True}
        line = ws.receive_json()
        assert line["type"] == "speech.line" and line["audio"] is True
        line_id = line["id"]
        state = client.get(f"/runs/{run_id}/speech").json()
        assert state["lines"][-1]["id"] == line_id and state["devices"] == 1
        pcm = client.get(f"/runs/{run_id}/speech/lines/{line_id}/audio?format=pcm")
        wav = client.get(f"/runs/{run_id}/speech/lines/{line_id}/audio")
        assert pcm.status_code == 200 and pcm.content == PCM
        assert pcm.headers["x-sample-rate"] == str(SAMPLE_RATE)
        assert wav.status_code == 200 and wav.content.startswith(b"RIFF")
        assert client.get(f"/runs/{run_id}/speech/lines/999/audio").status_code == 404
    deadline = time.monotonic() + 2
    while run.speech.devices and time.monotonic() < deadline:
        time.sleep(0.01)
    assert run.speech.devices == 0

    assert client.post(f"/runs/{run_id}/stop").json()["state"] == "stopped"
    stopped = client.post(f"/runs/{run_id}/speak", json={"text": "Dropped"})
    assert stopped.status_code == 409
    assert stopped.json()["detail"] == "run is not running"


def test_restored_run_uses_manager_speech_synth(client: TestClient) -> None:
    run_id = client.post(
        "/runs",
        json={"width": 32, "height": 16, "agents": 3, "cooldown": 600, "policy": "random"},
    ).json()["id"]
    manager = cast(FastAPI, client.app).state.runs
    run = manager.get(run_id)
    assert not run.speech.state()["enabled"]
    assert client.post(f"/runs/{run_id}/stop").json()["state"] == "stopped"
    manager.speech = FakeSpeech()
    manager.runs.pop(run_id)

    restored = client.get(f"/runs/{run_id}/speech")
    assert restored.status_code == 200
    assert restored.json()["enabled"] is True
