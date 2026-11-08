import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from worldwarpixel import interview
from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.interview import ascii_text, clean_reply, pick_agent, resample
from worldwarpixel.sim.agents import RandomWalkPolicy
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.run import Run, RunConfig

NAMES = ["Imran Qureshi", "José Álvarez", "Mei Lin", "Ngozi Okafor"]


def _named(run: Run) -> Run:
    for agent, name in zip(run.agents, NAMES, strict=False):
        agent.name = name
    run.agents[3].country = "Nigeria"
    return run


def test_pick_agent_by_name_near_miss_and_country() -> None:
    cfg = RunConfig(width=16, height=16, agents=4, cooldown=1, seed=1, societies=2)
    run = _named(Run("p", cfg, EventLog(), RandomWalkPolicy(0)))
    assert pick_agent(run, "Can I talk to Imran please?") is run.agents[0]
    assert pick_agent(run, "jose alvarez") is run.agents[1]
    assert pick_agent(run, "let me speak with Imram") is run.agents[0]
    assert pick_agent(run, "someone from Nigeria") is run.agents[3]
    assert pick_agent(run, "hello there") is None


def test_resample_and_clean_reply() -> None:
    assert len(resample(b"\x01\x00" * 2400, 24_000)) == 1600 * 2
    assert resample(b"\x01\x00" * 10, 16_000) == b"\x01\x00" * 10
    assert clean_reply('<think>hmm</think> "*waves* Hello!"') == "waves Hello!"
    assert ascii_text("Jos\u00e9 said \u201chi\u201d \u2014 ok\u2026") == 'Jose said "hi" - ok...'


class Transcripts:
    def __init__(self) -> None:
        self.queue: list[str] = []

    async def __call__(self, pcm: bytes, rate: int) -> str:
        return self.queue.pop(0)


class FakeSynth:
    sample_rate = 24_000

    async def synth(self, text: str, voice: str, instructions: str) -> bytes:
        return b"\x10\x00" * 24_000  # one second


class StreamingSynth(FakeSynth):
    async def stream(self, text: str, voice: str, instructions: str) -> AsyncIterator[bytes]:
        sent = 0
        for size in (1, 4095, 7, 12_000, 30_000):
            yield (b"\x10\x00" * 24_000)[sent : sent + size]
            sent += size
        yield (b"\x10\x00" * 24_000)[sent:]


def test_streamed_voice_resamples_uneven_chunks() -> None:
    class Speech:
        synth = StreamingSynth()

    class FakeRun:
        speech = Speech()

    agent = type("A", (), {"id": 1, "name": "Mei", "country": "X", "persona_card": {}})()

    async def collect() -> bytes:
        return b"".join([c async for c in interview.voice(FakeRun(), agent, "hi")])  # type: ignore[arg-type]

    pcm = asyncio.run(collect())
    assert len(pcm) == 32_000 and set(pcm[::2]) == {0x10}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EVENT_LOG_PATH", ":memory:")
    monkeypatch.setenv("DEVICE_TOKENS", "tok-a:0")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _until(ws: Any, kind: str) -> dict[str, Any]:
    while (msg := ws.receive_json())["type"] != kind:
        pass
    return msg  # type: ignore[no-any-return]


def test_interview_over_websocket(client: TestClient) -> None:
    state: Any = client.app.state  # type: ignore[attr-defined]
    heard = Transcripts()
    state.transcriber = heard
    asked: list[list[dict[str, str]]] = []

    async def chat(run: Run, messages: list[dict[str, str]]) -> str:
        asked.append(messages)
        return "*smiles* Hello, I paint blue waves."

    state.interview_chat = chat
    small = {"width": 32, "height": 16, "agents": 4, "cooldown": 60, "seed": 7}
    client.post("/runs", json={**small, "policy": "random", "societies": 2})
    run = _named(state.runs.active()[0])
    run.speech._synth = FakeSynth()

    with client.websocket_connect("/devices/ws?token=tok-a") as ws:
        ws.send_json({"type": "agents.list"})
        listed = _until(ws, "agents")
        assert listed["run"] == run.id and len(listed["agents"]) == 4
        assert {"id", "name", "country", "color", "society"} <= set(listed["agents"][0])

        ws.send_json({"type": "interview.select", "agent": 99})
        assert "99" in _until(ws, "interview.error")["error"]

        heard.queue.append("I'd like to talk to Mei")
        ws.send_json({"type": "voice.start", "rate": 16000, "purpose": "pick"})
        ws.send_bytes(b"\0\0" * 160)
        ws.send_json({"type": "voice.end"})
        picked = _until(ws, "interview.agent")
        assert picked["agent"]["name"] == "Mei Lin" and picked["heard"] == "I'd like to talk to Mei"
        mei = picked["agent"]["id"]

        heard.queue.append("What are you painting?")
        ws.send_json({"type": "voice.start", "rate": 16000, "purpose": "interview", "agent": mei})
        ws.send_bytes(b"\0\0" * 160)
        ws.send_json({"type": "voice.end"})
        assert _until(ws, "interview.heard")["text"] == "What are you painting?"
        reply = _until(ws, "interview.reply")
        assert reply["text"] == "smiles Hello, I paint blue waves."
        assert reply["audio"] and reply["rate"] == 16000
        got = 0
        while True:
            msg = ws.receive()
            if msg.get("bytes") is not None:
                got += len(msg["bytes"])
            elif '"interview.audio_end"' in (msg.get("text") or ""):
                break
        assert got == 32000

        heard.queue.append("And tomorrow?")
        ws.send_json({"type": "voice.start", "rate": 16000, "purpose": "interview", "agent": mei})
        ws.send_json({"type": "voice.end"})
        _until(ws, "interview.reply")

    system = asked[0][0]["content"]
    assert "visitor" in system and "society" in system
    assert [m["role"] for m in asked[1]] == ["system", "user", "assistant", "user"]
    memory = run.thoughts[mei][-2]["thought"]
    assert "What are you painting?" in memory and "blue waves" in memory
