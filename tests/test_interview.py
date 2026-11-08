import asyncio
import base64
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
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
    assert "visitor" in system and "society" not in system
    assert [m["role"] for m in asked[1]] == ["system", "user", "assistant", "user"]
    memory = run.thoughts[mei][-2]["thought"]
    assert "What are you painting?" in memory and "blue waves" in memory


class FakeRealtime:
    """Stands in for ``AsyncOpenAI`` with just enough of ``client.realtime.connect``."""

    def __init__(self, fail: bool = False, late: bool = False) -> None:
        self.fail = fail
        self.late = late
        self.sessions: list[dict[str, Any]] = []
        self.realtime = self

    def connect(self, model: str) -> "FakeRealtime":
        return self

    async def __aenter__(self) -> "FakeRealtime":
        if self.fail:
            raise ConnectionError("no realtime")
        self.session = SimpleNamespace(update=self._update)
        self.input_audio_buffer = SimpleNamespace(append=self._noop, commit=self._noop)
        self.response = SimpleNamespace(create=self._noop)
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def _update(self, session: dict[str, Any]) -> None:
        self.sessions.append(session)

    async def _noop(self, **kw: Any) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        audio = base64.b64encode(b"\1\0" * 2400).decode()  # 0.1 s at 24 kHz
        heard = SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed",
            transcript="What are you painting?",
        )
        yield SimpleNamespace(type="response.output_audio.delta", delta=audio)
        if not self.late:
            yield heard
        yield SimpleNamespace(type="response.output_audio.delta", delta=audio)
        yield SimpleNamespace(
            type="response.output_audio_transcript.done", transcript="Blue waves, mostly."
        )
        yield SimpleNamespace(type="response.done", response=SimpleNamespace(status="completed"))
        if self.late:
            yield heard


def test_realtime_interview_streams_in_screen_order(client: TestClient) -> None:
    state: Any = client.app.state  # type: ignore[attr-defined]
    rt = FakeRealtime()
    state.realtime = rt
    small = {"width": 32, "height": 16, "agents": 4, "cooldown": 60, "seed": 7}
    client.post("/runs", json={**small, "policy": "random", "societies": 2})
    run = _named(state.runs.active()[0])
    agent = run.agents[2]
    try:
        with client.websocket_connect("/devices/ws?token=tok-a") as ws:
            ws.send_json(
                {"type": "voice.start", "rate": 16000, "purpose": "interview", "agent": agent.id}
            )
            ws.send_bytes(b"\0\0" * 160)
            ws.send_json({"type": "voice.end"})
            assert _until(ws, "interview.heard")["text"] == "What are you painting?"
            reply = _until(ws, "interview.reply")
            assert reply["audio"]
            got = 0
            text = None
            while True:
                msg = ws.receive()
                if msg.get("bytes") is not None:
                    assert text is None
                    got += len(msg["bytes"])
                elif '"interview.text"' in (msg.get("text") or ""):
                    text = msg["text"]
                elif '"interview.audio_end"' in (msg.get("text") or ""):
                    break
            assert got == 2 * 3200 and text is not None and "Blue waves" in text

            state.realtime = FakeRealtime(fail=True)
            heard = Transcripts()
            state.transcriber = heard
            heard.queue.append("And tomorrow?")

            async def chat(run: Run, messages: list[dict[str, str]]) -> str:
                return "More waves."

            state.interview_chat = chat
            ws.send_json(
                {"type": "voice.start", "rate": 16000, "purpose": "interview", "agent": agent.id}
            )
            ws.send_json({"type": "voice.end"})
            assert _until(ws, "interview.reply")["text"] == "More waves."
    finally:
        state.realtime = None
    assert "visitor" in rt.sessions[0]["instructions"]
    assert any("Blue waves" in t["thought"] for t in run.thoughts[agent.id])


def test_realtime_releases_held_audio_when_answer_ends_before_transcript(
    client: TestClient,
) -> None:
    state: Any = client.app.state  # type: ignore[attr-defined]
    state.realtime = FakeRealtime(late=True)
    small = {"width": 32, "height": 16, "agents": 4, "cooldown": 60, "seed": 7}
    client.post("/runs", json={**small, "policy": "random", "societies": 2})
    agent = state.runs.active()[0].agents[1]
    try:
        with client.websocket_connect("/devices/ws?token=tok-a") as ws:
            ws.send_json(
                {"type": "voice.start", "rate": 16000, "purpose": "interview", "agent": agent.id}
            )
            ws.send_json({"type": "voice.end"})
            assert _until(ws, "interview.heard")["text"] == "..."
            assert _until(ws, "interview.reply")["audio"]
            got = 0
            while (msg := ws.receive()).get("bytes") is not None:
                got += len(msg["bytes"])
            assert got == 2 * 3200 and '"interview.text"' in msg["text"]
    finally:
        state.realtime = None


def test_interview_context_has_threads_people_and_long_memory() -> None:
    from worldwarpixel.interview import context

    cfg = RunConfig(width=16, height=16, agents=4, cooldown=1, seed=1, societies=2)
    run = _named(Run("c", cfg, EventLog(), RandomWalkPolicy(0)))
    me, a, b, c = run.agents
    root = run.board.post_thread(a.id, "Who smeared my wave?", "Somebody painted over it", 1.0)
    run.board.reply(me.id, root.id, "Not me, ask the green crowd", 2.0)
    run.board.reply(b.id, root.id, "It was definitely you", 3.0)
    other = run.board.post_thread(c.id, "Quiet thread", "nothing to do with you", 4.0)
    ping = run.board.post_thread(b.id, "Hey", f"@{me.name.split()[0]} come look", 5.0)
    run.meet(me.id, a.id)
    for i in range(12):
        run.note_thought(me.id, f"turn number {i}")
    long = run.board.post_thread(c.id, f"@{me.name.split()[0]} paints too much", "see title", 7.0)
    asked = run.board.reply(a.id, long.id, "What colour should the river be?", 8.0)
    run.board.reply(b.id, long.id, f"@{me.name.split()[0]} you owe me a pixel", 9.0)
    for i in range(15):
        run.board.reply(c.id, long.id, f"filler {i}", 10.0 + i)
    run.board.reply(me.id, long.id, "Blue, obviously", 30.0, parent_id=asked.id)

    text = context(run, me)
    assert text.startswith(f"You are {me.persona}")
    assert "Who smeared my wave?" in text and "Hey" in text and "Quiet thread" not in text
    assert f"{b.name}: It was definitely you" in text and "You: Not me" in text
    assert f"started by {a.name}" in text
    assert a.name in text.split("People you have")[1] and "society" not in text
    assert "turn number 0" in text and "turn number 11" in text
    assert "not an assistant" in text
    assert "paints too much" in text and "you owe me a pixel" in text
    assert f"{a.name}: What colour should the river be?" in text and "filler 0" not in text
    assert other.id and ping.id


@pytest.mark.parametrize("name", ["Leon Fattakhov", "Maddie Bernheim", "Ali Partovi"])
def test_hackathon_interview_treats_canvas_and_event_as_one_setting(name: str) -> None:
    cfg = RunConfig(width=64, height=64, roster="hackathon", seed=1)
    run = Run("hackathon-interview", cfg, EventLog(), RandomWalkPolicy(0))
    agent = next(a for a in run.agents if a.name == name)
    run.note_thought(agent.id, "I painted a blue robot on the canvas.")
    event = agent.persona_card["event"]
    highlight = next(
        h for h in agent.persona_card["highlights"] if h.startswith("Neo Hackathon 2026")
    )
    prompts = [
        interview.messages(run, agent, [], "How is the hackathon going?")[0]["content"],
        interview.realtime_instructions(run, agent, []),
    ]
    for prompt in prompts:
        assert f"The shared canvas and message board are part of {event}." in prompt
        assert "belong to this same event, not separate worlds" in prompt
        assert highlight in prompt
        assert "I painted a blue robot on the canvas." in prompt
        assert "world you live in" not in prompt and "You live in a shared world" not in prompt


def test_world_interview_does_not_invent_a_hackathon() -> None:
    run = Run("world-interview", RunConfig(agents=1), EventLog(), RandomWalkPolicy(0))
    text = interview.context(run, run.agents[0])
    assert "one of 1 people sharing" in text
    assert "same event" not in text and "hackathon" not in text.lower()
