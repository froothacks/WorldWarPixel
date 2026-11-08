import asyncio
import logging
import struct
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from openai import AsyncOpenAI

from worldwarpixel.sim.canvas import PALETTE
from worldwarpixel.sim.hub import Hub

if TYPE_CHECKING:
    from worldwarpixel.sim.agents import AgentSpec
    from worldwarpixel.sim.run import Run

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
"""OpenAI `pcm` speech is 24 kHz signed 16-bit little-endian mono."""
NARRATOR_ID = 0
NARRATOR_VOICE = "verse"
VOICES_FEMALE = ("coral", "nova", "shimmer", "sage")
VOICES_MALE = ("ash", "echo", "onyx", "ballad")
VOICES_ANY = ("alloy", "fable", "verse")
MAX_AUDIO_CLIPS = 20
MAX_LINE_CHARS = 280
WORDS_PER_SECOND = 2.6


class SpeechSynth(Protocol):
    """Text to raw PCM (s16le mono at `sample_rate`); tests swap in a fake."""

    sample_rate: int

    async def synth(self, text: str, voice: str, instructions: str) -> bytes: ...


class OpenAISpeech:
    """Any OpenAI-compatible /audio/speech endpoint (OpenAI, or a local Piper/Kokoro server)."""

    sample_rate = SAMPLE_RATE

    def __init__(self, client: AsyncOpenAI, model: str, timeout: float = 30.0) -> None:
        self._client = client
        self.model = model
        self._timeout = timeout

    async def synth(self, text: str, voice: str, instructions: str) -> bytes:
        extra: dict[str, Any] = {}
        if self.model.startswith("gpt-4o"):
            extra["instructions"] = instructions
        response = await self._client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text,
            response_format="pcm",
            timeout=self._timeout,
            **extra,
        )
        return response.content


def wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap s16le mono PCM in a WAV header so browsers can play it."""
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(pcm),
    )
    return header + pcm


def voice_for(agent: "AgentSpec", shift: int = 0) -> str:
    """A stable voice per agent, matched to the persona's sex where known."""
    sex = str(agent.persona_card.get("sex") or "").lower()
    pool = (
        VOICES_FEMALE if sex.startswith("f") else VOICES_MALE if sex.startswith("m") else VOICES_ANY
    )
    return pool[(agent.id + shift) % len(pool)]


class Speech:
    def __init__(
        self,
        run: "Run",
        synth: SpeechSynth | None,
        running: asyncio.Event,
        *,
        narrate: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._run = run
        self._synth = synth
        self._running = running
        self._narrate = narrate
        self._clock = clock
        self.hub = Hub()
        self.devices = 0
        self.lines: deque[dict[str, Any]] = deque(maxlen=MAX_AUDIO_CLIPS)
        self.stats = {"lines": 0, "audio": 0, "failures": 0}
        self._queue: asyncio.Queue[tuple[str, int | None]] = asyncio.Queue(MAX_AUDIO_CLIPS)
        self._audio: dict[int, bytes] = {}
        self._next_id = 1

    def say(self, text: str, agent_id: int | None = None) -> None:
        text = text.strip()
        if not text:
            raise ValueError("speech text must not be empty")
        if len(text) > MAX_LINE_CHARS:
            raise ValueError(f"speech text must be at most {MAX_LINE_CHARS} characters")
        if agent_id is not None and agent_id not in self._run.by_id:
            raise ValueError(f"unknown agent {agent_id}")
        if self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
        self._queue.put_nowait((text, agent_id))

    def narrate(self, update: dict[str, Any]) -> None:
        if not self._narrate or update.get("error"):
            return
        headline = str(update.get("headline") or "").strip()
        if not headline:
            return
        bullets = update.get("bullets")
        parts = [headline]
        if isinstance(bullets, list):
            parts.extend(str(b).strip() for b in bullets if str(b).strip())
        words: list[str] = []
        for word in " ".join(parts).split():
            candidate = " ".join((*words, word))
            if len(candidate) > MAX_LINE_CHARS:
                break
            words.append(word)
        if words:
            self.say(" ".join(words))

    async def run(self) -> None:
        while True:
            await self._running.wait()
            text, agent_id = await self._queue.get()
            await self._running.wait()
            try:
                await self._speak(text, agent_id)
            finally:
                self._queue.task_done()

    async def _speak(self, text: str, agent_id: int | None) -> None:
        agent = self._run.by_id[agent_id] if agent_id is not None else None
        speaker_id = agent.id if agent is not None else NARRATOR_ID
        name = agent.name if agent is not None else "Narrator"
        voice = voice_for(agent) if agent is not None else NARRATOR_VOICE
        instructions = (
            f"You are {agent.name} from {agent.country}. Speak naturally with an accent from "
            f"{agent.country}."
            if agent is not None
            else "You are the lively live commentator of a pixel-war game. Clear and energetic."
        )
        sample_rate = self._synth.sample_rate if self._synth is not None else SAMPLE_RATE
        pcm: bytes | None = None
        if self._synth is not None:
            try:
                pcm = await self._synth.synth(text, voice, instructions)
            except Exception as e:
                self.stats["failures"] += 1
                log.warning("speech synthesis failed: %s", e)
        duration = (
            len(pcm) / 2 / sample_rate
            if pcm is not None
            else max(2.0, len(text.split()) / WORDS_PER_SECOND)
        )
        line_id = self._next_id
        line = {
            "id": line_id,
            "speaker_id": speaker_id,
            "name": name,
            "color_hex": PALETTE[agent.color] if agent is not None else None,
            "text": text,
            "voice": voice,
            "audio": pcm is not None,
            "sample_rate": sample_rate,
            "duration": round(duration, 2),
            "ts": self._clock(),
        }
        self._next_id += 1
        self.lines.append(line)
        self.stats["lines"] += 1
        self.stats["audio"] += int(pcm is not None)
        if pcm is not None:
            self._audio[line_id] = pcm
            while len(self._audio) > MAX_AUDIO_CLIPS:
                del self._audio[next(iter(self._audio))]
        self._run.record("speech.line", agent_id, line)
        self.hub.publish("speech.line", **line)

    def audio(self, line_id: int) -> bytes | None:
        return self._audio.get(line_id)

    @property
    def sample_rate(self) -> int:
        return self._synth.sample_rate if self._synth is not None else SAMPLE_RATE

    def state(self) -> dict[str, Any]:
        return {
            "enabled": self._synth is not None,
            "devices": self.devices,
            "lines": list(self.lines),
            "stats": dict(self.stats),
        }

    def device_connected(self) -> None:
        self.devices += 1

    def device_disconnected(self) -> None:
        self.devices = max(0, self.devices - 1)
