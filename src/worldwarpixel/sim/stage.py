"""Stage: a live, wall-clock debate between two featured agents, voiced, with an audience.

The swarm keeps painting on its own cooldown clock. The stage picks two rivals (the pair that
painted over each other most), and plays their spoken lines one at a time in real time. A
debater's next canvas turn waits until its line has been spoken, so what it paints and what it
says stay in step. The audience votes, reacts and can interject as the narrator; the verdict
is announced to every agent through the board.
"""

import asyncio
import contextlib
import logging
import math
import random
import re
import struct
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from worldwarpixel.sim.board import BoardError
from worldwarpixel.sim.canvas import PALETTE
from worldwarpixel.sim.hub import Hub
from worldwarpixel.sim.narrator import NarratorModel, _extract_json

if TYPE_CHECKING:
    from worldwarpixel.sim.agents import AgentSpec
    from worldwarpixel.sim.run import Run

log = logging.getLogger(__name__)

Phase = Literal["idle", "debating", "voting"]
LineKind = Literal["debate", "narrator", "verdict"]

SAMPLE_RATE = 24_000
"""OpenAI `pcm` speech is 24 kHz signed 16-bit little-endian mono."""
GOD_ID = 65535
NARRATOR_ID = 0
NARRATOR_VOICE = "verse"
VOICES_FEMALE = ("coral", "nova", "shimmer", "sage")
VOICES_MALE = ("ash", "echo", "onyx", "ballad")
VOICES_ANY = ("alloy", "fable", "verse")
EMOTIONS = ("calm", "angry", "smug", "defiant", "pleading", "excited", "sarcastic")
REACTIONS = ("cheer", "boo", "laugh", "gasp")
MAX_TRANSCRIPT = 50
MAX_AUDIO_CLIPS = 20
MAX_LINE_CHARS = 280
WORDS_PER_SECOND = 2.6
ACTIONS_RE = re.compile(r"\s*\[[^\]]*\]\s*$")


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
        if self.model.startswith("gpt-4o"):  # only these models take style instructions
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


def clean_thought(thought: str) -> str:
    """An agent's thought without the trailing `[tool summary]` the policy appends."""
    text = ACTIONS_RE.sub("", thought).strip()
    return "" if text.startswith("(") else text


def parse_line(text: str) -> tuple[str, str]:
    """(spoken line, emotion) from the model's reply; tolerates plain text."""
    data = _extract_json(text)
    if isinstance(data, dict) and isinstance(data.get("line"), str):
        line, emotion = data["line"], str(data.get("emotion", "")).lower()
    else:
        line, emotion = text, ""
    line = " ".join(line.strip().strip('"').split())[:MAX_LINE_CHARS]
    return line, emotion if emotion in EMOTIONS else "defiant"


class StageBusy(Exception):
    """A debate is already on stage."""


class Stage:
    def __init__(
        self,
        run: "Run",
        model: NarratorModel | None,
        speech: SpeechSynth | None,
        running: asyncio.Event,
        auto: bool = False,
        lines_per_debate: int = 6,
        vote_seconds: float = 20.0,
        hold_seconds: float = 120.0,
        break_seconds: float = 15.0,
        line_gap: float = 1.2,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._run = run
        self._model = model
        self._speech = speech
        self._running = running
        self.auto = auto
        self.lines_per_debate = lines_per_debate
        self.vote_seconds = vote_seconds
        self.hold_seconds = hold_seconds
        self.break_seconds = break_seconds
        self.line_gap = line_gap
        self._clock = clock
        self._sleep = sleep
        self.hub = Hub()
        """Stage-only stream (podium screens, the stage page): no canvas traffic."""
        self.phase: Phase = "idle"
        self.debate_id = 0
        self.debaters: list[int] = []
        self.topic = ""
        self.region: dict[str, int] | None = None
        self.thread_id: int | None = None
        self.devices: Counter[int] = Counter()
        self.hold: dict[str, Any] | None = None
        self._picked_rivalry_count = 0
        self._picked_rivalry_at = (0, 0)
        self.vote_ends_at: float | None = None
        self.transcript: deque[dict[str, Any]] = deque(maxlen=MAX_TRANSCRIPT)
        self.stats = {
            "debates": 0,
            "lines": 0,
            "audio": 0,
            "speech_failures": 0,
            "model_failures": 0,
            "repeats": 0,
            "votes": 0,
            "reactions": 0,
            "interjections": 0,
        }
        self._spoken = 0
        self._votes: dict[str, int] = {}
        self._reactions: Counter[tuple[str, int | None]] = Counter()
        self._interjections: deque[dict[str, Any]] = deque(maxlen=10)
        self._audio: dict[int, bytes] = {}
        self._gates: dict[int, asyncio.Event] = {}
        self._prepared: (
            tuple[
                int,
                int,
                asyncio.Task[tuple[dict[str, Any], Counter[tuple[str, int | None]]]],
            ]
            | None
        ) = None
        self._next_line_id = 1
        self._next_auto_at = 0.0
        self._wake = asyncio.Event()
        self._rng = random.Random(run.config.seed)
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._next_auto_at = self._clock() + self.break_seconds
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._drop_prepared()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._release_all()

    # -- hooks from the run ----------------------------------------------------------------

    async def gate(self, agent_id: int) -> None:
        """Debaters take a canvas turn only after each line they speak."""
        ev = self._gates.get(agent_id)
        if ev is None:
            return
        await ev.wait()
        if self._gates.get(agent_id) is ev:
            ev.clear()

    # -- control ---------------------------------------------------------------------------

    def pick_rivals(self) -> tuple[int, int, str, dict[str, int] | None]:
        """The pair that overwrote each other most (both directions), else two strangers."""
        by_id = self._run.by_id
        pairs: Counter[tuple[int, int]] = Counter()
        for (a, b), n in self._run.narrator.rivalry.items():
            pairs[(min(a, b), max(a, b))] += n
        if pairs:
            (a, b), self._picked_rivalry_count = pairs.most_common(1)[0]
            rivalry_at = self._run.narrator.rivalry_at
            at = rivalry_at.get((a, b)) or rivalry_at.get((b, a)) or (0, 0)
            self._picked_rivalry_at = at
            x, y = at
            half = 24
            region = {
                "x": max(0, x - half),
                "y": max(0, y - half),
                "w": half * 2,
                "h": half * 2,
            }
            for k in list(self._run.narrator.rivalry):
                self._run.narrator.rivalry[k] //= 2
                if not self._run.narrator.rivalry[k]:
                    del self._run.narrator.rivalry[k]
            return a, b, f"Who owns the canvas around ({x}, {y})?", region
        self._picked_rivalry_count = 0
        self._picked_rivalry_at = (0, 0)
        agents = list(by_id.values())
        if len(agents) < 2:
            raise StageBusy("a debate needs at least two agents")
        a_spec = self._rng.choice(agents)
        others = [g for g in agents if g.country != a_spec.country] or [
            g for g in agents if g.id != a_spec.id
        ]
        b_spec = self._rng.choice(others)
        return a_spec.id, b_spec.id, "Whose country deserves more of the canvas?", None

    def start_debate(
        self, a: int | None = None, b: int | None = None, topic: str | None = None
    ) -> dict[str, Any]:
        if self.phase != "idle":
            raise StageBusy(f"a debate is already {self.phase}")
        by_id = self._run.by_id
        if (a is None) != (b is None):
            raise ValueError("give both debaters or neither")
        region: dict[str, int] | None = None
        if a is None or b is None:
            a, b, auto_topic, region = self.pick_rivals()
        else:
            if a == b or a not in by_id or b not in by_id:
                raise ValueError("debaters must be two different agents of this run")
            auto_topic = f"{by_id[a].country} vs {by_id[b].country}: who rules the canvas?"
        self.debate_id += 1
        self.debaters = [a, b]
        self.topic = (topic or "").strip() or auto_topic
        self.region = region
        self.phase = "debating"
        self.vote_ends_at = None
        self._spoken = 0
        self._votes.clear()
        self._reactions.clear()
        self._gates = {a: asyncio.Event(), b: asyncio.Event()}
        self.thread_id = None
        agent_a, agent_b = by_id[a], by_id[b]
        title = f"Live debate #{self.debate_id}: {agent_a.name} vs {agent_b.name}"
        body = f"{self.topic} Reply here to weigh in; the live audience picks the winner."
        with contextlib.suppress(BoardError):
            thread = self._run.god_announce(title[: self._run.board.max_title], body)
            self.thread_id = int(thread["thread_id"])
        intro = (
            f"Debate #{self.debate_id}: {agent_a.name} of {agent_a.country} "
            f"versus {agent_b.name} of {agent_b.country}."
        )
        if region is not None:
            x, y = self._picked_rivalry_at
            intro += (
                f" They have painted over each other {self._picked_rivalry_count} times "
                f"around ({x}, {y})."
            )
        intro += f" {self.topic}"
        self._interjections.appendleft({"name": "Narrator", "text": intro})
        state = self.state(transcript=False)
        self._run.record("stage.debate", None, state)
        self._publish("stage.state", **state)
        self._wake.set()
        return state

    def end_debate(self) -> None:
        self._drop_prepared()
        self.phase = "idle"
        self.debaters = []
        self.vote_ends_at = None
        self._release_all()
        self._next_auto_at = self._clock() + self.break_seconds
        self._publish("stage.state", **self.state(transcript=False))

    def set_auto(self, auto: bool) -> None:
        self.auto = auto
        self._next_auto_at = min(self._next_auto_at, self._clock() + 1.0)
        self._publish("stage.state", **self.state(transcript=False))
        self._wake.set()

    def device_connected(self, podium: int) -> None:
        self.devices[podium] += 1
        self._publish("stage.state", **self.state(transcript=False))

    def device_disconnected(self, podium: int) -> None:
        self.devices[podium] = max(0, self.devices[podium] - 1)
        self._publish("stage.state", **self.state(transcript=False))

    # -- audience --------------------------------------------------------------------------

    def vote(self, voter: str, agent_id: int) -> dict[int, int]:
        if self.phase == "idle":
            raise StageBusy("no debate on stage")
        if agent_id not in self.debaters:
            raise ValueError("vote for one of the two debaters")
        self._votes[voter] = agent_id
        self.stats["votes"] += 1
        tally = self.tally()
        self._publish("stage.votes", debate_id=self.debate_id, votes=tally)
        return tally

    def tally(self) -> dict[int, int]:
        counts = Counter(self._votes.values())
        return {a: counts.get(a, 0) for a in self.debaters}

    def react(self, reaction: str, agent_id: int | None) -> None:
        if reaction not in REACTIONS:
            raise ValueError(f"reaction must be one of {', '.join(REACTIONS)}")
        if agent_id is not None and agent_id not in self.debaters:
            agent_id = None
        self._reactions[(reaction, agent_id)] += 1
        self.stats["reactions"] += 1
        self._publish("stage.reaction", reaction=reaction, agent_id=agent_id)

    def narrate(self, text: str, name: str | None = None) -> None:
        """A live narrator line from the audience/MC; spoken next, and debaters respond to it."""
        text = " ".join(text.split())[:MAX_LINE_CHARS]
        if not text:
            raise ValueError("say something")
        self._interjections.append({"name": (name or "").strip()[:40] or "Narrator", "text": text})
        self.stats["interjections"] += 1
        self._drop_prepared()
        self._wake.set()

    # -- loop ------------------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            await self._running.wait()
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except Exception:  # keep the show going if one line blows up
                log.exception("stage step failed")
                await self._sleep(1.0)

    async def _step(self) -> None:
        if self._interjections:
            item = self._interjections.popleft()
            await self._play(await self._narrator_line(item["text"], item["name"], "narrator"))
            return
        if self.phase == "idle":
            if self.auto and self._clock() >= self._next_auto_at:
                try:
                    self.start_debate()
                except StageBusy:
                    self._next_auto_at = self._clock() + self.break_seconds
                    await self._idle(1.0)
                return
            await self._idle(1.0)
        elif self.phase == "debating":
            if self._spoken >= self.lines_per_debate:
                self.phase = "voting"
                self.vote_ends_at = self._clock() + self.vote_seconds
                self._publish("stage.state", **self.state(transcript=False))
                return
            await self._debate_line()
        else:
            remaining = (self.vote_ends_at or 0) - self._clock()
            if remaining <= 0:
                await self._verdict()
            else:
                await self._idle(min(1.0, remaining))

    async def _idle(self, seconds: float) -> None:
        self._wake.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), seconds)

    async def _debate_line(self) -> None:
        idx, debate = self._spoken, self.debate_id
        task = self._take_prepared(debate, idx) or asyncio.create_task(
            self._compose(self.debaters[idx % 2])
        )
        line, seen = await task
        if self.phase != "debating" or self.debate_id != debate or self._interjections:
            return  # stopped, replaced or interrupted while composing: drop the stale line
        self._emit(line)
        self._reactions -= seen
        self._spoken += 1
        if self._spoken < self.lines_per_debate:
            nxt = asyncio.create_task(self._compose(self.debaters[self._spoken % 2]))
            self._prepared = (debate, self._spoken, nxt)
        await self._sleep(line["duration"] + self.line_gap)
        ev = self._gates.get(line["speaker_id"])
        if ev is not None:
            ev.set()

    async def _verdict(self) -> None:
        debate = self.debate_id
        tally = self.tally()
        by_id = self._run.by_id
        a, b = (by_id[i] for i in self.debaters)
        va, vb = tally[a.id], tally[b.id]
        winner = a if va > vb else b if vb > va else None
        if winner is None:
            text = (
                f"It's a draw, {va} votes each! {a.name} and {b.name} will have to settle "
                "it on the canvas."
            )
        else:
            loser = b if winner is a else a
            hi, lo = max(va, vb), min(va, vb)
            text = (
                f"The crowd has spoken: {winner.name} of {winner.country} beats {loser.name} "
                f"of {loser.country}, {hi} vote{'' if hi == 1 else 's'} to {lo}!"
            )
            hold_seconds = self.hold_seconds
            if self.region is not None and hold_seconds > 0:
                self.hold = {
                    "agent_id": winner.id,
                    "region": dict(self.region),
                    "until": self._clock() + hold_seconds,
                }
                text += f" {winner.name} holds that ground for {hold_seconds:g}s."
        result = {
            "debate_id": self.debate_id,
            "votes": tally,
            "winner_id": winner.id if winner else None,
            "topic": self.topic,
        }
        self._run.record("stage.verdict", None, result)
        self._publish("stage.verdict", **result)
        self.stats["debates"] += 1
        await self._play(await self._narrator_line(text, "Narrator", "verdict"))
        if self.thread_id is not None:
            self._run.pin_announcement(
                self.thread_id,
                "Live debate verdict",
                f'On "{self.topic}": {text} (decided by the live audience)'[:600],
            )
        if self.debate_id == debate and self.phase == "voting":
            self.end_debate()

    async def _play(self, line: dict[str, Any]) -> None:
        self._emit(line)
        await self._sleep(line["duration"] + self.line_gap)

    def _take_prepared(
        self, debate: int, idx: int
    ) -> asyncio.Task[tuple[dict[str, Any], Counter[tuple[str, int | None]]]] | None:
        prepared, self._prepared = self._prepared, None
        if prepared is None:
            return None
        if prepared[:2] == (debate, idx):
            return prepared[2]
        prepared[2].cancel()
        return None

    def _drop_prepared(self) -> None:
        if self._prepared is not None:
            self._prepared[2].cancel()
            self._prepared = None

    def _release_all(self) -> None:
        for ev in self._gates.values():
            ev.set()
        self._gates = {}

    # -- composing lines -------------------------------------------------------------------

    def _messages(self, speaker: "AgentSpec", opponent: "AgentSpec") -> list[dict[str, Any]]:
        system = (
            f"You are {speaker.name} from {speaker.country}, on stage at WorldWarPixel, a live "
            "show where AI agents from every country fight over one shared pixel canvas. A live "
            f"audience is watching. You are debating {opponent.name} from {opponent.country}. "
            f"Topic: {self.topic}\n\n"
            f"Who you are (shapes your views; never recite it): {speaker.persona}\n\n"
            "Say one or two short spoken sentences (at most 30 words), first person, out loud "
            "to the crowd. This is a debate, not a group hug: be punchy, witty and combative, "
            "answer your opponent's last line directly, defend what you are painting with "
            "specifics, and play to the audience. React to anything the narrator or the crowd "
            "just did. Say something new each turn: never repeat a line already said. No stage "
            "directions, no emojis, no hashtags, no quotation marks.\n"
            'Reply with JSON only: {"line": "what you say", "emotion": one of '
            + ", ".join(f'"{e}"' for e in EMOTIONS)
            + "}"
        )
        parts = [f"What you were doing on the canvas: {self._thoughts(speaker.id, 3)}"]
        parts.append(f"What {opponent.name} was doing: {self._thoughts(opponent.id, 1)}")
        lines = [ln for ln in self.transcript if ln["debate_id"] == self.debate_id][-6:]
        if lines:
            parts.append(
                "The debate so far:\n" + "\n".join(f"{ln['name']}: {ln['text']}" for ln in lines)
            )
            last = lines[-1]
            who = "The narrator" if last["kind"] == "narrator" else last["name"]
            parts.append(f"{who} just said: {last['text']}")
        else:
            parts.append("You open the debate.")
        crowd = self._crowd(speaker, opponent)
        if crowd:
            parts.append(f"The crowd: {crowd}")
        parts.append(f"Your turn, {speaker.name}. Something new, in JSON.")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def _repeats(self, text: str) -> bool:
        """True when the line mostly repeats one already said in this debate."""
        words = set(re.findall(r"[a-z']+", text.lower()))
        for ln in self.transcript:
            if ln["debate_id"] != self.debate_id:
                continue
            said = set(re.findall(r"[a-z']+", ln["text"].lower()))
            if words and len(words & said) / len(words | said) > 0.6:
                return True
        return False

    def _thoughts(self, agent_id: int, n: int) -> str:
        texts = [clean_thought(t["thought"]) for t in list(self._run.thoughts[agent_id])[-n:]]
        return " / ".join(t for t in texts if t) or "(nothing yet)"

    def _crowd(self, speaker: "AgentSpec", opponent: "AgentSpec") -> str:
        names = {speaker.id: "you", opponent.id: opponent.name, None: "the stage"}
        bits = [
            f"{n} {reaction}{'s' if n > 1 else ''} for {names[target]}"
            for (reaction, target), n in self._reactions.most_common(4)
            if target in names
        ]
        tally = self.tally()
        if any(tally.values()):
            bits.append(
                f"votes so far: you {tally[speaker.id]}, {opponent.name} {tally[opponent.id]}"
            )
        return "; ".join(bits)

    def fallback_line(self, speaker: "AgentSpec", opponent: "AgentSpec") -> str:
        thought = (
            clean_thought(self._run.thoughts[speaker.id][-1]["thought"])
            if (self._run.thoughts[speaker.id])
            else ""
        )
        openers = (
            f"{opponent.name}, {speaker.country} is not going anywhere.",
            f"Listen up: {opponent.country} keeps painting over my work.",
            f"Everyone can see who is building and who is wrecking, {opponent.name}.",
            f"{speaker.country} was here first, and the canvas knows it.",
        )
        line = openers[(self._spoken + speaker.id) % len(openers)]
        # Only a short, plain thought is worth saying out loud (not tool output or reasoning).
        if "\n" in thought or len(thought) > 120 or re.search(r"\d", thought):
            thought = ""
        return f"{line} {thought}".strip()[:MAX_LINE_CHARS]

    async def _compose(
        self, speaker_id: int
    ) -> tuple[dict[str, Any], Counter[tuple[str, int | None]]]:
        by_id = self._run.by_id
        speaker = by_id[speaker_id]
        opponent = by_id[next(d for d in self.debaters if d != speaker_id)]
        text, emotion = "", "defiant"
        seen: Counter[tuple[str, int | None]] = Counter()
        if self._model is not None:
            try:
                seen = Counter(self._reactions)
                messages = cast(list[ChatCompletionMessageParam], self._messages(speaker, opponent))
                for _ in range(2):  # one retry if the model just echoes an earlier line
                    text, emotion = parse_line(await self._model.complete(messages))
                    if not self._repeats(text):
                        break
                    self.stats["repeats"] += 1
                else:
                    text = ""  # still echoing: a canned line beats a repeat
            except Exception as e:  # the show goes on with a canned line
                self.stats["model_failures"] += 1
                log.warning("stage model call failed: %s", e)
        if not text:
            text = self.fallback_line(speaker, opponent)
        voice = self._voice(speaker.id)
        instructions = (
            f"You are {speaker.name} from {speaker.country}, debating live on stage in front "
            f"of a crowd. Speak with a natural accent from {speaker.country}. "
            f"Emotion: {emotion}. Punchy and conversational."
        )
        return (
            await self._line(
                "debate",
                speaker_id,
                speaker.name,
                speaker.country,
                speaker.color,
                self.debaters.index(speaker_id),
                text,
                emotion,
                voice,
                instructions,
            ),
            seen,
        )

    async def _narrator_line(self, text: str, name: str, kind: LineKind) -> dict[str, Any]:
        instructions = "You are the energetic host of a live debate show. Clear and lively."
        return await self._line(
            kind, NARRATOR_ID, name, "", None, None, text, "excited", NARRATOR_VOICE, instructions
        )

    async def _line(
        self,
        kind: LineKind,
        speaker_id: int,
        name: str,
        country: str,
        color: int | None,
        podium: int | None,
        text: str,
        emotion: str,
        voice: str,
        instructions: str,
    ) -> dict[str, Any]:
        pcm: bytes | None = None
        if self._speech is not None:
            try:
                pcm = await self._speech.synth(text, voice, instructions)
            except Exception as e:
                self.stats["speech_failures"] += 1
                log.warning("speech synthesis failed: %s", e)
        rate = self.sample_rate
        duration = len(pcm) / 2 / rate if pcm else max(2.0, len(text.split()) / WORDS_PER_SECOND)
        line_id = self._next_line_id
        self._next_line_id += 1
        if pcm:
            self._audio[line_id] = pcm
            for old in sorted(self._audio)[:-MAX_AUDIO_CLIPS]:
                del self._audio[old]
        return {
            "id": line_id,
            "debate_id": self.debate_id,
            "kind": kind,
            "speaker_id": speaker_id,
            "name": name,
            "country": country,
            "color": color,
            "color_hex": PALETTE[color] if color is not None else None,
            "podium": podium,
            "text": text,
            "emotion": emotion,
            "voice": voice,
            "audio": bool(pcm),
            "sample_rate": rate,
            "duration": round(duration, 2),
        }

    def _emit(self, line: dict[str, Any]) -> None:
        line["ts"] = self._clock()
        self.transcript.append(line)
        self.stats["lines"] += 1
        self.stats["audio"] += int(line["audio"])
        self._run.record("stage.line", line["speaker_id"] or None, line)
        if self.thread_id is not None:
            author_id = int(line["speaker_id"]) if line["kind"] == "debate" else GOD_ID
            body = line["text"] if line["kind"] == "debate" else f"{line['name']}: {line['text']}"
            with contextlib.suppress(BoardError):
                self._run.post_reply(author_id, self.thread_id, body)
        self._publish("stage.line", **line)

    def _publish(self, type_: str, **data: Any) -> None:
        self.hub.publish(type_, **data)
        self._run.hub.publish(type_, **data)

    def _voice(self, agent_id: int) -> str:
        """The agent's voice, nudged so two rivals on stage never sound the same."""
        by_id = self._run.by_id
        voice = voice_for(by_id[agent_id])
        rival = len(self.debaters) == 2 and agent_id == self.debaters[1]
        if rival and voice == voice_for(by_id[self.debaters[0]]):
            return voice_for(by_id[agent_id], shift=1)
        return voice

    # -- read model ------------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._speech.sample_rate if self._speech is not None else SAMPLE_RATE

    def audio(self, line_id: int) -> bytes | None:
        return self._audio.get(line_id)

    def blocks(self, agent_id: int, x: int, y: int) -> bool:
        hold = self.hold
        if (
            hold is None
            or agent_id == hold["agent_id"]
            or agent_id == GOD_ID
            or agent_id not in self._run.by_id
            or hold["until"] <= self._clock()
        ):
            return False
        region = cast(dict[str, int], hold["region"])
        return (
            region["x"] <= x < region["x"] + region["w"]
            and region["y"] <= y < region["y"] + region["h"]
        )

    def hold_state(self) -> dict[str, Any] | None:
        hold = self.hold
        if hold is None or hold["until"] <= self._clock():
            return None
        return {
            **hold,
            "seconds_left": math.ceil(hold["until"] - self._clock()),
        }

    def frame(self, width: int, height: int) -> bytes:
        """Render the current debate region (or canvas) as letterboxed RGB565 pixels."""
        canvas = self._run.canvas
        if self.region is None:
            sx, sy, source_w, source_h = 0, 0, canvas.width, canvas.height
        else:
            sx, sy, source_w, source_h = (
                self.region["x"],
                self.region["y"],
                self.region["w"],
                self.region["h"],
            )
        scale = min(width / source_w, height / source_h)
        dest_w = max(1, round(source_w * scale))
        dest_h = max(1, round(source_h * scale))
        offset_x = (width - dest_w) // 2
        offset_y = (height - dest_h) // 2
        output = bytearray(width * height * 2)
        debaters = set(self.debaters)
        for dy in range(dest_h):
            source_y = sy + min(source_h - 1, int(dy * source_h / dest_h))
            for dx in range(dest_w):
                source_x = sx + min(source_w - 1, int(dx * source_w / dest_w))
                if not canvas.in_bounds(source_x, source_y):
                    continue
                index = source_y * canvas.width + source_x
                color = int(canvas.pixels[index])
                red, green, blue = (int(PALETTE[color][i : i + 2], 16) for i in (1, 3, 5))
                if canvas.owners[index] not in debaters:
                    red, green, blue = int(red * 0.4), int(green * 0.4), int(blue * 0.4)
                rgb565 = ((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3)
                target = ((offset_y + dy) * width + offset_x + dx) * 2
                struct.pack_into("<H", output, target, rgb565)
        return bytes(output)

    def state(self, transcript: bool = True) -> dict[str, Any]:
        by_id = self._run.by_id
        out: dict[str, Any] = {
            "phase": self.phase,
            "auto": self.auto,
            "debate_id": self.debate_id,
            "topic": self.topic,
            "region": self.region,
            "thread_id": self.thread_id,
            "hold": self.hold_state(),
            "devices": {"0": self.devices[0], "1": self.devices[1]},
            "debaters": [
                {
                    "id": a,
                    "name": by_id[a].name,
                    "country": by_id[a].country,
                    "color": by_id[a].color,
                    "color_hex": PALETTE[by_id[a].color],
                    "podium": i,
                    "voice": self._voice(a),
                }
                for i, a in enumerate(self.debaters)
            ],
            "votes": self.tally(),
            "vote_ends_at": self.vote_ends_at,
            "lines_per_debate": self.lines_per_debate,
            "spoken": self._spoken,
            "speech": self._speech is not None,
            "model": self._model is not None,
            "stats": dict(self.stats),
        }
        if transcript:
            out["transcript"] = list(self.transcript)
        return out
