"""Interview mode: a visitor at a screen picks one agent and talks with it out loud.

The visitor's clip is transcribed like a murmur, but instead of becoming a rumour it is put to
the chosen agent as a question. The agent answers in character with the run's own model, the
answer is voiced with the speech panel's synthesizer (resampled to the screen's 16 kHz), and
the exchange lands in the agent's memory like any other thought.
"""

import array
import asyncio
import base64
import difflib
import math
import re
import unicodedata
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from worldwarpixel.sim.agents import AgentSpec
from worldwarpixel.sim.canvas import PALETTE
from worldwarpixel.sim.run import Run
from worldwarpixel.sim.speech import voice_for

if TYPE_CHECKING:
    from openai import AsyncOpenAI

RATE = 16_000
"""Screens play what they record: 16 kHz signed 16-bit little-endian mono."""
MAX_TURNS = 6
MAX_REPLY = 400
MAX_AGENTS = 120
REALTIME_RATE: Literal[24000] = 24_000
REALTIME_VOICES = ("alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin")
REALTIME_TIMEOUT = 30.0

Chat = Callable[[Run, list[dict[str, str]]], Awaitable[str]]
"""Ask the run's model for one completion (the app wires the real client; tests fake it)."""


class InterviewError(RuntimeError):
    pass


_PUNCT = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
    | {"\u2026": "...", "\u00a0": " "}
)


def ascii_text(text: str) -> str:
    """What a screen's 5x7 ASCII font can draw: accents dropped, typographic punctuation plain."""
    folded = unicodedata.normalize("NFKD", text.translate(_PUNCT))
    return folded.encode("ascii", "ignore").decode()


def agent_brief(run: Run, agent: AgentSpec) -> dict[str, Any]:
    return {
        "id": agent.id,
        "name": ascii_text(agent.name).strip() or f"Agent {agent.id}",
        "country": ascii_text(agent.country),
        "color": PALETTE[agent.color],
        "society": run.societies.of(agent.id) if run.societies else None,
    }


def agent_list(run: Run) -> list[dict[str, Any]]:
    """Who a screen can talk to: grouped by society, then by name."""
    briefs = [agent_brief(run, a) for a in run.agents]
    briefs.sort(key=lambda b: (b["society"] if b["society"] is not None else 0, b["name"]))
    return briefs[:MAX_AGENTS]


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def pick_agent(run: Run, heard: str) -> AgentSpec | None:
    """Match a spoken request ("let me talk to Imran", "someone from Peru") to an agent.

    Speech recognition mangles foreign names, so near misses on a name count too.
    """
    words = re.findall(r"[a-z0-9]+", _fold(heard))
    text = f" {' '.join(words)} "
    best: AgentSpec | None = None
    best_score = 0.0
    for agent in run.agents:
        parts = re.findall(r"[a-z0-9]+", _fold(agent.name))
        if not parts:
            continue
        score = 0.0
        if f" {' '.join(parts)} " in text:
            score = 4.0
        elif parts[0] in words and len(parts[0]) > 2:
            score = 3.0
        elif any(p in words for p in parts[1:] if len(p) > 2):
            score = 2.5
        else:
            near = [
                difflib.SequenceMatcher(None, p, w).ratio()
                for p in parts
                if len(p) > 2
                for w in words
                if len(w) > 2
            ]
            if near and max(near) >= 0.75:
                score = 1.0 + max(near)
        country = " ".join(re.findall(r"[a-z0-9]+", _fold(agent.country)))
        if country and f" {country} " in text:
            score = max(score, 1.0)
        if score > best_score:
            best, best_score = agent, score
    return best


def resample(pcm: bytes, src: int, dst: int = RATE) -> bytes:
    """Linear-interpolation resample of s16le mono PCM."""
    if src == dst or len(pcm) < 4:
        return pcm
    s = array.array("h")
    s.frombytes(pcm[: len(pcm) // 2 * 2])
    n = len(s) * dst // src
    out = array.array("h", bytes(2 * n))
    step = src / dst
    last = len(s) - 1
    for i in range(n):
        p = i * step
        j = int(p)
        a = s[j]
        b = s[j + 1] if j < last else a
        out[i] = int(a + (b - a) * (p - j))
    return out.tobytes()


def messages(
    run: Run, agent: AgentSpec, history: list[tuple[str, str]], question: str
) -> list[dict[str, str]]:
    context = [
        agent.persona or f"You are {agent.name} from {agent.country}.",
        "",
        "Right now a human visitor is standing at a screen next to the world you live in and "
        "is talking to you out loud. Answer as yourself, in the first person, the way you would "
        "actually speak: one to three short sentences, plain words, no lists, no markdown, no "
        "emoji. Reply in the language the visitor uses.",
    ]
    if run.societies is not None:
        society = run.societies.get(run.societies.of(agent.id))
        context.append(f"You currently belong to the society called {society.name}.")
    recent = run.recall(agent.id, 5)
    if recent:
        context += ["", "What you have been doing lately:", *(f"- {r}" for r in recent)]
    out = [{"role": "system", "content": "\n".join(context)}]
    for asked, said in history[-MAX_TURNS:]:
        out += [{"role": "user", "content": asked}, {"role": "assistant", "content": said}]
    out.append({"role": "user", "content": question})
    return out


def clean_reply(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("*", "").strip().strip('"').strip()
    if len(text) > MAX_REPLY:
        cut = text[:MAX_REPLY]
        end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        text = cut[: end + 1] if end > MAX_REPLY // 2 else cut
    return text


async def answer(
    chat: Chat, run: Run, agent: AgentSpec, history: list[tuple[str, str]], question: str
) -> str:
    try:
        raw = await chat(run, messages(run, agent, history, question))
    except InterviewError:
        raise
    except Exception as e:
        raise InterviewError(f"{agent.name} could not answer ({type(e).__name__})") from e
    reply = clean_reply(raw)
    if not reply:
        raise InterviewError(f"{agent.name} said nothing")
    history.append((question, reply))
    del history[:-MAX_TURNS]
    run.note_thought(
        agent.id, f'A visitor at a screen asked me: "{question}" I answered: "{reply}"'
    )
    return reply


async def voice(run: Run, agent: AgentSpec, text: str) -> AsyncGenerator[bytes, None]:
    """The reply as 16 kHz PCM in the agent's speech-panel voice, streamed as it is synthesized
    when the synthesizer can stream; yields nothing without a synthesizer."""
    synth = run.speech.synth
    if synth is None:
        return
    voice = voice_for(agent)
    instructions = (
        f"You are {agent.name} from {agent.country}, talking to a visitor. Speak naturally."
    )
    stream = getattr(synth, "stream", None)
    if stream is None:
        yield resample(await synth.synth(text, voice, instructions), synth.sample_rate)
        return
    # Resample whole src:dst sample blocks so chunk edges land on output samples.
    block = 2 * (synth.sample_rate // math.gcd(synth.sample_rate, RATE))
    block *= max(1, 2400 // block)  # resample() passes clips under 2 samples through as-is
    pending = b""
    async for chunk in stream(text, voice, instructions):
        pending += chunk
        cut = len(pending) - len(pending) % block
        if cut:
            yield resample(pending[:cut], synth.sample_rate)
            pending = pending[cut:]
    if len(pending) >= 2:
        yield resample(pending, synth.sample_rate)


def realtime_voice(agent: AgentSpec) -> str:
    v = voice_for(agent)
    return v if v in REALTIME_VOICES else REALTIME_VOICES[agent.id % len(REALTIME_VOICES)]


def realtime_instructions(run: Run, agent: AgentSpec, history: list[tuple[str, str]]) -> str:
    text = messages(run, agent, [], "")[0]["content"]
    if history:
        turns = "\n".join(f'Visitor: "{q}"\nYou: "{a}"' for q, a in history[-MAX_TURNS:])
        text += f"\n\nEarlier in this conversation:\n{turns}"
    return text


async def live(
    client: "AsyncOpenAI",
    model: str,
    run: Run,
    agent: AgentSpec,
    history: list[tuple[str, str]],
    pcm: bytes,
    rate: int,
) -> AsyncGenerator[tuple[str, Any], None]:
    """Answer the visitor's clip speech-to-speech with a Realtime model.

    Yields ``("heard", question)`` once the clip is transcribed, ``("audio", pcm16k)`` as the
    answer is spoken, and finally ``("text", reply)`` with the transcript of what the agent said.
    ``("spoken", None)`` means the answer is fully spoken while the question's transcript is
    still pending.
    The exchange then lands in history and the agent's memory like a typed interview.
    """
    question = reply = ""
    try:
        async with asyncio.timeout(REALTIME_TIMEOUT), client.realtime.connect(model=model) as conn:
            await conn.session.update(
                session={
                    "type": "realtime",
                    "instructions": realtime_instructions(run, agent, history),
                    "output_modalities": ["audio"],
                    "max_output_tokens": 600,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                            "transcription": {"model": "gpt-4o-mini-transcribe"},
                            "turn_detection": None,
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                            "voice": realtime_voice(agent),
                        },
                    },
                }
            )
            clip = resample(pcm, rate, REALTIME_RATE)
            for i in range(0, len(clip), 480_000):
                await conn.input_audio_buffer.append(
                    audio=base64.b64encode(clip[i : i + 480_000]).decode()
                )
            await conn.input_audio_buffer.commit()
            await conn.response.create()
            heard = answered = False
            pending = b""
            async for event in conn:
                if event.type == "conversation.item.input_audio_transcription.completed":
                    question = " ".join(event.transcript.split())
                    heard = True
                    yield ("heard", question)
                elif event.type == "conversation.item.input_audio_transcription.failed":
                    heard = True
                    yield ("heard", "")
                elif event.type == "response.output_audio.delta":
                    pending += base64.b64decode(event.delta)
                    cut = len(pending) - len(pending) % 2400  # whole 3:2 sample blocks
                    if cut:
                        yield ("audio", resample(pending[:cut], REALTIME_RATE))
                        pending = pending[cut:]
                elif event.type == "response.output_audio_transcript.done":
                    reply = event.transcript
                elif event.type == "response.done":
                    if event.response.status == "failed":
                        raise InterviewError(f"{agent.name} could not answer")
                    answered = True
                    if not heard:
                        yield (
                            "spoken",
                            None,
                        )  # done talking; don't sit on audio for the transcript
                elif event.type == "error":
                    raise InterviewError(f"{agent.name} could not answer ({event.error.message})")
                if answered and heard:
                    break
            if len(pending) >= 2:
                yield ("audio", resample(pending, REALTIME_RATE))
    except InterviewError:
        raise
    except Exception as e:
        raise InterviewError(f"{agent.name} could not answer ({type(e).__name__})") from e
    reply = clean_reply(reply)
    yield ("text", reply)
    if reply:
        history.append((question or "(inaudible)", reply))
        del history[:-MAX_TURNS]
        run.note_thought(
            agent.id, f'A visitor at a screen asked me: "{question}" I answered: "{reply}"'
        )
