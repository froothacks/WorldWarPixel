"""Turning a human's spoken words into a society murmur: transcribe, then condense."""

import io
import logging
import sys
import wave
from array import array
from typing import Protocol

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

RATE = 16_000
MAX_SECONDS = 20
MAX_MURMUR = 200


class TranscriptionUnavailable(RuntimeError):
    pass


class Transcriber(Protocol):
    async def __call__(self, pcm: bytes, rate: int) -> str: ...


def wav_to_pcm(data: bytes) -> tuple[bytes, int]:
    """16-bit WAV (mono or stereo) -> mono 16-bit little-endian PCM and its sample rate."""
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("WAV must be 16-bit")
        frames, channels, rate = w.readframes(w.getnframes()), w.getnchannels(), w.getframerate()
    if channels == 1:
        return frames, rate
    samples = array("h", frames)
    if sys.byteorder == "big":
        samples.byteswap()
    mono = array(
        "h",
        (int(sum(samples[i : i + channels]) / channels) for i in range(0, len(samples), channels)),
    )
    if sys.byteorder == "big":
        mono.byteswap()
    return mono.tobytes(), rate


class NoTranscriber:
    """Stands in when no hosted transcription is configured."""

    async def __call__(self, pcm: bytes, rate: int) -> str:
        raise TranscriptionUnavailable("voice clips need TTS_API_KEY (an OpenAI key)")


class OpenAITranscriber:
    """Hosted transcription (OpenAI /audio/transcriptions): no CPU work on this server."""

    def __init__(self, client: AsyncOpenAI, model: str, timeout: float = 30.0) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout

    async def __call__(self, pcm: bytes, rate: int) -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        r = await self._client.audio.transcriptions.create(
            model=self._model,
            file=("clip.wav", buf.getvalue(), "audio/wav"),
            timeout=self._timeout,
        )
        return r.text.strip()


CONDENSE_PROMPT = (
    "Rewrite what someone said as a short rumour that is going around a community: what people "
    "there have been saying lately. Third person, no speaker, no quotes, at most 25 words, same "
    "language as the input. Output only the rumour."
)


class Murmurer:
    """Condenses a transcript into the unattributed murmur members of a society notice."""

    def __init__(self, client: AsyncOpenAI | None, model: str, timeout: float = 30.0) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout

    async def __call__(self, transcript: str) -> str:
        fallback = transcript.strip()[:MAX_MURMUR]
        if self._client is None:
            return fallback
        try:
            r = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": CONDENSE_PROMPT},
                    {"role": "user", "content": transcript[:2000]},
                ],
                max_tokens=80,
                temperature=0.7,
                timeout=self._timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:
            log.warning("murmur condense failed: %s", e)
            return fallback
        text = (r.choices[0].message.content or "").strip().strip('"')
        return text[:MAX_MURMUR] or fallback
