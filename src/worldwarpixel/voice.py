"""Turning a human's spoken words into a society murmur: transcribe, then condense."""

import asyncio
import io
import logging
import wave
from typing import Any, Protocol

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
    import numpy as np

    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, channels).mean(axis=1)
    return samples.astype("<i2").tobytes(), rate


class WhisperTranscriber:
    """Local faster-whisper on the CPU; the model loads on first use."""

    def __init__(self, model_size: str = "base") -> None:
        self._size = model_size
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _load(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TranscriptionUnavailable(
                "faster-whisper is not installed (uv sync --extra voice)"
            ) from e
        return WhisperModel(self._size, device="cpu", compute_type="int8")

    def _run(self, pcm: bytes, rate: int) -> str:
        import numpy as np

        if self._model is None:
            self._model = self._load()
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if rate != RATE and len(audio):
            n = int(len(audio) * RATE / rate)
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
            audio = audio.astype(np.float32)
        segments, _ = self._model.transcribe(audio, beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

    async def __call__(self, pcm: bytes, rate: int) -> str:
        async with self._lock:  # one clip at a time; whisper on CPU is not re-entrant-friendly
            return await asyncio.to_thread(self._run, pcm, rate)


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
