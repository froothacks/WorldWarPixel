import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from worldwarpixel.api import RunManager, random_policy
from worldwarpixel.api import router as runs_router
from worldwarpixel.config import Settings, get_settings
from worldwarpixel.llm import ChatService
from worldwarpixel.sim.agents import Policy
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.llm_policy import LLMPolicy
from worldwarpixel.sim.narrator import OpenAINarratorModel
from worldwarpixel.sim.run import RunConfig
from worldwarpixel.sim.stage import OpenAISpeech, SpeechSynth


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str | None = None


class ChatResponse(BaseModel):
    reply: str


class BatchChatRequest(BaseModel):
    conversations: list[list[Message]] = Field(min_length=1)
    model: str | None = None


class BatchChatResponse(BaseModel):
    replies: list[str]


def _to_params(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    return [m.model_dump() for m in messages]  # type: ignore[misc]


def tts_client(settings: Settings) -> AsyncOpenAI | None:
    """Stage voices need their own endpoint: an OpenAI key, or a local OpenAI-compatible TTS."""
    key = settings.tts_api_key.get_secret_value()
    if not key and not settings.tts_base_url:
        return None
    return AsyncOpenAI(
        api_key=key or "none",
        # Explicit, so an OPENAI_BASE_URL pointing at the chat endpoint is never used for TTS.
        base_url=settings.tts_base_url or "https://api.openai.com/v1",
        timeout=settings.request_timeout,
    )


def stage_model(
    settings: Settings, client: AsyncOpenAI, tts: AsyncOpenAI | None
) -> OpenAINarratorModel | None:
    """STAGE_MODEL on OpenAI when the TTS client is OpenAI, else on the main chat endpoint."""
    if not settings.stage_model:
        return None
    if tts is not None and not settings.tts_base_url:
        return OpenAINarratorModel(
            tts, settings.stage_model, settings.request_timeout, temperature=0.9, extra_body=None
        )
    return OpenAINarratorModel(client, settings.stage_model, settings.request_timeout, 0.9)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    tts = tts_client(settings)
    speech: SpeechSynth | None = (
        OpenAISpeech(tts, settings.tts_model, settings.request_timeout) if tts else None
    )
    async with AsyncOpenAI(
        api_key=settings.api_key(),
        base_url=settings.openai_base_url,
        timeout=settings.request_timeout,
        default_headers=settings.default_headers() or None,
    ) as client:
        app.state.chat_service = ChatService(
            client, settings.openai_model, settings.max_concurrency
        )
        agent_model = settings.agent_model or settings.openai_model

        def make_policy(config: RunConfig) -> Policy:
            if config.policy == "llm":
                return LLMPolicy(
                    client,
                    agent_model,
                    timeout=settings.request_timeout,
                    vision=settings.agent_vision,
                    max_tokens=settings.agent_max_tokens,
                )
            return random_policy(config)

        app.state.runs = RunManager(
            EventLog(settings.event_log_path),
            settings.max_active_runs,
            make_policy,
            OpenAINarratorModel(
                client,
                settings.narrator_model or settings.openai_model,
                settings.request_timeout,
                max_tokens=settings.narrator_max_tokens,
            ),
            speech,
            stage_model(settings, client, tts),
        )
        try:
            yield
        finally:
            await app.state.runs.shutdown()
            if tts is not None:
                await tts.close()


app = FastAPI(title="WorldWarPixel", lifespan=lifespan)
app.include_router(runs_router)


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/config")
async def health_config() -> dict[str, object]:
    """Deployment self-check: which env/config values the process sees (no secret values)."""
    s = get_settings()
    return {
        "openai_base_url": s.openai_base_url,
        "openai_model": s.openai_model,
        "is_modal_endpoint": s.is_modal_endpoint,
        "api_key": s.api_key() != "none",
        "modal_key": bool(s.modal_key.get_secret_value()),
        "modal_secret": bool(s.modal_secret.get_secret_value()),
        "god_token": bool(s.god_token.get_secret_value()),
        "env_keys": sorted(k for k in os.environ if k.startswith(("OPENAI_", "MODAL_", "GOD_"))),
    }


@app.post("/chat")
async def chat(body: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    reply = await service.chat(_to_params(body.messages), body.model)
    return ChatResponse(reply=reply)


@app.post("/chat/batch")
async def chat_batch(body: BatchChatRequest, service: ChatServiceDep) -> BatchChatResponse:
    replies = await service.chat_many([_to_params(c) for c in body.conversations], body.model)
    return BatchChatResponse(replies=replies)


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
API_PREFIXES = {"runs", "chat", "health", "docs", "redoc", "openapi.json"}


async def frontend(path: str) -> FileResponse:
    """SPA fallback: unknown non-API paths get index.html; unknown API paths stay 404."""
    if path.split("/", 1)[0] in API_PREFIXES:
        raise HTTPException(404, "not found")
    return FileResponse(FRONTEND_DIST / "index.html")


if FRONTEND_DIST.is_dir():
    # Serve the built React app from the same process. API routes above take precedence.
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
    app.add_api_route("/{path:path}", frontend, include_in_schema=False)
