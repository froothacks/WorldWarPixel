import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from worldwarpixel.api import RunManager, random_policy, rosters_router
from worldwarpixel.api import router as runs_router
from worldwarpixel.config import get_settings
from worldwarpixel.devices import router as devices_router
from worldwarpixel.llm import ChatService
from worldwarpixel.models import ModelSpec, find_model, model_available, model_registry
from worldwarpixel.sim.agents import Policy
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.llm_policy import LLMPolicy
from worldwarpixel.sim.narrator import NarratorModel, OpenAINarratorModel
from worldwarpixel.sim.run import RunConfig
from worldwarpixel.voice import Murmurer, WhisperTranscriber


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    specs = model_registry(settings)
    app.state.model_specs = specs
    # One client per model preset, created lazily and shared by every run using it.
    clients: dict[str, AsyncOpenAI] = {}

    def client_for(spec: ModelSpec) -> AsyncOpenAI:
        client = clients.get(spec.id)
        if client is None:
            if spec.auth == "modal":
                client = AsyncOpenAI(
                    api_key="none",
                    base_url=spec.base_url,
                    timeout=settings.request_timeout,
                    # Modal presets authenticate with their own headers, scoped to Modal
                    # hosts so an overridden preset URL never receives the credentials.
                    default_headers=settings.modal_headers_for(spec.base_url) or None,
                )
            else:
                client = AsyncOpenAI(
                    api_key=settings.openai_api_key.get_secret_value() or "none",
                    base_url=spec.base_url,
                    timeout=settings.request_timeout,
                )
            clients[spec.id] = client
        return client

    def resolve_model(config: RunConfig) -> ModelSpec:
        spec = find_model(settings, config.model)
        if spec is None:
            known = ", ".join(s.id for s in specs)
            raise HTTPException(422, f"unknown model {config.model!r}; choose one of: {known}")
        if not model_available(settings, spec):
            raise HTTPException(
                400, f"model {spec.id!r} is unavailable: this server lacks its credentials"
            )
        return spec

    async with AsyncOpenAI(
        api_key=settings.api_key(),
        base_url=settings.openai_base_url,
        timeout=settings.request_timeout,
        default_headers=settings.default_headers() or None,
    ) as client:
        app.state.chat_service = ChatService(
            client, settings.openai_model, settings.max_concurrency
        )

        def make_policy(config: RunConfig) -> Policy:
            if config.policy in ("llm", "llm-guided"):
                spec = resolve_model(config)
                return LLMPolicy(
                    client_for(spec),
                    spec.model,
                    timeout=settings.request_timeout,
                    vision=settings.agent_vision and spec.vision,
                    guided=config.policy == "llm-guided",
                    max_tokens=(
                        spec.max_tokens
                        if spec.max_tokens is not None
                        else settings.agent_max_tokens
                    ),
                    token_param=spec.token_param,
                    chat_template_kwargs=spec.chat_template_kwargs,
                    reasoning_effort=spec.reasoning_effort,
                )
            return random_policy(config)

        app.state.transcriber = WhisperTranscriber(settings.whisper_model)
        app.state.murmurer = Murmurer(
            client if settings.murmur_condense else None,
            settings.openai_model,
            settings.request_timeout,
        )

        def make_narrator(config: RunConfig) -> NarratorModel | None:
            if config.policy == "random":
                return None
            spec = resolve_model(config)
            return OpenAINarratorModel(
                client_for(spec),
                spec.model,
                settings.request_timeout,
                max_tokens=(
                    spec.narrator_max_tokens
                    if spec.narrator_max_tokens is not None
                    else settings.narrator_max_tokens
                ),
                token_param=spec.token_param,
                chat_template_kwargs=spec.chat_template_kwargs,
                reasoning_effort=spec.reasoning_effort,
                vision=spec.vision,
            )

        app.state.runs = RunManager(
            EventLog(settings.event_log_path),
            settings.max_active_runs,
            make_policy,
            make_narrator=make_narrator,
        )
        try:
            yield
        finally:
            await app.state.runs.shutdown()
            for model_client in clients.values():
                await model_client.close()


app = FastAPI(title="WorldWarPixel", lifespan=lifespan)
app.include_router(runs_router)
app.include_router(devices_router)
app.include_router(rosters_router)


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def list_models() -> list[dict[str, Any]]:
    """The model presets runs can pick, with whether this server has their credentials."""
    settings = get_settings()
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "model": spec.model,
            "available": model_available(settings, spec),
        }
        for spec in app.state.model_specs
    ]


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
API_PREFIXES = {"runs", "devices", "chat", "health", "models", "docs", "redoc", "openapi.json"}


async def frontend(path: str) -> FileResponse:
    """SPA fallback: unknown non-API paths get index.html; unknown API paths stay 404."""
    if path.split("/", 1)[0] in API_PREFIXES:
        raise HTTPException(404, "not found")
    return FileResponse(FRONTEND_DIST / "index.html")


if FRONTEND_DIST.is_dir():
    # Serve the built React app from the same process. API routes above take precedence.
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
    app.add_api_route("/{path:path}", frontend, include_in_schema=False)
