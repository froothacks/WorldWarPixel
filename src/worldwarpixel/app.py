import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import NOT_GIVEN, AsyncOpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from worldwarpixel.api import RunManager, random_policy, rosters_router, videos_router
from worldwarpixel.api import router as runs_router
from worldwarpixel.config import Settings, get_settings
from worldwarpixel.devices import router as devices_router
from worldwarpixel.interview import InterviewError
from worldwarpixel.llm import ChatService
from worldwarpixel.models import (
    ModelSpec,
    available_endpoints,
    find_model,
    model_available,
    model_registry,
)
from worldwarpixel.routing import EndpointPool, endpoint_client
from worldwarpixel.sim.agents import Policy
from worldwarpixel.sim.events import open_event_log
from worldwarpixel.sim.llm_policy import LLMPolicy
from worldwarpixel.sim.narrator import NarratorModel, OpenAINarratorModel
from worldwarpixel.sim.run import Run, RunConfig
from worldwarpixel.sim.societies import naming_messages, parse_names
from worldwarpixel.sim.speech import OpenAISpeech, SpeechSynth
from worldwarpixel.videos import VideoStore
from worldwarpixel.voice import Murmurer, NoTranscriber, OpenAITranscriber, Transcriber

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

SLOW_REQUEST_MS = 1000
http_logger = logging.getLogger("worldwarpixel.http")


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
    """Speech panel voices use their own OpenAI-compatible endpoint."""
    key = settings.tts_api_key.get_secret_value()
    if not key and not settings.tts_base_url:
        return None
    return AsyncOpenAI(
        api_key=key or "none",
        base_url=settings.tts_base_url or "https://api.openai.com/v1",
        timeout=settings.request_timeout,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    tts = tts_client(settings)
    speech: SpeechSynth | None = (
        OpenAISpeech(tts, settings.tts_model, settings.request_timeout) if tts else None
    )
    specs = model_registry()
    app.state.model_specs = specs
    # One pool per model preset over the endpoints this server has secrets for, created
    # lazily and shared by every run using it (so rate-limit cooldowns are shared too).
    pools: dict[str, EndpointPool] = {}
    app.state.model_pools = pools
    clients: list[AsyncOpenAI] = []

    def pool_for(spec: ModelSpec) -> EndpointPool:
        pool = pools.get(spec.id)
        if pool is None:
            members = []
            for endpoint in available_endpoints(settings, spec):
                client = endpoint_client(endpoint, settings.secret, settings.request_timeout)
                clients.append(client)
                members.append((endpoint, client))
            pool = pools[spec.id] = EndpointPool(members)
        return pool

    def resolve_model(config: RunConfig) -> ModelSpec:
        spec = find_model(config.model)
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
                    pool_for(spec),
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

        async def complete(
            spec: ModelSpec, key: str, messages: list[dict[str, str]], temperature: float
        ) -> str:
            limit = spec.max_tokens if spec.max_tokens is not None else settings.agent_max_tokens
            completion = spec.token_param == "max_completion_tokens"
            reasoning: Any = (
                spec.reasoning_effort if spec.reasoning_effort is not None else NOT_GIVEN
            )
            pool = pool_for(spec)
            try:
                response = await pool.create(
                    key,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    temperature=temperature,
                    timeout=settings.request_timeout,
                    max_tokens=omit if completion else limit,
                    max_completion_tokens=limit if completion else omit,
                    reasoning_effort=reasoning,
                    extra_body=(
                        {"chat_template_kwargs": {"enable_thinking": False}}
                        if spec.chat_template_kwargs
                        else None
                    ),
                )
            finally:
                pool.release(key)
            return response.choices[0].message.content or ""

        async def interview_chat(run: Run, messages: list[dict[str, str]]) -> str:
            if run.config.policy not in ("llm", "llm-guided"):
                raise InterviewError("agents in this world have no language model to talk with")
            spec = find_model(run.config.model)
            if spec is None or not model_available(settings, spec):
                raise InterviewError(f"model {run.config.model!r} is unavailable")
            return await complete(spec, f"interview-{run.id}", messages, 0.9)

        async def name_societies(config: RunConfig) -> list[str] | None:
            """Starting society names from the run's model; None falls back to random names."""
            spec = find_model(config.model)
            if config.policy == "random" or spec is None or not model_available(settings, spec):
                return None
            messages = naming_messages(config.societies, config.society_theme)
            try:
                raw = await asyncio.wait_for(complete(spec, "societies", messages, 1.0), 30)
            except Exception as e:
                logging.getLogger(__name__).warning("society naming failed: %r", e)
                return None
            return parse_names(raw, config.societies) or None

        app.state.interview_chat = interview_chat
        app.state.realtime = (
            tts
            if tts is not None and not settings.tts_base_url and settings.interview_realtime_model
            else None
        )
        app.state.realtime_model = settings.interview_realtime_model
        transcriber: Transcriber = NoTranscriber()
        if tts is not None and not settings.tts_base_url:
            transcriber = OpenAITranscriber(
                tts, settings.transcribe_model, settings.request_timeout
            )
        app.state.transcriber = transcriber
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
                pool_for(spec),
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
            open_event_log(settings),
            settings.max_active_runs,
            make_policy,
            make_narrator=make_narrator,
            speech=speech,
            name_societies=name_societies,
            videos=VideoStore(
                settings.timelapse_dir,
                s3_bucket=settings.timelapse_s3_bucket,
                s3_endpoint=settings.timelapse_s3_endpoint,
                s3_prefix=settings.timelapse_s3_prefix,
                duration=settings.timelapse_duration,
                ffmpeg=settings.ffmpeg_path,
                autosave_interval=settings.timelapse_autosave_interval,
            ),
            auto_video=settings.timelapse_auto,
        )
        if settings.resume_runs:
            app.state.runs.start_resuming()
        try:
            yield
        finally:
            await app.state.runs.shutdown()
            for model_client in clients:
                await model_client.close()
            if tts is not None:
                await tts.close()


app = FastAPI(title="WorldWarPixel", lifespan=lifespan)


@app.middleware("http")
async def log_http_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        http_logger.error(
            "%s %s -> error in %.0f ms", request.method, request.url.path, elapsed_ms, exc_info=True
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    level = logging.WARNING if elapsed_ms >= SLOW_REQUEST_MS else logging.INFO
    if request.url.path != "/health" or level == logging.WARNING:
        http_logger.log(
            level,
            "%s %s -> %d in %.0f ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    return response


app.include_router(runs_router)
app.include_router(devices_router)
app.include_router(rosters_router)
app.include_router(videos_router)


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def list_models() -> list[dict[str, Any]]:
    """The model presets runs can pick, with whether this server has their credentials, and
    per endpoint whether its secrets are present plus call/error/failover counts so far."""
    settings = get_settings()
    out: list[dict[str, Any]] = []
    for spec in app.state.model_specs:
        pool: EndpointPool | None = app.state.model_pools.get(spec.id)
        usage = pool.public_stats() if pool is not None else {}
        ready = {e.name for e in available_endpoints(settings, spec)}
        out.append(
            {
                "id": spec.id,
                "label": spec.label,
                "model": spec.model,
                "available": bool(ready),
                "endpoints": [
                    {
                        "name": e.name,
                        "base_url": e.base_url,
                        "secrets": list(e.secrets),
                        "available": e.name in ready,
                        **usage.get(
                            e.name, {"calls": 0, "errors": 0, "failovers": 0, "cooling": False}
                        ),
                    }
                    for e in spec.endpoints
                ],
            }
        )
    return out


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
        "model_secrets": {
            name: bool(s.secret(name))
            for spec in app.state.model_specs
            for e in spec.endpoints
            for name in e.secrets
        },
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
