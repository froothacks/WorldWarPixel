from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from worldwarpixel.api import RunManager
from worldwarpixel.api import router as runs_router
from worldwarpixel.config import get_settings
from worldwarpixel.llm import ChatService
from worldwarpixel.sim.events import EventLog


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
    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value() or None,
        base_url=settings.openai_base_url,
        timeout=settings.request_timeout,
    ) as client:
        app.state.chat_service = ChatService(
            client, settings.openai_model, settings.max_concurrency
        )
        app.state.runs = RunManager(EventLog(settings.event_log_path), settings.max_active_runs)
        try:
            yield
        finally:
            await app.state.runs.shutdown()


app = FastAPI(title="WorldWarPixel", lifespan=lifespan)
app.include_router(runs_router)


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(body: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    reply = await service.chat(_to_params(body.messages), body.model)
    return ChatResponse(reply=reply)


@app.post("/chat/batch")
async def chat_batch(body: BatchChatRequest, service: ChatServiceDep) -> BatchChatResponse:
    replies = await service.chat_many([_to_params(c) for c in body.conversations], body.model)
    return BatchChatResponse(replies=replies)


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    # Serve the built React app from the same process. API routes above take precedence.
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str) -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")
