import asyncio
from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionMessageParam

from worldwarpixel.app import app, get_chat_service
from worldwarpixel.config import get_settings
from worldwarpixel.llm import ChatService


class FakeChatService(ChatService):
    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(10)

    async def chat(
        self, messages: Sequence[ChatCompletionMessageParam], model: str | None = None
    ) -> str:
        last = messages[-1]
        return f"echo:{last.get('content')}"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    get_settings.cache_clear()
    app.dependency_overrides[get_chat_service] = FakeChatService
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_chat(client: TestClient) -> None:
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.json() == {"reply": "echo:hi"}


def test_chat_batch(client: TestClient) -> None:
    resp = client.post(
        "/chat/batch",
        json={
            "conversations": [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"replies": ["echo:a", "echo:b"]}


def test_chat_rejects_empty(client: TestClient) -> None:
    assert client.post("/chat", json={"messages": []}).status_code == 422
