import asyncio
import logging
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


def test_http_request_logging_skips_fast_health_and_warns_when_slow(
    client: TestClient, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("worldwarpixel.app.SLOW_REQUEST_MS", 60_000)
    with caplog.at_level(logging.INFO, logger="worldwarpixel.http"):
        config = client.get("/health/config")
        health = client.get("/health")
    assert config.status_code == health.status_code == 200
    info_records = [record for record in caplog.records if record.name == "worldwarpixel.http"]
    assert len(info_records) == 1
    assert info_records[0].levelno == logging.INFO
    assert "GET /health/config -> 200" in info_records[0].getMessage()

    caplog.clear()
    monkeypatch.setattr("worldwarpixel.app.SLOW_REQUEST_MS", 0)
    with caplog.at_level(logging.INFO, logger="worldwarpixel.http"):
        health = client.get("/health")
    assert health.status_code == 200
    [record] = [r for r in caplog.records if r.name == "worldwarpixel.http"]
    assert record.levelno == logging.WARNING
    assert "GET /health -> 200" in record.getMessage()


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
