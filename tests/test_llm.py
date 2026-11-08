import asyncio
from types import SimpleNamespace
from typing import Any

from openai import AsyncOpenAI

from worldwarpixel.llm import ChatService


class FakeCompletions:
    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def create(self, *, model: str, messages: list[dict[str, Any]]) -> Any:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        content = f"{model}:{messages[-1]['content']}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _service(completions: FakeCompletions, max_concurrency: int) -> ChatService:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    fake: AsyncOpenAI = client  # type: ignore[assignment]
    return ChatService(fake, max_concurrency)


def test_chat_many_preserves_order_and_limits_concurrency() -> None:
    completions = FakeCompletions()
    service = _service(completions, max_concurrency=3)
    conversations = [[{"role": "user", "content": str(i)}] for i in range(10)]

    replies = asyncio.run(service.chat_many(conversations, "test-model"))  # type: ignore[arg-type]

    assert replies == [f"test-model:{i}" for i in range(10)]
    assert completions.peak == 3
