import asyncio
from collections.abc import Sequence

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


class ChatService:
    def __init__(self, client: AsyncOpenAI, max_concurrency: int) -> None:
        self._client = client
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def chat(self, messages: Sequence[ChatCompletionMessageParam], model: str) -> str:
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=model, messages=list(messages)
            )
        return response.choices[0].message.content or ""

    async def chat_many(
        self,
        conversations: Sequence[Sequence[ChatCompletionMessageParam]],
        model: str,
    ) -> list[str]:
        return await asyncio.gather(*(self.chat(c, model) for c in conversations))
