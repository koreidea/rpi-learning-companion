from typing import AsyncIterator

from loguru import logger

from llm.base import BaseLLM


class OpenAIProvider(BaseLLM):
    """OpenAI ChatGPT provider with streaming support."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream tokens from OpenAI API."""
        if not self.api_key:
            logger.error("OpenAI API key not configured.")
            yield "I need to be connected to the internet for this. Ask your parent to check the settings!"
            return

        self._ensure_client()

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.4,
                stream=True,
            )

            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            logger.error("OpenAI streaming error: {}", e)
            yield "Hmm, I couldn't think of an answer right now. Let's try again!"
