from typing import AsyncIterator

from loguru import logger

from llm.base import BaseLLM


class ClaudeProvider(BaseLLM):
    """Anthropic Claude provider with streaming support."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream tokens from Claude API."""
        if not self.api_key:
            logger.error("Claude API key not configured.")
            yield "I need to be connected to the internet for this. Ask your parent to check the settings!"
            return

        self._ensure_client()

        try:
            # Separate system message from conversation
            system_msg = ""
            conversation = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    conversation.append(msg)

            async with self._client.messages.stream(
                model=self.model,
                max_tokens=150,
                temperature=0.4,
                system=system_msg,
                messages=conversation,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error("Claude streaming error: {}", e)
            yield "Hmm, I couldn't think of an answer right now. Let's try again!"
