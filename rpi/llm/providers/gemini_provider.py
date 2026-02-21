from typing import AsyncIterator

from loguru import logger

from llm.base import BaseLLM


class GeminiProvider(BaseLLM):
    """Google Gemini provider with streaming support."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream tokens from Gemini API."""
        if not self.api_key:
            logger.error("Gemini API key not configured.")
            yield "I need to be connected to the internet for this. Ask your parent to check the settings!"
            return

        self._ensure_client()

        try:
            # Convert messages to Gemini format
            # Gemini uses a different message format: system instruction + contents
            system_msg = ""
            user_msg = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]

            # Combine system + user for Gemini (it handles system differently)
            prompt = f"{system_msg}\n\nChild says: {user_msg}"

            response = self._client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 150,
                    "temperature": 0.4,
                },
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error("Gemini streaming error: {}", e)
            yield "Hmm, I couldn't think of an answer right now. Let's try again!"
