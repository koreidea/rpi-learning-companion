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
            # Convert OpenAI-style messages to Gemini format.
            # Gemini uses system_instruction on the model and a list of
            # {"role": "user"/"model", "parts": [text]} for conversation.
            system_msg = ""
            gemini_history = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    gemini_history.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    gemini_history.append({"role": "model", "parts": [msg["content"]]})

            # Use system instruction if supported, otherwise prepend to first user msg
            if system_msg and gemini_history:
                first = gemini_history[0]
                first["parts"] = [f"{system_msg}\n\nChild says: {first['parts'][0]}"]

            # The last message is the current user turn
            # Start a chat with the history (all but last) and send the last message
            if len(gemini_history) > 1:
                chat = self._client.start_chat(history=gemini_history[:-1])
                last_msg = gemini_history[-1]["parts"][0]
            else:
                chat = self._client.start_chat()
                last_msg = gemini_history[0]["parts"][0] if gemini_history else ""

            response = chat.send_message(
                last_msg,
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
