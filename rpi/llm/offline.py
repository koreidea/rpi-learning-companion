import asyncio
from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from llm.base import BaseLLM


class OfflineLLM(BaseLLM):
    """Local LLM using llama-cpp-python.

    Runs Llama 3.2 1B (Q4_K_M) on the RPi 5 with streaming output.
    """

    def __init__(
        self,
        model_dir: Path,
        model_name: str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        n_ctx: int = 2048,
        n_threads: int = 4,
        max_tokens: int = 150,
        temperature: float = 0.4,
    ):
        self.model_dir = model_dir
        self.model_name = model_name
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._llm = None

    async def load(self):
        """Load the GGUF model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        from llama_cpp import Llama

        model_path = self.model_dir / self.model_name
        if not model_path.exists():
            logger.error("Model not found: {}. Run download_models.sh first.", model_path)
            return

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            use_mmap=True,
            verbose=False,
        )
        logger.info("Offline LLM loaded: {} (ctx={})", self.model_name, self.n_ctx)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream tokens from the local LLM."""
        if self._llm is None:
            logger.error("Offline LLM not loaded!")
            yield "Sorry, I'm having trouble thinking right now. Let's try again!"
            return

        loop = asyncio.get_event_loop()

        # llama-cpp-python's create_chat_completion with stream=True
        # returns a generator, so we process it in an executor
        import queue
        token_queue = queue.Queue()
        done_event = asyncio.Event()

        def generate():
            try:
                response = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True,
                )
                for chunk in response:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        token_queue.put(content)
            except Exception as e:
                logger.error("Offline LLM generation error: {}", e)
            finally:
                token_queue.put(None)  # Sentinel

        # Run generation in background thread
        loop.run_in_executor(None, generate)

        # Yield tokens as they arrive
        while True:
            try:
                token = await loop.run_in_executor(None, token_queue.get, True, 0.1)
            except Exception:
                await asyncio.sleep(0.05)
                continue

            if token is None:
                break
            yield token

    async def unload(self):
        """Unload the model to free RAM."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("Offline LLM unloaded.")
