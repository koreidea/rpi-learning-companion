import asyncio
import time
from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from llm.base import BaseLLM


class OfflineLLM(BaseLLM):
    """Local LLM using llama-cpp-python.

    Runs Qwen 2.5 3B Instruct (Q4_K_M) on the RPi 5 with streaming output.
    Uses standard attention (no SWA), giving ~6 tok/s on Pi 5.
    Falls back to Llama 3.2 1B if primary model is not found.
    """

    # Model preference order (name, ctx, threads, max_tokens)
    # Qwen 2.5 3B: standard attention, fast on CPU, great quality
    # Gemma 3 4B: SWA architecture = ~30s prompt processing on Pi 5, too slow
    # Llama 3.2 1B: smallest fallback
    MODELS = [
        ("Qwen2.5-3B-Instruct-Q4_K_M.gguf", 1024, 4, 80),
        ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", 768, 3, 60),
    ]

    def __init__(
        self,
        model_dir: Path,
        model_name: str | None = None,
        n_ctx: int = 512,
        n_threads: int = 4,
        max_tokens: int = 60,
        temperature: float = 0.5,
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

        # Auto-detect best available model if no explicit name given
        if self.model_name is None:
            for name, ctx, threads, max_tok in self.MODELS:
                if (self.model_dir / name).exists():
                    self.model_name = name
                    self.n_ctx = ctx
                    self.n_threads = threads
                    self.max_tokens = max_tok
                    break
            else:
                logger.error("No LLM model found in {}. Run download_models.sh.", self.model_dir)
                return

        model_path = self.model_dir / self.model_name
        if not model_path.exists():
            logger.error("Model not found: {}. Run download_models.sh first.", model_path)
            return

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=64,
            flash_attn=True,
            use_mmap=True,
            use_mlock=True,  # Lock model in RAM — prevents page faults during inference
            verbose=False,
        )
        logger.info("Offline LLM loaded: {} (ctx={}, threads={}, max_tokens={})",
                     self.model_name, self.n_ctx, self.n_threads, self.max_tokens)

        # Warmup: run a full inference with the real system prompt to prime
        # all memory pages, KV-cache, and prompt processing pipeline.
        # Without this, the first real query takes ~25s extra on Pi 5.
        t0 = time.monotonic()
        try:
            from llm.prompts import build_system_prompt
            warmup_system = build_system_prompt()
            self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": warmup_system},
                    {"role": "user", "content": "Tell me something fun!"},
                ],
                max_tokens=10,
                temperature=0.0,
            )
            logger.info("LLM warmup done in {:.1f}s", time.monotonic() - t0)
        except Exception as e:
            logger.warning("LLM warmup failed (non-fatal): {}", e)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream tokens from the local LLM."""
        if self._llm is None:
            logger.error("Offline LLM not loaded!")
            yield "Sorry, I'm having trouble thinking right now. Let's try again!"
            return

        loop = asyncio.get_event_loop()

        # Use asyncio.Queue for zero-overhead async token passing
        token_queue: asyncio.Queue = asyncio.Queue()

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
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put(content), loop
                        )
            except Exception as e:
                logger.error("Offline LLM generation error: {}", e)
            finally:
                asyncio.run_coroutine_threadsafe(
                    token_queue.put(None), loop
                )

        # Run generation in background thread
        loop.run_in_executor(None, generate)

        # Yield tokens as they arrive — no polling, true async await
        while True:
            token = await token_queue.get()
            if token is None:
                break
            yield token

    async def unload(self):
        """Unload the model to free RAM."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("Offline LLM unloaded.")
