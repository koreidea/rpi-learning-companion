from abc import ABC, abstractmethod
from typing import AsyncIterator

from loguru import logger

from core.config import ConfigManager
from core.state import LLMMode, SharedState
from llm.prompts import build_system_prompt


class BaseLLM(ABC):
    """Abstract base class for all LLM providers (local and cloud)."""

    @abstractmethod
    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream tokens from the LLM.

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Yields:
            Token strings one at a time.
        """
        ...

    async def load(self):
        """Load the model (for local LLMs)."""
        pass

    async def unload(self):
        """Unload the model to free RAM."""
        pass


class LLMRouter:
    """Routes LLM requests to the appropriate provider based on config."""

    def __init__(self, config_manager: ConfigManager, state: SharedState):
        self.config_manager = config_manager
        self.state = state
        self._offline_llm = None
        self._providers: dict[str, BaseLLM] = {}

    async def load(self):
        """Load the appropriate LLM based on current mode."""
        if self.state.llm_mode == LLMMode.OFFLINE:
            await self._ensure_offline_llm()
        logger.info("LLM Router ready. Mode: {}", self.state.llm_mode.value)

    async def _ensure_offline_llm(self):
        """Load the local LLM if not already loaded."""
        if self._offline_llm is None:
            from llm.offline import OfflineLLM
            from core.main import MODELS_DIR

            self._offline_llm = OfflineLLM(model_dir=MODELS_DIR / "llm")
            await self._offline_llm.load()

    def _get_online_provider(self, name: str) -> BaseLLM:
        """Get or create a cloud LLM provider.

        Re-reads the API key from config each time so that key updates
        via the settings page take effect without restart.
        """
        api_keys = self.config_manager.config.api_keys
        current_key = getattr(api_keys, name, "")

        # Recreate the provider if the key changed or first time
        cached = self._providers.get(name)
        if cached is not None and getattr(cached, "api_key", None) == current_key:
            return cached

        if name == "openai":
            from llm.providers.openai_provider import OpenAIProvider
            self._providers[name] = OpenAIProvider(api_key=current_key)
        elif name == "gemini":
            from llm.providers.gemini_provider import GeminiProvider
            self._providers[name] = GeminiProvider(api_key=current_key)
        elif name == "claude":
            from llm.providers.claude_provider import ClaudeProvider
            self._providers[name] = ClaudeProvider(api_key=current_key)

        logger.info("Online provider '{}' initialized.", name)
        return self._providers[name]

    def get_provider(self) -> BaseLLM:
        """Get the active LLM provider based on current mode and settings."""
        if self.state.llm_mode == LLMMode.ONLINE:
            provider_name = self.state.active_provider
            return self._get_online_provider(provider_name)

        # Offline mode — ensure local model is ready
        if self._offline_llm is None:
            logger.warning("Offline LLM not loaded yet!")
        return self._offline_llm

    def build_messages(self, user_text: str) -> list[dict]:
        """Build the message list with system prompt and user input."""
        config = self.config_manager.config
        system_prompt = build_system_prompt(
            age_min=config.child.age_min,
            age_max=config.child.age_max,
            language=config.child.language,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

    async def unload(self):
        """Unload all models."""
        if self._offline_llm:
            await self._offline_llm.unload()
            self._offline_llm = None
