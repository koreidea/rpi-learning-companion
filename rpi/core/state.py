import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BotState(str, Enum):
    SETUP = "setup"            # First boot, waiting for parental consent
    LOADING = "loading"        # Models loading
    READY = "ready"            # Listening for wake word
    LISTENING = "listening"    # Wake word detected, capturing speech
    PROCESSING = "processing"  # STT + LLM generating response
    SPEAKING = "speaking"      # TTS playing response
    ERROR = "error"            # Something went wrong


class LLMMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass
class SharedState:
    """Thread-safe shared state for all components."""

    bot_state: BotState = BotState.SETUP
    llm_mode: LLMMode = LLMMode.OFFLINE
    active_provider: str = "openai"
    mic_enabled: bool = True
    camera_enabled: bool = True

    # Set by orchestrator for inter-component signaling
    wake_event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Current interaction info (ephemeral, not persisted)
    current_transcript: Optional[str] = None
    is_model_loaded: bool = False
    last_error: Optional[str] = None

    # Latency tracking
    interaction_start_time: Optional[float] = None

    def set_state(self, state: BotState) -> None:
        self.bot_state = state

    def request_stop(self) -> None:
        self.stop_event.set()

    @property
    def is_running(self) -> bool:
        return not self.stop_event.is_set()

    @property
    def is_ready(self) -> bool:
        return self.bot_state == BotState.READY
