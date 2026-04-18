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
    DANCING = "dancing"        # Dance mode — synced car + display
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
    cloud_stt: bool = False

    # Set by orchestrator for inter-component signaling
    wake_event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Remote text input (from phone/web dashboard)
    remote_text: Optional[str] = None

    # Image-to-sketch tracing data (from phone upload)
    sketch_from_image: Optional[dict] = None
    # Line art trace image (PIL Image from phone upload)
    trace_image: Optional[dict] = None

    # Current interaction info (ephemeral, not persisted)
    current_transcript: Optional[str] = None
    current_response: Optional[str] = None
    is_model_loaded: bool = False
    last_error: Optional[str] = None
    in_follow_up: bool = False

    # Vision: base64 PNG of last captured image + detected objects
    current_image_b64: Optional[str] = None
    current_detections: Optional[list] = None

    # Conversation history for the live dashboard display.
    # Each item: {"role": "user"/"assistant", "content": "...", "image": optional_b64}
    conversation_messages: list = field(default_factory=list)

    # Volume (0-100)
    volume: int = 80

    # Touch
    touch_enabled: bool = False         # True when touch sensor is initialized
    last_touch_event: Optional[str] = None  # Last touch event name for display reaction

    # TFT Menu (activated by extra_long press on touch sensor)
    menu_open: bool = False
    menu_index: int = 0                 # Currently highlighted menu item

    # Card UI (replaces old settings panel on long press)
    card_mode: str = "off"              # off, cards, arts, encyclopedia, settings
    card_index: int = 0                 # Selected card in horizontal scroll
    card_sub_index: int = 0             # Selected item within a card's sub-screen
    card_scroll_offset: int = 0         # Scroll offset for sub-screens
    card_action: Optional[str] = None   # Pending action from card selection

    # AI-generated story (from Imagine Story)
    generated_story: Optional[dict] = None  # {"title":..., "pages":[{"text":..., "image": PIL},...]}

    # Projector
    projector_connected: bool = False   # True when HDMI projector detected
    projector_mode: str = "off"         # off, flashcard, alphabet, numbers, imagine
    projector_imagine_prompt: Optional[str] = None  # Current imagination prompt

    # Car chassis
    car_connected: bool = False       # True when Pi is connected to HC-05
    car_connecting: bool = False      # True while scanning/connecting
    car_mac: Optional[str] = None     # Saved HC-05 MAC address
    follow_mode: bool = False         # True when follow mode is active

    # WiFi info (refreshed periodically)
    wifi_ssid: Optional[str] = None
    wifi_ip: Optional[str] = None
    wifi_signal: int = 0              # Signal strength 0-100

    # Language
    language: str = "en"              # en, hi, te

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
