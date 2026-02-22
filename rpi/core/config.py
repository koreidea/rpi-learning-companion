import json
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field


class DeviceConfig(BaseModel):
    device_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    first_boot_complete: bool = False


class ChildConfig(BaseModel):
    age_min: int = 3
    age_max: int = 6
    language: str = "en"


class APIKeysConfig(BaseModel):
    openai: str = ""
    gemini: str = ""
    claude: str = ""


class PrivacyConfig(BaseModel):
    consent_given: bool = False
    consent_timestamp: Optional[str] = None
    session_log_retention_days: int = 30


class HardwareConfig(BaseModel):
    mic_enabled: bool = True
    camera_enabled: bool = True
    wake_word: str = "hey jarvis"
    cloud_stt: bool = False  # When True + online mode, use OpenAI Whisper API


class AppConfig(BaseModel):
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    mode: str = "offline"  # "offline" or "online"
    provider: str = "openai"  # "openai", "gemini", or "claude"
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    child: ChildConfig = Field(default_factory=ChildConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)


class ConfigManager:
    """Manages application configuration with encrypted persistence."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.config_path = data_dir / "config.json"
        self._config: Optional[AppConfig] = None
        self._encryption = None  # Set after privacy module init

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            self._config = self._load()
        return self._config

    def set_encryption(self, encryption):
        """Attach encryption handler once privacy module is initialized."""
        self._encryption = encryption

    def _load(self) -> AppConfig:
        """Load config from disk. Returns defaults if no config exists."""
        if self.config_path.exists():
            try:
                raw = self.config_path.read_text()
                if self._encryption:
                    raw = self._encryption.decrypt(raw)
                data = json.loads(raw)
                logger.info("Configuration loaded from {}", self.config_path)
                return AppConfig(**data)
            except Exception as e:
                logger.error("Failed to load config: {}. Using defaults.", e)
        logger.info("No existing config found. Using defaults.")
        return AppConfig()

    def save(self) -> None:
        """Persist current config to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        raw = self.config.model_dump_json(indent=2)
        if self._encryption:
            raw = self._encryption.encrypt(raw)
        self.config_path.write_text(raw)
        logger.debug("Configuration saved to {}", self.config_path)

    def update(self, **kwargs) -> AppConfig:
        """Update top-level config fields and save."""
        current = self.config.model_dump()
        for key, value in kwargs.items():
            if key in current:
                if isinstance(current[key], dict) and isinstance(value, dict):
                    current[key].update(value)
                else:
                    current[key] = value
        self._config = AppConfig(**current)
        self.save()
        return self._config

    def update_nested(self, section: str, **kwargs) -> AppConfig:
        """Update fields within a nested config section."""
        current = self.config.model_dump()
        if section in current and isinstance(current[section], dict):
            current[section].update(kwargs)
        self._config = AppConfig(**current)
        self.save()
        return self._config

    def reset(self) -> None:
        """Reset config to defaults (used for data erasure)."""
        self._config = AppConfig()
        if self.config_path.exists():
            self.config_path.unlink()
        logger.info("Configuration reset to defaults.")

    @property
    def is_online(self) -> bool:
        return self.config.mode == "online"

    @property
    def has_consent(self) -> bool:
        return self.config.privacy.consent_given

    @property
    def is_setup_complete(self) -> bool:
        return self.config.device.first_boot_complete
