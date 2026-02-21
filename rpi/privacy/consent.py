from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from core.config import ConfigManager


class ConsentManager:
    """Manages DPDP Act parental consent state.

    Key principles:
    - No interaction with the child until consent is given
    - Consent must be explicit and verifiable
    - Consent can be revoked at any time
    - Consent record includes timestamp and scope
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    @property
    def has_consent(self) -> bool:
        return self.config_manager.config.privacy.consent_given

    @property
    def consent_timestamp(self) -> str | None:
        return self.config_manager.config.privacy.consent_timestamp

    def grant_consent(self) -> None:
        """Record parental consent with timestamp."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self.config_manager.update_nested(
            "privacy",
            consent_given=True,
            consent_timestamp=timestamp,
        )
        logger.info("Parental consent granted at {}", timestamp)

    def revoke_consent(self) -> None:
        """Revoke parental consent."""
        self.config_manager.update_nested(
            "privacy",
            consent_given=False,
            consent_timestamp=None,
        )
        logger.info("Parental consent revoked.")

    def get_consent_info(self) -> dict:
        """Return consent details for the parent dashboard."""
        config = self.config_manager.config
        return {
            "consent_given": config.privacy.consent_given,
            "consent_timestamp": config.privacy.consent_timestamp,
            "data_collected": [
                "Session timestamps and duration",
                "Topic categories (e.g., 'math', 'animals')",
            ],
            "data_not_collected": [
                "Voice recordings (processed in memory only)",
                "Conversation text (discarded after response)",
                "Personal information",
                "Behavioral profiles",
            ],
            "data_retention": f"{config.privacy.session_log_retention_days} days",
            "online_mode_note": (
                "When online mode is enabled, the text of the child's question "
                "(not audio) is sent to a cloud AI provider for processing. "
                "No data is stored by the cloud provider beyond the request."
            ),
        }
