import asyncio
from pathlib import Path

from loguru import logger

from core.config import ConfigManager
from privacy.audit_log import AuditLog


class DataLifecycleManager:
    """Manages automatic data expiry and cleanup.

    Runs as a background task to enforce data retention policies.
    """

    def __init__(self, config_manager: ConfigManager, data_dir: Path):
        self.config_manager = config_manager
        self.audit_log = AuditLog(data_dir / "sessions")
        self._task = None

    async def start(self):
        """Start the background cleanup task."""
        self._task = asyncio.create_task(self._cleanup_loop())
        logger.info("Data lifecycle manager started.")

    async def _cleanup_loop(self):
        """Periodically clean up expired data."""
        while True:
            try:
                retention_days = self.config_manager.config.privacy.session_log_retention_days
                removed = await self.audit_log.cleanup_old(retention_days)
                if removed > 0:
                    logger.info("Auto-cleanup removed {} old log files.", removed)
            except Exception as e:
                logger.error("Data cleanup error: {}", e)

            # Run cleanup once per hour
            await asyncio.sleep(3600)

    async def full_erasure(self):
        """Complete data erasure (Right to Erasure / DPDP compliance)."""
        logger.info("Executing full data erasure...")

        # Erase session logs
        await self.audit_log.erase_all()

        # Reset configuration
        self.config_manager.reset()

        logger.info("Full data erasure complete.")

    def stop(self):
        if self._task:
            self._task.cancel()
