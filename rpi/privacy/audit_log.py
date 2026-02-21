import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger


class AuditLog:
    """Session audit log — stores ONLY metadata, never conversation content.

    DPDP compliance: We log timestamps, durations, and topic categories
    but NEVER the actual speech or text of conversations.
    """

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    async def log_session(
        self,
        duration_seconds: float,
        topic_category: str = "general",
        mode: str = "offline",
        had_vision: bool = False,
    ) -> None:
        """Log a session interaction (metadata only)."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 1),
            "topic_category": topic_category,
            "mode": mode,
            "had_vision": had_vision,
        }

        # Store in daily log files
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.sessions_dir / f"{today}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.debug("Session logged: {}s, topic={}", duration_seconds, topic_category)

    async def get_recent_sessions(self, limit: int = 50) -> list[dict]:
        """Get recent session entries."""
        sessions = []
        log_files = sorted(self.sessions_dir.glob("*.jsonl"), reverse=True)

        for log_file in log_files:
            with open(log_file) as f:
                for line in f:
                    try:
                        sessions.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
                    if len(sessions) >= limit:
                        return sessions

        return sessions

    async def get_all_sessions(self) -> list[dict]:
        """Get all session entries (for data export)."""
        return await self.get_recent_sessions(limit=100_000)

    async def get_stats(self) -> dict:
        """Calculate usage statistics for the dashboard."""
        sessions = await self.get_all_sessions()

        if not sessions:
            return {
                "total_sessions": 0,
                "total_duration_minutes": 0,
                "topics": {},
                "sessions_today": 0,
                "avg_session_seconds": 0,
            }

        total_duration = sum(s.get("duration_seconds", 0) for s in sessions)
        topics = {}
        for s in sessions:
            cat = s.get("topic_category", "general")
            topics[cat] = topics.get(cat, 0) + 1

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sessions_today = sum(1 for s in sessions if s["timestamp"].startswith(today))

        return {
            "total_sessions": len(sessions),
            "total_duration_minutes": round(total_duration / 60, 1),
            "topics": topics,
            "sessions_today": sessions_today,
            "avg_session_seconds": round(total_duration / len(sessions), 1),
        }

    async def cleanup_old(self, retention_days: int = 30) -> int:
        """Delete session logs older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        removed = 0

        for log_file in self.sessions_dir.glob("*.jsonl"):
            date_str = log_file.stem  # filename is YYYY-MM-DD.jsonl
            if date_str < cutoff_str:
                log_file.unlink()
                removed += 1

        if removed > 0:
            logger.info("Cleaned up {} old session log files.", removed)
        return removed

    async def erase_all(self) -> None:
        """Delete all session data (Right to Erasure)."""
        import shutil
        if self.sessions_dir.exists():
            shutil.rmtree(self.sessions_dir)
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        logger.info("All session data erased.")
