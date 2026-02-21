import asyncio
import subprocess
import tempfile
from pathlib import Path

from loguru import logger


class AudioPlayer:
    """Plays audio through PipeWire/PulseAudio (paplay).

    Uses paplay for reliable Bluetooth speaker output via PipeWire.

    Supports:
    - Playing WAV bytes (from TTS)
    - Playing WAV files (for sound effects)
    - Queued playback for streaming TTS
    """

    def __init__(self):
        pass

    async def play(self, wav_bytes: bytes) -> None:
        """Play WAV audio bytes through the speaker.

        Args:
            wav_bytes: Complete WAV file bytes (with header).
        """
        if not wav_bytes:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_sync, wav_bytes)

    def _play_sync(self, wav_bytes: bytes) -> None:
        """Synchronous WAV playback using paplay."""
        try:
            # Write WAV bytes to a temp file, then play with paplay
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(wav_bytes)
                tmp.flush()
                result = subprocess.run(
                    ["paplay", tmp.name],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    stderr = result.stderr.decode().strip()
                    logger.error("paplay error: {}", stderr)
        except subprocess.TimeoutExpired:
            logger.error("Audio playback timed out (30s)")
        except FileNotFoundError:
            logger.error("paplay not found. Install pulseaudio-utils.")
        except Exception as e:
            logger.error("Audio playback error: {}", e)

    async def play_file(self, path: Path) -> None:
        """Play a WAV file from disk."""
        if not path.exists():
            logger.warning("Sound file not found: {}", path)
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_file_sync, path)

    def _play_file_sync(self, path: Path) -> None:
        """Play a file directly with paplay (no temp file needed)."""
        try:
            result = subprocess.run(
                ["paplay", str(path)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode().strip()
                logger.error("paplay error: {}", stderr)
        except subprocess.TimeoutExpired:
            logger.error("Audio playback timed out (30s)")
        except FileNotFoundError:
            logger.error("paplay not found. Install pulseaudio-utils.")
        except Exception as e:
            logger.error("Audio playback error: {}", e)

    def close(self):
        pass

    def __del__(self):
        pass
