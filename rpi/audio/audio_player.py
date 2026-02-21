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
    - Stop (kill) currently playing audio
    """

    def __init__(self):
        self._current_process: subprocess.Popen | None = None

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
                proc = subprocess.Popen(
                    ["paplay", tmp.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self._current_process = proc
                proc.wait(timeout=30)
                self._current_process = None
                if proc.returncode != 0 and proc.returncode != -9:
                    stderr = proc.stderr.read().decode().strip()
                    logger.error("paplay error: {}", stderr)
        except subprocess.TimeoutExpired:
            if self._current_process:
                self._current_process.kill()
                self._current_process = None
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
            proc = subprocess.Popen(
                ["paplay", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._current_process = proc
            proc.wait(timeout=30)
            self._current_process = None
            if proc.returncode != 0 and proc.returncode != -9:
                stderr = proc.stderr.read().decode().strip()
                logger.error("paplay error: {}", stderr)
        except subprocess.TimeoutExpired:
            if self._current_process:
                self._current_process.kill()
                self._current_process = None
            logger.error("Audio playback timed out (30s)")
        except FileNotFoundError:
            logger.error("paplay not found. Install pulseaudio-utils.")
        except Exception as e:
            logger.error("Audio playback error: {}", e)

    async def stop(self) -> None:
        """Stop any currently playing audio immediately."""
        proc = self._current_process
        if proc is not None:
            try:
                proc.kill()
                logger.info("Audio playback stopped (killed paplay).")
            except Exception as e:
                logger.debug("Error killing paplay: {}", e)
            self._current_process = None

    def close(self):
        if self._current_process:
            try:
                self._current_process.kill()
            except Exception:
                pass
            self._current_process = None

    def __del__(self):
        self.close()
