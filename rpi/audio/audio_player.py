import asyncio
import subprocess
import tempfile
from pathlib import Path

from loguru import logger


class AudioPlayer:
    """Plays audio through ALSA (aplay) targeting dual I2S MAX98357A speakers.

    Two MAX98357A modules on the same I2S bus — one for Left, one for Right.
    All audio is converted to stereo before playback so both speakers fire.

    Supports:
    - Playing WAV bytes (from TTS)
    - Playing WAV files (for sound effects)
    - Queued playback for streaming TTS
    - Stop (kill) currently playing audio
    - Software volume control (scales PCM samples before playback)
    - Mono-to-stereo conversion for dual speaker output
    """

    # ALSA device for I2S MAX98357A speakers
    PLAY_CMD = ["aplay", "-q", "-D", "plughw:2,0"]

    def __init__(self):
        self._current_process: subprocess.Popen | None = None
        self._volume: int = 80  # 0-100, default 80%

    def set_volume(self, level: int) -> int:
        """Set playback volume (0-100). Returns clamped value."""
        self._volume = max(0, min(100, level))
        logger.info("Volume set to {}%", self._volume)
        return self._volume

    def get_volume(self) -> int:
        """Get current volume level (0-100)."""
        return self._volume

    @staticmethod
    def _ensure_stereo(wav_bytes: bytes) -> bytes:
        """Convert mono WAV to stereo by duplicating the channel.

        If the audio is already stereo (or more channels), returns unchanged.
        This ensures both L and R MAX98357A modules receive audio.
        """
        import io
        import wave

        try:
            with io.BytesIO(wav_bytes) as inp:
                with wave.open(inp, "rb") as wf:
                    params = wf.getparams()
                    raw = wf.readframes(params.nframes)

            if params.nchannels >= 2:
                return wav_bytes  # Already stereo

            # Duplicate mono samples: L=R for each frame
            import numpy as np
            if params.sampwidth == 2:
                samples = np.frombuffer(raw, dtype=np.int16)
            elif params.sampwidth == 4:
                samples = np.frombuffer(raw, dtype=np.int32)
            else:
                return wav_bytes  # Unsupported sample width

            # Interleave: [s0, s0, s1, s1, s2, s2, ...]
            stereo = np.column_stack((samples, samples)).flatten()

            with io.BytesIO() as out:
                with wave.open(out, "wb") as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(params.sampwidth)
                    wf.setframerate(params.framerate)
                    wf.writeframes(stereo.tobytes())
                return out.getvalue()
        except Exception as e:
            logger.debug("Stereo conversion failed: {}, playing as-is", e)
            return wav_bytes

    def _apply_volume(self, wav_bytes: bytes) -> bytes:
        """Apply software volume scaling to WAV audio data.

        Reads the WAV, scales PCM samples by volume factor, re-encodes.
        This works regardless of ALSA mixer availability (MAX98357A has none).
        """
        import io
        import wave
        import numpy as np

        if self._volume >= 100:
            return wav_bytes  # No scaling needed

        try:
            with io.BytesIO(wav_bytes) as inp:
                with wave.open(inp, "rb") as wf:
                    params = wf.getparams()
                    raw = wf.readframes(params.nframes)

            # Scale PCM samples
            dtype = np.int16 if params.sampwidth == 2 else np.int32
            samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            factor = self._volume / 100.0
            if params.sampwidth == 2:
                samples = np.clip(samples * factor, -32768, 32767).astype(np.int16)
            else:
                samples = np.clip(samples * factor, -2147483648, 2147483647).astype(np.int32)

            with io.BytesIO() as out:
                with wave.open(out, "wb") as wf:
                    wf.setparams(params)
                    wf.writeframes(samples.tobytes())
                return out.getvalue()
        except Exception as e:
            logger.debug("Volume scaling failed: {}, playing at full volume", e)
            return wav_bytes

    def _prepare(self, wav_bytes: bytes) -> bytes:
        """Apply stereo conversion + volume scaling."""
        wav_bytes = self._ensure_stereo(wav_bytes)
        wav_bytes = self._apply_volume(wav_bytes)
        return wav_bytes

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
        """Synchronous WAV playback using aplay (ALSA direct)."""
        try:
            wav_bytes = self._prepare(wav_bytes)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(wav_bytes)
                tmp.flush()
                proc = subprocess.Popen(
                    self.PLAY_CMD + [tmp.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self._current_process = proc
                proc.wait(timeout=30)
                self._current_process = None
                if proc.returncode != 0 and proc.returncode != -9:
                    stderr = proc.stderr.read().decode().strip()
                    logger.error("aplay error: {}", stderr)
        except subprocess.TimeoutExpired:
            if self._current_process:
                self._current_process.kill()
                self._current_process = None
            logger.error("Audio playback timed out (30s)")
        except FileNotFoundError:
            logger.error("aplay not found. Install alsa-utils.")
        except Exception as e:
            logger.error("Audio playback error: {}", e)

    async def play_file(self, path: Path, timeout: int = 30) -> None:
        """Play a WAV file from disk.

        Args:
            path: Path to WAV file.
            timeout: Max playback time in seconds (default 30, use 300 for songs).
        """
        if not path.exists():
            logger.warning("Sound file not found: {}", path)
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_file_sync, path, timeout)

    def _play_file_sync(self, path: Path, timeout: int = 30) -> None:
        """Play a file directly with aplay (no temp file needed)."""
        try:
            wav_bytes = path.read_bytes()
            wav_bytes = self._prepare(wav_bytes)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(wav_bytes)
                tmp.flush()
                proc = subprocess.Popen(
                    self.PLAY_CMD + [tmp.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self._current_process = proc
                proc.wait(timeout=timeout)
                self._current_process = None
            if proc.returncode != 0 and proc.returncode != -9:
                stderr = proc.stderr.read().decode().strip()
                logger.error("aplay error: {}", stderr)
        except subprocess.TimeoutExpired:
            if self._current_process:
                self._current_process.kill()
                self._current_process = None
            logger.error("Audio playback timed out ({}s)", timeout)
        except FileNotFoundError:
            logger.error("aplay not found. Install alsa-utils.")
        except Exception as e:
            logger.error("Audio playback error: {}", e)

    async def stop(self) -> None:
        """Stop any currently playing audio immediately."""
        proc = self._current_process
        if proc is not None:
            try:
                proc.kill()
                logger.info("Audio playback stopped.")
            except Exception as e:
                logger.debug("Error killing aplay: {}", e)
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
