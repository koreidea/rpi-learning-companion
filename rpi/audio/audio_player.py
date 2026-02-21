import asyncio
import io
import wave
from pathlib import Path

import numpy as np
from loguru import logger


class AudioPlayer:
    """Plays audio through the Bluetooth speaker (or default audio output).

    Supports:
    - Playing WAV bytes (from TTS)
    - Playing WAV files (for sound effects)
    - Queued playback for streaming TTS
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._pa = None
        self._stream = None

    def _ensure_output(self, rate: int = None):
        """Lazily open the PyAudio output stream."""
        rate = rate or self.sample_rate
        if self._stream is not None:
            return

        import pyaudio
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            output=True,
        )

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
        """Synchronous WAV playback."""
        try:
            wav_buffer = io.BytesIO(wav_bytes)
            with wave.open(wav_buffer, "rb") as wf:
                rate = wf.getframerate()
                self._ensure_output(rate)

                chunk_size = 4096
                data = wf.readframes(chunk_size)
                while data:
                    self._stream.write(data)
                    data = wf.readframes(chunk_size)
        except Exception as e:
            logger.error("Audio playback error: {}", e)

    async def play_file(self, path: Path) -> None:
        """Play a WAV file from disk."""
        if not path.exists():
            logger.warning("Sound file not found: {}", path)
            return

        wav_bytes = path.read_bytes()
        await self.play(wav_bytes)

    def close(self):
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def __del__(self):
        self.close()
