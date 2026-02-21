import asyncio
from typing import AsyncIterator

import numpy as np
from loguru import logger

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024  # ~64ms at 16kHz
FORMAT_DTYPE = np.int16


class AudioCapture:
    """Captures audio from USB microphone as a continuous stream of chunks."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._stream = None
        self._pa = None

    def _ensure_stream(self):
        """Lazily open the PyAudio stream."""
        if self._stream is not None:
            return

        import pyaudio
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        logger.info(
            "Audio capture started: {}Hz, chunk_size={}",
            self.sample_rate, self.chunk_size,
        )

    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Yield audio chunks as numpy arrays. Non-blocking via executor."""
        self._ensure_stream()
        loop = asyncio.get_event_loop()

        while True:
            # Read from mic in a thread to avoid blocking the event loop
            raw = await loop.run_in_executor(
                None, self._stream.read, self.chunk_size, False
            )
            chunk = np.frombuffer(raw, dtype=FORMAT_DTYPE)
            yield chunk

    def read_chunk(self) -> np.ndarray:
        """Synchronous read of one chunk (for wake word thread)."""
        self._ensure_stream()
        raw = self._stream.read(self.chunk_size, exception_on_overflow=False)
        return np.frombuffer(raw, dtype=FORMAT_DTYPE)

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
