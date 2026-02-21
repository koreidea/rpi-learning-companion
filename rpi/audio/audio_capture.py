import asyncio
from typing import AsyncIterator

import numpy as np
from loguru import logger

# Target sample rate for all ML models (Whisper, VAD, OpenWakeWord)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024  # ~64ms at 16kHz (output chunk size after resampling)
FORMAT_DTYPE = np.int16

# Capture at the mic's native rate and resample down
_NATIVE_RATE = 44100


class AudioCapture:
    """Captures audio from USB microphone as a continuous stream of chunks.

    Records at the mic's native 44100Hz and resamples to 16kHz for ML models.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._stream = None
        self._pa = None
        self._native_rate = _NATIVE_RATE
        # How many native samples to capture per chunk to yield chunk_size output samples
        self._native_chunk_size = int(chunk_size * self._native_rate / self.sample_rate)

    def _ensure_stream(self):
        """Lazily open the PyAudio stream."""
        if self._stream is not None:
            return

        import pyaudio
        self._pa = pyaudio.PyAudio()

        # Try native rate first, fall back to target rate
        for rate in [self._native_rate, self.sample_rate]:
            try:
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=rate,
                    input=True,
                    frames_per_buffer=(
                        self._native_chunk_size if rate == self._native_rate
                        else self.chunk_size
                    ),
                )
                self._native_rate = rate
                self._native_chunk_size = int(
                    self.chunk_size * self._native_rate / self.sample_rate
                )
                logger.info(
                    "Audio capture started: native={}Hz, output={}Hz, chunk_size={}",
                    self._native_rate, self.sample_rate, self.chunk_size,
                )
                return
            except Exception as e:
                logger.debug("Sample rate {}Hz not supported: {}", rate, e)

        raise RuntimeError(
            f"Could not open audio stream at {self._native_rate}Hz or {self.sample_rate}Hz"
        )

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        """Resample from native rate to target rate using linear interpolation."""
        if self._native_rate == self.sample_rate:
            return chunk

        ratio = self.sample_rate / self._native_rate
        n_out = int(len(chunk) * ratio)
        indices = np.arange(n_out) / ratio
        indices = np.clip(indices, 0, len(chunk) - 1)
        # Linear interpolation
        idx_floor = indices.astype(np.int32)
        idx_ceil = np.minimum(idx_floor + 1, len(chunk) - 1)
        frac = indices - idx_floor
        resampled = chunk[idx_floor] * (1 - frac) + chunk[idx_ceil] * frac
        return resampled.astype(FORMAT_DTYPE)

    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Yield audio chunks as numpy arrays at 16kHz. Non-blocking via executor."""
        self._ensure_stream()
        loop = asyncio.get_event_loop()

        while True:
            raw = await loop.run_in_executor(
                None, self._stream.read, self._native_chunk_size, False
            )
            chunk = np.frombuffer(raw, dtype=FORMAT_DTYPE)
            yield self._resample(chunk)

    def read_chunk(self) -> np.ndarray:
        """Synchronous read of one chunk at 16kHz (for wake word thread)."""
        self._ensure_stream()
        raw = self._stream.read(self._native_chunk_size, exception_on_overflow=False)
        chunk = np.frombuffer(raw, dtype=FORMAT_DTYPE)
        return self._resample(chunk)

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
