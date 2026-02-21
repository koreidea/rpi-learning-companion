import asyncio
from typing import AsyncIterator

import numpy as np
from loguru import logger

# Target sample rate for all ML models (Whisper, VAD, OpenWakeWord)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1280  # 80ms at 16kHz — matches OpenWakeWord's preferred size
FORMAT_DTYPE = np.int16


class AudioCapture:
    """Captures audio from USB microphone as a continuous stream of chunks.

    Tries 16kHz first (PipeWire does high-quality resampling), falls back
    to native 44100Hz with manual resampling if needed.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._stream = None
        self._pa = None
        self._capture_rate = sample_rate  # actual rate we opened the stream at
        self._capture_chunk = chunk_size  # actual chunk size for the capture rate

    def _ensure_stream(self):
        """Lazily open the PyAudio stream."""
        if self._stream is not None:
            return

        import pyaudio
        self._pa = pyaudio.PyAudio()

        # Try target rate first (PipeWire resamples for us), then native
        for rate in [self.sample_rate, 44100, 48000]:
            try:
                capture_chunk = (
                    self.chunk_size if rate == self.sample_rate
                    else int(self.chunk_size * rate / self.sample_rate)
                )
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=rate,
                    input=True,
                    frames_per_buffer=capture_chunk,
                )
                self._capture_rate = rate
                self._capture_chunk = capture_chunk
                logger.info(
                    "Audio capture started: capture={}Hz, output={}Hz, chunk={}",
                    rate, self.sample_rate, self.chunk_size,
                )
                return
            except Exception as e:
                logger.debug("Sample rate {}Hz not supported: {}", rate, e)

        raise RuntimeError("Could not open audio input stream at any supported rate")

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        """Resample from capture rate to target rate using linear interpolation."""
        if self._capture_rate == self.sample_rate:
            return chunk

        ratio = self.sample_rate / self._capture_rate
        n_out = self.chunk_size
        indices = np.arange(n_out) / ratio
        indices = np.clip(indices, 0, len(chunk) - 1)
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
                None, self._stream.read, self._capture_chunk, False
            )
            chunk = np.frombuffer(raw, dtype=FORMAT_DTYPE)
            yield self._resample(chunk)

    def read_chunk(self) -> np.ndarray:
        """Synchronous read of one chunk at 16kHz (for wake word thread)."""
        self._ensure_stream()
        raw = self._stream.read(self._capture_chunk, exception_on_overflow=False)
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
