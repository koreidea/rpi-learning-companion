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
    """Captures audio from I2S INMP441 microphone (or USB mic fallback).

    Opens the default ALSA capture device. The INMP441 outputs 32-bit I2S
    data which is converted to 16-bit int for downstream ML models.
    Falls back to trying multiple sample rates if needed.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._stream = None
        self._pa = None
        self._capture_rate = sample_rate  # actual rate we opened the stream at
        self._capture_chunk = chunk_size  # actual chunk size for the capture rate
        self._capture_format = None  # pyaudio format used
        self._is_32bit = False  # whether we need 32→16 bit conversion

    def _find_i2s_device(self, pa) -> int | None:
        """Find the I2S input device index (googlevoicehat / INMP441)."""
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            if info["maxInputChannels"] > 0 and ("voicehat" in name or "googlevoice" in name):
                logger.info("Found I2S mic device {}: {}", i, info["name"])
                return i
        return None

    def _ensure_stream(self):
        """Lazily open the PyAudio stream."""
        if self._stream is not None:
            return

        import pyaudio
        self._pa = pyaudio.PyAudio()

        # Try I2S device first (INMP441 via googlevoicehat overlay)
        i2s_idx = self._find_i2s_device(self._pa)

        if i2s_idx is not None:
            # INMP441 requires: 32-bit, 48kHz, 2 channels (stereo)
            rate = 48000
            capture_chunk = int(self.chunk_size * rate / self.sample_rate)
            try:
                self._stream = self._pa.open(
                    format=pyaudio.paInt32,
                    channels=2,
                    rate=rate,
                    input=True,
                    input_device_index=i2s_idx,
                    frames_per_buffer=capture_chunk,
                )
                self._capture_rate = rate
                self._capture_chunk = capture_chunk
                self._capture_format = pyaudio.paInt32
                self._is_32bit = True
                self._is_stereo = True
                logger.info(
                    "Audio capture started: I2S 32-bit stereo@{}Hz, output={}Hz, chunk={}",
                    rate, self.sample_rate, self.chunk_size,
                )
                return
            except Exception as e:
                logger.warning("Failed to open I2S mic: {}", e)

        # Fallback: try default device with various formats
        self._is_stereo = False
        formats = [
            (pyaudio.paInt32, True, "32-bit"),
            (pyaudio.paInt16, False, "16-bit"),
        ]

        for pa_fmt, is_32, fmt_name in formats:
            for rate in [self.sample_rate, 48000, 44100]:
                try:
                    capture_chunk = (
                        self.chunk_size if rate == self.sample_rate
                        else int(self.chunk_size * rate / self.sample_rate)
                    )
                    self._stream = self._pa.open(
                        format=pa_fmt,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        frames_per_buffer=capture_chunk,
                    )
                    self._capture_rate = rate
                    self._capture_chunk = capture_chunk
                    self._capture_format = pa_fmt
                    self._is_32bit = is_32
                    logger.info(
                        "Audio capture started: {}@{}Hz, output={}Hz, chunk={}",
                        fmt_name, rate, self.sample_rate, self.chunk_size,
                    )
                    return
                except Exception as e:
                    logger.debug("{} @ {}Hz not supported: {}", fmt_name, rate, e)

        raise RuntimeError("Could not open audio input stream at any supported format/rate")

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

    def _convert_chunk(self, raw: bytes) -> np.ndarray:
        """Convert raw audio bytes to int16 mono numpy array.

        Handles: stereo→mono, 32-bit→16-bit, resampling (48kHz→16kHz).
        The INMP441 outputs 24-bit data in 32-bit frames. A simple >>16
        loses 8 bits of signal, making the audio too quiet for VAD/STT.
        We use >>14 for ~4x gain, then clip to int16 range.
        """
        if self._is_32bit:
            chunk32 = np.frombuffer(raw, dtype=np.int32)
            # Stereo→mono: take left channel only (INMP441 is on L/R=GND=left)
            if getattr(self, '_is_stereo', False):
                chunk32 = chunk32[0::2]  # every other sample = left channel
            # 32-bit→16-bit with gain: >>14 instead of >>16 gives ~4x boost
            # INMP441 24-bit data is quiet at >>16; >>14 brings levels up
            # for reliable VAD and STT while avoiding clipping in normal use
            shifted = chunk32 >> 14
            chunk = np.clip(shifted, -32768, 32767).astype(FORMAT_DTYPE)
        else:
            chunk = np.frombuffer(raw, dtype=FORMAT_DTYPE)
        return self._resample(chunk)

    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Yield audio chunks as numpy arrays at 16kHz. Non-blocking via executor."""
        self._ensure_stream()
        loop = asyncio.get_event_loop()

        while True:
            raw = await loop.run_in_executor(
                None, self._stream.read, self._capture_chunk, False
            )
            yield self._convert_chunk(raw)

    def read_chunk(self) -> np.ndarray:
        """Synchronous read of one chunk at 16kHz (for wake word thread)."""
        self._ensure_stream()
        raw = self._stream.read(self._capture_chunk, exception_on_overflow=False)
        return self._convert_chunk(raw)

    def pause(self):
        """Stop the mic stream so it doesn't buffer audio during playback."""
        if self._stream is not None and self._stream.is_active():
            self._stream.stop_stream()
            logger.debug("Mic paused")

    def resume(self):
        """Restart the mic stream and discard any stale data."""
        if self._stream is None:
            # Stream was closed entirely, re-open it
            self._ensure_stream()
            logger.debug("Mic re-opened after close")
            return

        if self._stream.is_stopped():
            try:
                self._stream.start_stream()
            except Exception as e:
                # Stream is broken, close and re-create
                logger.warning("Failed to resume stream ({}), re-opening...", e)
                self.close()
                self._ensure_stream()
                return

            # Drain anything that was left over
            try:
                avail = self._stream.get_read_available()
                if avail > 0:
                    self._stream.read(avail, exception_on_overflow=False)
            except Exception:
                pass
            logger.debug("Mic resumed")

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
