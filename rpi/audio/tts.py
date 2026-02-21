import asyncio
import io
import wave
from pathlib import Path

import numpy as np
from loguru import logger


class TextToSpeech:
    """Text-to-speech using Piper TTS.

    Optimized for sentence-level streaming: synthesize one sentence at a time
    for low-latency output while the LLM continues generating.
    """

    def __init__(
        self,
        model_dir: Path,
        voice: str = "en_US-lessac-medium",
        sample_rate: int = 22050,
    ):
        self.model_dir = model_dir
        self.voice = voice
        self.sample_rate = sample_rate
        self._piper = None

    async def load(self):
        """Load the Piper TTS voice model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        try:
            from piper import PiperVoice

            model_path = self.model_dir / f"{self.voice}.onnx"
            config_path = self.model_dir / f"{self.voice}.onnx.json"

            if model_path.exists():
                self._piper = PiperVoice.load(str(model_path), config_path=str(config_path))
                logger.info("Piper TTS loaded: {}", self.voice)
            else:
                logger.warning(
                    "Piper voice model not found at {}. Run download_models.sh first.",
                    model_path,
                )
        except ImportError:
            logger.warning("piper-tts not installed. TTS will be unavailable.")

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV audio bytes.

        Designed to be called per-sentence for streaming playback.

        Args:
            text: A single sentence to synthesize.

        Returns:
            WAV audio bytes ready for playback.
        """
        if not text or not text.strip():
            return b""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous synthesis."""
        if self._piper is None:
            logger.warning("TTS not loaded. Returning empty audio.")
            return b""

        # Piper synthesize() returns an iterable of AudioChunk objects
        # Each chunk has audio_float_array (float32 normalized to [-1, 1])
        all_audio = []
        for chunk in self._piper.synthesize(text):
            # Convert float32 [-1, 1] to int16
            audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
            all_audio.append(audio_int16)

        if not all_audio:
            logger.warning("TTS produced no audio for: '{}'", text[:50])
            return b""

        audio_data = np.concatenate(all_audio)

        # Write to WAV buffer
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_data.tobytes())

        audio_bytes = wav_buffer.getvalue()
        logger.debug("TTS: synthesized {} bytes for '{}'", len(audio_bytes), text[:50])
        return audio_bytes
