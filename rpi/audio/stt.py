import asyncio
from pathlib import Path

import numpy as np
from loguru import logger

from audio.audio_capture import SAMPLE_RATE


class SpeechToText:
    """Speech-to-text using Whisper.cpp (via pywhispercpp).

    Uses whisper-tiny.en for fast inference on RPi 5.
    """

    def __init__(self, model_dir: Path, model_name: str = "tiny.en"):
        self.model_dir = model_dir
        self.model_name = model_name
        self._model = None

    async def load(self):
        """Load the Whisper model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        from pywhispercpp.model import Model

        model_path = self.model_dir / f"ggml-{self.model_name}.bin"

        if not model_path.exists():
            logger.info("Whisper model not found at {}. Will download on first use.", model_path)
            # pywhispercpp downloads the model automatically if not found
            self._model = Model(self.model_name, models_dir=str(self.model_dir))
        else:
            self._model = Model(str(model_path))

        logger.info("Whisper STT loaded: {}", self.model_name)

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio numpy array to text.

        Args:
            audio: int16 numpy array at 16kHz

        Returns:
            Transcribed text string
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription."""
        if self._model is None:
            return ""

        # Whisper expects float32 audio normalized to [-1, 1]
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        segments = self._model.transcribe(audio_float)

        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.debug("STT result: '{}'", text)
        return text
