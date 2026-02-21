import asyncio
from pathlib import Path

import numpy as np
from loguru import logger


class WakeWordDetector:
    """Detects wake word using OpenWakeWord."""

    def __init__(self, model_dir: Path, wake_word: str = "hey buddy"):
        self.model_dir = model_dir
        self.wake_word = wake_word
        self._model = None
        self._threshold = 0.5

    async def load(self):
        """Load the OpenWakeWord model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        import openwakeword
        from openwakeword.model import Model

        # Download default models if not present
        openwakeword.utils.download_models()

        self._model = Model(
            wakeword_models=[],  # Use built-in models
            inference_framework="onnx",
        )
        logger.info("Wake word detector loaded. Listening for: '{}'", self.wake_word)

    async def listen(self, audio_stream) -> None:
        """Listen to audio stream until wake word is detected.

        Args:
            audio_stream: async iterator yielding np.ndarray audio chunks
        """
        loop = asyncio.get_event_loop()

        async for chunk in audio_stream:
            # Run prediction in executor (it's a CPU-bound operation)
            detected = await loop.run_in_executor(None, self._predict, chunk)
            if detected:
                return

    def _predict(self, chunk: np.ndarray) -> bool:
        """Run wake word prediction on a single audio chunk."""
        if self._model is None:
            return False

        # OpenWakeWord expects int16 numpy array
        self._model.predict(chunk)

        # Check all model scores
        for model_name, score in self._model.prediction_buffer.items():
            if len(score) > 0 and score[-1] > self._threshold:
                logger.debug("Wake word '{}' detected (score: {:.2f})", model_name, score[-1])
                self._model.reset()
                return True

        return False

    def detect_sync(self, chunk: np.ndarray) -> bool:
        """Synchronous detection for use in dedicated thread."""
        return self._predict(chunk)
