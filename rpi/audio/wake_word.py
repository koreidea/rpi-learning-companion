import asyncio
from pathlib import Path

import numpy as np
from loguru import logger


class WakeWordDetector:
    """Detects wake word using OpenWakeWord."""

    # Map friendly names to bundled model filenames
    _BUILTIN_MODELS = {
        "hey jarvis": "hey_jarvis_v0.1",
        "alexa": "alexa_v0.1",
        "hey mycroft": "hey_mycroft_v0.1",
        "hey marvin": "hey_marvin_v0.1",
    }

    def __init__(self, model_dir: Path, wake_word: str = "hey jarvis"):
        self.model_dir = model_dir
        self.wake_word = wake_word.lower().strip()
        self._model = None
        self._threshold = 0.5

    async def load(self):
        """Load the OpenWakeWord model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        from openwakeword.model import Model

        model_paths = []

        # Check for custom model in model_dir
        custom_onnx = self.model_dir / f"{self.wake_word.replace(' ', '_')}.onnx"
        if custom_onnx.exists():
            model_paths = [str(custom_onnx)]
            logger.info("Using custom wake word model: {}", custom_onnx)
        elif self.wake_word in self._BUILTIN_MODELS:
            # Use bundled model from openwakeword package
            import openwakeword
            pkg_models = Path(openwakeword.__file__).parent / "resources" / "models"
            builtin = pkg_models / f"{self._BUILTIN_MODELS[self.wake_word]}.onnx"
            if builtin.exists():
                model_paths = [str(builtin)]
                logger.info("Using built-in wake word model: {}", builtin.name)

        # If no specific model found, load all bundled models
        if not model_paths:
            logger.info("No specific model for '{}', loading all built-in models.", self.wake_word)

        self._model = Model(wakeword_model_paths=model_paths)
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
