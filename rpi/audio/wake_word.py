import asyncio
from pathlib import Path

import numpy as np
from loguru import logger


class WakeWordDetector:
    """Detects wake word using OpenWakeWord."""

    # Map config names to the label OpenWakeWord uses internally
    _LABEL_MAP = {
        "hey jarvis": "hey_jarvis",
        "alexa": "alexa",
        "hey mycroft": "hey_mycroft",
        "hey marvin": "hey_marvin",
    }

    def __init__(self, model_dir: Path, wake_word: str = "hey jarvis"):
        self.model_dir = model_dir
        self.wake_word = wake_word.lower().strip()
        self._model = None
        self._threshold = 0.4  # Balanced: rejects false positives (0.3x) while catching real wake words (0.4+)
        self._target_label = self._LABEL_MAP.get(self.wake_word, self.wake_word)

    async def load(self):
        """Load the OpenWakeWord model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        from openwakeword.model import Model

        # Check for custom model in model_dir
        custom_onnx = self.model_dir / f"{self.wake_word.replace(' ', '_')}.onnx"
        if custom_onnx.exists():
            self._model = Model(wakeword_model_paths=[str(custom_onnx)])
            logger.info("Using custom wake word model: {}", custom_onnx)
        else:
            # Use all bundled models (most reliable loading method)
            self._model = Model()
            logger.info("Loaded bundled OpenWakeWord models")

        # Prime the model with a dummy prediction to initialize labels
        dummy = np.zeros(1280, dtype=np.int16)
        self._model.predict(dummy)

        labels = list(self._model.prediction_buffer.keys())
        logger.info(
            "Wake word detector ready. Target: '{}', Available: {}",
            self._target_label, labels,
        )

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

        self._model.predict(chunk)

        # Check the target wake word label first
        for model_name, scores in self._model.prediction_buffer.items():
            if self._target_label not in model_name:
                continue
            if len(scores) > 0 and scores[-1] > self._threshold:
                logger.info(
                    "Wake word '{}' detected (score: {:.2f})",
                    model_name, scores[-1],
                )
                self._model.reset()
                return True

        return False

    def detect_sync(self, chunk: np.ndarray) -> bool:
        """Synchronous detection for use in dedicated thread."""
        return self._predict(chunk)
