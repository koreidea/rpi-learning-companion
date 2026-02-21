import asyncio
from typing import Optional

import numpy as np
from loguru import logger

from audio.audio_capture import SAMPLE_RATE


class VADDetector:
    """Voice Activity Detection using Silero-VAD.

    Detects when the speaker stops talking, allowing faster
    end-of-utterance detection than simple silence thresholds.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        silence_duration: float = 0.8,  # seconds of silence to end capture
        max_duration: float = 15.0,     # max recording length in seconds
        speech_threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.speech_threshold = speech_threshold
        self._model = None

    def _ensure_model(self):
        """Load Silero-VAD model lazily."""
        if self._model is not None:
            return

        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        self._get_speech_timestamps = utils[0]
        logger.info("Silero-VAD loaded.")

    async def capture_until_silence(self, audio_capture) -> Optional[np.ndarray]:
        """Record audio until the speaker stops talking.

        Uses VAD to detect speech activity. Recording ends when:
        - silence_duration of non-speech is detected after speech started
        - max_duration is reached
        - No speech detected within first 3 seconds

        Returns:
            numpy array of the captured audio, or None if no speech.
        """
        self._ensure_model()

        loop = asyncio.get_event_loop()
        frames = []
        speech_started = False
        silence_frames = 0
        total_frames = 0

        frames_per_chunk = audio_capture.chunk_size
        chunks_per_second = self.sample_rate / frames_per_chunk
        silence_chunks_needed = int(self.silence_duration * chunks_per_second)
        max_chunks = int(self.max_duration * chunks_per_second)
        initial_wait_chunks = int(3.0 * chunks_per_second)

        while total_frames < max_chunks:
            chunk = await loop.run_in_executor(None, audio_capture.read_chunk)
            frames.append(chunk)
            total_frames += 1

            # Check VAD
            is_speech = await loop.run_in_executor(None, self._is_speech, chunk)

            if is_speech:
                speech_started = True
                silence_frames = 0
            elif speech_started:
                silence_frames += 1
                if silence_frames >= silence_chunks_needed:
                    logger.debug(
                        "End of speech detected after {:.1f}s",
                        total_frames / chunks_per_second,
                    )
                    break
            elif total_frames >= initial_wait_chunks and not speech_started:
                logger.debug("No speech detected within 3s timeout.")
                return None

        if not speech_started:
            return None

        audio = np.concatenate(frames)
        duration = len(audio) / self.sample_rate
        logger.debug("Captured {:.1f}s of audio.", duration)
        return audio

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """Check if an audio chunk contains speech."""
        import torch

        # Silero-VAD expects float32 tensor normalized to [-1, 1]
        audio_float = chunk.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float)

        confidence = self._model(tensor, self.sample_rate).item()
        return confidence > self.speech_threshold
