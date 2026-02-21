import asyncio
from typing import Optional

import numpy as np
from loguru import logger

from audio.audio_capture import SAMPLE_RATE

# Silero-VAD requires exactly this many samples per call at 16kHz
_VAD_CHUNK_SAMPLES = 512


class VADDetector:
    """Voice Activity Detection using Silero-VAD.

    Detects when the speaker stops talking, allowing faster
    end-of-utterance detection than simple silence thresholds.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        silence_duration: float = 0.4,  # seconds of silence to end capture
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

    async def capture_until_silence(
        self, audio_capture, initial_wait: float = 3.0
    ) -> Optional[np.ndarray]:
        """Record audio until the speaker stops talking.

        Uses VAD to detect speech activity. Recording ends when:
        - silence_duration of non-speech is detected after speech started
        - max_duration is reached
        - No speech detected within initial_wait seconds

        Args:
            audio_capture: AudioCapture instance to read from.
            initial_wait: Seconds to wait for speech before giving up.

        Returns:
            numpy array of the captured audio, or None if no speech.
        """
        self._ensure_model()

        # Reset Silero-VAD internal state between captures
        self._model.reset_states()

        loop = asyncio.get_event_loop()
        frames = []
        speech_started = False
        silence_time = 0.0
        total_time = 0.0

        # Buffer to accumulate samples for VAD (needs exactly 512 samples)
        vad_buffer = np.array([], dtype=np.int16)

        while total_time < self.max_duration:
            chunk = await loop.run_in_executor(None, audio_capture.read_chunk)
            frames.append(chunk)
            total_time += len(chunk) / self.sample_rate

            # Accumulate into VAD buffer
            vad_buffer = np.concatenate([vad_buffer, chunk])

            # Process all complete 512-sample windows in the buffer
            while len(vad_buffer) >= _VAD_CHUNK_SAMPLES:
                vad_chunk = vad_buffer[:_VAD_CHUNK_SAMPLES]
                vad_buffer = vad_buffer[_VAD_CHUNK_SAMPLES:]

                is_speech = await loop.run_in_executor(
                    None, self._is_speech, vad_chunk
                )

                if is_speech:
                    speech_started = True
                    silence_time = 0.0
                elif speech_started:
                    silence_time += _VAD_CHUNK_SAMPLES / self.sample_rate

                    if silence_time >= self.silence_duration:
                        logger.debug(
                            "End of speech detected after {:.1f}s",
                            total_time,
                        )
                        audio = np.concatenate(frames)
                        duration = len(audio) / self.sample_rate
                        logger.debug("Captured {:.1f}s of audio.", duration)
                        return audio

            # Timeout if no speech within initial_wait
            if total_time >= initial_wait and not speech_started:
                logger.debug("No speech detected within 3s timeout.")
                return None

        if not speech_started:
            return None

        audio = np.concatenate(frames)
        duration = len(audio) / self.sample_rate
        logger.debug("Captured {:.1f}s of audio.", duration)
        return audio

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """Check if an audio chunk contains speech.

        Args:
            chunk: Must be exactly 512 samples (int16) at 16kHz.
        """
        import torch

        # Silero-VAD expects float32 tensor normalized to [-1, 1]
        audio_float = chunk.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float)

        confidence = self._model(tensor, self.sample_rate).item()
        return confidence > self.speech_threshold
