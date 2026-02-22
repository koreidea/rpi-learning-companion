import asyncio
import io
import struct
import wave
from pathlib import Path

import numpy as np
from loguru import logger

from audio.audio_capture import SAMPLE_RATE


class SpeechToText:
    """Speech-to-text using Whisper.cpp (via pywhispercpp).

    Uses multilingual whisper-tiny for Hindi + English on RPi 5.
    For low-resource languages like Telugu, uses the larger whisper-base
    model for better accuracy.
    """

    # Languages where tiny model works well enough
    _TINY_LANGS = {"en", "hi"}

    def __init__(self, model_dir: Path, model_name: str = "tiny", language: str = "en",
                 n_threads: int = 4):
        self.model_dir = model_dir
        self.language = language
        self.n_threads = n_threads
        self._model = None

        # Auto-select model: use 'base' for languages where 'tiny' is too weak
        if language not in self._TINY_LANGS:
            base_path = model_dir / "ggml-base.bin"
            if base_path.exists():
                self.model_name = "base"
            else:
                self.model_name = model_name  # fallback to tiny
                logger.warning("Whisper base model not found for '{}'; using tiny (accuracy may be poor)", language)
        else:
            self.model_name = model_name

    async def load(self):
        """Load the Whisper model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        from pywhispercpp.model import Model

        model_path = self.model_dir / f"ggml-{self.model_name}.bin"

        if not model_path.exists():
            logger.info("Whisper model not found at {}. Will download on first use.", model_path)
            self._model = Model(self.model_name, models_dir=str(self.model_dir),
                                n_threads=self.n_threads)
        else:
            self._model = Model(str(model_path), n_threads=self.n_threads)

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

        segments = self._model.transcribe(audio_float, language=self.language)

        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.debug("STT result: '{}'", text)
        return text


class CloudSpeechToText:
    """Cloud STT using OpenAI Whisper API.

    Faster than local Whisper (~0.5s vs ~3s) but sends audio off-device.
    Only used when cloud_stt is enabled AND mode is online.
    Falls back to local STT on failure.
    """

    def __init__(self, api_key: str, language: str = "en"):
        self.api_key = api_key
        self.language = language
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)

    def update_api_key(self, api_key: str):
        """Update API key (e.g., when user changes it in settings)."""
        if api_key != self.api_key:
            self.api_key = api_key
            self._client = None  # Force re-creation

    @staticmethod
    def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
        """Convert int16 numpy audio array to WAV bytes in memory."""
        if audio.dtype != np.int16:
            audio = (audio * 32768.0).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        buf.seek(0)
        return buf.read()

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio via OpenAI Whisper API.

        Args:
            audio: int16 numpy array at 16kHz

        Returns:
            Transcribed text string, or empty string on failure.
        """
        if not self.api_key:
            logger.warning("Cloud STT: No API key configured.")
            return ""

        self._ensure_client()

        try:
            wav_bytes = self._audio_to_wav_bytes(audio)

            # OpenAI expects a file-like object with a name
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"

            # OpenAI Whisper API supported languages (ISO 639-1 codes).
            # Telugu ('te') is NOT supported — omit language param to let
            # the API auto-detect, which works for most languages.
            api_kwargs = {
                "model": "whisper-1",
                "file": audio_file,
            }
            # Only pass language if the API supports it; auto-detect otherwise
            OPENAI_WHISPER_LANGUAGES = {
                "en", "hi", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt",
                "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "ms", "tl",
                "uk", "ro", "el", "cs", "da", "fi", "hu", "no", "th", "vi",
            }
            if self.language in OPENAI_WHISPER_LANGUAGES:
                api_kwargs["language"] = self.language

            response = await self._client.audio.transcriptions.create(**api_kwargs)

            text = response.text.strip()
            logger.debug("Cloud STT result: '{}'", text)
            return text

        except Exception as e:
            logger.error("Cloud STT error: {}. Falling back to local.", e)
            return ""  # Empty string signals caller to fall back
