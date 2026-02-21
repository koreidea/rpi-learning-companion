import asyncio
import time
from pathlib import Path

from loguru import logger

from core.config import ConfigManager
from core.state import BotState, LLMMode, SharedState

# Base directory for the rpi package
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SOUNDS_DIR = BASE_DIR / "audio" / "sounds"


class Orchestrator:
    """Main orchestrator: manages the streaming voice interaction pipeline."""

    def __init__(self):
        self.config_manager = ConfigManager(DATA_DIR)
        self.state = SharedState()

        # Components (initialized lazily)
        self._wake_word = None
        self._vad = None
        self._stt = None
        self._tts = None
        self._llm_router = None
        self._audio_capture = None
        self._audio_player = None
        self._safety_filter = None
        self._camera = None
        self._api_server = None

    async def start(self):
        """Boot sequence: load config, check consent, load models, start listening."""
        logger.info("=== RPi Learning Companion starting ===")

        config = self.config_manager.config

        # Sync state from config
        self.state.llm_mode = LLMMode(config.mode)
        self.state.active_provider = config.provider
        self.state.mic_enabled = config.hardware.mic_enabled
        self.state.camera_enabled = config.hardware.camera_enabled

        # Always start the API server (needed for setup mode too)
        await self._start_api_server()

        if not self.config_manager.has_consent:
            logger.info("No parental consent yet. Entering setup mode.")
            self.state.set_state(BotState.SETUP)
            # Wait until consent is given via the parent app
            await self._wait_for_consent()

        # Consent given — load all models
        self.state.set_state(BotState.LOADING)
        await self._load_components()

        # Play ready chime
        await self._play_sound("ready")
        logger.info("=== Bot is ready! Listening for wake word. ===")

        # Main interaction loop
        self.state.set_state(BotState.READY)
        await self._interaction_loop()

    async def _start_api_server(self):
        """Start the FastAPI server in the background."""
        from api.server import create_app

        app = create_app(self.config_manager, self.state)
        self._api_server = app

        import uvicorn
        server_config = uvicorn.Config(
            app, host="0.0.0.0", port=8080, log_level="warning"
        )
        server = uvicorn.Server(server_config)
        asyncio.create_task(server.serve())
        logger.info("API server started on port 8080")

    async def _wait_for_consent(self):
        """Block until parental consent is given via the parent app."""
        while not self.config_manager.has_consent:
            await asyncio.sleep(2)
        logger.info("Parental consent received.")

    async def _load_components(self):
        """Load all ML models and audio components."""
        logger.info("Loading components...")

        from audio.audio_capture import AudioCapture
        from audio.audio_player import AudioPlayer
        from audio.wake_word import WakeWordDetector
        from audio.vad import VADDetector
        from audio.stt import SpeechToText
        from audio.tts import TextToSpeech
        from audio.sentence_buffer import SentenceBuffer
        from llm.base import LLMRouter
        from llm.safety_filter import SafetyFilter

        self._audio_capture = AudioCapture()
        self._audio_player = AudioPlayer()
        self._wake_word = WakeWordDetector(
            model_dir=MODELS_DIR / "wake_word",
            wake_word=self.config_manager.config.hardware.wake_word,
        )
        self._vad = VADDetector()
        self._stt = SpeechToText(model_dir=MODELS_DIR / "stt")
        self._tts = TextToSpeech(model_dir=MODELS_DIR / "tts")
        self._llm_router = LLMRouter(self.config_manager, self.state)
        self._safety_filter = SafetyFilter(self.config_manager)
        self._sentence_buffer = SentenceBuffer()

        # Load models
        await self._wake_word.load()
        await self._stt.load()
        await self._tts.load()
        await self._llm_router.load()

        self.state.is_model_loaded = True
        logger.info("All components loaded.")

    async def _interaction_loop(self):
        """Main loop: wake word → capture → STT → LLM stream → TTS stream → speak."""
        while self.state.is_running:
            try:
                self.state.set_state(BotState.READY)

                # Step 1: Wait for wake word
                audio_stream = self._audio_capture.stream()
                await self._wake_word.listen(audio_stream)
                logger.info("Wake word detected!")

                if not self.state.mic_enabled:
                    continue

                # Step 2: Play acknowledgment sound immediately
                self.state.set_state(BotState.LISTENING)
                self.state.interaction_start_time = time.monotonic()
                asyncio.create_task(self._play_sound("ding"))

                # Step 3: Capture speech with VAD
                audio_data = await self._vad.capture_until_silence(
                    self._audio_capture
                )
                if audio_data is None or len(audio_data) == 0:
                    logger.debug("No speech detected after wake word.")
                    continue

                # Step 4: Speech-to-text
                self.state.set_state(BotState.PROCESSING)
                transcript = await self._stt.transcribe(audio_data)
                logger.info("Child said: '{}'", transcript)

                if not transcript or len(transcript.strip()) < 2:
                    continue

                self.state.current_transcript = transcript

                # Step 5: Safety check on input
                safe_input = self._safety_filter.check_input(transcript)
                if safe_input.blocked:
                    await self._speak_text(safe_input.redirect_response)
                    continue

                # Step 6: Check if this is a vision request
                if self._is_vision_request(transcript) and self.state.camera_enabled:
                    await self._handle_vision_request(transcript)
                    continue

                # Step 7: Stream LLM → sentence buffer → TTS → speak
                await self._stream_response(transcript)

                # Log interaction metadata (not content)
                elapsed = time.monotonic() - self.state.interaction_start_time
                logger.info("Interaction complete. Latency: {:.1f}s", elapsed)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in interaction loop: {}", e)
                self.state.set_state(BotState.ERROR)
                self.state.last_error = str(e)
                await asyncio.sleep(1)

    async def _stream_response(self, transcript: str):
        """Stream LLM tokens → buffer sentences → TTS each sentence → play."""
        self.state.set_state(BotState.PROCESSING)

        provider = self._llm_router.get_provider()
        messages = self._llm_router.build_messages(transcript)
        self._sentence_buffer.reset()

        # Queue for TTS audio chunks ready to play
        audio_queue: asyncio.Queue = asyncio.Queue()
        speaking_done = asyncio.Event()

        # Task: continuously play audio chunks from queue
        async def play_audio_chunks():
            self.state.set_state(BotState.SPEAKING)
            while True:
                chunk = await audio_queue.get()
                if chunk is None:  # Sentinel: done
                    break
                await self._audio_player.play(chunk)
            speaking_done.set()

        player_task = asyncio.create_task(play_audio_chunks())

        # Stream tokens from LLM
        async for token in provider.stream(messages):
            sentence = self._sentence_buffer.feed(token)
            if sentence:
                # Safety check on each sentence
                safe = self._safety_filter.check_output(sentence)
                if safe.blocked:
                    sentence = safe.redirect_response

                # TTS the sentence and queue audio for playback
                audio_data = await self._tts.synthesize(sentence)
                await audio_queue.put(audio_data)

        # Flush any remaining text in the buffer
        remaining = self._sentence_buffer.flush()
        if remaining:
            safe = self._safety_filter.check_output(remaining)
            text = safe.redirect_response if safe.blocked else remaining
            audio_data = await self._tts.synthesize(text)
            await audio_queue.put(audio_data)

        # Signal player to stop
        await audio_queue.put(None)
        await speaking_done.wait()
        player_task.cancel()

    async def _handle_vision_request(self, transcript: str):
        """Capture image, detect/OCR, send to LLM with visual context."""
        if self._camera is None:
            from vision.camera import Camera
            self._camera = Camera()

        image = await self._camera.capture()

        from vision.object_detector import ObjectDetector
        detector = ObjectDetector(model_dir=MODELS_DIR / "vision")
        detections = await detector.detect(image)

        # Build a visual context description
        objects = ", ".join(d["label"] for d in detections) if detections else "unknown object"
        visual_prompt = (
            f"The child is showing me something and asked: '{transcript}'. "
            f"I can see: {objects}. Describe what this is in simple terms for a "
            f"{self.config_manager.config.child.age_min}-"
            f"{self.config_manager.config.child.age_max} year old child."
        )

        await self._stream_response(visual_prompt)

    def _is_vision_request(self, transcript: str) -> bool:
        """Check if the child's speech is asking about something visual."""
        vision_triggers = [
            "what is this", "what's this", "what do you see",
            "look at this", "can you see", "what am i holding",
            "tell me about this", "what color is this",
        ]
        lower = transcript.lower()
        return any(trigger in lower for trigger in vision_triggers)

    async def _speak_text(self, text: str):
        """Synthesize and speak a single text string."""
        self.state.set_state(BotState.SPEAKING)
        audio_data = await self._tts.synthesize(text)
        await self._audio_player.play(audio_data)

    async def _play_sound(self, sound_name: str):
        """Play a pre-loaded sound effect (ding, thinking, ready)."""
        sound_path = SOUNDS_DIR / f"{sound_name}.wav"
        if sound_path.exists():
            await self._audio_player.play_file(sound_path)

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.state.request_stop()
        if self._llm_router:
            await self._llm_router.unload()
        logger.info("Shutdown complete.")


def main():
    """Entry point."""
    import sys
    from loguru import logger as log

    log.remove()
    log.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
    log.add(DATA_DIR / "bot.log", rotation="10 MB", retention="7 days", level="DEBUG")

    orchestrator = Orchestrator()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(orchestrator.start())
    except KeyboardInterrupt:
        loop.run_until_complete(orchestrator.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
