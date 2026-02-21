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
        """Main loop: wake word → capture → STT → LLM stream → TTS stream → speak.

        After each response, opens a follow-up window (5s) so the child
        can keep talking without repeating the wake word.
        """
        follow_up = False  # True when in follow-up listening mode

        while self.state.is_running:
            try:
                if not follow_up:
                    self.state.set_state(BotState.READY)
                    self.state.in_follow_up = False
                    self.state.current_transcript = None
                    self.state.current_response = None
                    self.state.current_image_b64 = None
                    self.state.current_detections = None

                    # Step 1: Wait for wake word
                    audio_stream = self._audio_capture.stream()
                    await self._wake_word.listen(audio_stream)
                    logger.info("Wake word detected!")

                    if not self.state.mic_enabled:
                        continue

                # Step 2: Play acknowledgment sound
                self.state.set_state(BotState.LISTENING)
                self.state.interaction_start_time = time.monotonic()
                if not follow_up:
                    asyncio.create_task(self._play_sound("ding"))

                # Step 3: Capture speech with VAD
                # Give more time in follow-up mode (child needs time to think)
                wait_time = 10.0 if follow_up else 3.0
                audio_data = await self._vad.capture_until_silence(
                    self._audio_capture, initial_wait=wait_time
                )
                if audio_data is None or len(audio_data) == 0:
                    if follow_up:
                        logger.debug("No follow-up speech. Back to wake word mode.")
                        follow_up = False
                        continue
                    logger.debug("No speech detected after wake word.")
                    continue

                # Step 4: Speech-to-text
                self.state.set_state(BotState.PROCESSING)
                transcript = await self._stt.transcribe(audio_data)
                logger.info("Child said: '{}'", transcript)

                if not transcript or len(transcript.strip()) < 2:
                    if follow_up:
                        follow_up = False
                    continue

                # Echo detection: if follow-up transcript sounds like the
                # bot's own previous response (speaker → mic bleed), skip it.
                if follow_up and self._is_echo(transcript):
                    logger.info("Detected echo of bot's own speech, ignoring.")
                    continue

                self.state.current_transcript = transcript

                # Step 5: Safety check on input
                safe_input = self._safety_filter.check_input(transcript)
                if safe_input.blocked:
                    await self._speak_text(safe_input.redirect_response)
                    follow_up = True
                    continue

                # Step 6: Check if this is a vision request
                if self._is_vision_request(transcript) and self.state.camera_enabled:
                    await self._handle_vision_request(transcript)
                    follow_up = True
                    continue

                # Step 7: Stream LLM → sentence buffer → TTS → speak
                await self._stream_response(transcript)

                # Log interaction metadata (not content)
                elapsed = time.monotonic() - self.state.interaction_start_time
                logger.info("Interaction complete. Latency: {:.1f}s", elapsed)

                # Enter follow-up mode: listen again without wake word
                follow_up = True
                self.state.in_follow_up = True
                logger.info("Listening for follow-up (no wake word needed)...")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in interaction loop: {}", e)
                self.state.set_state(BotState.ERROR)
                self.state.last_error = str(e)
                follow_up = False
                await asyncio.sleep(1)

    async def _stream_response(self, transcript: str):
        """Stream LLM tokens → buffer sentences → TTS each sentence → play.

        Supports interruption via self.state.interrupt_event — when set,
        stops LLM streaming, cancels pending audio, and returns to listening.
        """
        self.state.set_state(BotState.PROCESSING)
        self.state.current_response = ""
        self.state.interrupt_event.clear()  # Reset at start

        provider = self._llm_router.get_provider()
        messages = self._llm_router.build_messages(transcript)
        self._sentence_buffer.reset()

        # Pause mic while bot is speaking to prevent echo feedback
        self._audio_capture.pause()

        interrupted = False

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
                # Check for interrupt before playing each chunk
                if self.state.interrupt_event.is_set():
                    break
                await self._audio_player.play(chunk)
                # Check again after playing (user may have pressed stop mid-sentence)
                if self.state.interrupt_event.is_set():
                    break
            speaking_done.set()

        player_task = asyncio.create_task(play_audio_chunks())

        # Stream tokens from LLM
        response_text = []
        async for token in provider.stream(messages):
            # Check for interrupt during LLM streaming
            if self.state.interrupt_event.is_set():
                logger.info("Response interrupted by user during LLM streaming.")
                interrupted = True
                break

            sentence = self._sentence_buffer.feed(token)
            if sentence:
                # Safety check on each sentence
                safe = self._safety_filter.check_output(sentence)
                if safe.blocked:
                    sentence = safe.redirect_response

                response_text.append(sentence)
                self.state.current_response = " ".join(response_text)

                # TTS the sentence and queue audio for playback
                audio_data = await self._tts.synthesize(sentence)
                await audio_queue.put(audio_data)

        # Flush any remaining text in the buffer (only if not interrupted)
        if not interrupted:
            remaining = self._sentence_buffer.flush()
            if remaining:
                safe = self._safety_filter.check_output(remaining)
                text = safe.redirect_response if safe.blocked else remaining
                response_text.append(text)
                self.state.current_response = " ".join(response_text)
                audio_data = await self._tts.synthesize(text)
                await audio_queue.put(audio_data)

        # Signal player to stop and wait for it to finish
        await audio_queue.put(None)
        await speaking_done.wait()
        player_task.cancel()

        if interrupted:
            # Stop any currently playing audio
            await self._audio_player.stop()
            logger.info("Response stopped. Returning to listening mode.")

        # Clear the interrupt flag
        self.state.interrupt_event.clear()

        # Wait for BT speaker buffer to fully flush, then resume mic
        await asyncio.sleep(1.0)
        self._audio_capture.resume()

    async def _handle_vision_request(self, transcript: str):
        """Capture image, detect/OCR, send to LLM with visual context."""
        if self._camera is None:
            from vision.camera import Camera
            self._camera = Camera()

        image = await self._camera.capture()

        # Encode image to base64 PNG for the dashboard
        if image is not None:
            self.state.current_image_b64 = self._image_to_base64(image)
        else:
            self.state.current_image_b64 = None

        from vision.object_detector import ObjectDetector
        detector = ObjectDetector(model_dir=MODELS_DIR / "vision")
        detections = await detector.detect(image)

        # Store detections for dashboard display
        self.state.current_detections = detections

        # Build a visual context description
        objects = ", ".join(d["label"] for d in detections) if detections else "unknown object"
        visual_prompt = (
            f"The child is showing me something and asked: '{transcript}'. "
            f"I can see: {objects}. Describe what this is in simple terms for a "
            f"{self.config_manager.config.child.age_min}-"
            f"{self.config_manager.config.child.age_max} year old child."
        )

        await self._stream_response(visual_prompt)

    @staticmethod
    def _image_to_base64(image) -> str:
        """Convert a captured image to a base64-encoded JPEG string."""
        import base64
        import io
        from PIL import Image

        # picamera2 capture_array() returns BGR despite RGB888 config
        # Swap R and B channels: [:, :, ::-1] converts BGR → RGB
        rgb_image = image[:, :, ::-1]
        img = Image.fromarray(rgb_image)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _is_echo(self, transcript: str) -> bool:
        """Check if transcript is an echo of the bot's own last response.

        When the mic picks up the speaker output, STT will transcribe it.
        We detect this by checking if significant words from the transcript
        overlap with the bot's last response.
        """
        last_response = self.state.current_response
        if not last_response:
            return False

        # Normalize both strings
        t_words = set(transcript.lower().split())
        r_words = set(last_response.lower().split())

        # Remove very common short words
        stop = {"the", "a", "an", "is", "it", "to", "and", "of", "in", "i", "you", "that", "this"}
        t_words -= stop
        r_words -= stop

        if not t_words:
            return False

        # If most of the transcript words appear in the response, it's echo
        overlap = t_words & r_words
        ratio = len(overlap) / len(t_words)
        if ratio > 0.5:
            logger.debug("Echo ratio: {:.0%} ({} / {})", ratio, len(overlap), len(t_words))
            return True
        return False

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
        """Synthesize and speak a single text string. Supports interruption."""
        self._audio_capture.pause()
        self.state.set_state(BotState.SPEAKING)
        self.state.interrupt_event.clear()

        audio_data = await self._tts.synthesize(text)

        if not self.state.interrupt_event.is_set():
            await self._audio_player.play(audio_data)

        if self.state.interrupt_event.is_set():
            await self._audio_player.stop()
            self.state.interrupt_event.clear()

        await asyncio.sleep(1.0)
        self._audio_capture.resume()

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
