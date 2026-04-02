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
SONGS_DIR = BASE_DIR / "audio" / "songs"


class Orchestrator:
    """Main orchestrator: manages the streaming voice interaction pipeline."""

    def __init__(self):
        self.config_manager = ConfigManager(DATA_DIR)
        self.state = SharedState()

        # Components (initialized lazily)
        self._wake_word = None
        self._vad = None
        self._stt = None
        self._cloud_stt = None
        self._tts = None
        self._llm_router = None
        self._audio_capture = None
        self._audio_player = None
        self._safety_filter = None
        self._camera = None
        self._object_detector = None
        self._api_server = None
        self._display = None
        self._car = None  # Bluetooth car chassis module
        self._car_command_active = False  # True while executing a car command
        self._last_car_command = None  # Last executed car command (for reverse)
        self._follower = None  # Follow mode module
        self._touch = None  # Capacitive touch controller
        self._servos = None  # SG90 servo hands
        self._projector = None  # HDMI projector module
        self._image_generator = None  # DALL-E 3 image generator
        self._loop = None  # Main asyncio event loop (for thread callbacks)

        # Conversation history: rolling buffer of user/assistant message pairs.
        # Gives the LLM context of prior turns within a single conversation session.
        # Cleared when follow-up mode times out or a new wake word starts.
        self._conversation_history: list[dict] = []

        # Max history turns (user+assistant pairs).
        # Online providers have huge context windows, offline is limited.
        self._max_history_online = 10  # 10 pairs = 20 messages
        self._max_history_offline = 4  # 4 pairs = 8 messages

        # Song playback state
        self._song_interrupted = False

        # Pending touch-triggered speech (handled by main loop, not concurrently)
        self._pending_speech: str | None = None

    async def start(self):
        """Boot sequence: load config, check consent, load models, start listening."""
        logger.info("=== RPi Learning Companion starting ===")

        # Store event loop reference for thread callbacks (touch sensor, etc.)
        self._loop = asyncio.get_running_loop()

        config = self.config_manager.config

        # Sync state from config
        self.state.llm_mode = LLMMode(config.mode)
        self.state.active_provider = config.provider
        self.state.mic_enabled = config.hardware.mic_enabled
        self.state.camera_enabled = config.hardware.camera_enabled
        self.state.cloud_stt = config.hardware.cloud_stt

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
        """Start the FastAPI server in the background (HTTP + HTTPS)."""
        from api.server import create_app

        app = create_app(self.config_manager, self.state)
        self._api_server = app

        import uvicorn

        # HTTP server on port 8080
        server_config = uvicorn.Config(
            app, host="0.0.0.0", port=8080, log_level="warning"
        )
        server = uvicorn.Server(server_config)
        asyncio.create_task(server.serve())
        logger.info("API server started on port 8080 (HTTP)")

        # HTTPS server on port 8443 (for Web Speech API which requires secure context)
        cert_dir = Path(__file__).resolve().parent.parent
        cert_file = cert_dir / "cert.pem"
        key_file = cert_dir / "key.pem"
        if cert_file.exists() and key_file.exists():
            ssl_config = uvicorn.Config(
                app, host="0.0.0.0", port=8443, log_level="warning",
                ssl_certfile=str(cert_file), ssl_keyfile=str(key_file),
            )
            ssl_server = uvicorn.Server(ssl_config)
            asyncio.create_task(ssl_server.serve())
            logger.info("API server started on port 8443 (HTTPS)")
        else:
            logger.warning("No SSL certs found — HTTPS not available")

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
        language = self.config_manager.config.child.language
        self._vad = VADDetector()
        self._stt = SpeechToText(model_dir=MODELS_DIR / "stt", language=language)
        self._tts = TextToSpeech(model_dir=MODELS_DIR / "tts", language=language)
        self._llm_router = LLMRouter(self.config_manager, self.state)
        self._safety_filter = SafetyFilter(self.config_manager)
        self._sentence_buffer = SentenceBuffer()

        # Load models
        await self._wake_word.load()
        await self._stt.load()
        await self._tts.load()
        await self._llm_router.load()

        self.state.is_model_loaded = True

        # Expose audio_player to API for volume control
        if self._api_server:
            self._api_server.state.audio_player = self._audio_player

        # Sync initial volume
        self._audio_player.set_volume(self.state.volume)

        # Start TFT display (non-blocking, runs in background thread)
        try:
            from display.tft_display import TFTDisplay
            self._display = TFTDisplay()
            self._display.start(self.state)
        except Exception as e:
            logger.warning("TFT display not available: {}", e)

        # Try to connect to car chassis via Bluetooth (non-blocking, best-effort at boot)
        # If it fails, the user can reconnect from the Settings menu → Car item
        asyncio.create_task(self._connect_car_module())

        # Start touch controller (non-blocking, runs in background thread)
        try:
            from display.touch import TouchController
            self._touch = TouchController()
            self._touch.add_callback(self._on_touch_event)
            self._touch.start()
            self.state.touch_enabled = True
            logger.info("Touch controller started")
        except Exception as e:
            logger.warning("Touch controller not available: {}", e)

        # Start servo hands
        try:
            from modules.servos import ServoController
            self._servos = ServoController()
            self._servos.start()
            logger.info("Servo hands started")
        except Exception as e:
            logger.warning("Servo controller not available: {}", e)

        # Detect and start projector if connected
        try:
            from modules.projector import Projector
            self._projector = Projector()
            if self._projector.detect():
                self._projector.start()
                self.state.projector_connected = True
                logger.info("Projector detected and started")
            else:
                logger.info("No projector detected (will check on command)")
        except Exception as e:
            logger.warning("Projector module not available: {}", e)

        logger.info("All components loaded.")

    async def _interaction_loop(self):
        """Main loop: wake word → capture → STT → LLM stream → TTS stream → speak.

        After each response, opens a follow-up window (5s) so the child
        can keep talking without repeating the wake word.
        """
        follow_up = False  # True when in follow-up listening mode
        _from_remote = False  # True when current command came from phone

        while self.state.is_running:
            try:
                # Remote commands never enter follow-up mode
                if _from_remote and follow_up:
                    follow_up = False
                    self.state.in_follow_up = False

                # If mic is disabled, pause until it's re-enabled
                if not self.state.mic_enabled:
                    self.state.set_state(BotState.READY)
                    follow_up = False
                    self._clear_conversation_history()
                    await asyncio.sleep(1)
                    continue

                if not follow_up:
                    self.state.set_state(BotState.READY)
                    self.state.in_follow_up = False
                    self.state.current_transcript = None
                    self.state.current_response = None
                    self.state.current_image_b64 = None
                    self.state.current_detections = None
                    self._clear_conversation_history()

                    # Step 1: Wait for wake word OR touch-to-wake
                    self.state.wake_event.clear()
                    audio_stream = self._audio_capture.stream()
                    wake_word_task = asyncio.create_task(
                        self._wake_word.listen(audio_stream)
                    )
                    touch_wake_task = asyncio.create_task(
                        self.state.wake_event.wait()
                    )
                    done, pending = await asyncio.wait(
                        [wake_word_task, touch_wake_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass
                    # Close the async generator to release the audio stream
                    await audio_stream.aclose()
                    # Reset mic stream so next iteration gets a clean one
                    self._audio_capture.pause()
                    self._audio_capture.resume()
                    self.state.wake_event.clear()
                    logger.info("Wake detected (voice or touch)!")

                    # Check if this wake was triggered by a touch-speech action
                    # (pet reaction, menu select, etc.) — handle it and loop back
                    if self._pending_speech:
                        text = self._pending_speech
                        self._pending_speech = None
                        logger.info("Touch speech: {}", text)
                        await self._speak_text(text)
                        self.state.set_state(BotState.READY)
                        continue

                # Check for remote text from phone/web dashboard.
                # This can arrive during wake word wait OR follow-up mode.
                transcript = None
                if self.state.remote_text:
                    transcript = self.state.remote_text
                    self.state.remote_text = None
                    _from_remote = True
                    logger.info("Remote text from phone: '{}'", transcript)
                    self.state.set_state(BotState.PROCESSING)
                    self.state.interaction_start_time = time.monotonic()
                    asyncio.create_task(self._play_sound("ding"))
                    self.state.current_transcript = transcript
                    follow_up = False
                else:
                    _from_remote = False
                    # Step 2: Play acknowledgment sound
                    self.state.set_state(BotState.LISTENING)
                    self.state.interaction_start_time = time.monotonic()
                    if not follow_up:
                        asyncio.create_task(self._play_sound("ding"))

                    # Step 3: Capture speech with VAD
                    wait_time = 5.0 if follow_up else 3.0
                    t0 = time.monotonic()
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

                    import numpy as np
                    rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                    if rms < 300:
                        logger.debug("Audio too quiet (RMS={:.0f}), skipping.", rms)
                        if follow_up:
                            follow_up = False
                        continue

                    t_vad = time.monotonic()
                    logger.info("[TIMING] VAD capture: {:.1f}s", t_vad - t0)

                    # Step 4: Speech-to-text
                    self.state.set_state(BotState.PROCESSING)
                    transcript = ""
                    if self.state.cloud_stt and self.state.llm_mode == LLMMode.ONLINE:
                        cloud = self._get_cloud_stt()
                        if cloud:
                            transcript = await cloud.transcribe(audio_data)
                        if not transcript:
                            logger.info("Cloud STT returned empty, falling back to local.")
                            transcript = await self._stt.transcribe(audio_data)
                    else:
                        transcript = await self._stt.transcribe(audio_data)
                    t_stt = time.monotonic()
                    logger.info("[TIMING] STT: {:.1f}s", t_stt - t_vad)
                    logger.info("Child said: '{}'", transcript)

                    if not transcript or len(transcript.strip()) < 2:
                        if follow_up:
                            follow_up = False
                        continue

                    self.state.current_transcript = transcript

                # Whisper hallucination filter: tiny.en often hallucinates
                # these phrases from silence or background noise.
                if self._is_whisper_hallucination(transcript):
                    logger.info("Whisper hallucination filtered: '{}'", transcript)
                    if follow_up:
                        follow_up = False
                    continue

                # Echo detection: if follow-up transcript sounds like the
                # bot's own previous response (speaker → mic bleed), skip it.
                if follow_up and self._is_echo(transcript):
                    logger.info("Detected echo of bot's own speech, ignoring.")
                    continue

                self.state.current_transcript = transcript

                # ── Follow mode intercept ──
                # When follow mode is active, ANY speech stops it first.
                # Then we continue to process the actual command normally.
                if self._follower and self._follower.active:
                    logger.info("Follow mode active — stopping before processing command")
                    await self._stop_follow_mode()
                    # If they just said "stop" / "stop following" etc, we're done
                    lower_check = transcript.lower().strip()
                    stop_words = ["stop", "stay", "halt", "enough", "ruk", "ఆగు", "ఆపు"]
                    if any(lower_check.startswith(w) or lower_check == w for w in stop_words):
                        follow_up = True
                        continue
                    # Otherwise, fall through to handle whatever else they asked

                # Step 5: Safety check on input
                safe_input = self._safety_filter.check_input(transcript)
                if safe_input.blocked:
                    await self._speak_text(safe_input.redirect_response)
                    follow_up = True
                    continue

                # Step 6a: Check if this is a volume command
                volume_level = self._parse_volume_command(transcript)
                if volume_level is not None:
                    await self._handle_volume_command(volume_level)
                    follow_up = True
                    continue

                # Step 6b: Check car/movement commands FIRST (before generic stop)
                # so "stop following" isn't eaten by the generic stop handler
                car_cmd = self._parse_car_command(transcript)
                if car_cmd is not None:
                    await self._handle_car_command(car_cmd)
                    follow_up = True
                    continue

                # Step 6c: Check if this is a stop command (stop song/audio)
                if self._is_stop_command(transcript):
                    await self._audio_player.stop()
                    await self._speak_text("Okay, stopped!")
                    follow_up = True
                    continue

                # Step 6d: Check if this is a song/music request
                song_match = self._parse_song_command(transcript)
                if song_match is not None:
                    await self._handle_song_command(song_match)
                    follow_up = getattr(self, '_song_interrupted', False) or True
                    self._song_interrupted = False
                    continue

                # Step 6e: Check if this is a projector command
                proj_cmd = self._parse_projector_command(transcript)
                if proj_cmd is not None:
                    await self._handle_projector_command(proj_cmd)
                    follow_up = True
                    continue

                # Step 6: Check if this is a vision request
                if self._is_vision_request(transcript) and self.state.camera_enabled:
                    await self._handle_vision_request(transcript)
                    # Record vision exchange in history too
                    if self.state.current_response:
                        self._append_to_history(
                            transcript, self.state.current_response,
                            image_b64=self.state.current_image_b64,
                        )
                    follow_up = True
                    continue

                # Step 7: Stream LLM → sentence buffer → TTS → speak
                await self._stream_response(transcript)

                # Record exchange in conversation history for context
                if self.state.current_response:
                    self._append_to_history(transcript, self.state.current_response)

                # Log interaction metadata (not content)
                elapsed = time.monotonic() - self.state.interaction_start_time
                logger.info("Interaction complete. Latency: {:.1f}s", elapsed)

                # Enter follow-up mode: listen again without wake word
                # But NOT for remote commands — those go back to wake word mode
                if _from_remote:
                    follow_up = False
                    self.state.in_follow_up = False
                    logger.info("Remote command done. Back to wake word mode.")
                else:
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

                # Attempt to recover audio stream if it was closed/broken
                err_msg = str(e).lower()
                if "stream" in err_msg and ("closed" in err_msg or "stopped" in err_msg):
                    try:
                        self._audio_capture.resume()
                        logger.info("Audio stream recovered after error.")
                    except Exception:
                        logger.warning("Audio stream recovery failed, will retry.")

                await asyncio.sleep(2)

    async def _stream_response(self, transcript: str):
        """Stream LLM tokens → buffer sentences → TTS each sentence → play.

        Three concurrent stages forming a pipeline:
          1. LLM streaming → feeds sentences to TTS queue
          2. TTS synthesis  → processes sentences into audio, feeds play queue
          3. Audio playback → plays WAV chunks through speaker

        All three run concurrently so LLM keeps generating while TTS
        synthesizes while audio plays — minimizing time-to-first-sound.

        Supports interruption via self.state.interrupt_event.
        """
        t_start = time.monotonic()
        self.state.set_state(BotState.PROCESSING)
        self.state.current_response = ""
        self.state.interrupt_event.clear()

        provider = await self._llm_router.get_provider()
        messages = self._llm_router.build_messages(transcript, self._conversation_history)
        self._sentence_buffer.reset()

        # Pause mic while bot is speaking to prevent echo feedback
        self._audio_capture.pause()

        # Start hand + car gestures while speaking
        self._servos.start_speaking_gestures()
        if self._car and self._car.connected and not self._car_command_active:
            self._car.start_speaking_gestures()

        interrupted = False
        first_audio_logged = False

        # Queue: sentences waiting for TTS synthesis
        tts_queue: asyncio.Queue = asyncio.Queue()
        # Queue: WAV audio waiting for playback
        play_queue: asyncio.Queue = asyncio.Queue()
        speaking_done = asyncio.Event()

        # --- Stage 3: Audio playback task ---
        async def play_audio_chunks():
            while True:
                chunk = await play_queue.get()
                if chunk is None:
                    break
                if self.state.interrupt_event.is_set():
                    break
                self.state.set_state(BotState.SPEAKING)
                await self._audio_player.play(chunk)
                if self.state.interrupt_event.is_set():
                    break
            speaking_done.set()

        # --- Stage 2: TTS synthesis task ---
        async def tts_worker():
            nonlocal first_audio_logged
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    break
                if self.state.interrupt_event.is_set():
                    break
                audio_data = await self._tts.synthesize(sentence)
                if not first_audio_logged:
                    logger.info("[TIMING] First audio ready: {:.1f}s",
                                time.monotonic() - t_start)
                    first_audio_logged = True
                await play_queue.put(audio_data)
            # Signal player that no more audio is coming
            await play_queue.put(None)

        player_task = asyncio.create_task(play_audio_chunks())
        tts_task = asyncio.create_task(tts_worker())

        # --- Stage 1: LLM streaming (runs in this coroutine) ---
        response_text = []
        first_token_time = None
        async for token in provider.stream(messages):
            if self.state.interrupt_event.is_set():
                logger.info("Response interrupted by user during LLM streaming.")
                interrupted = True
                break

            if first_token_time is None:
                first_token_time = time.monotonic()
                logger.info("[TIMING] LLM first token: {:.1f}s",
                            first_token_time - t_start)

            sentence = self._sentence_buffer.feed(token)
            if sentence:
                safe = self._safety_filter.check_output(sentence)
                if safe.blocked:
                    sentence = safe.redirect_response
                response_text.append(sentence)
                self.state.current_response = " ".join(response_text)
                # Send to TTS pipeline (non-blocking)
                await tts_queue.put(sentence)

        # Flush remaining text
        if not interrupted:
            remaining = self._sentence_buffer.flush()
            if remaining:
                safe = self._safety_filter.check_output(remaining)
                text = safe.redirect_response if safe.blocked else remaining
                response_text.append(text)
                self.state.current_response = " ".join(response_text)
                await tts_queue.put(text)

        # Signal TTS worker that LLM is done
        await tts_queue.put(None)

        # Wait for all audio to finish playing
        await speaking_done.wait()
        player_task.cancel()
        tts_task.cancel()

        # Stop hand + car gestures
        self._servos.stop_speaking_gestures()
        if self._car and self._car.connected and not self._car_command_active:
            self._car.stop_speaking_gestures()

        if interrupted:
            await self._audio_player.stop()
            logger.info("Response stopped. Returning to listening mode.")

        self.state.interrupt_event.clear()

        logger.info("[TIMING] Total response: {:.1f}s", time.monotonic() - t_start)

        # Wait for BT speaker buffer to fully flush, then resume mic
        await asyncio.sleep(1.0)
        self._audio_capture.resume()

    async def _handle_vision_request(self, transcript: str):
        """Capture image, blur faces for privacy, send to GPT-4o-mini vision.

        Hybrid approach: ALWAYS uses GPT-4o-mini for vision (even in offline
        mode) because local YOLO is too inaccurate. Faces are blurred and EXIF
        stripped before sending for DPDP compliance.

        Falls back to YOLO + local LLM only if OpenAI API key is not set.
        """
        t_start = time.monotonic()

        if self._camera is None:
            from vision.camera import Camera
            self._camera = Camera()

        image = await self._camera.capture()
        t_capture = time.monotonic()
        logger.info("[TIMING] Camera capture: {:.1f}s", t_capture - t_start)

        if image is None:
            await self._speak_text("I couldn't see anything. Can you show me again?")
            return

        age_min = self.config_manager.config.child.age_min
        age_max = self.config_manager.config.child.age_max

        # Check if we can use GPT-4o-mini vision (need OpenAI API key)
        api_key = self.config_manager.config.api_keys.openai
        if api_key:
            # --- Hybrid: blur faces → send to GPT-4o-mini (works in any mode) ---
            blurred_image = self._blur_faces(image)
            t_blur = time.monotonic()
            logger.info("[TIMING] Face blur: {:.1f}s", t_blur - t_capture)

            image_b64 = self._image_to_base64(blurred_image)
            self.state.current_image_b64 = image_b64
            self.state.current_detections = None

            logger.info("[VISION] Using GPT-4o-mini vision (faces blurred)")
            logger.info("[TIMING] Vision pipeline total: {:.1f}s (before LLM)",
                        time.monotonic() - t_start)

            from llm.prompts import build_system_prompt
            system_prompt = build_system_prompt(
                age_min=age_min, age_max=age_max,
                language=self.config_manager.config.child.language,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "low",
                    }},
                    {"type": "text", "text": (
                        f"The child asked: '{transcript}'. "
                        f"You CAN see this image. Describe what you see directly. "
                        f"Never say you cannot see or view the image. "
                        f"Use simple, fun words for a {age_min}-{age_max} year old. "
                        f"Start with the object name, not 'I can see'."
                    )},
                ]},
            ]

            # Always use OpenAI provider for vision, even in offline mode
            from llm.providers.openai_provider import OpenAIProvider
            vision_provider = OpenAIProvider(api_key=api_key)
            logger.info("[LLM] Using ONLINE provider: openai (vision override)")
            await self._stream_response_with_messages(messages, provider=vision_provider)
        else:
            # --- Fallback: YOLO + local LLM (no API key available) ---
            logger.info("[VISION] No API key — falling back to YOLO (offline)")
            image_b64 = self._image_to_base64(image)
            self.state.current_image_b64 = image_b64

            if self._object_detector is None:
                from vision.object_detector import ObjectDetector
                self._object_detector = ObjectDetector(model_dir=MODELS_DIR / "vision")
            detections = await self._object_detector.detect(image)
            t_detect = time.monotonic()
            logger.info("[TIMING] Object detection: {:.1f}s", t_detect - t_capture)

            self.state.current_detections = detections

            objects = ", ".join(d["label"] for d in detections) if detections else "unknown object"
            logger.info("[TIMING] Vision pipeline total: {:.1f}s (before LLM)",
                        t_detect - t_start)

            messages = [
                {"role": "user", "content": (
                    f"I see: {objects}. A {age_min}-{age_max} year old child asked: "
                    f"'{transcript}'. Reply in 2 simple sentences."
                )},
            ]
            logger.info("[LLM] Using OFFLINE for vision (no API key)")
            await self._stream_response_with_messages(messages)

    @staticmethod
    def _blur_faces(image):
        """Detect and blur all faces in an image for privacy.

        Uses OpenCV's Haar cascade (fast, runs on Pi in ~50ms).
        Returns a copy with all detected faces heavily blurred.
        """
        import cv2

        blurred = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Expand region slightly to cover full face area
            pad = int(0.2 * max(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(blurred.shape[1], x + w + pad)
            y2 = min(blurred.shape[0], y + h + pad)

            face_region = blurred[y1:y2, x1:x2]
            # Heavy Gaussian blur — makes face completely unrecognizable
            blurred[y1:y2, x1:x2] = cv2.GaussianBlur(face_region, (99, 99), 30)

        n = len(faces)
        if n > 0:
            logger.info("[PRIVACY] Blurred {} face(s) before sending to cloud", n)
        return blurred

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

    def _append_to_history(self, user_text: str, assistant_text: str,
                           image_b64: str | None = None):
        """Add a user/assistant exchange to conversation history.

        Keeps only the most recent N exchanges, where N depends on mode
        (online has a larger window than offline).
        Also updates state.conversation_messages for the live dashboard.
        """
        self._conversation_history.append({"role": "user", "content": user_text})
        self._conversation_history.append({"role": "assistant", "content": assistant_text})
        logger.debug("[HISTORY] {} turns in conversation", len(self._conversation_history) // 2)

        max_pairs = (
            self._max_history_online
            if self.state.llm_mode == LLMMode.ONLINE
            else self._max_history_offline
        )
        max_messages = max_pairs * 2
        if len(self._conversation_history) > max_messages:
            self._conversation_history = self._conversation_history[-max_messages:]

        # Push to dashboard-visible conversation messages (no limit — show all in session)
        user_msg = {"role": "user", "content": user_text}
        if image_b64:
            user_msg["image"] = image_b64
        self.state.conversation_messages.append(user_msg)
        self.state.conversation_messages.append({"role": "assistant", "content": assistant_text})

        # Keep dashboard messages trimmed to last 50 messages (25 exchanges)
        # so the API response doesn't grow unbounded
        max_dashboard_messages = 50
        if len(self.state.conversation_messages) > max_dashboard_messages:
            self.state.conversation_messages = self.state.conversation_messages[-max_dashboard_messages:]

    def _clear_conversation_history(self):
        """Clear conversation history (new session)."""
        if self._conversation_history:
            logger.debug("Conversation history cleared ({} messages)", len(self._conversation_history))
        self._conversation_history.clear()
        self.state.conversation_messages.clear()

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

    @staticmethod
    def _is_whisper_hallucination(transcript: str) -> bool:
        """Detect common Whisper tiny.en hallucinations from silence/noise."""
        t = transcript.strip().lower().rstrip(".")
        # Common phantom phrases Whisper produces from silence
        hallucinations = {
            "thank you", "thanks", "thanks for watching",
            "thank you for watching", "thanks for listening",
            "thank you for listening", "bye", "goodbye",
            "you", "the end", "subscribe",
            "like and subscribe", "see you next time",
            "i'm not sure", "okay", "oh",
        }
        return t in hallucinations

    def _parse_volume_command(self, transcript: str) -> int | None:
        """Parse volume commands from transcript.

        Recognizes patterns like:
        - "set volume to 50%", "volume 50 percent"
        - "increase volume", "louder", "volume up"
        - "decrease volume", "quieter", "volume down"
        - "mute", "unmute"
        - "full volume", "maximum volume"

        Returns target volume (0-100) or None if not a volume command.
        """
        import re
        lower = transcript.lower().strip()

        # Direct percentage: "set volume to 50", "volume 50 percent", "volume to 80"
        m = re.search(r'volume\s+(?:to\s+)?(\d+)\s*(?:%|percent)?', lower)
        if m:
            return int(m.group(1))

        m = re.search(r'(?:set|change)\s+(?:the\s+)?volume\s+(?:to\s+)?(\d+)', lower)
        if m:
            return int(m.group(1))

        # "50 percent volume"
        m = re.search(r'(\d+)\s*(?:%|percent)\s*volume', lower)
        if m:
            return int(m.group(1))

        # Relative commands
        current = self.state.volume
        up_words = ["increase volume", "louder", "volume up", "turn up", "raise volume", "more volume"]
        down_words = ["decrease volume", "quieter", "volume down", "turn down", "lower volume", "less volume", "reduce volume"]

        for phrase in up_words:
            if phrase in lower:
                return min(100, current + 20)

        for phrase in down_words:
            if phrase in lower:
                return max(0, current - 20)

        # Special commands
        if "mute" in lower and "unmute" not in lower:
            return 0
        if "unmute" in lower:
            return 80
        if any(w in lower for w in ["full volume", "maximum volume", "max volume"]):
            return 100

        return None

    async def _handle_volume_command(self, level: int):
        """Set volume and confirm to the child."""
        level = max(0, min(100, level))
        actual = self._audio_player.set_volume(level)
        self.state.volume = actual
        logger.info("Volume changed to {}%", actual)

        # Speak confirmation at the new volume
        if actual == 0:
            # Can't speak at 0 volume, briefly set to 50 to confirm
            self._audio_player.set_volume(50)
            await self._speak_text("Okay, I'm muted now. Say unmute to hear me again!")
            self._audio_player.set_volume(0)
        else:
            await self._speak_text(f"Okay! Volume is now at {actual} percent.")

    def _parse_song_command(self, transcript: str) -> str | None:
        """Parse song/music requests from transcript.

        Recognizes patterns like:
        - "sing me a song", "sing a song", "play a song"
        - "sing twinkle twinkle", "play mary had a little lamb"
        - "play song number 2", "next song", "another song"
        - "sing nursery rhyme", "play rhyme"

        Returns song filename (without path) or None if not a song request.
        """
        import re
        lower = transcript.lower().strip()

        # Check if this is a song/music request at all
        song_triggers = [
            "sing me a song", "sing a song", "sing song",
            "play a song", "play song", "play music",
            "sing me something", "sing something",
            "nursery rhyme", "play rhyme",
            "another song", "next song", "one more song",
            "sing for me", "can you sing",
            # Hindi
            "गाना सुनाओ", "गाना गाओ", "एक गाना",
            # Telugu
            "పాట పాడు", "పాట వినిపించు",
        ]

        # Get available songs from disk
        if not SONGS_DIR.exists():
            return None
        available = sorted(SONGS_DIR.glob("*.wav"))
        if not available:
            return None

        # Check for specific song name match first
        for song_path in available:
            # Match filename (without extension) against transcript
            # e.g. "twinkle_twinkle.wav" matches "twinkle twinkle"
            song_name = song_path.stem.replace("_", " ").replace("-", " ")
            if song_name in lower:
                return song_path.name

        # Check for "song number N" pattern
        m = re.search(r'(?:song|rhyme)\s*(?:number\s*)?(\d+)', lower)
        if m:
            idx = int(m.group(1)) - 1  # 1-indexed
            if 0 <= idx < len(available):
                return available[idx].name

        # Generic song request → pick a random song
        import random
        for trigger in song_triggers:
            if trigger in lower:
                return random.choice(available).name

        return None

    async def _handle_song_command(self, song_filename: str):
        """Play a song from the songs library.

        Runs song playback in the background while listening for the wake word.
        If the child says the wake word during playback, the song stops
        immediately and the bot goes into listening mode.
        """
        song_path = SONGS_DIR / song_filename
        if not song_path.exists():
            await self._speak_text("Sorry, I couldn't find that song.")
            return

        song_name = song_path.stem.replace("_", " ").replace("-", " ").title()
        logger.info("Playing song: {} ({})", song_name, song_filename)

        # Brief announcement
        await self._speak_text(f"Okay! Here's {song_name}!")

        # Set state to speaking for display
        self.state.set_state(BotState.SPEAKING)

        # Start song playback as a background task
        play_task = asyncio.create_task(
            self._audio_player.play_file(song_path, timeout=300)
        )

        # Listen for wake word while song plays — child can interrupt
        try:
            audio_stream = self._audio_capture.stream()
            wake_task = asyncio.create_task(
                self._wake_word.listen(audio_stream)
            )

            # Wait for either: song finishes OR wake word detected
            done, pending = await asyncio.wait(
                [play_task, wake_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if wake_task in done:
                # Wake word detected — stop the song!
                logger.info("Wake word during song — stopping playback.")
                await self._audio_player.stop()
                play_task.cancel()
                try:
                    await play_task
                except (asyncio.CancelledError, Exception):
                    pass
                # Signal that we should go straight to listening mode
                self._song_interrupted = True
            else:
                # Song finished naturally
                wake_task.cancel()
                try:
                    await wake_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._song_interrupted = False

        except Exception as e:
            logger.error("Error during song playback: {}", e)
            await self._audio_player.stop()
            play_task.cancel()
            self._song_interrupted = False

        logger.info("Song finished: {}", song_name)

    def _is_stop_command(self, transcript: str) -> bool:
        """Check if the child wants to stop the current song."""
        lower = transcript.lower().strip()
        stop_triggers = [
            "stop", "stop the song", "stop the music", "stop singing",
            "stop playing", "be quiet", "quiet", "enough",
            "no more", "that's enough", "shut up",
            # Hindi
            "बंद करो", "रुको", "बस",
            # Telugu
            "ఆపు", "ఆపండి", "చాలు",
        ]
        return any(trigger in lower for trigger in stop_triggers)

    def _is_vision_request(self, transcript: str) -> bool:
        """Check if the child's speech is asking about something visual."""
        vision_triggers = [
            # English
            "what is this", "what's this", "what do you see",
            "look at this", "can you see", "what am i holding",
            "tell me about this", "what color is this",
            # Hindi
            "ये क्या है", "यह क्या है", "क्या दिख रहा",
            "क्या देख रहे", "कैमरा में", "केमरा में",
            "इसे देखो", "क्या दिखाई", "दिखाओ",
            # Telugu
            "ఇది ఏమిటి", "ఏమిటి ఇది", "ఏం కనిపిస్తుంది",
            "చూడు", "కెమెరా లో", "ఇది చూపించు",
            "ఏం చూస్తున్నావు", "ఏమి కనిపిస్తుంది",
        ]
        lower = transcript.lower()
        return any(trigger in lower for trigger in vision_triggers)

    # ── Car chassis module ──────────────────────────────────────────

    async def _connect_car_module(self, mac: str = None):
        """Try to connect to car chassis via Bluetooth (background task).

        Args:
            mac: Optional MAC address to connect to directly.
                 If None, uses saved MAC or scans for HC-05.
        """
        self.state.car_connecting = True
        self.state.car_connected = False
        try:
            from modules.car import CarChassis
            if self._car is None:
                self._car = CarChassis()
            connected = await self._car.connect(mac=mac)
            if connected:
                self.state.car_connected = True
                self.state.car_mac = self._car._mac
                logger.info("Car chassis connected via Bluetooth!")
            else:
                logger.info("Car chassis not found (will retry on car command)")
                self._car = None
                self.state.car_connected = False
        except Exception as e:
            logger.debug("Car module init skipped: {}", e)
            self._car = None
            self.state.car_connected = False
        finally:
            self.state.car_connecting = False

    async def _disconnect_car_module(self):
        """Disconnect from the car chassis and update state."""
        if self._car:
            await self._car.disconnect()
            self._car = None
        self.state.car_connected = False
        self.state.car_connecting = False
        logger.info("Car chassis disconnected from settings")

    def _parse_car_command(self, transcript: str) -> dict | None:
        """Parse movement commands from child's speech.

        Returns dict with 'action' and optional 'speed'/'duration', or None.
        Supports:
        - Duration: "move forward for 5 seconds"
        - Sequences: "forward for 5 seconds then left for 3 seconds then right for 4 seconds"
        """
        import re
        # Clean up punctuation from STT
        lower = transcript.lower().strip().rstrip('.')

        # ── Check for "reverse" command (replay last command backwards) ──
        reverse_triggers = [
            "reverse", "go back home", "come back home", "retrace",
            "go home", "come home", "undo", "reverse path",
            "वापस जाओ", "वापस आओ", "घर जाओ",  # Hindi
            "వెనక్కి రా", "ఇంటికి వెళ్ళు",  # Telugu
        ]
        for t in reverse_triggers:
            if t in lower:
                if self._last_car_command:
                    return self._reverse_command(self._last_car_command)
                return None

        # ── Check for sequential commands (contains "then") ──
        if " then " in lower:
            parts = [p.strip().strip(',.') for p in lower.split(" then ")]
            all_steps = []
            for part in parts:
                parsed = self._parse_single_car_command(part)
                if parsed:
                    # Flatten nested sequences (e.g. "turn left for 10s" returns a sequence)
                    if parsed.get("action") == "sequence":
                        all_steps.extend(parsed["steps"])
                    else:
                        all_steps.append(parsed)
            if all_steps:
                logger.info("Parsed car sequence: {} steps", len(all_steps))
                for i, s in enumerate(all_steps):
                    logger.info("  Step {}: {} speed={} dur={}", i+1, s["action"], s.get("speed"), s.get("duration"))
                return {"action": "sequence", "steps": all_steps}
            return None

        result = self._parse_single_car_command(lower)
        if result:
            logger.info("Parsed car command: {}", result)
        return result

    def _extract_duration(self, text: str, default: float) -> float:
        """Extract duration in seconds from text like 'for 5 seconds' or 'for 10 sec'."""
        import re
        m = re.search(r'(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)', text)
        if m:
            return float(m.group(1))
        # Check for spoken numbers
        word_nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                     "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                     "fifteen": 15, "twenty": 20, "thirty": 30}
        m2 = re.search(r'for\s+(\w+)\s+(?:seconds?|secs?)', text)
        if m2 and m2.group(1) in word_nums:
            return float(word_nums[m2.group(1)])
        return default

    def _parse_single_car_command(self, lower: str) -> dict | None:
        """Parse a single car movement command (no 'then' sequences)."""

        # Forward
        forward_triggers = [
            "go forward", "move forward", "go ahead", "move ahead",
            "go straight", "drive forward", "drive ahead", "forward",
            "आगे चलो", "आगे जाओ",  # Hindi
            "ముందుకు వెళ్ళు", "ముందుకు",  # Telugu
        ]
        for t in forward_triggers:
            if t in lower:
                dur = self._extract_duration(lower, 1.0)
                return {"action": "forward", "speed": 180, "duration": dur}

        # Backward
        backward_triggers = [
            "go back", "move back", "go backward", "move backward",
            "drive back", "back up", "backward",
            "पीछे जाओ", "पीछे चलो",  # Hindi
            "వెనక్కు వెళ్ళు", "వెనక్కు",  # Telugu
        ]
        for t in backward_triggers:
            if t in lower:
                dur = self._extract_duration(lower, 1.0)
                return {"action": "backward", "speed": 180, "duration": dur}

        # Turn left (swapped for backward-mounted bot)
        # "turn left" = spin 2s then forward for remaining time
        TURN_SPIN_DURATION = 2.0
        left_triggers = [
            "turn left", "go left", "move left", "left",
            "बाएं मुड़ो", "बाएं जाओ",  # Hindi
            "ఎడమకు తిరుగు", "ఎడమకు",  # Telugu
        ]
        for t in left_triggers:
            if t in lower:
                total_dur = self._extract_duration(lower, TURN_SPIN_DURATION)
                if total_dur > TURN_SPIN_DURATION:
                    return {"action": "sequence", "steps": [
                        {"action": "spin_right", "speed": 180, "duration": TURN_SPIN_DURATION},
                        {"action": "forward", "speed": 180, "duration": total_dur - TURN_SPIN_DURATION},
                    ]}
                return {"action": "spin_right", "speed": 180, "duration": total_dur}

        # Turn right (swapped for backward-mounted bot)
        right_triggers = [
            "turn right", "go right", "move right", "right",
            "दाएं मुड़ो", "दाएं जाओ",  # Hindi
            "కుడికి తిరుగు", "కుడికి",  # Telugu
        ]
        for t in right_triggers:
            if t in lower:
                total_dur = self._extract_duration(lower, TURN_SPIN_DURATION)
                if total_dur > TURN_SPIN_DURATION:
                    return {"action": "sequence", "steps": [
                        {"action": "spin_left", "speed": 180, "duration": TURN_SPIN_DURATION},
                        {"action": "forward", "speed": 180, "duration": total_dur - TURN_SPIN_DURATION},
                    ]}
                return {"action": "spin_left", "speed": 180, "duration": total_dur}

        # Spin
        spin_triggers = [
            "spin", "turn around", "rotate", "do a spin",
            "घूमो", "चक्कर",  # Hindi
            "తిరుగు", "గిర గిర",  # Telugu
        ]
        for t in spin_triggers:
            if t in lower:
                return {"action": "spin_right", "speed": 200, "duration": 1.0}

        # Stop car
        car_stop_triggers = [
            "stop the car", "car stop", "stop moving", "stop driving",
            "brake", "halt",
            "गाड़ी रोको", "रुक जाओ",  # Hindi
            "కారు ఆపు", "ఆగు",  # Telugu
        ]
        for t in car_stop_triggers:
            if t in lower:
                return {"action": "stop"}

        # Dance
        dance_triggers = [
            "dance", "do a dance", "car dance", "wiggle",
            "नाचो", "डांस",  # Hindi
            "డ్యాన్స్", "నాట్యం",  # Telugu
        ]
        for t in dance_triggers:
            if t in lower:
                return {"action": "dance"}

        # Come here / come to me
        come_triggers = [
            "come here", "come to me", "come over",
            "इधर आओ", "मेरे पास आओ",  # Hindi
            "ఇక్కడికి రా", "నా దగ్గరికి రా",  # Telugu
        ]
        for t in come_triggers:
            if t in lower:
                return {"action": "forward", "speed": 150, "duration": 1.5}

        # Stop following (MUST be checked before "follow" triggers)
        stop_follow_triggers = [
            "stop following", "don't follow", "stay there",
            "stop follow", "no follow", "quit following",
            "रुक जाओ", "मत आओ",  # Hindi
            "ఆగిపో", "రావద్దు",  # Telugu
        ]
        for t in stop_follow_triggers:
            if t in lower:
                return {"action": "follow_stop"}

        # Follow me
        follow_triggers = [
            "follow me", "follow", "chase me", "come with me",
            "मेरे पीछे आओ", "मेरे साथ आओ",  # Hindi
            "నన్ను అనుసరించు", "నా వెంట రా",  # Telugu
        ]
        for t in follow_triggers:
            if t in lower:
                return {"action": "follow_start"}

        return None

    async def _handle_car_command(self, cmd: dict):
        """Execute a parsed car command (single or sequence)."""
        action = cmd["action"]

        # ── Follow mode stop — handle BEFORE car connection check ──
        if action == "follow_stop":
            await self._stop_follow_mode()
            return

        # Lazy reconnect if not connected
        if not self._car or not self._car.connected:
            self.state.car_connected = False
            await self._connect_car_module()

        if not self._car or not self._car.connected:
            self.state.car_connected = False
            await self._speak_text("I can't find my wheels! Is the car turned on?")
            return

        # ── Follow mode start ──
        if action == "follow_start":
            await self._start_follow_mode()
            return

        # If follow mode is active and a manual command comes in, stop following first
        if self._follower and self._follower.active:
            await self._stop_follow_mode()

        # Save command for "reverse" (skip stop/dance/follow)
        if action not in ("stop", "dance", "follow_start", "follow_stop"):
            self._last_car_command = cmd
            logger.info("Saved car command for reverse: {}", action)

        # ── Sequential commands ──
        if action == "sequence":
            steps = cmd.get("steps", [])
            if not steps:
                return
            # Announce the sequence
            parts = []
            for s in steps:
                name = self._action_display_name(s["action"])
                dur = s.get("duration", 1.0)
                parts.append(f"{name} for {dur:.0f} seconds")
            announcement = ", then ".join(parts)
            self._car_command_active = True
            try:
                await self._speak_text(f"Okay! {announcement}!")
                for s in steps:
                    method = getattr(self._car, s["action"], None)
                    if method:
                        await method(speed=s.get("speed", 180), duration=s.get("duration", 1.0))
                await self._car.stop()
            finally:
                self._car_command_active = False
            return

        # ── Single command ──
        speed = cmd.get("speed", 180)
        duration = cmd.get("duration", 0)

        # Fun verbal responses
        responses = {
            "forward": "Here I go!",
            "backward": "Going backwards!",
            "spin_left": "Turning right!",
            "spin_right": "Turning left!",
            "stop": "Stopping!",
            "dance": "Let me dance for you!",
        }

        response = responses.get(action, "Okay!")
        if duration > 1.5:
            response = f"{self._action_display_name(action)} for {duration:.0f} seconds!"

        # Set flag so _speak_text won't start car speaking gestures
        self._car_command_active = True
        try:
            if action == "dance":
                await self._speak_text(response)
                self.state.set_state(BotState.DANCING)
                # Start servo hand dance (runs in background thread)
                if self._servos:
                    self._servos.dance()
                await self._car.dance()
                self.state.set_state(BotState.READY)
            elif action == "stop":
                await self._car.stop()
                await self._speak_text(response)
            else:
                # Speak and move concurrently
                method = getattr(self._car, action, None)
                if method:
                    speak_task = asyncio.create_task(self._speak_text(response))
                    await method(speed=speed, duration=duration)
                    await speak_task
                else:
                    await self._speak_text("I don't know how to do that yet.")
        finally:
            self._car_command_active = False

    def _action_display_name(self, action: str) -> str:
        """Human-friendly name for a car action."""
        names = {
            "forward": "Forward",
            "backward": "Backward",
            "spin_left": "Turning right",
            "spin_right": "Turning left",
            "stop": "Stop",
            "dance": "Dance",
        }
        return names.get(action, action)

    _REVERSE_ACTION = {
        "forward": "backward",
        "backward": "forward",
        "spin_left": "spin_right",
        "spin_right": "spin_left",
        "forward_left": "backward_right",
        "forward_right": "backward_left",
        "backward_left": "forward_right",
        "backward_right": "forward_left",
    }

    def _reverse_command(self, cmd: dict) -> dict:
        """Reverse a car command so the bot retraces its path home.

        - Single command: swap forward↔backward, left↔right, same duration
        - Sequence: reverse the order AND reverse each step
        """
        action = cmd.get("action")

        if action == "sequence":
            steps = cmd.get("steps", [])
            reversed_steps = []
            for s in reversed(steps):
                rev_action = self._REVERSE_ACTION.get(s["action"], s["action"])
                reversed_steps.append({
                    **s,
                    "action": rev_action,
                })
            return {"action": "sequence", "steps": reversed_steps}

        rev_action = self._REVERSE_ACTION.get(action, action)
        return {**cmd, "action": rev_action}

    # ── Follow mode ───────────────────────────────────────────────

    async def _start_follow_mode(self):
        """Initialize and start follow mode (camera + sound tracking)."""
        if self._follower and self._follower.active:
            await self._speak_text("I'm already following you!")
            return

        if not self._car or not self._car.connected:
            await self._speak_text("I can't find my wheels! Is the car turned on?")
            return

        # Lazy-init camera if not already loaded
        if not self._camera:
            try:
                from vision.camera import Camera
                self._camera = Camera()
                logger.info("Camera initialized for follow mode")
            except Exception as e:
                logger.warning("Camera not available for follow mode: {}", e)

        # Create follower (uses lightweight OpenCV HOG detector, no YOLO/PyTorch)
        from modules.follow import FollowMode
        self._follower = FollowMode(
            car=self._car,
            camera=self._camera,
            audio_capture=self._audio_capture,
        )

        self.state.follow_mode = True
        self._car_command_active = True  # Prevent speaking gestures from interfering

        await self._speak_text("Okay, I'll follow you! Say stop following when you're done.")
        await self._follower.start()

    async def _stop_follow_mode(self):
        """Stop follow mode."""
        if self._follower and self._follower.active:
            await self._follower.stop()
            self.state.follow_mode = False
            self._car_command_active = False
            await self._speak_text("Okay, I'll stay here!")
        else:
            await self._speak_text("I'm not following anyone right now.")

    # ── Touch handling ─────────────────────────────────────────────

    def _on_touch_event(self, event):
        """Callback from capacitive touch sensor (runs in touch thread).

        Context-aware: same gesture does different things depending on bot state.

        Tap:        Wake bot / stop speech / next menu item / next flashcard
        Double tap: Volume up / select menu item / previous flashcard
        Triple tap: Volume down
        Long press: Pet reaction (giggle)
        Extra long: Toggle menu on TFT display
        """
        from display.touch import TouchEvent

        self.state.last_touch_event = event.value
        logger.info("Touch: {}", event.value)

        loop = self._loop
        if loop is None:
            return

        # ── Extra long press: toggle menu ──
        if event == TouchEvent.EXTRA_LONG:
            self.state.menu_open = not self.state.menu_open
            if self.state.menu_open:
                self.state.menu_index = 0
            logger.info("Menu {}", "opened" if self.state.menu_open else "closed")
            return

        # ── If menu is open, route to menu navigation ──
        if self.state.menu_open:
            if event == TouchEvent.TAP:
                # Move to next menu item
                menu_count = 7  # Volume, Mode, Mic, Projector, Flashcards, Car, Sleep
                self.state.menu_index = (self.state.menu_index + 1) % menu_count

            elif event == TouchEvent.DOUBLE_TAP:
                # Select current menu item — queue for main loop
                loop.call_soon_threadsafe(
                    asyncio.ensure_future, self._handle_menu_select_safe()
                )

            elif event == TouchEvent.LONG_PRESS:
                # Close menu
                self.state.menu_open = False
                logger.info("Menu closed (long press)")

            return  # Don't process further when menu is open

        # ── Normal mode: context-aware touch actions ──

        if event == TouchEvent.TAP:
            if self.state.bot_state == BotState.SPEAKING:
                # Stop current speech
                self.state.interrupt_event.set()
            elif self.state.bot_state == BotState.READY:
                # Wake the bot — same as saying the wake word
                self.state.wake_event.set()
            elif self._projector and self._projector.mode.value == "flashcard":
                # Next flashcard
                self._projector.next_flashcard()

        elif event == TouchEvent.DOUBLE_TAP:
            if self._projector and self._projector.mode.value == "flashcard":
                # Previous flashcard
                self._projector.prev_flashcard()
            else:
                # Volume up
                new_vol = min(100, self.state.volume + 10)
                self.state.volume = new_vol
                if self._audio_player:
                    self._audio_player.set_volume(new_vol)
                logger.info("Volume up: {}%", new_vol)

        elif event == TouchEvent.TRIPLE_TAP:
            # Volume down
            new_vol = max(0, self.state.volume - 10)
            self.state.volume = new_vol
            if self._audio_player:
                self._audio_player.set_volume(new_vol)
            logger.info("Volume down: {}%", new_vol)

        elif event == TouchEvent.LONG_PRESS:
            # Pet the bot — queue speech for main loop to handle safely
            import random
            reactions = [
                "Hee hee, that tickles!",
                "Haha, be gentle!",
                "That's so nice!",
                "I like when you pet me!",
                "Ooh, you found me!",
                "That feels great!",
                "Hey! That's funny!",
            ]
            self._pending_speech = random.choice(reactions)
            self.state.wake_event.set()  # break main loop out of wake word wait
            # Trigger servo hand wave animation
            if self._servos:
                self._servos.tickle_wave()

    async def _handle_menu_select_safe(self):
        """Handle menu item selection — applies action and queues speech for main loop.

        Does NOT call _speak_text directly (that would crash the audio stream).
        Instead sets _pending_speech and wakes the main loop.
        """
        idx = self.state.menu_index
        speech = None
        # Menu items: 0=Volume, 1=Mode, 2=Mic, 3=Projector, 4=Flashcards, 5=Car, 6=Sleep

        if idx == 0:
            # Volume — cycle: 20 → 40 → 60 → 80 → 100 → 20
            new_vol = (self.state.volume + 20) % 120
            if new_vol == 0:
                new_vol = 20
            self.state.volume = new_vol
            if self._audio_player:
                self._audio_player.set_volume(new_vol)
            speech = f"Volume {new_vol} percent"

        elif idx == 1:
            # Toggle online/offline mode
            if self.state.llm_mode == LLMMode.ONLINE:
                self.config_manager.update_nested("llm", mode="offline")
                self.state.llm_mode = LLMMode.OFFLINE
                speech = "Offline mode"
            else:
                self.config_manager.update_nested("llm", mode="online")
                self.state.llm_mode = LLMMode.ONLINE
                speech = "Online mode"

        elif idx == 2:
            # Toggle microphone
            self.state.mic_enabled = not self.state.mic_enabled
            status = "on" if self.state.mic_enabled else "off"
            speech = f"Microphone {status}"

        elif idx == 3:
            # Toggle projector
            if self._projector:
                from modules.projector import ProjectorMode
                if self._projector.mode == ProjectorMode.OFF:
                    self._projector.set_mode(ProjectorMode.BLANK)
                    self.state.projector_mode = "blank"
                    speech = "Projector on"
                else:
                    self._projector.set_mode(ProjectorMode.OFF)
                    self.state.projector_mode = "off"
                    speech = "Projector off"

        elif idx == 4:
            # Show flashcards on projector
            if self._projector and self._projector.connected:
                from modules.projector import ProjectorMode
                self._projector.set_mode(ProjectorMode.FLASHCARD)
                self.state.projector_mode = "flashcard"
                self.state.menu_open = False
                speech = "Let's learn! Tap to see the next one."

        elif idx == 5:
            # Car — connect or disconnect
            if self.state.car_connected and self._car:
                # Already connected → disconnect
                asyncio.ensure_future(self._disconnect_car_module())
                speech = "Disconnecting the car."
            elif self.state.car_connecting:
                # Already in progress — ignore tap
                speech = "Still connecting, please wait."
            else:
                # Not connected → start scanning and connecting
                asyncio.ensure_future(self._connect_car_module())
                speech = "Looking for the car. Please wait."

        elif idx == 6:
            # Sleep — disable mic
            self.state.mic_enabled = False
            self.state.menu_open = False
            speech = "Good night! Long press to wake me up."

        if speech:
            self._pending_speech = speech
            self.state.wake_event.set()  # break main loop out of wake word wait

    # ── Projector commands ────────────────────────────────────────

    def _parse_projector_command(self, transcript: str) -> dict | None:
        """Parse projector-related voice commands."""
        lower = transcript.lower().strip()

        # Turn on/start projector
        on_triggers = [
            "projector on", "turn on projector", "start projector",
            "show on the wall", "project on wall", "show on wall",
        ]
        for t in on_triggers:
            if t in lower:
                return {"action": "on"}

        # Turn off projector
        off_triggers = [
            "projector off", "turn off projector", "stop projector",
            "stop projecting",
        ]
        for t in off_triggers:
            if t in lower:
                return {"action": "off"}

        # Flashcard mode
        flashcard_triggers = [
            "show flashcard", "flashcard", "show me flashcard",
            "teach me animals", "teach me colors", "teach me shapes",
            "teach me fruits", "show animals", "show colors",
            "show shapes", "show fruits",
        ]
        for t in flashcard_triggers:
            if t in lower:
                # Detect category
                category = "animals"  # default
                for cat in ("animals", "colors", "shapes", "fruits"):
                    if cat in lower:
                        category = cat
                        break
                return {"action": "flashcard", "category": category}

        # Next flashcard
        next_triggers = ["next", "next one", "next card", "show next"]
        for t in next_triggers:
            if t in lower:
                return {"action": "next"}

        # Previous
        prev_triggers = ["previous", "go back", "last one", "show previous"]
        for t in prev_triggers:
            if t in lower:
                return {"action": "previous"}

        # Alphabet mode
        alpha_triggers = [
            "show alphabet", "teach me alphabet", "show letters",
            "teach me letters", "abc", "a b c",
        ]
        for t in alpha_triggers:
            if t in lower:
                return {"action": "alphabet"}

        # Specific letter highlight
        if "show letter" in lower or "letter " in lower:
            # Extract single letter
            for word in lower.split():
                if len(word) == 1 and word.isalpha():
                    return {"action": "highlight_letter", "letter": word}

        # Numbers mode
        number_triggers = [
            "show numbers", "teach me numbers", "teach me counting",
            "count with me", "show counting",
        ]
        for t in number_triggers:
            if t in lower:
                return {"action": "numbers"}

        # Specific number highlight
        if "show number" in lower or "number " in lower:
            for word in lower.split():
                if word.isdigit():
                    num = int(word)
                    if 1 <= num <= 20:
                        return {"action": "highlight_number", "number": num}

        # Imagination / image generation (must be AFTER flashcard/alphabet/number checks)
        # Only triggers if projector is connected and we're in online mode
        imagine_triggers = [
            "show me a picture of", "show me a picture",
            "draw me a picture of", "draw me a picture",
            "draw a picture of", "draw a picture",
            "show me a", "show me an", "show me the",
            "draw a", "draw an", "draw the",
            "imagine a", "imagine an", "imagine the",
            "picture of a", "picture of an", "picture of the",
            "i want to see a", "i want to see an", "i want to see the",
            "can you show me a", "can you show me an",
            "can you draw a", "can you draw an",
            "generate a", "generate an",
            "create a", "create an",
        ]
        for t in imagine_triggers:
            if t in lower:
                # Extract the description after the trigger phrase
                idx = lower.index(t) + len(t)
                prompt = transcript[idx:].strip().rstrip("?.!")
                if prompt:
                    return {"action": "imagine", "prompt": prompt}

        return None

    async def _handle_projector_command(self, cmd: dict):
        """Execute a parsed projector command."""
        from modules.projector import ProjectorMode

        # Lazy init projector if needed
        if not self._projector:
            try:
                from modules.projector import Projector
                self._projector = Projector()
            except Exception as e:
                await self._speak_text("I can't find my projector module.")
                return

        # Start projector if not running
        if not self._projector.connected:
            if self._projector.detect():
                self._projector.start()
                self.state.projector_connected = True
            else:
                await self._speak_text("I don't see a projector connected. Is it plugged in?")
                return

        action = cmd["action"]

        if action == "on":
            self._projector.set_mode(ProjectorMode.BLANK)
            self.state.projector_mode = "blank"
            await self._speak_text("Projector is on! What would you like to see?")

        elif action == "off":
            self._projector.set_mode(ProjectorMode.OFF)
            self.state.projector_mode = "off"
            await self._speak_text("Projector is off!")

        elif action == "flashcard":
            category = cmd.get("category", "animals")
            self._projector.set_flashcard_category(category)
            self._projector.set_mode(ProjectorMode.FLASHCARD)
            self.state.projector_mode = "flashcard"
            card = self._projector.get_current_flashcard()
            if card:
                await self._speak_text(
                    f"Let's learn about {category}! This is a {card[0]}. {card[1]}"
                )

        elif action == "next":
            self._projector.next_flashcard()
            card = self._projector.get_current_flashcard()
            if card:
                await self._speak_text(f"This is a {card[0]}. {card[1]}")

        elif action == "previous":
            self._projector.prev_flashcard()
            card = self._projector.get_current_flashcard()
            if card:
                await self._speak_text(f"This is a {card[0]}. {card[1]}")

        elif action == "alphabet":
            self._projector.set_mode(ProjectorMode.ALPHABET)
            self.state.projector_mode = "alphabet"
            await self._speak_text("Here's the alphabet! Tell me a letter and I'll show you!")

        elif action == "highlight_letter":
            letter = cmd.get("letter", "a").upper()
            self._projector.highlight_letter(letter)
            await self._speak_text(f"{letter}! That's the letter {letter}!")

        elif action == "numbers":
            self._projector.set_mode(ProjectorMode.NUMBERS)
            self.state.projector_mode = "numbers"
            await self._speak_text(
                "Here are the numbers! Tell me a number and I'll highlight it!"
            )

        elif action == "highlight_number":
            num = cmd.get("number", 1)
            self._projector.highlight_number(num)
            await self._speak_text(f"Number {num}!")

        elif action == "imagine":
            await self._handle_imagine(cmd.get("prompt", ""))

    async def _handle_imagine(self, prompt: str):
        """Handle imagination requests — generate image via DALL-E 3 and show on projector."""
        from modules.projector import ProjectorMode

        # Check prerequisites
        if self.state.llm_mode != LLMMode.ONLINE:
            await self._speak_text(
                "I need internet to draw pictures! Ask a grown-up to turn on online mode."
            )
            return

        api_key = self.config_manager.config.api_keys.openai
        if not api_key:
            await self._speak_text(
                "I need an API key to draw pictures. Ask a grown-up to set it up!"
            )
            return

        if not self._projector or not self._projector.connected:
            await self._speak_text(
                "I need a projector to show you pictures! Is it plugged in?"
            )
            return

        # Lazy init image generator
        if self._image_generator is None:
            from vision.image_generator import ImageGenerator
            save_dir = DATA_DIR / "images" / "generated"
            self._image_generator = ImageGenerator(api_key=api_key, save_dir=save_dir)
        else:
            self._image_generator.update_api_key(api_key)

        # Show loading screen and speak
        self._projector.show_loading_message(prompt)
        self.state.projector_mode = "imagine"
        self.state.projector_imagine_prompt = prompt
        await self._speak_text(f"Let me imagine that for you!")

        # Generate the image (this takes 5-10 seconds)
        logger.info("Generating imagination image: '{}'", prompt)
        image = await self._image_generator.generate(prompt)

        if image:
            self._projector.show_generated_image(image, prompt)
            await self._speak_text(f"Here's your {prompt}!")
        else:
            self._projector.set_mode(ProjectorMode.BLANK)
            await self._speak_text(
                "Oops, I couldn't draw that right now. Let me try to describe it instead!"
            )
            # Fall back to LLM verbal description
            await self._stream_response(
                f"Describe in vivid, child-friendly detail what {prompt} would look like. "
                "Use simple words for a 4-year-old."
            )

    def _get_cloud_stt(self):
        """Get or create the cloud STT instance, updating API key and language."""
        api_key = self.config_manager.config.api_keys.openai
        if not api_key:
            return None

        language = self.config_manager.config.child.language
        if self._cloud_stt is None:
            from audio.stt import CloudSpeechToText
            self._cloud_stt = CloudSpeechToText(api_key=api_key, language=language)
        else:
            self._cloud_stt.update_api_key(api_key)
            self._cloud_stt.language = language

        return self._cloud_stt

    async def _stream_response_with_messages(self, messages: list[dict], provider=None):
        """Like _stream_response but with pre-built messages (for vision).

        Used when we need custom message format (e.g., image content).
        Accepts optional provider override (e.g., for hybrid vision).
        """
        t_start = time.monotonic()
        self.state.set_state(BotState.PROCESSING)
        self.state.current_response = ""
        self.state.interrupt_event.clear()

        if provider is None:
            provider = await self._llm_router.get_provider()
        self._sentence_buffer.reset()
        self._audio_capture.pause()

        interrupted = False
        first_audio_logged = False

        tts_queue: asyncio.Queue = asyncio.Queue()
        play_queue: asyncio.Queue = asyncio.Queue()
        speaking_done = asyncio.Event()

        async def play_audio_chunks():
            while True:
                chunk = await play_queue.get()
                if chunk is None:
                    break
                if self.state.interrupt_event.is_set():
                    break
                self.state.set_state(BotState.SPEAKING)
                await self._audio_player.play(chunk)
                if self.state.interrupt_event.is_set():
                    break
            speaking_done.set()

        async def tts_worker():
            nonlocal first_audio_logged
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    break
                if self.state.interrupt_event.is_set():
                    break
                audio_data = await self._tts.synthesize(sentence)
                if not first_audio_logged:
                    logger.info("[TIMING] First audio ready: {:.1f}s",
                                time.monotonic() - t_start)
                    first_audio_logged = True
                await play_queue.put(audio_data)
            await play_queue.put(None)

        player_task = asyncio.create_task(play_audio_chunks())
        tts_task = asyncio.create_task(tts_worker())

        response_text = []
        first_token_time = None
        async for token in provider.stream(messages):
            if self.state.interrupt_event.is_set():
                interrupted = True
                break
            if first_token_time is None:
                first_token_time = time.monotonic()
                logger.info("[TIMING] LLM first token: {:.1f}s",
                            first_token_time - t_start)
            sentence = self._sentence_buffer.feed(token)
            if sentence:
                safe = self._safety_filter.check_output(sentence)
                if safe.blocked:
                    sentence = safe.redirect_response
                response_text.append(sentence)
                self.state.current_response = " ".join(response_text)
                await tts_queue.put(sentence)

        if not interrupted:
            remaining = self._sentence_buffer.flush()
            if remaining:
                safe = self._safety_filter.check_output(remaining)
                text = safe.redirect_response if safe.blocked else remaining
                response_text.append(text)
                self.state.current_response = " ".join(response_text)
                await tts_queue.put(text)

        await tts_queue.put(None)
        await speaking_done.wait()
        player_task.cancel()
        tts_task.cancel()

        if interrupted:
            await self._audio_player.stop()

        self.state.interrupt_event.clear()
        logger.info("[TIMING] Total response: {:.1f}s", time.monotonic() - t_start)

        await asyncio.sleep(1.0)
        self._audio_capture.resume()

    async def _speak_text(self, text: str):
        """Synthesize and speak a single text string. Supports interruption."""
        self._audio_capture.pause()
        self.state.set_state(BotState.SPEAKING)
        self.state.interrupt_event.clear()

        self._servos.start_speaking_gestures()
        if self._car and self._car.connected and not self._car_command_active:
            self._car.start_speaking_gestures()

        audio_data = await self._tts.synthesize(text)

        if not self.state.interrupt_event.is_set():
            await self._audio_player.play(audio_data)

        self._servos.stop_speaking_gestures()
        if self._car and self._car.connected and not self._car_command_active:
            self._car.stop_speaking_gestures()

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
        if self._display:
            self._display.stop()
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
