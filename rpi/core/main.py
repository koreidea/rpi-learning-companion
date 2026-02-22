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
        self._cloud_stt = None
        self._tts = None
        self._llm_router = None
        self._audio_capture = None
        self._audio_player = None
        self._safety_filter = None
        self._camera = None
        self._object_detector = None
        self._api_server = None

        # Conversation history: rolling buffer of user/assistant message pairs.
        # Gives the LLM context of prior turns within a single conversation session.
        # Cleared when follow-up mode times out or a new wake word starts.
        self._conversation_history: list[dict] = []

        # Max history turns (user+assistant pairs).
        # Online providers have huge context windows, offline is limited.
        self._max_history_online = 10  # 10 pairs = 20 messages
        self._max_history_offline = 4  # 4 pairs = 8 messages

    async def start(self):
        """Boot sequence: load config, check consent, load models, start listening."""
        logger.info("=== RPi Learning Companion starting ===")

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
        logger.info("All components loaded.")

    async def _interaction_loop(self):
        """Main loop: wake word → capture → STT → LLM stream → TTS stream → speak.

        After each response, opens a follow-up window (5s) so the child
        can keep talking without repeating the wake word.
        """
        follow_up = False  # True when in follow-up listening mode

        while self.state.is_running:
            try:
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

                    # Step 1: Wait for wake word
                    audio_stream = self._audio_capture.stream()
                    await self._wake_word.listen(audio_stream)
                    logger.info("Wake word detected!")

                # Step 2: Play acknowledgment sound
                self.state.set_state(BotState.LISTENING)
                self.state.interaction_start_time = time.monotonic()
                if not follow_up:
                    asyncio.create_task(self._play_sound("ding"))

                # Step 3: Capture speech with VAD
                # Give slightly more time in follow-up mode
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

                # Audio energy check: reject low-energy captures (noise/silence)
                # that VAD may have falsely triggered on. Without this, Whisper
                # hallucinates text from silence ("Thank you", "Thanks for watching").
                import numpy as np
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                if rms < 300:
                    logger.debug("Audio too quiet (RMS={:.0f}), skipping.", rms)
                    if follow_up:
                        follow_up = False
                    continue

                t_vad = time.monotonic()
                logger.info("[TIMING] VAD capture: {:.1f}s", t_vad - t0)

                # Step 4: Speech-to-text (cloud or local)
                self.state.set_state(BotState.PROCESSING)
                transcript = ""
                if self.state.cloud_stt and self.state.llm_mode == LLMMode.ONLINE:
                    # Cloud STT — faster but sends audio off-device
                    cloud = self._get_cloud_stt()
                    if cloud:
                        transcript = await cloud.transcribe(audio_data)
                    # Fall back to local if cloud returned empty
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

                # Step 5: Safety check on input
                safe_input = self._safety_filter.check_input(transcript)
                if safe_input.blocked:
                    await self._speak_text(safe_input.redirect_response)
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
