import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../activities/activity_base.dart';
import '../../activities/activity_registry.dart';
import '../../activities/activity_session.dart';
import '../../audio/audio_capture.dart';
import '../../audio/cloud_stt.dart';
import '../../audio/sentence_buffer.dart';
import '../../audio/song_player.dart';
import '../../audio/tts_service.dart';
import '../../audio/wake_word_detector.dart';
import '../../bluetooth/bluetooth_manager.dart';
import '../../bluetooth/speaking_gestures.dart';
import '../../vision/camera_manager.dart';
import '../../vision/follow_mode.dart';
import '../../vision/person_detector.dart';
import '../../vision/vision_describer.dart';
import '../config/config_manager.dart';
import '../llm/llm_router.dart';
import '../llm/safety_filter.dart';
import '../state/bot_state.dart';
import '../state/shared_state.dart';
import 'command_parser.dart';

/// Main orchestrator: manages the voice interaction pipeline.
/// Port of rpi/core/main.py for Flutter.
///
/// Pipeline: Tap → Record → STT → Command check → LLM (streaming) → Sentence buffer → TTS → Speak
class Orchestrator {
  final StateNotifierProvider<SharedStateNotifier, SharedState> stateProvider;
  final Ref ref;

  // Components
  late final ConfigManager config;
  late final AudioCaptureService audioCapture;
  late final CloudSTT cloudStt;
  late final TtsService tts;
  late final LlmRouter llmRouter;
  late final SentenceBuffer sentenceBuffer;
  late final CommandParser commandParser;
  late final SongPlayer songPlayer;

  // Vision
  late final CameraManager cameraManager;
  late final VisionDescriber visionDescriber;

  // Follow mode (person-following via camera + car)
  FollowMode? _followMode;
  final PersonDetector _personDetector = PersonDetector();

  // Bluetooth (shared with UI)
  BluetoothManager? btManager;

  // Speaking gestures (car wiggles while bot talks)
  SpeakingGestures? _gestures;

  // Activities
  ActivityRegistry? _activityRegistry;
  final ActivitySession _activitySession = ActivitySession();
  Activity? _currentActivity;

  // Conversation history: rolling buffer of user/assistant pairs
  final List<Map<String, String>> _history = [];
  static const _maxHistoryPairs = 10;

  bool _initialized = false;
  bool _isProcessing = false;
  bool _carCommandActive = false; // True during car command execution
  String _lastTtsLang = 'en'; // Track current TTS language

  // Continuous conversation mode
  bool _continuousMode = false;
  // _continuousLoopRunning removed -- unified into _interactionLoop
  static const _followUpDuration = Duration(seconds: 5);

  // Wake word detection (always-on via Android SpeechRecognizer)
  WakeWordDetector? _wakeWordDetector;
  bool _wakeWordActive = false;

  // Wake word phrase (checked in STT transcript during idle listening)
  static const _wakePhrase = 'hey buddy';

  Orchestrator({required this.stateProvider, required this.ref});

  SharedStateNotifier get _notifier => ref.read(stateProvider.notifier);

  /// Initialize all components.
  Future<void> init() async {
    if (_initialized) return;

    debugPrint('[Orchestrator] Initializing...');

    config = ConfigManager();
    await config.init();

    audioCapture = AudioCaptureService();
    tts = TtsService();
    sentenceBuffer = SentenceBuffer();
    commandParser = CommandParser();
    songPlayer = SongPlayer();

    // Init TTS with configured language
    await tts.init(language: config.language);

    // Setup LLM router with saved API keys
    llmRouter = LlmRouter(
      activeProvider: config.provider,
      openaiKey: config.openaiKey,
      geminiKey: config.geminiKey,
      claudeKey: config.claudeKey,
    );

    // Setup cloud STT
    cloudStt = CloudSTT(apiKey: config.openaiKey);

    // Setup vision (camera init is lazy — happens on first vision command)
    cameraManager = CameraManager();
    visionDescriber = VisionDescriber();

    // Load persistent conversation history
    _notifier.loadConversationHistory(config);

    // Restore continuous mode setting
    _continuousMode = config.continuousMode;
    _notifier.setContinuousMode(_continuousMode);

    // Initialize activity session (loads stats from SharedPreferences)
    await _activitySession.init();

    _initialized = true;
    debugPrint('[Orchestrator] Ready');

    // Start wake word detection (always-on, free on-device recognition)
    _startWakeWordDetection();
  }

  /// Lazily build the activity registry, creating it with the current
  /// car reference and other dependencies so activities can use them.
  ActivityRegistry _getActivityRegistry() {
    _activityRegistry ??= ActivityRegistry(
      car: btManager?.car,
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
      cameraManager: cameraManager,
      visionDescriber: visionDescriber,
      openaiApiKey: config.openaiKey,
    );
    return _activityRegistry!;
  }

  /// Start always-on wake word detection.
  /// Uses Android's built-in SpeechRecognizer (free, on-device).
  void _startWakeWordDetection() {
    _wakeWordDetector?.dispose();
    _wakeWordDetector = WakeWordDetector(
      onWakeWord: _onWakeWordDetected,
    );
    _wakeWordDetector!.start().then((started) {
      _wakeWordActive = started;
      debugPrint('[Orchestrator] Wake word detection: ${started ? "ON" : "FAILED"}');
    });
  }

  /// Called when the wake word "hai robo" is detected.
  void _onWakeWordDetected() {
    debugPrint('[Orchestrator] Wake word "hai robo" detected!');

    if (_isProcessing) {
      debugPrint('[Orchestrator] Already processing, ignoring wake word');
      Future.delayed(const Duration(seconds: 1), () {
        if (!_isProcessing) _resumeWakeWord();
      });
      return;
    }

    // Acknowledge the wake word, then enter interaction loop
    _notifier.setBotState(BotState.speaking);
    tts.speak("I'm here!").then((_) {
      // Use the unified interaction loop — keeps listening while
      // activity is active or child keeps talking, then returns to wake word
      _pauseWakeWord().then((_) {
        _interactionLoop().then((_) {
          _resumeWakeWord();
        });
      });
    });
  }

  /// Resume wake word detection after a voice interaction.
  void _resumeWakeWord() {
    if (_wakeWordDetector != null && _wakeWordActive) {
      _wakeWordDetector!.resume();
    }
  }

  /// Pause wake word detection (e.g., during recording to avoid mic conflict).
  Future<void> _pauseWakeWord() async {
    if (_wakeWordDetector != null && _wakeWordActive) {
      await _wakeWordDetector!.stop();
    }
  }

  /// Whether continuous mode is currently enabled.
  bool get continuousMode => _continuousMode;

  /// Start continuous conversation mode.
  /// After each response, the bot automatically listens again.
  Future<void> startContinuousMode() async {
    if (!_initialized) await init();
    _continuousMode = true;
    _notifier.setContinuousMode(true);
    await config.setContinuousMode(true);
    debugPrint('[Orchestrator] Continuous mode ON');
  }

  /// Stop continuous conversation mode.
  Future<void> stopContinuousMode() async {
    _continuousMode = false;
    _notifier.setContinuousMode(false);
    await config.setContinuousMode(false);
    debugPrint('[Orchestrator] Continuous mode OFF');
  }

  /// Handle tap-to-talk: record speech -> STT -> command check -> LLM -> TTS.
  /// After interaction, if an activity is active it auto-listens again.
  /// Otherwise goes back to wake word detection.
  Future<void> handleTapToTalk() async {
    if (_isProcessing) {
      debugPrint('[Orchestrator] Already processing, ignoring tap');
      return;
    }
    if (!_initialized) await init();

    // Pause wake word during interaction
    await _pauseWakeWord();

    // Interaction loop: keeps going while an activity is active
    // or while in continuous mode
    await _interactionLoop();

    // Resume wake word after all interactions end
    _resumeWakeWord();
  }

  /// Main interaction loop. After each interaction:
  /// - If activity is active -> auto-listen again (no wake word needed)
  /// - If speech was detected -> follow-up window (5s) for more speech
  /// - If no more speech -> return to wake word detection
  Future<void> _interactionLoop() async {
    while (true) {
      final hadSpeech = await _singleInteraction();

      // Check if an activity is still running -- keep listening automatically
      if (_currentActivity != null && _currentActivity!.isActive) {
        debugPrint('[Orchestrator] Activity active, auto-listening...');
        await Future.delayed(const Duration(milliseconds: 300));
        continue;
      }

      // If speech was detected, do follow-up window to see if child wants
      // to say more (regardless of continuous mode)
      if (hadSpeech) {
        debugPrint('[Orchestrator] Follow-up window (${_followUpDuration.inSeconds}s)');
        final followUpSpeech = await _singleInteraction(
          maxWaitForSpeech: _followUpDuration,
          isFollowUp: true,
        );

        // If activity started during follow-up, loop back
        if (_currentActivity != null && _currentActivity!.isActive) {
          continue;
        }

        if (followUpSpeech) {
          continue; // Child spoke again, keep going
        }
        // No speech in follow-up -- conversation over for now
        debugPrint('[Orchestrator] No follow-up speech, returning to wake word');
      }

      // No more speech -- exit loop and return to wake word detection
      break;
    }
  }

  /// Perform a single listen -> process -> respond cycle.
  /// Returns true if speech was detected and processed, false otherwise.
  Future<bool> _singleInteraction({
    Duration maxWaitForSpeech = const Duration(seconds: 5),
    bool isFollowUp = false,
  }) async {
    // Check if we have an API key
    if (!config.hasApiKey) {
      debugPrint('[Orchestrator] No API key configured');
      _notifier.setBotState(BotState.error);
      await tts.speak('Please set up an API key in settings first.');
      _notifier.setBotState(BotState.ready);
      return false;
    }

    _isProcessing = true;

    // Pause wake word detection to free the mic
    await _pauseWakeWord();

    try {
      // Step 1: Listen
      _notifier.setBotState(BotState.listening);
      debugPrint('[Orchestrator] Recording${isFollowUp ? " (follow-up)" : ""}...');

      final hasPermission = await audioCapture.hasPermission();
      if (!hasPermission) {
        debugPrint('[Orchestrator] No mic permission');
        await tts.speak('I need microphone permission to hear you.');
        _notifier.setBotState(BotState.ready);
        return false;
      }

      final audioData = await audioCapture.recordWithVAD(
        maxDuration: const Duration(seconds: 15),
        silenceAfterSpeech: const Duration(milliseconds: 1500),
        maxWaitForSpeech: maxWaitForSpeech,
      );

      if (audioData == null || audioData.isEmpty) {
        debugPrint('[Orchestrator] No audio captured');
        _notifier.setBotState(BotState.ready);
        return false;
      }

      debugPrint('[Orchestrator] Captured ${audioData.length} bytes');

      // Step 2: Speech-to-Text
      _notifier.setBotState(BotState.processing);
      final transcript = await cloudStt.transcribe(audioData);

      if (transcript.isEmpty || transcript.length < 2) {
        debugPrint('[Orchestrator] Empty transcript');
        _notifier.setBotState(BotState.ready);
        return false;
      }

      debugPrint('[Orchestrator] Transcript: "$transcript"');
      _notifier.setTranscript(transcript);

      // Check for stop commands that should also stop continuous mode
      if (_continuousMode && _isStopContinuousPhrase(transcript)) {
        debugPrint('[Orchestrator] Stop phrase detected, exiting continuous mode');
        await stopContinuousMode();
        _notifier.setBotState(BotState.speaking);
        await tts.speak('Okay, I will wait for your tap!');
        _notifier.setBotState(BotState.ready);
        return true;
      }

      // Step 3a: If an activity is running, route to it first
      if (_currentActivity != null && _currentActivity!.isActive) {
        // Still check for stop commands during activities
        final stopCheck = commandParser.parse(
          transcript,
          currentVolume: ref.read(stateProvider).volume,
        );
        if (stopCheck != null && stopCheck.type == CommandType.stop) {
          await _handleStopCommand();
        } else {
          await _routeToActivity(transcript);
        }
      } else {
        // Step 3b: Check for voice commands BEFORE sending to LLM
        final command = commandParser.parse(
          transcript,
          currentVolume: ref.read(stateProvider).volume,
        );

        if (command != null) {
          await _handleCommand(command, transcript);
        } else {
          // Step 3c: Check activity registry for voice-trigger match
          // (specific activity triggers like "coding game", "mystery game"
          // are defined on each Activity's voiceTriggers, not in the
          // command parser)
          final registry = _getActivityRegistry();
          final matchedActivity = registry.findByVoiceTrigger(transcript);
          if (matchedActivity != null) {
            _currentActivity = matchedActivity;
            debugPrint('[Orchestrator] Activity matched by voice trigger: ${matchedActivity.id}');
            _notifier.setBotState(BotState.speaking);
            _startSpeakingGestures();
            try {
              final intro = await matchedActivity.start();
              await tts.speak(intro);
            } finally {
              _stopSpeakingGestures();
            }
          } else {
            // Step 4: No command, no activity match -- stream LLM response
            await _streamResponse(transcript);
          }
        }
      }

      // Step 5: Back to ready (caller decides next action)
      _notifier.setBotState(BotState.ready);
      return true;
    } catch (e) {
      debugPrint('[Orchestrator] Error: $e');
      _notifier.setBotState(BotState.error);
      await Future.delayed(const Duration(seconds: 2));
      _notifier.setBotState(BotState.ready);
      return false;
    } finally {
      _isProcessing = false;
    }
  }

  /// Listen passively for the wake word ("hai robo") via STT.
  /// Returns true if wake word was detected, false if mode stopped.
  Future<bool> _idleListenForWakeWord() async {
    _notifier.setBotState(BotState.ready);
    debugPrint('[Orchestrator] Idle listening for wake word...');

    // Resume on-device wake word detector
    _resumeWakeWord();

    const listenWindow = Duration(seconds: 8);
    const maxIdleCycles = 50; // ~400 seconds max idle

    for (int cycle = 0; cycle < maxIdleCycles; cycle++) {
      if (!_continuousMode) return false;
      // Check if wake word detector already triggered (via _onWakeWordDetected)
      // If so, _isProcessing would be set. We rely on the detector callback.

      final hasPermission = await audioCapture.hasPermission();
      if (!hasPermission) return false;

      final audioData = await audioCapture.recordWithVAD(
        maxDuration: const Duration(seconds: 10),
        silenceAfterSpeech: const Duration(milliseconds: 1000),
        maxWaitForSpeech: listenWindow,
      );

      if (!_continuousMode) return false;

      if (audioData == null || audioData.isEmpty) continue;

      try {
        final transcript = await cloudStt.transcribe(audioData);
        if (transcript.isEmpty) continue;

        debugPrint('[Orchestrator] Idle heard: "$transcript"');

        final lower = transcript.toLowerCase().trim();
        if (_containsWakeWord(lower)) {
          debugPrint('[Orchestrator] Wake word detected via STT!');
          await _pauseWakeWord();
          _notifier.setBotState(BotState.speaking);
          await tts.speak("I'm here!");
          return true;
        }
      } catch (e) {
        debugPrint('[Orchestrator] Wake word STT error: $e');
      }
    }

    return false;
  }

  /// Check if text contains the wake word "hai robo" or common variants.
  bool _containsWakeWord(String lower) {
    const wakePhrases = [
      'hai robo',
      'hi robo',
      'hey robo',
      'hai robot',
      'hi robot',
      'hey robot',
      'hairobo',
    ];
    for (final phrase in wakePhrases) {
      if (lower.contains(phrase)) return true;
    }
    return false;
  }

  /// Check if a transcript is a phrase to stop continuous mode.
  bool _isStopContinuousPhrase(String transcript) {
    final lower = transcript.toLowerCase().trim();
    const stopPhrases = [
      'stop listening',
      'stop talking',
      'go to sleep',
      'goodbye',
      'bye bye',
      'good night',
      'be quiet',
      'shut up',
      'stop continuous',
    ];
    for (final phrase in stopPhrases) {
      if (lower.contains(phrase)) return true;
    }
    return false;
  }

  // ── Command handling ──

  Future<void> _handleCommand(ParsedCommand cmd, String transcript) async {
    switch (cmd.type) {
      case CommandType.car:
        await _handleCarCommand(cmd);
      case CommandType.volume:
        await _handleVolumeCommand(cmd.volumeLevel!);
      case CommandType.song:
        await _handleSongCommand(cmd.songName!);
      case CommandType.follow:
        await _handleFollowCommand(cmd);
      case CommandType.stop:
        await _handleStopCommand();
      case CommandType.vision:
        await _handleVisionCommand(transcript);
      case CommandType.activity:
        await _handleActivityCommand(cmd, transcript);
      case CommandType.none:
        break;
    }
  }

  Future<void> _handleCarCommand(ParsedCommand cmd) async {
    final car = btManager?.car;
    if (car == null || !car.connected) {
      _notifier.setBotState(BotState.speaking);
      await tts.speak("I can't find my wheels! Is the car turned on?");
      return;
    }

    // Fun verbal responses
    const responses = {
      'forward': "Here I go!",
      'backward': "Going backwards!",
      'spinLeft': "Turning left!",
      'spinRight': "Turning right!",
      'stop': "Stopping!",
      'dance': "Let me dance for you!",
    };

    final action = cmd.action!;
    final response = responses[action] ?? "Okay!";
    final speed = cmd.speed ?? 200;
    final duration = cmd.duration ?? 1.0;

    _carCommandActive = true;
    _notifier.setBotState(BotState.speaking);

    try {
      if (action == 'dance') {
        await tts.speak(response);
        await car.dance();
      } else if (action == 'stop') {
        await car.stop();
        await tts.speak(response);
      } else {
        // Speak and move
        final speakFuture = tts.speak(response);

        switch (action) {
          case 'forward':
            await car.forward(speed: speed, duration: Duration(milliseconds: (duration * 1000).round()));
          case 'backward':
            await car.backward(speed: speed, duration: Duration(milliseconds: (duration * 1000).round()));
          case 'spinLeft':
            await car.spinLeft(speed: speed, duration: Duration(milliseconds: (duration * 1000).round()));
          case 'spinRight':
            await car.spinRight(speed: speed, duration: Duration(milliseconds: (duration * 1000).round()));
        }

        await speakFuture;
      }
    } finally {
      _carCommandActive = false;
    }
  }

  Future<void> _handleVolumeCommand(int level) async {
    final clamped = level.clamp(0, 100);
    _notifier.setVolume(clamped);
    debugPrint('[Orchestrator] Volume → $clamped%');

    // Actually apply volume to TTS engine
    await tts.setVolume(clamped);

    _notifier.setBotState(BotState.speaking);
    if (clamped == 0) {
      // Temporarily raise volume to confirm, then mute
      await tts.setVolume(50);
      await tts.speak("Okay, I'm muted now. Say unmute to hear me again!");
      await tts.setVolume(0);
    } else {
      await tts.speak("Okay! Volume is now at $clamped percent.");
    }
  }

  Future<void> _handleSongCommand(String songName) async {
    final displayName = SongPlayer.displayName(songName);
    debugPrint('[Orchestrator] Song request: $displayName');

    _notifier.setBotState(BotState.speaking);
    await tts.speak("Okay! Here's $displayName!");

    await songPlayer.play(songName);
  }

  Future<void> _handleStopCommand() async {
    // Stop follow mode if active
    if (_followMode != null && _followMode!.isActive) {
      await _followMode!.stop();
      _followMode = null;
      _notifier.setFollowMode(false);
    }
    // Stop any playing song
    if (songPlayer.isPlaying) {
      await songPlayer.stop();
    }
    // Stop car if moving
    if (btManager?.car.connected == true) {
      await btManager!.car.stop();
    }
    // Stop TTS
    await tts.stop();

    // End current activity if one is running
    if (_currentActivity != null && _currentActivity!.isActive) {
      final goodbye = await _currentActivity!.end();
      _currentActivity = null;
      _notifier.setBotState(BotState.speaking);
      await tts.speak(goodbye);
      return;
    }

    _notifier.setBotState(BotState.speaking);
    await tts.speak("Okay, stopped!");
  }

  // ── Activities ──

  Future<void> _handleActivityCommand(
      ParsedCommand cmd, String transcript) async {
    final registry = _getActivityRegistry();

    Activity? activity;
    if (cmd.activityId != null) {
      activity = registry.getById(cmd.activityId!);
    } else {
      // Try to match from the full transcript
      activity = registry.findByVoiceTrigger(transcript);
    }

    if (activity == null) {
      // Show activity menu — list a sample of available activities
      _notifier.setBotState(BotState.speaking);
      final all = registry.getAll();
      if (all.length <= 5) {
        final names = all.map((a) => a.name).join(', ');
        await tts.speak("I know these games: $names. Which one do you want to play?");
      } else {
        // Pick a few representative names
        final sample = (all.toList()..shuffle()).take(4).map((a) => a.name).toList();
        await tts.speak(
          "I know lots of games! For example: ${sample.join(', ')}, "
          "and more! Just tell me what you want to play!",
        );
      }
      return;
    }

    // Start the activity
    _currentActivity = activity;
    debugPrint('[Orchestrator] Starting activity: ${activity.id}');
    _notifier.setBotState(BotState.speaking);

    _startSpeakingGestures();
    try {
      final intro = await activity.start();
      await tts.speak(intro);
    } finally {
      _stopSpeakingGestures();
    }
  }

  /// Start a specific activity by its ID (called from UI).
  Future<void> startActivityById(String activityId) async {
    if (_isProcessing) return;
    if (!_initialized) await init();
    _isProcessing = true;

    try {
      final registry = _getActivityRegistry();
      final activity = registry.getById(activityId);
      if (activity == null) {
        debugPrint('[Orchestrator] Activity not found: $activityId');
        _isProcessing = false;
        return;
      }

      _currentActivity = activity;
      debugPrint('[Orchestrator] Starting activity from UI: ${activity.id}');
      _notifier.setBotState(BotState.speaking);

      _startSpeakingGestures();
      try {
        final intro = await activity.start();
        await tts.speak(intro);
      } finally {
        _stopSpeakingGestures();
      }

      // After intro, go to continuous listening for this activity
      _notifier.setBotState(BotState.ready);
      // Start wake word / continuous mode so child can respond
      _resumeWakeWord();
    } catch (e) {
      debugPrint('[Orchestrator] Error starting activity: $e');
    } finally {
      _isProcessing = false;
    }
  }

  /// Route a transcript to the currently active activity.
  /// Returns true if the activity handled it, false if the activity ended.
  Future<bool> _routeToActivity(String transcript) async {
    if (_currentActivity == null || !_currentActivity!.isActive) {
      _currentActivity = null;
      return false;
    }

    _notifier.setBotState(BotState.processing);
    final response = await _currentActivity!.processResponse(transcript);

    if (response == null) {
      // Activity is done
      final activityId = _currentActivity!.id;
      // Simple score extraction: use the activity's progress summary length
      // as a proxy, or parse score from the activity.
      _currentActivity = null;
      debugPrint('[Orchestrator] Activity $activityId completed');
      return false;
    }

    // Speak the activity response
    _notifier.setResponse(response);
    _notifier.setBotState(BotState.speaking);

    _startSpeakingGestures();
    try {
      await tts.speak(response);
    } finally {
      _stopSpeakingGestures();
    }

    // Check if the activity ended after processing
    if (_currentActivity != null && !_currentActivity!.isActive) {
      final activityId = _currentActivity!.id;
      // Record completion
      await _activitySession.recordCompletion(activityId, 1);
      _currentActivity = null;
      debugPrint('[Orchestrator] Activity $activityId completed after response');
    }

    return true;
  }

  // ── Follow mode ──

  Future<void> _handleFollowCommand(ParsedCommand cmd) async {
    final action = cmd.action ?? 'start';

    if (action == 'stop') {
      await _stopFollowMode();
    } else {
      await _startFollowMode();
    }
  }

  Future<void> _startFollowMode() async {
    final car = btManager?.car;
    if (car == null || !car.connected) {
      _notifier.setBotState(BotState.speaking);
      await tts.speak("I can't follow you without my wheels. Is the car turned on?");
      return;
    }

    // If already following, just acknowledge
    if (_followMode != null && _followMode!.isActive) {
      _notifier.setBotState(BotState.speaking);
      await tts.speak("I'm already following you!");
      return;
    }

    _notifier.setBotState(BotState.speaking);
    await tts.speak("Okay, I'll follow you!");

    // Lazy-init camera
    final cameraReady = await cameraManager.init();
    if (!cameraReady) {
      await tts.speak("Sorry, I cannot access the camera right now.");
      return;
    }

    // Create follow mode instance
    _followMode = FollowMode(
      car: car,
      camera: cameraManager,
      detector: _personDetector,
    );
    _followMode!.onStateChanged = (state) {
      debugPrint('[Orchestrator] Follow mode state: $state');
    };

    final started = await _followMode!.start();
    if (started) {
      _notifier.setFollowMode(true);
      debugPrint('[Orchestrator] Follow mode started');
    } else {
      _notifier.setBotState(BotState.speaking);
      await tts.speak("Hmm, I had trouble starting follow mode. Let me try again later.");
      _followMode = null;
    }
  }

  Future<void> _stopFollowMode() async {
    if (_followMode == null || !_followMode!.isActive) {
      _notifier.setBotState(BotState.speaking);
      await tts.speak("I'm not following anyone right now.");
      return;
    }

    await _followMode!.stop();
    _followMode = null;
    _notifier.setFollowMode(false);

    _notifier.setBotState(BotState.speaking);
    await tts.speak("Okay, I'll stay here!");
    debugPrint('[Orchestrator] Follow mode stopped');
  }

  // ── Vision command handling ──

  Future<void> _handleVisionCommand(String transcript) async {
    debugPrint('[Orchestrator] Vision command detected');
    _notifier.setBotState(BotState.processing);

    // Lazy-init camera on first vision request
    final cameraReady = await cameraManager.init();
    if (!cameraReady) {
      debugPrint('[Orchestrator] Camera not available');
      _notifier.setBotState(BotState.speaking);
      await tts.speak('Sorry, I cannot access the camera right now.');
      return;
    }

    // Capture a frame
    final frame = await cameraManager.captureFrame();
    if (frame == null || frame.isEmpty) {
      debugPrint('[Orchestrator] Failed to capture frame');
      _notifier.setBotState(BotState.speaking);
      await tts.speak('Hmm, I could not take a picture. Let me try again!');
      return;
    }

    debugPrint('[Orchestrator] Captured ${frame.length} bytes, sending to vision API...');

    // Send to GPT-4o-mini vision
    final apiKey = config.openaiKey;
    final language = _detectLanguage(transcript);
    final description = await visionDescriber.describe(
      imageBytes: frame,
      apiKey: apiKey,
      userPrompt: transcript,
      language: language,
    );

    debugPrint('[Orchestrator] Vision description: "${description.substring(0, description.length.clamp(0, 80))}"');

    // Speak the description
    _notifier.setResponse(description);
    _notifier.setBotState(BotState.speaking);

    _startSpeakingGestures();
    try {
      await tts.speak(description);
    } finally {
      _stopSpeakingGestures();
    }

    // Record in history
    _appendToHistory(transcript, description);
    _notifier.addConversation(
      userText: transcript,
      botResponse: description,
      language: language,
    );
    // Persist conversation history
    await _notifier.saveConversationHistory(config);
  }

  // ── LLM streaming ──

  /// Stream LLM tokens → buffer sentences → TTS each sentence.
  Future<void> _streamResponse(String transcript) async {
    debugPrint('[Orchestrator] Streaming response...');
    _notifier.setBotState(BotState.processing);
    _notifier.setResponse('');

    // Refresh keys from config
    llmRouter.openaiKey = config.openaiKey;
    llmRouter.geminiKey = config.geminiKey;
    llmRouter.claudeKey = config.claudeKey;
    llmRouter.activeProvider = config.provider;

    final provider = llmRouter.getProvider();
    final messages = llmRouter.buildMessages(
      transcript,
      history: _history.isNotEmpty ? _history : null,
    );
    sentenceBuffer.reset();

    final responseText = <String>[];
    bool firstSentenceSpoken = false;

    // Start speaking gestures (car wiggles while talking)
    _startSpeakingGestures();

    try {
      await for (final token in provider.stream(messages)) {
        final sentence = sentenceBuffer.feed(token);
        if (sentence != null) {
          // Apply safety filter before speaking
          final filtered = SafetyFilter.filter(sentence);
          responseText.add(filtered);
          _notifier.setResponse(responseText.join(' '));

          if (!firstSentenceSpoken) {
            _notifier.setBotState(BotState.speaking);
            firstSentenceSpoken = true;
          }

          await tts.speak(filtered);

          // If the safety filter replaced the sentence, stop streaming
          // and use the safe replacement for the whole response.
          if (filtered != sentence) {
            debugPrint('[Orchestrator] Safety filter triggered, stopping stream');
            break;
          }
        }
      }

      // Flush remaining text
      final remaining = sentenceBuffer.flush();
      if (remaining != null) {
        final filtered = SafetyFilter.filter(remaining);
        responseText.add(filtered);
        _notifier.setResponse(responseText.join(' '));
        if (!firstSentenceSpoken) {
          _notifier.setBotState(BotState.speaking);
        }
        await tts.speak(filtered);
      }

      // Record in conversation history
      final fullResponse = responseText.join(' ');
      if (fullResponse.isNotEmpty) {
        _appendToHistory(transcript, fullResponse);
        // Also add to shared state for dashboard
        final detectedLang = _detectLanguage(transcript);
        _notifier.addConversation(
          userText: transcript,
          botResponse: fullResponse,
          language: detectedLang,
        );
        // Persist conversation history
        await _notifier.saveConversationHistory(config);
      }

      debugPrint('[Orchestrator] Response complete: "${fullResponse.substring(0, fullResponse.length.clamp(0, 80))}"');
    } catch (e) {
      debugPrint('[Orchestrator] Stream error: $e');
      if (responseText.isEmpty) {
        _notifier.setBotState(BotState.speaking);
        await tts.speak("Oops, I had a little trouble. Can you try again?");
      }
    } finally {
      _stopSpeakingGestures();
    }
  }

  // ── Speaking gestures ──

  void _startSpeakingGestures() {
    if (_carCommandActive) return;
    final car = btManager?.car;
    if (car == null || !car.connected) return;

    _gestures ??= SpeakingGestures(car);
    _gestures!.start();
  }

  void _stopSpeakingGestures() {
    _gestures?.stop();
  }

  // ── History ──

  void _appendToHistory(String userText, String assistantText) {
    _history.add({'role': 'user', 'content': userText});
    _history.add({'role': 'assistant', 'content': assistantText});

    while (_history.length > _maxHistoryPairs * 2) {
      _history.removeAt(0);
      _history.removeAt(0);
    }
  }

  void clearHistory() {
    _history.clear();
  }

  /// Detect language from transcript text using Unicode script ranges.
  /// Telugu: \u0C00-\u0C7F, Hindi/Devanagari: \u0900-\u097F
  String _detectLanguage(String text) {
    int telugu = 0, hindi = 0, latin = 0;
    for (final c in text.runes) {
      if (c >= 0x0C00 && c <= 0x0C7F) {
        telugu++;
      } else if (c >= 0x0900 && c <= 0x097F) {
        hindi++;
      } else if ((c >= 0x0041 && c <= 0x007A)) {
        latin++;
      }
    }
    if (telugu > hindi && telugu > latin) return 'te';
    if (hindi > telugu && hindi > latin) return 'hi';
    return 'en';
  }

  /// Stop current interaction.
  Future<void> stop() async {
    // Stop continuous mode if active
    if (_continuousMode) {
      await stopContinuousMode();
    }
    // End active activity
    if (_currentActivity != null && _currentActivity!.isActive) {
      await _currentActivity!.end();
      _currentActivity = null;
    }
    await tts.stop();
    await audioCapture.stop();
    await songPlayer.stop();
    if (_followMode != null && _followMode!.isActive) {
      await _followMode!.stop();
      _notifier.setFollowMode(false);
    }
    _stopSpeakingGestures();
    _isProcessing = false;
  }

  /// Stop everything (speaking, listening, activities) and resume wake word.
  /// Used when user double-taps to interrupt and go back to idle.
  Future<void> stopAndResumeWakeWord() async {
    debugPrint('[Orchestrator] Double-tap: stopping all, resuming wake word');
    // Stop continuous mode if active
    if (_continuousMode) {
      _continuousMode = false;
      _notifier.setContinuousMode(false);
    }
    // End active activity
    if (_currentActivity != null && _currentActivity!.isActive) {
      await _currentActivity!.end();
      _currentActivity = null;
    }
    await tts.stop();
    await audioCapture.stop();
    await songPlayer.stop();
    if (_followMode != null && _followMode!.isActive) {
      await _followMode!.stop();
      _notifier.setFollowMode(false);
    }
    _stopSpeakingGestures();
    _isProcessing = false;
    _notifier.setBotState(BotState.ready);

    // Resume wake word detection
    _resumeWakeWord();
  }

  void dispose() {
    _wakeWordDetector?.dispose();
    tts.dispose();
    audioCapture.dispose();
    songPlayer.dispose();
    _followMode?.dispose();
    _personDetector.dispose();
    cameraManager.dispose();
    _gestures?.stop();
  }
}

/// Riverpod provider for the orchestrator.
final orchestratorProvider = Provider<Orchestrator>((ref) {
  final orchestrator = Orchestrator(
    stateProvider: sharedStateProvider,
    ref: ref,
  );
  ref.onDispose(() => orchestrator.dispose());
  return orchestrator;
});
