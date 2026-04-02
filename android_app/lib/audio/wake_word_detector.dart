import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:speech_to_text/speech_to_text.dart';

/// Callback type for wake word detection events.
typedef WakeWordCallback = void Function();

/// Always-on wake word detection using Android's built-in SpeechRecognizer.
///
/// Listens continuously for "hai robo" (free, on-device, no API key needed).
/// Uses the speech_to_text package which wraps Android SpeechRecognizer.
///
/// The detector automatically restarts listening after each recognition cycle
/// ends (Android stops after a period of silence). This creates a continuous
/// listening loop that runs as long as the detector is active.
class WakeWordDetector {
  static const String _tag = '[WakeWord]';

  /// Wake phrases to detect (all lowercase).
  /// Includes common misrecognitions by Android SpeechRecognizer.
  static const List<String> _wakePhrases = [
    'hai robo',
    'hi robo',
    'hey robo',
    'hai robot',
    'hi robot',
    'hey robot',
    'hairobo',
    'hi robho',
    'high robo',
    'hire obo',
    'hair obo',
  ];

  /// Callback invoked when the wake word is detected.
  final WakeWordCallback onWakeWord;

  final SpeechToText _speech = SpeechToText();
  bool _isListening = false;
  bool _shouldBeListening = false;
  bool _initialized = false;
  Timer? _restartTimer;

  // Prevent rapid restarts
  DateTime _lastRestartTime = DateTime.now();
  static const _minRestartInterval = Duration(milliseconds: 500);

  // Track if we're currently in a callback to avoid re-entrancy
  bool _inCallback = false;

  WakeWordDetector({required this.onWakeWord});

  /// Whether the detector is currently active (may be between listen cycles).
  bool get isListening => _shouldBeListening;

  /// Initialize the speech recognizer.
  Future<bool> _ensureInitialized() async {
    if (_initialized) return true;
    try {
      _initialized = await _speech.initialize(
        onStatus: _onStatus,
        onError: _onError,
        debugLogging: false,
      );
      debugPrint('$_tag Initialized: $_initialized');
      return _initialized;
    } catch (e) {
      debugPrint('$_tag Init error: $e');
      return false;
    }
  }

  /// Start listening for the wake word.
  /// Returns true if listening started successfully.
  Future<bool> start() async {
    if (_shouldBeListening) {
      debugPrint('$_tag Already active');
      return true;
    }

    final ready = await _ensureInitialized();
    if (!ready) {
      debugPrint('$_tag Speech recognition not available');
      return false;
    }

    _shouldBeListening = true;
    debugPrint('$_tag Starting wake word detection for "hai robo"');
    return _startListening();
  }

  /// Internal: start a single listen cycle.
  Future<bool> _startListening() async {
    if (!_shouldBeListening) return false;
    if (_isListening) return true;

    // Throttle restarts
    final now = DateTime.now();
    if (now.difference(_lastRestartTime) < _minRestartInterval) {
      _scheduleRestart(const Duration(milliseconds: 600));
      return true;
    }
    _lastRestartTime = now;

    try {
      await _speech.listen(
        onResult: _onResult,
        listenFor: const Duration(seconds: 30),
        pauseFor: const Duration(seconds: 3),
        partialResults: true,
        listenMode: ListenMode.dictation,
        cancelOnError: false,
      );
      _isListening = true;
      debugPrint('$_tag Listening cycle started');
      return true;
    } catch (e) {
      debugPrint('$_tag Listen error: $e');
      _isListening = false;
      // Try to restart after a delay
      _scheduleRestart(const Duration(seconds: 2));
      return false;
    }
  }

  /// Called when speech is recognized.
  void _onResult(result) {
    if (!_shouldBeListening || _inCallback) return;

    final text = result.recognizedWords?.toLowerCase()?.trim() ?? '';
    if (text.isEmpty) return;

    debugPrint('$_tag Heard: "$text" (final: ${result.finalResult})');

    // Check for wake phrase
    for (final phrase in _wakePhrases) {
      if (text.contains(phrase)) {
        debugPrint('$_tag Wake word detected in: "$text"');
        _inCallback = true;

        // Stop listening before triggering callback
        _stopListeningCycle();

        // Trigger the callback
        onWakeWord();

        _inCallback = false;
        return;
      }
    }
  }

  /// Called when recognition status changes.
  void _onStatus(String status) {
    debugPrint('$_tag Status: $status');

    if (status == 'notListening' || status == 'done') {
      _isListening = false;
      // Auto-restart if we should still be listening
      if (_shouldBeListening && !_inCallback) {
        _scheduleRestart(const Duration(milliseconds: 300));
      }
    }
  }

  /// Called on recognition errors.
  void _onError(dynamic error) {
    debugPrint('$_tag Error: $error');
    _isListening = false;

    // Restart on recoverable errors
    if (_shouldBeListening && !_inCallback) {
      _scheduleRestart(const Duration(seconds: 2));
    }
  }

  /// Schedule a restart of the listen cycle.
  void _scheduleRestart(Duration delay) {
    _restartTimer?.cancel();
    _restartTimer = Timer(delay, () {
      if (_shouldBeListening && !_isListening && !_inCallback) {
        debugPrint('$_tag Restarting listen cycle');
        _startListening();
      }
    });
  }

  /// Stop the current listen cycle (but don't deactivate).
  void _stopListeningCycle() {
    _restartTimer?.cancel();
    if (_isListening) {
      _speech.stop();
      _isListening = false;
    }
  }

  /// Stop listening completely.
  Future<void> stop() async {
    _shouldBeListening = false;
    _restartTimer?.cancel();
    _restartTimer = null;
    if (_isListening) {
      await _speech.stop();
      _isListening = false;
    }
    debugPrint('$_tag Stopped');
  }

  /// Resume listening after a conversation ends.
  /// Call this after the orchestrator finishes processing a voice interaction.
  Future<void> resume() async {
    if (!_shouldBeListening) return;
    debugPrint('$_tag Resuming wake word detection');
    // Small delay to let the mic be freed by audio capture
    await Future.delayed(const Duration(milliseconds: 500));
    _startListening();
  }

  /// Release all resources.
  Future<void> dispose() async {
    await stop();
    if (_initialized) {
      _speech.cancel();
      _initialized = false;
    }
    debugPrint('$_tag Disposed');
  }
}
