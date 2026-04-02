import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';

/// Text-to-speech service using platform TTS (Android/iOS built-in).
/// Phase 2: platform TTS first, Piper/sherpa_onnx added later.
class TtsService {
  final FlutterTts _tts = FlutterTts();
  bool _initialized = false;
  bool _isSpeaking = false;
  final Completer<void> _speakCompleter = Completer<void>();

  bool get isSpeaking => _isSpeaking;

  /// Initialize TTS engine with child-friendly settings.
  /// Don't force a language — Android Google TTS auto-detects
  /// Telugu/Hindi/English from the text script automatically.
  Future<void> init({String language = 'en'}) async {
    if (_initialized) return;

    await _tts.setSpeechRate(0.45); // Slow for children
    await _tts.setPitch(1.15); // Slightly higher pitch — friendlier
    await _tts.setVolume(1.0);

    // Try to pick a female voice if available (more friendly for kids)
    try {
      final voices = await _tts.getVoices;
      if (voices is List) {
        final langCode = _languageCode(language);
        // Find voices matching our language
        final matching = voices.where((v) {
          final locale = (v as Map)['locale']?.toString() ?? '';
          return locale.startsWith(langCode.split('-')[0]);
        }).toList();
        if (matching.isNotEmpty) {
          debugPrint('[TTS] Available voices: ${matching.length}');
        }
      }
    } catch (e) {
      debugPrint('[TTS] Could not list voices: $e');
    }

    _tts.setStartHandler(() {
      _isSpeaking = true;
    });

    _tts.setCompletionHandler(() {
      _isSpeaking = false;
    });

    _tts.setErrorHandler((msg) {
      debugPrint('[TTS] Error: $msg');
      _isSpeaking = false;
    });

    _initialized = true;
    debugPrint('[TTS] Initialized for language: $language');
  }

  /// Speak a sentence. Returns a Future that completes when speech is done.
  Future<void> speak(String text) async {
    if (!_initialized) await init();
    if (text.trim().isEmpty) return;

    final completer = Completer<void>();

    _tts.setCompletionHandler(() {
      _isSpeaking = false;
      if (!completer.isCompleted) completer.complete();
    });

    _tts.setErrorHandler((msg) {
      _isSpeaking = false;
      if (!completer.isCompleted) completer.complete();
    });

    _tts.setCancelHandler(() {
      _isSpeaking = false;
      if (!completer.isCompleted) completer.complete();
    });

    _isSpeaking = true;
    await _tts.speak(text);

    return completer.future;
  }

  /// Stop speaking immediately.
  Future<void> stop() async {
    await _tts.stop();
    _isSpeaking = false;
  }

  /// Set volume (0-100 mapped to 0.0-1.0).
  Future<void> setVolume(int percent) async {
    final vol = (percent.clamp(0, 100)) / 100.0;
    await _tts.setVolume(vol);
    debugPrint('[TTS] Volume set to $percent% ($vol)');
  }

  /// Update language and log available voices for debugging.
  Future<void> setLanguage(String language) async {
    final code = _languageCode(language);
    debugPrint('[TTS] Switching to language: $code');

    // Log all available voices for this language
    try {
      final voices = await _tts.getVoices;
      if (voices is List) {
        final matching = voices.where((v) {
          final locale = (v as Map)['locale']?.toString() ?? '';
          return locale.startsWith(language == 'te' ? 'te' : language == 'hi' ? 'hi' : 'en');
        }).toList();
        for (final v in matching) {
          debugPrint('[TTS] Voice: ${v}');
        }
      }
    } catch (_) {}

    await _tts.setLanguage(code);
  }

  /// Map short language codes to BCP-47.
  String _languageCode(String lang) {
    switch (lang) {
      case 'hi':
        return 'hi-IN';
      case 'te':
        return 'te-IN';
      case 'en':
      default:
        return 'en-IN';
    }
  }

  void dispose() {
    _tts.stop();
  }
}
