import 'package:flutter/foundation.dart';

import 'llm_provider.dart';
import 'openai_provider.dart';
import 'gemini_provider.dart';
import 'claude_provider.dart';
import 'offline_llm.dart';
import 'prompts.dart';

/// Routes LLM requests to the appropriate provider based on config.
///
/// Supports both online (cloud) and offline (local) modes.
/// In offline mode, routes to OfflineLlm which uses llama.cpp on-device.
class LlmRouter {
  final Map<String, LlmProvider> _providers = {};

  /// Active provider name: 'openai', 'gemini', 'claude', or 'offline'.
  String activeProvider;

  /// Current mode: 'online' or 'offline'.
  String mode;

  /// API keys (can be updated at runtime from settings).
  String openaiKey;
  String geminiKey;
  String claudeKey;

  /// Offline LLM instance (lazily created).
  OfflineLlm? _offlineLlm;

  LlmRouter({
    this.activeProvider = 'openai',
    this.mode = 'online',
    this.openaiKey = '',
    this.geminiKey = '',
    this.claudeKey = '',
  });

  /// Whether offline LLM model is available on disk.
  bool get offlineAvailable => _offlineLlm?.isAvailable ?? false;

  /// Initialize the offline LLM provider.
  /// Call this early so [offlineAvailable] reflects the true state.
  Future<void> initOffline() async {
    _offlineLlm ??= OfflineLlm();
    await _offlineLlm!.init();
    if (_offlineLlm!.isAvailable) {
      _providers['offline'] = _offlineLlm!;
      debugPrint('[LlmRouter] Offline LLM ready');
    } else {
      debugPrint('[LlmRouter] Offline LLM model not found');
    }
  }

  /// Get the active provider, creating it if needed.
  ///
  /// When [mode] is 'offline', returns the OfflineLlm provider.
  /// Falls back to cloud provider if offline model is not available.
  LlmProvider getProvider() {
    if (mode == 'offline') {
      if (_offlineLlm != null && _offlineLlm!.isAvailable) {
        return _offlineLlm!;
      }
      debugPrint(
        '[LlmRouter] Offline mode requested but model not available. '
        'Falling back to cloud provider "$activeProvider".',
      );
    }

    switch (activeProvider) {
      case 'gemini':
        return _getOrCreate('gemini', () => GeminiProvider(apiKey: geminiKey));
      case 'claude':
        return _getOrCreate('claude', () => ClaudeProvider(apiKey: claudeKey));
      case 'openai':
      default:
        return _getOrCreate('openai', () => OpenAIProvider(apiKey: openaiKey));
    }
  }

  LlmProvider _getOrCreate(String name, LlmProvider Function() factory) {
    final existing = _providers[name];
    if (existing != null) {
      // Update API key if changed
      _updateKey(existing);
      return existing;
    }
    final provider = factory();
    _providers[name] = provider;
    debugPrint('[LlmRouter] Created provider: $name');
    return provider;
  }

  void _updateKey(LlmProvider provider) {
    if (provider is OpenAIProvider) {
      provider.apiKey = openaiKey;
    } else if (provider is GeminiProvider) {
      provider.apiKey = geminiKey;
    } else if (provider is ClaudeProvider) {
      provider.apiKey = claudeKey;
    }
  }

  /// Build the full message list with system prompt + history + user input.
  List<Map<String, String>> buildMessages(
    String userText, {
    List<Map<String, String>>? history,
    int ageMin = 3,
    int ageMax = 6,
  }) {
    final systemPrompt = buildSystemPrompt(
      ageMin: ageMin,
      ageMax: ageMax,
    );

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
    ];

    if (history != null) {
      messages.addAll(history);
    }

    messages.add({'role': 'user', 'content': userText});
    return messages;
  }

  /// Dispose offline resources.
  void dispose() {
    _offlineLlm?.dispose();
    _offlineLlm = null;
  }
}
