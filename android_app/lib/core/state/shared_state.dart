import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../config/config_manager.dart';
import 'bot_state.dart';

/// A single conversation exchange between user and bot.
class ConversationEntry {
  final String userText;
  final String botResponse;
  final DateTime timestamp;
  final String language; // en, hi, te

  ConversationEntry({
    required this.userText,
    required this.botResponse,
    required this.timestamp,
    required this.language,
  });

  Map<String, dynamic> toJson() => {
        'userText': userText,
        'botResponse': botResponse,
        'timestamp': timestamp.toIso8601String(),
        'language': language,
      };

  factory ConversationEntry.fromJson(Map<String, dynamic> json) {
    return ConversationEntry(
      userText: json['userText'] as String? ?? '',
      botResponse: json['botResponse'] as String? ?? '',
      timestamp: DateTime.tryParse(json['timestamp'] as String? ?? '') ??
          DateTime.now(),
      language: json['language'] as String? ?? 'en',
    );
  }
}

/// Observable app state -- Riverpod providers for reactive UI.
class SharedState {
  BotState botState;
  String currentTranscript;
  String currentResponse;
  bool micEnabled;
  bool cameraEnabled;
  int volume; // 0-100
  String language; // en, hi, te
  String mode; // online, offline

  // Car
  bool carConnected;
  bool carConnecting;
  String? carMac;
  bool followMode;

  // Continuous conversation mode
  bool continuousMode;

  // Conversation history
  List<ConversationEntry> conversationHistory;

  SharedState({
    this.botState = BotState.ready,
    this.currentTranscript = '',
    this.currentResponse = '',
    this.micEnabled = true,
    this.cameraEnabled = true,
    this.volume = 80,
    this.language = 'en',
    this.mode = 'online',
    this.carConnected = false,
    this.carConnecting = false,
    this.carMac,
    this.followMode = false,
    this.continuousMode = false,
    List<ConversationEntry>? conversationHistory,
  }) : conversationHistory = conversationHistory ?? [];

  SharedState copyWith({
    BotState? botState,
    String? currentTranscript,
    String? currentResponse,
    bool? micEnabled,
    bool? cameraEnabled,
    int? volume,
    String? language,
    String? mode,
    bool? carConnected,
    bool? carConnecting,
    String? carMac,
    bool? followMode,
    bool? continuousMode,
    List<ConversationEntry>? conversationHistory,
  }) {
    return SharedState(
      botState: botState ?? this.botState,
      currentTranscript: currentTranscript ?? this.currentTranscript,
      currentResponse: currentResponse ?? this.currentResponse,
      micEnabled: micEnabled ?? this.micEnabled,
      cameraEnabled: cameraEnabled ?? this.cameraEnabled,
      volume: volume ?? this.volume,
      language: language ?? this.language,
      mode: mode ?? this.mode,
      carConnected: carConnected ?? this.carConnected,
      carConnecting: carConnecting ?? this.carConnecting,
      carMac: carMac ?? this.carMac,
      followMode: followMode ?? this.followMode,
      continuousMode: continuousMode ?? this.continuousMode,
      conversationHistory: conversationHistory ?? this.conversationHistory,
    );
  }
}

/// Main state notifier
class SharedStateNotifier extends StateNotifier<SharedState> {
  SharedStateNotifier() : super(SharedState());

  void setBotState(BotState s) => state = state.copyWith(botState: s);
  void setTranscript(String t) => state = state.copyWith(currentTranscript: t);
  void setResponse(String r) => state = state.copyWith(currentResponse: r);
  void setVolume(int v) => state = state.copyWith(volume: v);
  void setCarConnected(bool c) => state = state.copyWith(carConnected: c);
  void setCarConnecting(bool c) => state = state.copyWith(carConnecting: c);
  void setCarMac(String? m) => state = state.copyWith(carMac: m);
  void setFollowMode(bool f) => state = state.copyWith(followMode: f);
  void setMode(String m) => state = state.copyWith(mode: m);
  void setLanguage(String l) => state = state.copyWith(language: l);
  void setContinuousMode(bool c) => state = state.copyWith(continuousMode: c);

  /// Add a conversation entry to the history.
  void addConversation({
    required String userText,
    required String botResponse,
    required String language,
  }) {
    final entry = ConversationEntry(
      userText: userText,
      botResponse: botResponse,
      timestamp: DateTime.now(),
      language: language,
    );
    final updated = List<ConversationEntry>.from(state.conversationHistory)
      ..add(entry);
    state = state.copyWith(conversationHistory: updated);
  }

  /// Clear all conversation history.
  void clearHistory() {
    state = state.copyWith(conversationHistory: []);
  }

  // -- Persistent conversation history --

  /// Maximum number of conversation entries to persist.
  static const _maxPersistedEntries = 100;

  /// Save conversation history to SharedPreferences via ConfigManager.
  Future<void> saveConversationHistory(ConfigManager config) async {
    try {
      final entries = state.conversationHistory;
      // Keep only the last N entries
      final toSave = entries.length > _maxPersistedEntries
          ? entries.sublist(entries.length - _maxPersistedEntries)
          : entries;
      final jsonList = toSave.map((e) => e.toJson()).toList();
      await config.setConversationHistoryJson(jsonEncode(jsonList));
      debugPrint(
          '[SharedState] Saved ${toSave.length} conversation entries');
    } catch (e) {
      debugPrint('[SharedState] Failed to save conversation history: $e');
    }
  }

  /// Load conversation history from SharedPreferences via ConfigManager.
  void loadConversationHistory(ConfigManager config) {
    try {
      final jsonStr = config.conversationHistoryJson;
      if (jsonStr.isEmpty || jsonStr == '[]') return;
      final jsonList = jsonDecode(jsonStr) as List<dynamic>;
      final entries = jsonList
          .map((e) => ConversationEntry.fromJson(e as Map<String, dynamic>))
          .toList();
      // Keep only the last N entries
      final trimmed = entries.length > _maxPersistedEntries
          ? entries.sublist(entries.length - _maxPersistedEntries)
          : entries;
      state = state.copyWith(conversationHistory: trimmed);
      debugPrint(
          '[SharedState] Loaded ${trimmed.length} conversation entries');
    } catch (e) {
      debugPrint('[SharedState] Failed to load conversation history: $e');
    }
  }
}

/// Riverpod provider
final sharedStateProvider =
    StateNotifierProvider<SharedStateNotifier, SharedState>(
  (ref) => SharedStateNotifier(),
);
