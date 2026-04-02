import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../models/age_band.dart';

/// Manages app configuration backed by SharedPreferences.
class ConfigManager {
  static const _keyOpenaiKey = 'api_key_openai';
  static const _keyGeminiKey = 'api_key_gemini';
  static const _keyClaudeKey = 'api_key_claude';
  static const _keyProvider = 'llm_provider';
  static const _keyMode = 'llm_mode';
  static const _keyLanguage = 'language';
  static const _keyVolume = 'volume';
  static const _keyAgeMin = 'age_min';
  static const _keyAgeMax = 'age_max';
  static const _keyCloudStt = 'cloud_stt';
  static const _keyPin = 'parent_pin';
  static const _keySpeechRate = 'speech_rate';
  static const _keyFirstRunDone = 'first_run_done';
  static const _keyPorcupineAccessKey = 'porcupine_access_key';
  static const _keyWakeWordEnabled = 'wake_word_enabled';
  static const _keyOfflineMode = 'offline_mode';
  static const _keyContinuousMode = 'continuous_mode';
  static const _keyConversationHistory = 'conversation_history';
  static const _keyChildAge = 'child_age';
  static const _keyChildName = 'child_name';

  SharedPreferences? _prefs;

  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
    debugPrint('[Config] Initialized');
  }

  SharedPreferences get _p {
    if (_prefs == null) throw StateError('ConfigManager not initialized');
    return _prefs!;
  }

  // API Keys
  String get openaiKey => _p.getString(_keyOpenaiKey) ?? '';
  String get geminiKey => _p.getString(_keyGeminiKey) ?? '';
  String get claudeKey => _p.getString(_keyClaudeKey) ?? '';

  Future<void> setOpenaiKey(String key) => _p.setString(_keyOpenaiKey, key);
  Future<void> setGeminiKey(String key) => _p.setString(_keyGeminiKey, key);
  Future<void> setClaudeKey(String key) => _p.setString(_keyClaudeKey, key);

  // Provider & mode
  String get provider => _p.getString(_keyProvider) ?? 'openai';
  String get mode => _p.getString(_keyMode) ?? 'online';

  Future<void> setProvider(String p) => _p.setString(_keyProvider, p);
  Future<void> setMode(String m) => _p.setString(_keyMode, m);

  // Language
  String get language => _p.getString(_keyLanguage) ?? 'en';
  Future<void> setLanguage(String l) => _p.setString(_keyLanguage, l);

  // Volume
  int get volume => _p.getInt(_keyVolume) ?? 80;
  Future<void> setVolume(int v) => _p.setInt(_keyVolume, v);

  // Child age range
  int get ageMin => _p.getInt(_keyAgeMin) ?? 3;
  int get ageMax => _p.getInt(_keyAgeMax) ?? 6;

  // Child age range
  Future<void> setAgeMin(int v) => _p.setInt(_keyAgeMin, v);
  Future<void> setAgeMax(int v) => _p.setInt(_keyAgeMax, v);

  // Cloud STT toggle
  bool get cloudStt => _p.getBool(_keyCloudStt) ?? true;
  Future<void> setCloudStt(bool v) => _p.setBool(_keyCloudStt, v);

  // Parent PIN (default: 1234)
  String get pin => _p.getString(_keyPin) ?? '1234';
  Future<void> setPin(String p) => _p.setString(_keyPin, p);

  // Speech rate (0.5 - 2.0, default 1.0)
  double get speechRate => _p.getDouble(_keySpeechRate) ?? 1.0;
  Future<void> setSpeechRate(double r) => _p.setDouble(_keySpeechRate, r);

  // First-run flag
  bool get firstRunDone => _p.getBool(_keyFirstRunDone) ?? false;
  Future<void> setFirstRunDone(bool v) => _p.setBool(_keyFirstRunDone, v);

  // Porcupine wake word access key
  String get porcupineAccessKey =>
      _p.getString(_keyPorcupineAccessKey) ?? '';
  Future<void> setPorcupineAccessKey(String key) =>
      _p.setString(_keyPorcupineAccessKey, key);

  // Wake word detection toggle
  bool get wakeWordEnabled => _p.getBool(_keyWakeWordEnabled) ?? false;
  Future<void> setWakeWordEnabled(bool v) =>
      _p.setBool(_keyWakeWordEnabled, v);

  // Offline mode toggle (use local models instead of cloud APIs)
  bool get offlineMode => _p.getBool(_keyOfflineMode) ?? false;
  Future<void> setOfflineMode(bool v) => _p.setBool(_keyOfflineMode, v);

  // Continuous conversation mode
  bool get continuousMode => _p.getBool(_keyContinuousMode) ?? false;
  Future<void> setContinuousMode(bool v) => _p.setBool(_keyContinuousMode, v);

  // Persistent conversation history (JSON string)
  String get conversationHistoryJson =>
      _p.getString(_keyConversationHistory) ?? '[]';
  Future<void> setConversationHistoryJson(String json) =>
      _p.setString(_keyConversationHistory, json);

  // Child profile
  int get childAge => _p.getInt(_keyChildAge) ?? 5;
  Future<void> setChildAge(int v) => _p.setInt(_keyChildAge, v);

  String get childName => _p.getString(_keyChildName) ?? '';
  Future<void> setChildName(String n) => _p.setString(_keyChildName, n);

  /// Computed age band derived from [childAge].
  AgeBand get ageBand => AgeBandExt.fromAge(childAge);

  /// Check if any API key is configured.
  bool get hasApiKey => openaiKey.isNotEmpty || geminiKey.isNotEmpty || claudeKey.isNotEmpty;

  /// Get API key for the active provider.
  String getActiveApiKey() {
    switch (provider) {
      case 'gemini':
        return geminiKey;
      case 'claude':
        return claudeKey;
      case 'openai':
      default:
        return openaiKey;
    }
  }
}
