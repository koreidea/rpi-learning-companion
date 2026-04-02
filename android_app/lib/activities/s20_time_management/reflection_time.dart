import 'dart:convert';
import 'dart:math';

import 'package:shared_preferences/shared_preferences.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// ReflectionTime: end-of-day reflection and gratitude practice.
///
/// Teaches: self-reflection, gratitude, emotional awareness, growth mindset.
///
/// Flow:
/// 1. Bot asks: what went well today?
/// 2. What was challenging?
/// 3. What did you learn?
/// 4. What are you grateful for?
/// 5. Summarize and encourage.
///
/// Persists reflections via SharedPreferences for tracking over time.
class ReflectionTime extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _score = 0;
  int _reflectionsDone = 0;

  /// 0=went well, 1=challenge, 2=learned, 3=grateful, 4=summary
  int _phase = 0;

  String _wentWell = '';
  String _challenge = '';
  String _learned = '';
  String _grateful = '';

  static const String _prefKeyHistory = 'reflection_time_history';
  static const String _prefKeyCount = 'reflection_time_count';

  ReflectionTime();

  @override
  String get id => 'reflection_time';

  @override
  String get name => 'Reflection Time';

  @override
  String get category => 'time-management';

  @override
  String get description =>
      'End your day with reflection and gratitude to grow stronger.';

  @override
  List<String> get skills =>
      ['self-reflection', 'gratitude', 'emotional awareness', 'growth mindset'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.timeManagement;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'reflection time',
          'daily reflection',
          'end of day',
          'how was my day',
          'gratitude',
        ],
        'hi': [
          'आज का चिंतन',
          'दिन कैसा रहा',
          'आभार',
        ],
        'te': [
          'ఈ రోజు ఆలోచన',
          'రోజు ఎలా గడిచింది',
          'కృతజ్ఞత',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    return '$_reflectionsDone reflection${_reflectionsDone != 1 ? 's' : ''} '
        'completed. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _score = 0;
    _phase = 0;
    _wentWell = '';
    _challenge = '';
    _learned = '';
    _grateful = '';

    await _loadCount();

    return 'Welcome to Reflection Time! Taking a few minutes to think about '
        'your day helps you learn and grow. Let us start with a nice question: '
        'What is one thing that went well today?';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      return await end();
    }

    switch (_phase) {
      case 0:
        // What went well
        _wentWell = childSaid.trim();
        _phase = 1;
        _score += 10;
        return 'That sounds wonderful! It is important to notice the good '
            'things. Now, was there anything that was a little challenging '
            'or difficult today?';

      case 1:
        // What was challenging
        _challenge = childSaid.trim();
        _phase = 2;
        _score += 10;
        return 'Thank you for sharing that. Challenges are how we get '
            'stronger. Everyone faces hard things sometimes. '
            'What is one thing you learned today? It could be anything, '
            'big or small.';

      case 2:
        // What you learned
        _learned = childSaid.trim();
        _phase = 3;
        _score += 10;
        return 'That is great! Learning something new every day is '
            'wonderful. Last question: what is one thing you are '
            'grateful for today?';

      case 3:
        // Gratitude
        _grateful = childSaid.trim();
        _phase = 4;
        _score += 15;
        _reflectionsDone++;

        await _saveReflection();

        _active = false;
        return _buildSummary();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_phase == 0) {
      return 'Come back tonight for reflection time. Looking back on your '
          'day is one of the best ways to grow!';
    }

    // Partial reflection is still valuable
    _reflectionsDone++;
    await _saveReflection();
    return 'Even a little reflection is valuable! '
        'Score: $_score points. Sweet dreams!';
  }

  String _buildSummary() {
    return 'Here is your reflection for today. '
        'You said something good was: $_wentWell. '
        'A challenge was: $_challenge. '
        'You learned: $_learned. '
        'And you are grateful for: $_grateful. '
        'That is $_reflectionsDone reflection'
        '${_reflectionsDone != 1 ? 's' : ''} total! '
        'Score: $_score points! '
        'Thinking about your day makes tomorrow even better. '
        'Sweet dreams!';
  }

  // -- Persistence --

  Future<void> _loadCount() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _reflectionsDone = prefs.getInt(_prefKeyCount) ?? 0;
    } catch (_) {
      _reflectionsDone = 0;
    }
  }

  Future<void> _saveReflection() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt(_prefKeyCount, _reflectionsDone);

      // Save today's reflection to history
      final historyJson = prefs.getString(_prefKeyHistory);
      List<Map<String, dynamic>> history = [];
      if (historyJson != null) {
        history = List<Map<String, dynamic>>.from(
          (json.decode(historyJson) as List).map((e) => Map<String, dynamic>.from(e)),
        );
      }

      history.add({
        'date': DateTime.now().toIso8601String().substring(0, 10),
        'went_well': _wentWell,
        'challenge': _challenge,
        'learned': _learned,
        'grateful': _grateful,
      });

      // Keep only last 30 reflections
      if (history.length > 30) {
        history = history.sublist(history.length - 30);
      }

      await prefs.setString(_prefKeyHistory, json.encode(history));
    } catch (_) {
      // Silently fail; non-critical.
    }
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'finish', 'enough',
      'bye', 'skip'];
    return quitWords.any((w) => text.contains(w));
  }
}
