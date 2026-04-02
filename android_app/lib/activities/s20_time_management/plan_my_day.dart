import 'dart:convert';
import 'dart:math';

import 'package:shared_preferences/shared_preferences.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// PlanMyDay: morning and evening planning routine.
///
/// Teaches: planning, prioritization, time awareness, self-organization.
///
/// Flow:
/// 1. Morning mode: help child plan 3-5 things for the day.
/// 2. Evening mode: review the plan, check what was done.
/// 3. Celebrate accomplishments, reflect on what to improve.
///
/// Persists today's plan via SharedPreferences.
class PlanMyDay extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _score = 0;
  List<String> _todaysPlan = [];
  List<bool> _completed = [];

  /// 0=detect mode, 1=adding items, 2=confirming plan,
  /// 3=reviewing items, 4=reflection
  int _phase = 0;
  bool _isEvening = false;
  int _reviewIndex = 0;

  static const String _prefKeyPlan = 'plan_my_day_plan';
  static const String _prefKeyCompleted = 'plan_my_day_completed';
  static const String _prefKeyDate = 'plan_my_day_date';

  static const int _maxItems = 5;

  PlanMyDay();

  @override
  String get id => 'plan_my_day';

  @override
  String get name => 'Plan My Day';

  @override
  String get category => 'time-management';

  @override
  String get description =>
      'Plan your day in the morning and review it in the evening.';

  @override
  List<String> get skills =>
      ['planning', 'prioritization', 'time awareness', 'self-organization'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.timeManagement;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'plan my day',
          'daily plan',
          'morning plan',
          'evening review',
          'what should i do today',
        ],
        'hi': [
          'दिन की योजना',
          'आज का प्लान',
          'सुबह की योजना',
        ],
        'te': [
          'రోజు ప్లాన్',
          'ఈ రోజు ప్లాన్',
          'ఉదయం ప్లాన్',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_todaysPlan.isEmpty) return 'No plan set for today.';
    final doneCount = _completed.where((c) => c).length;
    return '$doneCount of ${_todaysPlan.length} tasks done. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _score = 0;

    await _loadState();

    final hour = DateTime.now().hour;
    final hasPlanForToday = _todaysPlan.isNotEmpty && await _isPlanFromToday();

    if (hasPlanForToday && hour >= 16) {
      // Evening review mode
      _isEvening = true;
      _phase = 3;
      _reviewIndex = 0;
      return 'Good evening! Let us review your day. You planned '
          '${_todaysPlan.length} things today. Let me go through them. '
          '${_askAboutItem()}';
    } else if (hasPlanForToday) {
      // Already has a plan for today
      _phase = 0;
      final planList = _todaysPlan
          .asMap()
          .entries
          .map((e) => '${e.key + 1}. ${e.value}')
          .join('. ');
      return 'You already have a plan for today! Here it is: $planList. '
          'Would you like to review how it is going, or make a new plan?';
    } else {
      // Morning planning mode
      _isEvening = false;
      _todaysPlan = [];
      _completed = [];
      _phase = 1;
      return 'Good morning! Let us plan your day. What are the most '
          'important things you want to do today? Tell me one at a time. '
          'You can plan up to $_maxItems things. What is the first one?';
    }
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
        // Existing plan: review or new?
        if (lower.contains('review') || lower.contains('how') ||
            lower.contains('check')) {
          _isEvening = true;
          _phase = 3;
          _reviewIndex = 0;
          return _askAboutItem();
        } else if (lower.contains('new') || lower.contains('different') ||
            lower.contains('change')) {
          _todaysPlan = [];
          _completed = [];
          _phase = 1;
          return 'Let us make a fresh plan! What is the first thing '
              'you want to do today?';
        }
        return 'Would you like to review how your plan is going, '
            'or make a new plan?';

      case 1:
        // Adding items to plan
        _todaysPlan.add(childSaid.trim());
        _completed.add(false);
        _score += 5;

        if (_todaysPlan.length >= _maxItems) {
          _phase = 2;
          return _summarizePlan();
        }

        return 'Got it! ${_todaysPlan.length} '
            'thing${_todaysPlan.length != 1 ? 's' : ''} so far. '
            'What else? Or say "that is all" if you are done planning.';

      case 2:
        // Confirming plan (or they said "that is all" during adding)
        if (_isAffirmative(lower) || lower.contains('that is all') ||
            lower.contains('done') || lower.contains('good')) {
          await _saveState();
          _active = false;
          return 'Your plan is set! You have ${_todaysPlan.length} things '
              'to do today. Come back this evening and we will see how you '
              'did. Have a great day! Score: $_score points.';
        }
        // They want to add more
        if (_todaysPlan.length < _maxItems) {
          _todaysPlan.add(childSaid.trim());
          _completed.add(false);
          _score += 5;
          return _summarizePlan();
        }
        await _saveState();
        _active = false;
        return 'Your plan is full! ${_todaysPlan.length} things for today. '
            'Come back tonight to review. Score: $_score points.';

      case 3:
        // Reviewing items
        if (_isAffirmative(lower) || lower.contains('did') ||
            lower.contains('finished') || lower.contains('completed')) {
          _completed[_reviewIndex] = true;
          _score += 10;
        } else {
          _score += 3;
        }

        _reviewIndex++;
        if (_reviewIndex >= _todaysPlan.length) {
          _phase = 4;
          final doneCount = _completed.where((c) => c).length;
          await _saveState();
          return '${ doneCount == _todaysPlan.length ? "You finished everything! That is incredible!" : "You got $doneCount out of ${_todaysPlan.length} done!"} '
              'What is one thing you learned about your day?';
        }

        return '${_completed[_reviewIndex - 1] ? _randomCelebration() : "That is okay, tomorrow is a new chance!"} '
            '${_askAboutItem()}';

      case 4:
        // Reflection
        _score += 10;
        _active = false;
        final doneCount = _completed.where((c) => c).length;
        return 'That is great reflection! You completed $doneCount out of '
            '${_todaysPlan.length} tasks today. Score: $_score points. '
            'Planning your day helps you do more of what matters. '
            'See you tomorrow for a new plan!';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_todaysPlan.isNotEmpty) {
      await _saveState();
    }
    if (_todaysPlan.isEmpty) {
      return 'Come back to plan your day! A good plan makes a great day.';
    }
    final doneCount = _completed.where((c) => c).length;
    return 'Your plan has ${_todaysPlan.length} items. '
        '$doneCount done so far. Score: $_score points. See you later!';
  }

  // -- Helpers --

  String _askAboutItem() {
    if (_reviewIndex >= _todaysPlan.length) return '';
    return 'Number ${_reviewIndex + 1}: ${_todaysPlan[_reviewIndex]}. '
        'Did you get this done?';
  }

  String _summarizePlan() {
    final planList = _todaysPlan
        .asMap()
        .entries
        .map((e) => '${e.key + 1}. ${e.value}')
        .join('. ');
    if (_todaysPlan.length >= _maxItems) {
      return 'Here is your plan: $planList. That looks great! '
          'Shall I save this plan?';
    }
    return 'Here is your plan so far: $planList. '
        'Would you like to add more, or is this good?';
  }

  // -- Persistence --

  Future<void> _loadState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final planJson = prefs.getString(_prefKeyPlan);
      final completedJson = prefs.getString(_prefKeyCompleted);
      if (planJson != null) {
        _todaysPlan = List<String>.from(json.decode(planJson));
      }
      if (completedJson != null) {
        _completed = List<bool>.from(json.decode(completedJson));
      }
      // Ensure lists are same length
      while (_completed.length < _todaysPlan.length) {
        _completed.add(false);
      }
    } catch (_) {
      _todaysPlan = [];
      _completed = [];
    }
  }

  Future<void> _saveState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_prefKeyPlan, json.encode(_todaysPlan));
      await prefs.setString(_prefKeyCompleted, json.encode(_completed));
      final today = DateTime.now().toIso8601String().substring(0, 10);
      await prefs.setString(_prefKeyDate, today);
    } catch (_) {
      // Silently fail.
    }
  }

  Future<bool> _isPlanFromToday() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final savedDate = prefs.getString(_prefKeyDate);
      final today = DateTime.now().toIso8601String().substring(0, 10);
      return savedDate == today;
    } catch (_) {
      return false;
    }
  }

  String _randomCelebration() {
    const phrases = [
      'Great job!',
      'Well done!',
      'You did it!',
      'Awesome!',
      'Nice work!',
      'That is wonderful!',
    ];
    return phrases[_rng.nextInt(phrases.length)];
  }

  bool _isAffirmative(String text) {
    const words = ['yes', 'yeah', 'sure', 'okay', 'yep', 'i did',
      'finished', 'completed', 'done',
      'हाँ', 'हां', 'అవును'];
    return words.any((w) => text.contains(w));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'finish', 'enough', 'bye'];
    return quitWords.any((w) => text.contains(w));
  }
}
