import 'dart:math';

import 'package:shared_preferences/shared_preferences.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// GoalBuddy: weekly goal setting with daily check-ins.
///
/// Teaches: goal setting, self-reflection, perseverance, planning.
///
/// Flow:
/// 1. If no current goal: help child set a weekly goal.
/// 2. If goal exists: daily check-in on progress.
/// 3. Celebrate milestones and encourage persistence.
/// 4. At week end: reflect on achievement and set a new goal.
///
/// Persists current goal and progress via SharedPreferences.
class GoalBuddy extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _checkIns = 0;

  /// 0=check state, 1=setting goal, 2=confirming goal, 3=daily check-in,
  /// 4=check-in follow-up
  int _phase = 0;

  String _pendingGoal = '';
  String _currentGoal = '';
  int _daysCompleted = 0;
  int _totalDays = 7;
  int _score = 0;

  static const String _prefKeyGoal = 'goal_buddy_current_goal';
  static const String _prefKeyDays = 'goal_buddy_days_completed';
  static const String _prefKeyLastCheckin = 'goal_buddy_last_checkin';

  GoalBuddy();

  @override
  String get id => 'goal_buddy';

  @override
  String get name => 'Goal Buddy';

  @override
  String get category => 'self-direction';

  @override
  String get description =>
      'Set weekly goals and check in daily to build self-direction.';

  @override
  List<String> get skills =>
      ['goal setting', 'self-reflection', 'perseverance', 'planning'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.selfDirection;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'goal buddy',
          'set a goal',
          'my goals',
          'weekly goal',
          'check in',
        ],
        'hi': [
          'लक्ष्य दोस्त',
          'लक्ष्य बनाओ',
          'मेरा लक्ष्य',
        ],
        'te': [
          'లక్ష్యం స్నేహితుడు',
          'లక్ష్యం పెట్టు',
          'నా లక్ష్యం',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_currentGoal.isEmpty) return 'No goal set yet.';
    return 'Goal: $_currentGoal. $_daysCompleted of $_totalDays days done. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _checkIns = 0;
    _score = 0;
    _phase = 0;

    await _loadState();

    if (_currentGoal.isNotEmpty) {
      if (_daysCompleted >= _totalDays) {
        _phase = 1;
        final oldGoal = _currentGoal;
        _currentGoal = '';
        _daysCompleted = 0;
        await _saveState();
        return 'Welcome back to Goal Buddy! Congratulations, you finished '
            'your goal of $oldGoal! That took real determination. '
            'Would you like to set a new goal for this week?';
      }

      final alreadyCheckedInToday = await _hasCheckedInToday();
      if (alreadyCheckedInToday) {
        _phase = 3;
        return 'Welcome back! Your goal is: $_currentGoal. '
            'You have already checked in today. You are on day '
            '${_daysCompleted + 1} of $_totalDays. Keep going, you are '
            'doing great! Would you like to talk about how it is going?';
      }

      _phase = 3;
      return 'Welcome back to Goal Buddy! Your goal this week is: '
          '$_currentGoal. You are on day ${_daysCompleted + 1} of '
          '$_totalDays. How did it go today? Did you work on your goal?';
    }

    _phase = 1;
    return 'Welcome to Goal Buddy! I am here to help you set a goal and '
        'check in every day to see how you are doing. What is something '
        'you would like to get better at this week? It could be anything, '
        'like reading more, being kind, or practicing a skill.';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      return await end();
    }

    switch (_phase) {
      case 1:
        // Child is stating their goal
        _pendingGoal = childSaid.trim();
        _phase = 2;
        return 'So your goal is: $_pendingGoal. That sounds like a wonderful '
            'goal! Shall I set that as your goal for the week?';

      case 2:
        // Confirming the goal
        if (_isAffirmative(lower)) {
          _currentGoal = _pendingGoal;
          _daysCompleted = 0;
          await _saveState();
          _phase = 3;
          _score += 20;
          return 'Your goal is set! I will check in with you every day for '
              '$_totalDays days. You got this! Remember, even small steps '
              'count. Come talk to me tomorrow and tell me how it went!';
        } else {
          _phase = 1;
          return 'No problem! What would you like your goal to be instead?';
        }

      case 3:
        // Daily check-in response
        _checkIns++;
        if (_isAffirmative(lower) || _containsProgress(lower)) {
          _daysCompleted++;
          _score += 15;
          await _saveState();
          await _recordCheckin();
          _phase = 4;

          if (_daysCompleted >= _totalDays) {
            _active = false;
            return 'You did it! $_totalDays days of working on your goal! '
                'That shows incredible dedication. Your goal was: '
                '$_currentGoal. You should be very proud of yourself! '
                'Score: $_score points!';
          }

          return '${_randomEncouragement()} That is $_daysCompleted out of '
              '$_totalDays days done. What is one thing you did today '
              'toward your goal?';
        } else {
          _score += 5;
          await _recordCheckin();
          _phase = 4;
          return 'That is okay! Some days are harder than others. '
              'The important thing is that you came back to check in. '
              'What could help you work on it tomorrow?';
        }

      case 4:
        // Follow-up after check-in
        _score += 5;
        _active = false;
        return 'Thank you for sharing! Keep working on your goal: '
            '$_currentGoal. You are on day ${_daysCompleted} of '
            '$_totalDays. Come back tomorrow for another check-in! '
            'Score: $_score points.';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_currentGoal.isEmpty) {
      return 'Come back anytime to set a goal! Having a goal gives you '
          'something to work toward every day.';
    }
    return 'Great check-in! Your goal: $_currentGoal. '
        '$_daysCompleted of $_totalDays days done. '
        'Score: $_score points. See you next time!';
  }

  // -- Persistence --

  Future<void> _loadState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _currentGoal = prefs.getString(_prefKeyGoal) ?? '';
      _daysCompleted = prefs.getInt(_prefKeyDays) ?? 0;
    } catch (_) {
      _currentGoal = '';
      _daysCompleted = 0;
    }
  }

  Future<void> _saveState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_prefKeyGoal, _currentGoal);
      await prefs.setInt(_prefKeyDays, _daysCompleted);
    } catch (_) {
      // Silently fail; non-critical.
    }
  }

  Future<bool> _hasCheckedInToday() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final lastCheckin = prefs.getString(_prefKeyLastCheckin);
      if (lastCheckin == null) return false;
      final today = DateTime.now().toIso8601String().substring(0, 10);
      return lastCheckin == today;
    } catch (_) {
      return false;
    }
  }

  Future<void> _recordCheckin() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final today = DateTime.now().toIso8601String().substring(0, 10);
      await prefs.setString(_prefKeyLastCheckin, today);
    } catch (_) {
      // Silently fail.
    }
  }

  // -- Helpers --

  String _randomEncouragement() {
    const phrases = [
      'Fantastic!',
      'That is wonderful!',
      'You are doing great!',
      'Way to go!',
      'Keep it up!',
      'I am proud of you!',
      'That is real progress!',
      'Awesome work!',
    ];
    return phrases[_rng.nextInt(phrases.length)];
  }

  bool _isAffirmative(String text) {
    const words = ['yes', 'yeah', 'sure', 'okay', 'ok', 'yep', 'yup',
      'absolutely', 'definitely', 'of course', 'i did', 'done',
      'हाँ', 'हां', 'अवुनु', 'అవును'];
    return words.any((w) => text.contains(w));
  }

  bool _containsProgress(String text) {
    const words = ['worked', 'practiced', 'tried', 'finished', 'completed',
      'did it', 'made progress', 'got better', 'read', 'studied'];
    return words.any((w) => text.contains(w));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough',
      'bye', 'end'];
    return quitWords.any((w) => text.contains(w));
  }
}
