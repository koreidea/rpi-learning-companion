import 'dart:convert';
import 'dart:math';

import 'package:shared_preferences/shared_preferences.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// HabitTracker: 21-day habit building challenge.
///
/// Teaches: discipline, consistency, self-monitoring, perseverance.
///
/// Flow:
/// 1. If no habit: help child pick a habit to build.
/// 2. Daily check-in: did you do your habit today?
/// 3. Track streak and celebrate milestones (3, 7, 14, 21 days).
/// 4. At 21 days: celebrate habit completion.
///
/// Persists habit and streak via SharedPreferences.
class HabitTracker extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _score = 0;

  /// 0=check state, 1=choosing habit, 2=confirming, 3=daily check-in,
  /// 4=follow-up
  int _phase = 0;

  String _pendingHabit = '';
  String _currentHabit = '';
  int _streak = 0;
  List<String> _completedDates = [];

  static const int _goalDays = 21;
  static const String _prefKeyHabit = 'habit_tracker_habit';
  static const String _prefKeyStreak = 'habit_tracker_streak';
  static const String _prefKeyDates = 'habit_tracker_dates';

  static const List<String> _habitSuggestions = [
    'reading for 10 minutes',
    'drinking a glass of water first thing in the morning',
    'saying something kind to someone',
    'tidying your desk before bed',
    'doing 10 jumping jacks',
    'writing one sentence in a journal',
    'practicing deep breathing for 1 minute',
    'drawing something small every day',
  ];

  HabitTracker();

  @override
  String get id => 'habit_tracker';

  @override
  String get name => 'Habit Tracker';

  @override
  String get category => 'self-direction';

  @override
  String get description =>
      'Build a new habit with a 21-day challenge and daily check-ins.';

  @override
  List<String> get skills =>
      ['discipline', 'consistency', 'self-monitoring', 'perseverance'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.selfDirection;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'habit tracker',
          'build a habit',
          'habit challenge',
          'twenty one days',
          'daily habit',
        ],
        'hi': [
          'आदत ट्रैकर',
          'आदत बनाओ',
          'रोज़ का काम',
        ],
        'te': [
          'అలవాటు ట్రాకర్',
          'అలవాటు పెట్టు',
          'రోజువారీ పని',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_currentHabit.isEmpty) return 'No habit set yet.';
    return 'Habit: $_currentHabit. Streak: $_streak of $_goalDays days. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _score = 0;
    _phase = 0;

    await _loadState();

    if (_currentHabit.isNotEmpty) {
      if (_streak >= _goalDays) {
        final oldHabit = _currentHabit;
        _currentHabit = '';
        _streak = 0;
        _completedDates = [];
        await _saveState();
        _phase = 1;
        return 'Amazing! You completed $_goalDays days of $oldHabit! '
            'That habit is now a part of who you are. '
            'Would you like to start a new habit challenge?';
      }

      final checkedInToday = _hasCheckedInToday();
      if (checkedInToday) {
        return 'Welcome back! Your habit is: $_currentHabit. '
            'You are on day $_streak of $_goalDays and you have already '
            'checked in today. ${_milestoneMessage()} '
            'Keep going, you are building something great!';
      }

      _phase = 3;
      return 'Welcome back to your habit challenge! Your habit is: '
          '$_currentHabit. You are on a $_streak day streak. '
          'Did you do your habit today?';
    }

    _phase = 1;
    final suggestion =
        _habitSuggestions[_rng.nextInt(_habitSuggestions.length)];
    return 'Welcome to the 21-Day Habit Challenge! It takes about 21 days '
        'to build a new habit. What habit would you like to build? '
        'For example, you could try $suggestion. '
        'What sounds good to you?';
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
        _pendingHabit = childSaid.trim();
        _phase = 2;
        return 'So you want to build the habit of: $_pendingHabit. '
            'For $_goalDays days in a row. Does that sound right?';

      case 2:
        if (_isAffirmative(lower)) {
          _currentHabit = _pendingHabit;
          _streak = 0;
          _completedDates = [];
          await _saveState();
          _score += 15;
          _active = false;
          return 'Your habit challenge starts now! Every day, try to do: '
              '$_currentHabit. Come talk to me each day to check in. '
              'Day 1 starts tomorrow! I believe in you!';
        } else {
          _phase = 1;
          return 'No problem! What habit would you like to try instead?';
        }

      case 3:
        if (_isAffirmative(lower)) {
          _streak++;
          _completedDates.add(
            DateTime.now().toIso8601String().substring(0, 10),
          );
          await _saveState();
          _score += 10;
          _phase = 4;

          if (_streak >= _goalDays) {
            _active = false;
            return 'You did it! $_goalDays days of $_currentHabit! '
                'This is a huge accomplishment. You have proven that you '
                'can stick with something and build it into your life. '
                'Score: $_score points! You are incredible!';
          }

          return '${_randomCelebration()} That is $_streak out of '
              '$_goalDays days! ${_milestoneMessage()} '
              'How did it feel doing it today?';
        } else {
          _score += 3;
          _phase = 4;
          return 'That is okay! Missing a day happens to everyone. '
              'The most important thing is to try again tomorrow. '
              'Your streak is still at $_streak days. '
              'What got in the way today?';
        }

      case 4:
        _score += 5;
        _active = false;
        return 'Thank you for checking in! Your habit: $_currentHabit. '
            'Streak: $_streak of $_goalDays days. '
            'Score: $_score points. See you tomorrow!';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_currentHabit.isEmpty) {
      return 'Come back when you are ready to start a habit challenge!';
    }
    return 'Keep going with your habit: $_currentHabit! '
        'Streak: $_streak of $_goalDays days. '
        'Score: $_score points. See you next time!';
  }

  // -- Persistence --

  Future<void> _loadState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _currentHabit = prefs.getString(_prefKeyHabit) ?? '';
      _streak = prefs.getInt(_prefKeyStreak) ?? 0;
      final datesJson = prefs.getString(_prefKeyDates);
      if (datesJson != null) {
        _completedDates = List<String>.from(json.decode(datesJson));
      } else {
        _completedDates = [];
      }
    } catch (_) {
      _currentHabit = '';
      _streak = 0;
      _completedDates = [];
    }
  }

  Future<void> _saveState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_prefKeyHabit, _currentHabit);
      await prefs.setInt(_prefKeyStreak, _streak);
      await prefs.setString(_prefKeyDates, json.encode(_completedDates));
    } catch (_) {
      // Silently fail; non-critical.
    }
  }

  bool _hasCheckedInToday() {
    final today = DateTime.now().toIso8601String().substring(0, 10);
    return _completedDates.contains(today);
  }

  String _milestoneMessage() {
    if (_streak == 3) {
      return 'Three days in a row! You are off to a great start!';
    } else if (_streak == 7) {
      return 'One whole week! You are building real consistency!';
    } else if (_streak == 14) {
      return 'Two weeks! You are more than halfway there!';
    } else if (_streak == 21) {
      return 'Twenty-one days! You have built a real habit!';
    }
    return '';
  }

  String _randomCelebration() {
    const phrases = [
      'Fantastic!',
      'Well done!',
      'You are amazing!',
      'Keep it up!',
      'Brilliant!',
      'That is dedication!',
      'You showed up today!',
      'Wonderful!',
    ];
    return phrases[_rng.nextInt(phrases.length)];
  }

  bool _isAffirmative(String text) {
    const words = ['yes', 'yeah', 'sure', 'okay', 'ok', 'yep', 'i did',
      'done', 'did it', 'completed',
      'हाँ', 'हां', 'అవును'];
    return words.any((w) => text.contains(w));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'finish', 'enough', 'bye'];
    return quitWords.any((w) => text.contains(w));
  }
}
