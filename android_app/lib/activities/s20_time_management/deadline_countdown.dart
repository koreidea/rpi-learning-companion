import 'dart:convert';
import 'dart:math';

import 'package:shared_preferences/shared_preferences.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// DeadlineCountdown: backward planning for deadlines and big tasks.
///
/// Teaches: backward planning, task decomposition, deadline awareness,
/// project management.
///
/// Flow:
/// 1. Child describes what they need to finish and when.
/// 2. Bot helps break it into smaller steps.
/// 3. Assigns steps to days working backward from the deadline.
/// 4. Daily check-in to see if the step for today is done.
///
/// Persists deadline and steps via SharedPreferences.
class DeadlineCountdown extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _score = 0;

  /// 0=check state, 1=set task, 2=set deadline days, 3=break into steps,
  /// 4=confirm plan, 5=daily check-in, 6=check-in follow-up
  int _phase = 0;

  String _taskName = '';
  int _daysUntilDeadline = 0;
  List<String> _steps = [];
  int _currentStepIndex = 0;
  int _stepsCompleted = 0;

  static const String _prefKeyTask = 'deadline_task';
  static const String _prefKeyDays = 'deadline_days_remaining';
  static const String _prefKeySteps = 'deadline_steps';
  static const String _prefKeyCurrentStep = 'deadline_current_step';
  static const String _prefKeyStepsCompleted = 'deadline_steps_completed';

  DeadlineCountdown();

  @override
  String get id => 'deadline_countdown';

  @override
  String get name => 'Deadline Countdown';

  @override
  String get category => 'time-management';

  @override
  String get description =>
      'Plan backward from a deadline to break big tasks into small steps.';

  @override
  List<String> get skills => [
        'backward planning',
        'task decomposition',
        'deadline awareness',
        'project management',
      ];

  @override
  int get minAge => 7;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.timeManagement;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'deadline countdown',
          'plan for deadline',
          'backward planning',
          'big project',
          'break it down',
        ],
        'hi': [
          'डेडलाइन प्लान',
          'समय सीमा योजना',
          'बड़ा काम',
        ],
        'te': [
          'డెడ్లైన్ ప్లాన్',
          'గడువు ప్రణాళిక',
          'పెద్ద పని',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.senior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_taskName.isEmpty) return 'No deadline project set.';
    return 'Project: $_taskName. $_stepsCompleted of ${_steps.length} '
        'steps done. $_daysUntilDeadline days remaining. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _score = 0;

    await _loadState();

    if (_taskName.isNotEmpty && _steps.isNotEmpty) {
      if (_currentStepIndex >= _steps.length) {
        final oldTask = _taskName;
        await _clearState();
        _phase = 1;
        return 'You finished your deadline project: $oldTask! '
            'Amazing planning and execution! '
            'Would you like to set up a new project?';
      }

      _phase = 5;
      return 'Welcome back! Your project: $_taskName. '
          '$_daysUntilDeadline days left. '
          'Today\'s step is: ${_steps[_currentStepIndex]}. '
          'Have you worked on it?';
    }

    _phase = 1;
    return 'Welcome to Deadline Countdown! I help you take a big task '
        'with a deadline and break it into small daily steps. '
        'What is the project or task you need to finish?';
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
        // Set task name
        _taskName = childSaid.trim();
        _phase = 2;
        return 'Got it! Your project is: $_taskName. '
            'How many days do you have until it is due? '
            'Just tell me the number of days.';

      case 2:
        // Set deadline days
        final days = _extractNumber(lower);
        if (days == null || days < 1) {
          return 'I need a number of days. How many days until '
              '$_taskName is due?';
        }
        _daysUntilDeadline = days;
        _phase = 3;
        _score += 10;

        final stepsCount = _daysUntilDeadline > 7
            ? 7
            : _daysUntilDeadline > 1 ? _daysUntilDeadline : 2;

        return 'So you have $_daysUntilDeadline days to finish $_taskName. '
            'Let us break this into about $stepsCount smaller steps. '
            'Think about what needs to happen first, then second, and so on. '
            'What is the very first step?';

      case 3:
        // Adding steps
        _steps.add(childSaid.trim());
        _score += 5;

        final targetSteps = _daysUntilDeadline > 7
            ? 7
            : _daysUntilDeadline > 1 ? _daysUntilDeadline : 2;

        if (_steps.length >= targetSteps) {
          _phase = 4;
          return _summarizePlan();
        }

        return 'Step ${_steps.length} is: ${_steps.last}. '
            'What is step ${_steps.length + 1}? '
            'Or say "that is all" if you have listed all the steps.';

      case 4:
        // Confirm plan
        if (_isAffirmative(lower) || lower.contains('that is all') ||
            lower.contains('good') || lower.contains('save')) {
          _currentStepIndex = 0;
          _stepsCompleted = 0;
          await _saveState();
          _active = false;
          _score += 15;
          return 'Your deadline countdown plan is saved! '
              'You have ${_steps.length} steps over $_daysUntilDeadline days. '
              'Come back each day and I will tell you which step to work on. '
              'Score: $_score points. You have got this!';
        }

        // Add more steps
        if (lower.contains('that is all') || lower.contains('done')) {
          _currentStepIndex = 0;
          _stepsCompleted = 0;
          await _saveState();
          _active = false;
          _score += 15;
          return 'Plan saved! ${_steps.length} steps, '
              '$_daysUntilDeadline days. Come back tomorrow to start! '
              'Score: $_score points.';
        }

        _steps.add(childSaid.trim());
        _score += 5;
        return _summarizePlan();

      case 5:
        // Daily check-in
        if (_isAffirmative(lower)) {
          _stepsCompleted++;
          _currentStepIndex++;
          _daysUntilDeadline--;
          _score += 15;
          await _saveState();
          _phase = 6;

          if (_currentStepIndex >= _steps.length) {
            _active = false;
            return 'You finished all the steps for $_taskName! '
                'You planned it, broke it down, and completed it step by '
                'step. That is real project management! '
                'Score: $_score points!';
          }

          return '${_randomCelebration()} $_stepsCompleted of '
              '${_steps.length} steps done. $_daysUntilDeadline days left. '
              'Tomorrow\'s step will be: ${_steps[_currentStepIndex]}. '
              'You are right on track!';
        } else {
          _score += 3;
          _phase = 6;
          return 'That is okay! The step is still: '
              '${_steps[_currentStepIndex]}. '
              'Try to work on it before the end of the day. '
              'You are $_stepsCompleted of ${_steps.length} steps in. '
              'You can do this!';
        }

      case 6:
        // Follow-up
        _active = false;
        _score += 5;
        return 'Keep going with $_taskName! '
            '$_stepsCompleted of ${_steps.length} steps done. '
            '$_daysUntilDeadline days remaining. '
            'Score: $_score points. See you tomorrow!';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_taskName.isNotEmpty && _steps.isNotEmpty) {
      await _saveState();
      return 'Your deadline countdown for $_taskName is saved. '
          '$_stepsCompleted of ${_steps.length} steps done. '
          '$_daysUntilDeadline days left. Score: $_score points. '
          'Come back tomorrow!';
    }
    return 'Come back when you have a deadline to plan for! '
        'Breaking big tasks into small steps makes everything easier.';
  }

  String _summarizePlan() {
    final stepList = _steps
        .asMap()
        .entries
        .map((e) => 'Step ${e.key + 1}: ${e.value}')
        .join('. ');
    return 'Here is your plan: $stepList. '
        'That is ${_steps.length} steps over $_daysUntilDeadline days. '
        'Does this look good, or would you like to add more steps?';
  }

  // -- Persistence --

  Future<void> _loadState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _taskName = prefs.getString(_prefKeyTask) ?? '';
      _daysUntilDeadline = prefs.getInt(_prefKeyDays) ?? 0;
      _currentStepIndex = prefs.getInt(_prefKeyCurrentStep) ?? 0;
      _stepsCompleted = prefs.getInt(_prefKeyStepsCompleted) ?? 0;
      final stepsJson = prefs.getString(_prefKeySteps);
      if (stepsJson != null) {
        _steps = List<String>.from(json.decode(stepsJson));
      } else {
        _steps = [];
      }
    } catch (_) {
      _taskName = '';
      _steps = [];
      _daysUntilDeadline = 0;
      _currentStepIndex = 0;
      _stepsCompleted = 0;
    }
  }

  Future<void> _saveState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_prefKeyTask, _taskName);
      await prefs.setInt(_prefKeyDays, _daysUntilDeadline);
      await prefs.setString(_prefKeySteps, json.encode(_steps));
      await prefs.setInt(_prefKeyCurrentStep, _currentStepIndex);
      await prefs.setInt(_prefKeyStepsCompleted, _stepsCompleted);
    } catch (_) {
      // Silently fail.
    }
  }

  Future<void> _clearState() async {
    _taskName = '';
    _steps = [];
    _daysUntilDeadline = 0;
    _currentStepIndex = 0;
    _stepsCompleted = 0;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_prefKeyTask);
      await prefs.remove(_prefKeyDays);
      await prefs.remove(_prefKeySteps);
      await prefs.remove(_prefKeyCurrentStep);
      await prefs.remove(_prefKeyStepsCompleted);
    } catch (_) {
      // Silently fail.
    }
  }

  int? _extractNumber(String text) {
    final digitMatch = RegExp(r'\d+').firstMatch(text);
    if (digitMatch != null) {
      return int.tryParse(digitMatch.group(0)!);
    }
    const wordToNum = <String, int>{
      'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
      'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
      'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
      'fifteen': 15, 'twenty': 20, 'thirty': 30,
    };
    for (final entry in wordToNum.entries) {
      if (text.contains(entry.key)) return entry.value;
    }
    return null;
  }

  String _randomCelebration() {
    const phrases = [
      'Excellent!',
      'Well done!',
      'Great progress!',
      'You are on fire!',
      'Brilliant!',
      'Way to go!',
    ];
    return phrases[_rng.nextInt(phrases.length)];
  }

  bool _isAffirmative(String text) {
    const words = ['yes', 'yeah', 'sure', 'okay', 'ok', 'yep', 'i did',
      'done', 'finished', 'completed', 'did it',
      'हाँ', 'हां', 'అవును'];
    return words.any((w) => text.contains(w));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'finish', 'enough', 'bye'];
    return quitWords.any((w) => text.contains(w));
  }
}
