import 'dart:async';
import 'dart:math';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// PomodoroBuddy: focus timer with 25-minute work + 5-minute break cycles.
///
/// Teaches: focus, time management, work-break balance, self-regulation.
///
/// Flow:
/// 1. Bot asks what the child wants to focus on.
/// 2. Starts a 25-minute focus timer (or shorter for younger kids).
/// 3. Checks in periodically with encouragement.
/// 4. Break time: 5-minute guided break.
/// 5. Option to do another cycle.
///
/// Note: Since this is voice-based, the timer is managed internally and
/// the activity provides check-ins at appropriate intervals. The actual
/// timing relies on the caller invoking processResponse at intervals.
class PomodoroBuddy extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _cyclesCompleted = 0;
  int _score = 0;
  String _focusTask = '';

  /// 0=pick task, 1=focus mode, 2=check-in, 3=break time,
  /// 4=break done, 5=another cycle?
  int _phase = 0;
  int _checkInsThisCycle = 0;
  DateTime? _cycleStartTime;

  static const int _focusMinutes = 25;
  static const int _breakMinutes = 5;
  static const int _checkInsPerCycle = 3;

  PomodoroBuddy();

  @override
  String get id => 'pomodoro_buddy';

  @override
  String get name => 'Pomodoro Buddy';

  @override
  String get category => 'time-management';

  @override
  String get description =>
      'Stay focused with timed work sessions and fun breaks.';

  @override
  List<String> get skills =>
      ['focus', 'time management', 'work-break balance', 'self-regulation'];

  @override
  int get minAge => 6;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.timeManagement;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'pomodoro',
          'focus timer',
          'help me focus',
          'study timer',
          'work timer',
          'pomodoro buddy',
        ],
        'hi': [
          'पोमोडोरो',
          'फोकस टाइमर',
          'पढ़ाई टाइमर',
        ],
        'te': [
          'పొమొడోరో',
          'ఫోకస్ టైమర్',
          'చదువు టైమర్',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_cyclesCompleted == 0 && _focusTask.isEmpty) {
      return 'No focus sessions yet.';
    }
    return '$_cyclesCompleted Pomodoro cycle${_cyclesCompleted != 1 ? 's' : ''} '
        'completed. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _cyclesCompleted = 0;
    _score = 0;
    _focusTask = '';
    _phase = 0;
    _checkInsThisCycle = 0;

    return 'Welcome to Pomodoro Buddy! The Pomodoro technique helps you '
        'focus by working for $_focusMinutes minutes, then taking a '
        '$_breakMinutes minute break. What would you like to work on? '
        'It could be homework, reading, drawing, or anything!';
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
        // Child picked a task
        _focusTask = childSaid.trim();
        _phase = 1;
        _checkInsThisCycle = 0;
        _cycleStartTime = DateTime.now();
        _score += 10;
        return 'Great! Your focus task is: $_focusTask. '
            'Your $_focusMinutes minute timer starts now! '
            'I will check in with you a few times to see how it is going. '
            'Go focus! You can talk to me when you want a check-in.';

      case 1:
        // Focus mode check-in
        _checkInsThisCycle++;
        _score += 5;

        final elapsed = _cycleStartTime != null
            ? DateTime.now().difference(_cycleStartTime!).inMinutes
            : 0;

        if (_checkInsThisCycle >= _checkInsPerCycle || elapsed >= _focusMinutes) {
          // Time for a break
          _phase = 3;
          _score += 15;
          return '${_randomFocusPraise()} Your focus time is done! '
              'You worked on $_focusTask. Now it is break time! '
              'Take $_breakMinutes minutes to rest. You could stretch, '
              'get water, or just relax. Tell me when your break is done!';
        }

        return '${_randomCheckIn()} '
            'About ${_focusMinutes - elapsed} minutes left. Keep going!';

      case 3:
        // Break is done
        _cyclesCompleted++;
        _phase = 5;
        _score += 10;
        return 'Welcome back from your break! You have completed '
            '$_cyclesCompleted Pomodoro '
            'cycle${_cyclesCompleted != 1 ? 's' : ''}. '
            'Would you like to do another focus session?';

      case 5:
        // Another cycle?
        if (_isAffirmative(lower)) {
          _phase = 0;
          _focusTask = '';
          return 'What would you like to focus on this time? '
              'Same thing or something different?';
        }
        return await end();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_cyclesCompleted == 0 && _checkInsThisCycle == 0) {
      return 'Come back when you need help focusing! The Pomodoro technique '
          'is a powerful way to get things done.';
    }

    if (_checkInsThisCycle > 0 && _phase == 1) {
      _cyclesCompleted++;
    }

    return 'Great focus session! You completed $_cyclesCompleted Pomodoro '
        'cycle${_cyclesCompleted != 1 ? 's' : ''}. '
        'Score: $_score points! Focused work plus good breaks is the '
        'secret to getting things done!';
  }

  String _randomCheckIn() {
    const phrases = [
      'Quick check-in! You are doing great.',
      'How is it going? You are staying focused!',
      'Checking in! Keep up the great work.',
      'You are doing awesome! Stay with it.',
      'Great focus! You are making progress.',
    ];
    return phrases[_rng.nextInt(phrases.length)];
  }

  String _randomFocusPraise() {
    const phrases = [
      'Incredible focus!',
      'You stayed with it!',
      'That was excellent concentration!',
      'What amazing discipline!',
      'You are a focus champion!',
    ];
    return phrases[_rng.nextInt(phrases.length)];
  }

  bool _isAffirmative(String text) {
    const words = ['yes', 'yeah', 'sure', 'okay', 'ok', 'another',
      'one more', 'again', 'let us go',
      'हाँ', 'और', 'అవును'];
    return words.any((w) => text.contains(w));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough',
      'no', 'no thanks'];
    return quitWords.any((w) => text.contains(w));
  }
}
