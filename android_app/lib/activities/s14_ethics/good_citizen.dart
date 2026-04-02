import 'dart:math';

import 'package:shared_preferences/shared_preferences.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_types.dart';

/// Good Citizen: weekly micro civic duty assignments with follow-up.
///
/// Teaches: civic responsibility, kindness, community awareness,
/// habit building, empathy.
///
/// Flow:
/// 1. If a task was previously assigned, ask for follow-up first.
/// 2. Assign a new micro civic duty task.
/// 3. Celebrate completions and track streaks.
/// 4. Persists current task and completion status.
class GoodCitizen extends EpisodicActivity {
  final Random _rng = Random();

  bool _active = false;
  int _episode = 1;
  bool _todayDone = false;
  int _streak = 0;
  int _score = 0;

  String? _currentTask;
  bool _taskCompleted = false;

  /// 0=follow-up (if task exists), 1=assign new task, 2=done
  int _phase = 0;

  static const String _prefsTaskKey = 'good_citizen_task';
  static const String _prefsCompletedKey = 'good_citizen_completed';
  static const String _prefsStreakKey = 'good_citizen_streak';
  static const String _prefsDateKey = 'good_citizen_date';

  static const List<String> _tasks = [
    'Help someone today without being asked. It can be small, like holding a door or carrying something.',
    'Pick up 5 pieces of litter from a park or street. Every piece makes the world cleaner!',
    'Say thank you to 3 different people today. Really mean it!',
    'Share something with a friend or sibling today. It could be food, a toy, or just your time.',
    'Learn one interesting thing about your neighborhood. Maybe the name of a street or a tree.',
    'Give a sincere compliment to 3 people today. Tell them something nice you noticed.',
    'Help with a chore at home without being asked. Surprise your family!',
    'Smile at 5 strangers today. A smile can make someone\'s whole day better!',
    'Write a thank you note or draw a thank you picture for someone who helped you.',
    'Teach something you know to someone younger. It could be a game, a song, or a fact.',
    'Water a plant or feed a bird today. Taking care of nature is a civic duty!',
    'Listen carefully to someone today without interrupting. Really hear what they say.',
    'Find something that can be recycled in your house and put it in the right bin.',
    'Be extra kind to someone who seems sad or lonely today.',
    'Clean up a space in your house that is messy. A clean space makes everyone happy!',
  ];

  GoodCitizen();

  @override
  String get id => 'ethics_good_citizen';

  @override
  String get name => 'Good Citizen';

  @override
  String get category => 'ethics';

  @override
  String get description =>
      'Weekly micro civic duty tasks to make the world a little better.';

  @override
  List<String> get skills => [
        'civic responsibility',
        'kindness',
        'community awareness',
        'empathy',
      ];

  @override
  int get minAge => 4;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.ethics;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'good citizen',
          'civic duty',
          'be a good citizen',
          'help the community',
          'kindness task',
        ],
        'hi': [
          'अच्छा नागरिक',
          'समुदाय की मदद',
          'दया का काम',
        ],
        'te': [
          'మంచి పౌరుడు',
          'సమాజ సేవ',
          'దయ పని',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  int get currentEpisode => _episode;

  @override
  String get episodeKey => _prefsTaskKey;

  @override
  bool get hasNewEpisode => !_todayDone;

  @override
  String get progressSummary {
    return 'Good citizen streak: $_streak days. Score: $_score points.';
  }

  @override
  Future<void> saveProgress() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsTaskKey, _currentTask ?? '');
    await prefs.setBool(_prefsCompletedKey, _taskCompleted);
    await prefs.setInt(_prefsStreakKey, _streak);
    await prefs.setString(
      _prefsDateKey,
      DateTime.now().toIso8601String().substring(0, 10),
    );
  }

  @override
  Future<void> loadProgress() async {
    final prefs = await SharedPreferences.getInstance();
    _currentTask = prefs.getString(_prefsTaskKey);
    _taskCompleted = prefs.getBool(_prefsCompletedKey) ?? false;
    _streak = prefs.getInt(_prefsStreakKey) ?? 0;
    final dateStr = prefs.getString(_prefsDateKey);
    final today = DateTime.now().toIso8601String().substring(0, 10);
    _todayDone = dateStr == today;
    if (_currentTask?.isEmpty ?? true) _currentTask = null;
  }

  @override
  Future<String> start() async {
    await loadProgress();
    _active = true;
    _score = 0;

    // If there is an existing task that was not followed up on
    if (_currentTask != null && !_taskCompleted) {
      _phase = 0;
      return 'Welcome back, good citizen! Last time your task was: '
          '$_currentTask Did you do it? Tell me about it!';
    }

    // No pending task, assign a new one
    _phase = 1;
    return _assignNewTask();
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
        // Follow-up on previous task
        _phase = 1;
        if (_didComplete(lower)) {
          _streak++;
          _score += 20;
          _taskCompleted = true;
          await saveProgress();

          String streakMessage = '';
          if (_streak >= 7) {
            streakMessage = ' You have a $_streak day streak! You are an incredible citizen!';
          } else if (_streak >= 3) {
            streakMessage = ' $_streak day streak! Keep it going!';
          }

          return 'That is wonderful! You made a real difference! $streakMessage '
              'Ready for a new task?';
        } else {
          return 'That is okay! Sometimes we forget or do not get the chance. '
              'The important thing is you are thinking about it. '
              'Want a new task?';
        }

      case 1:
        // After hearing about readiness, assign task
        _phase = 2;
        return _assignNewTask();

      case 2:
        // After task is assigned
        _active = false;
        _score += 5;
        await saveProgress();
        return 'Great! Remember your task: $_currentTask '
            'I will ask you about it next time! Good luck, good citizen! '
            'Score: $_score points.';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    await saveProgress();
    if (_currentTask != null) {
      return 'Remember your good citizen task: $_currentTask '
          'I will ask you about it next time! Streak: $_streak days.';
    }
    return 'Being a good citizen is about small acts of kindness. '
        'Come back for a new task anytime!';
  }

  String _assignNewTask() {
    _currentTask = _tasks[_rng.nextInt(_tasks.length)];
    _taskCompleted = false;
    return 'Here is your good citizen task! $_currentTask '
        'Can you do this before we talk next time?';
  }

  bool _didComplete(String lower) {
    const yesWords = [
      'yes', 'yeah', 'yep', 'i did', 'done', 'completed', 'finished',
      'sure', 'of course',
    ];
    return yesWords.any((w) => lower.contains(w));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'finish', 'end'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
