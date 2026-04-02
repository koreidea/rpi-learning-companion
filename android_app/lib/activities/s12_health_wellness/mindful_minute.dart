import 'dart:async';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_types.dart';

/// Breathing technique definition.
class _BreathingTechnique {
  final String name;
  final String description;
  final int inhaleSeconds;
  final int holdSeconds;
  final int exhaleSeconds;
  final int cycles;

  const _BreathingTechnique({
    required this.name,
    required this.description,
    required this.inhaleSeconds,
    required this.holdSeconds,
    required this.exhaleSeconds,
    required this.cycles,
  });
}

/// Mindful Minute: guided breathing exercise with timing.
///
/// Teaches: mindfulness, self-regulation, body awareness, calm focus.
///
/// Flow:
/// 1. Bot introduces the breathing technique.
/// 2. Guides through each breath cycle with counting.
/// 3. After all cycles, asks how the child feels.
/// 4. Positive reinforcement about the power of breathing.
///
/// Timer tracks breath cycles, not clock time. Uses face state changes
/// to show expanding/contracting visually.
class MindfulMinute extends TimerActivity {
  bool _active = false;
  bool _paused = false;
  int _currentCycle = 0;
  int _techniqueIndex = 0;
  Duration _elapsed = Duration.zero;
  int _score = 0;

  /// 0=intro, 1=breathing cycles, 2=feeling check, 3=done
  int _phase = 0;

  static const List<_BreathingTechnique> _techniques = [
    _BreathingTechnique(
      name: 'Belly Breathing',
      description: 'Put your hands on your belly. When you breathe in, feel your belly grow like a balloon. When you breathe out, feel it shrink back down.',
      inhaleSeconds: 4,
      holdSeconds: 0,
      exhaleSeconds: 4,
      cycles: 5,
    ),
    _BreathingTechnique(
      name: 'Box Breathing',
      description: 'We are going to breathe in a square pattern. Breathe in, hold, breathe out, hold. Like drawing a box with your breath!',
      inhaleSeconds: 4,
      holdSeconds: 4,
      exhaleSeconds: 4,
      cycles: 4,
    ),
    _BreathingTechnique(
      name: 'Four Seven Eight Breathing',
      description: 'This one is special. We breathe in for 4, hold for 7, and breathe out slowly for 8. It makes you super calm!',
      inhaleSeconds: 4,
      holdSeconds: 7,
      exhaleSeconds: 8,
      cycles: 3,
    ),
    _BreathingTechnique(
      name: 'Flower Breathing',
      description: 'Pretend you are smelling a beautiful flower. Breathe in through your nose nice and slow, then breathe out through your mouth like you are blowing a dandelion!',
      inhaleSeconds: 3,
      holdSeconds: 1,
      exhaleSeconds: 5,
      cycles: 5,
    ),
  ];

  MindfulMinute();

  @override
  String get id => 'health_mindful_minute';

  @override
  String get name => 'Mindful Minute';

  @override
  String get category => 'wellness';

  @override
  String get description =>
      'Guided breathing exercises to help you feel calm and focused.';

  @override
  List<String> get skills => [
        'mindfulness',
        'self-regulation',
        'body awareness',
        'calm focus',
      ];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.healthWellness;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'mindful minute',
          'breathing exercise',
          'calm down',
          'help me relax',
          'deep breathing',
          'calm breathing',
        ],
        'hi': [
          'शांत हो जाओ',
          'गहरी सांस',
          'सांस का अभ्यास',
        ],
        'te': [
          'ప్రశాంతంగా',
          'లోతు శ్వాస',
          'శ్వాస వ్యాయామం',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  Duration get totalDuration {
    final t = _techniques[_techniqueIndex];
    final cycleTime = t.inhaleSeconds + t.holdSeconds + t.exhaleSeconds;
    return Duration(seconds: cycleTime * t.cycles);
  }

  @override
  Duration get elapsed => _elapsed;

  @override
  bool get isPaused => _paused;

  @override
  String get progressSummary {
    final t = _techniques[_techniqueIndex];
    return 'Cycle $_currentCycle of ${t.cycles}. Score: $_score points.';
  }

  @override
  Future<void> pause() async {
    _paused = true;
  }

  @override
  Future<void> resume() async {
    _paused = false;
  }

  @override
  void onTimerTick(Duration remaining) {
    _elapsed = totalDuration - remaining;
  }

  @override
  Future<String> onTimerComplete() async {
    return 'All breathing cycles complete!';
  }

  @override
  Future<String> start() async {
    _active = true;
    _paused = false;
    _currentCycle = 0;
    _elapsed = Duration.zero;
    _score = 0;
    _phase = 0;
    _techniqueIndex = DateTime.now().millisecondsSinceEpoch % _techniques.length;

    final t = _techniques[_techniqueIndex];
    return 'Let us do a mindful minute together. We are going to try '
        '${t.name}. ${t.description} We will do ${t.cycles} breaths. '
        'Close your eyes if you want. Are you ready?';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    final t = _techniques[_techniqueIndex];

    switch (_phase) {
      case 0:
        // Start first breathing cycle
        _phase = 1;
        _currentCycle = 1;
        return _buildBreathCycleText(t);

      case 1:
        // Continue breathing cycles
        _currentCycle++;
        _score += 10;
        if (_currentCycle > t.cycles) {
          _phase = 2;
          return 'Beautiful! You did all ${t.cycles} breathing cycles! '
              'How do you feel now? Calmer? More relaxed?';
        }
        return _buildBreathCycleText(t);

      case 2:
        // Feeling check response
        _phase = 3;
        _score += 20;
        _active = false;
        return 'That is the power of breathing! Whenever you feel worried, '
            'angry, or just need a moment, you can always do this breathing '
            'exercise. Your body and mind will thank you! '
            'Score: $_score points! Great mindfulness!';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_currentCycle == 0) {
      return 'That is okay. Whenever you need a calm moment, I am here '
          'to breathe with you.';
    }
    return 'You did $_currentCycle breathing cycles! Remember, breathing '
        'is a superpower that you always have with you. Score: $_score points!';
  }

  String _buildBreathCycleText(_BreathingTechnique t) {
    final buffer = StringBuffer();
    buffer.write('Breath $_currentCycle of ${t.cycles}. ');
    buffer.write('Breathe in. ');
    for (int i = 2; i <= t.inhaleSeconds; i++) {
      buffer.write('$i. ');
    }
    if (t.holdSeconds > 0) {
      buffer.write('Hold. ');
      for (int i = 2; i <= t.holdSeconds; i++) {
        buffer.write('$i. ');
      }
    }
    buffer.write('Breathe out slowly. ');
    for (int i = 2; i <= t.exhaleSeconds; i++) {
      buffer.write('$i. ');
    }
    buffer.write('Good.');
    return buffer.toString();
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop', 'quit', 'exit', "i'm done", 'no more', 'finish', 'end',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
