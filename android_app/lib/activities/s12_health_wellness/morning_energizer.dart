import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../bluetooth/car_chassis.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// A single exercise with instructions and counting dialog.
class _Exercise {
  final String name;
  final String instruction;
  final String countText;
  final bool canUseCar;

  const _Exercise({
    required this.name,
    required this.instruction,
    required this.countText,
    this.canUseCar = false,
  });
}

/// Morning Energizer: a 5-minute guided exercise routine for kids.
///
/// Teaches: physical coordination, counting, following instructions,
/// body awareness, healthy habits.
///
/// Flow:
/// 1. Bot introduces the exercise session.
/// 2. Shuffles from 12+ exercises and picks 5.
/// 3. For each: instruction, counting/cheering, celebration.
/// 4. Encouraging summary at the end.
///
/// If a car chassis is connected, it dances along during compatible exercises.
class MorningEnergizer extends Activity {
  final CarChassis? _car;
  final Random _rng = Random();

  bool _active = false;
  int _exercisesCompleted = 0;
  int _score = 0;
  int _currentIndex = -1;
  bool _waitingForReady = false;

  final List<int> _selectedExercises = [];
  int _currentSelectedPos = 0;

  static const int _exercisesPerSession = 5;

  static const List<_Exercise> _exercises = [
    _Exercise(
      name: 'Jumping Jacks',
      instruction: 'Let us do 5 jumping jacks! Spread your arms and legs wide, then jump!',
      countText: 'Jump! 1! 2! 3! 4! 5! Woohoo, great jumping!',
      canUseCar: true,
    ),
    _Exercise(
      name: 'Toe Touches',
      instruction: 'Touch your toes 5 times! Bend down nice and slow.',
      countText: 'Reach down! 1! 2! 3! 4! 5! Amazing flexibility!',
    ),
    _Exercise(
      name: 'Spin Around',
      instruction: 'Spin around 3 times like a tornado! Not too fast!',
      countText: 'Spin! 1! 2! 3! Whee! Are you dizzy?',
      canUseCar: true,
    ),
    _Exercise(
      name: 'March in Place',
      instruction: 'March in place like a soldier! Lift those knees high for 10 seconds!',
      countText: 'Left right left right! 1, 2, 3, 4, 5, 6, 7, 8, 9, 10! Great marching soldier!',
      canUseCar: true,
    ),
    _Exercise(
      name: 'Giraffe Stretch',
      instruction: 'Stretch up high like a giraffe! Reach your arms to the sky!',
      countText: 'Stretch stretch stretch! Higher! Higher! Touch the clouds! Wow, you are so tall!',
    ),
    _Exercise(
      name: 'Hop on One Foot',
      instruction: 'Hop on one foot! First your left foot, then your right!',
      countText: 'Left foot! Hop hop hop! Now right foot! Hop hop hop! Amazing balance!',
    ),
    _Exercise(
      name: 'Squats',
      instruction: 'Let us do 5 squats! Go down and up like you are sitting in an invisible chair!',
      countText: 'Down and up! 1! 2! 3! 4! 5! Your legs are super strong!',
    ),
    _Exercise(
      name: 'Arm Circles',
      instruction: 'Make big circles with your arms! Like a helicopter getting ready to fly!',
      countText: 'Circle circle circle! Forward forward! Now backward! Great arm circles!',
    ),
    _Exercise(
      name: 'Bunny Hops',
      instruction: 'Hop forward like a bunny rabbit! Keep your feet together!',
      countText: 'Hop! Hop! Hop! Hop! Hop! What a cute bunny you are!',
      canUseCar: true,
    ),
    _Exercise(
      name: 'Windmill Touches',
      instruction: 'Touch your left foot with your right hand, then switch! Like a windmill!',
      countText: 'Right hand to left foot! Switch! Left hand to right foot! Switch! 1! 2! 3! 4! 5! Super windmill!',
    ),
    _Exercise(
      name: 'Star Jumps',
      instruction: 'Jump up and spread your arms and legs like a star!',
      countText: 'Star jump! 1! 2! 3! 4! 5! You are a shining star!',
      canUseCar: true,
    ),
    _Exercise(
      name: 'Bear Crawl',
      instruction: 'Get on your hands and feet and walk like a big strong bear!',
      countText: 'Walk like a bear! Growl! Left paw right paw! You are a mighty bear!',
    ),
  ];

  MorningEnergizer({CarChassis? car}) : _car = car;

  @override
  String get id => 'health_morning_energizer';

  @override
  String get name => 'Morning Energizer';

  @override
  String get category => 'physical';

  @override
  String get description =>
      'A fun 5-minute guided exercise routine to start the day with energy.';

  @override
  List<String> get skills => [
        'physical coordination',
        'counting',
        'following instructions',
        'healthy habits',
      ];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 8;

  @override
  SkillId? get skillId => SkillId.healthWellness;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'morning exercise',
          'morning energizer',
          'wake up exercise',
          'let us exercise',
          'workout time',
          'get moving',
        ],
        'hi': [
          'सुबह की कसरत',
          'व्यायाम करो',
          'शरीर हिलाओ',
        ],
        'te': [
          'ఉదయం వ్యాయామం',
          'శరీరం కదుపు',
          'వ్యాయామం చేద్దాం',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_exercisesCompleted == 0) return 'No exercises done yet.';
    return '$_exercisesCompleted exercises completed. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _exercisesCompleted = 0;
    _score = 0;
    _waitingForReady = false;
    _active = true;
    _currentSelectedPos = 0;

    // Shuffle and pick exercises for this session
    final indices = List.generate(_exercises.length, (i) => i);
    indices.shuffle(_rng);
    _selectedExercises
      ..clear()
      ..addAll(indices.take(_exercisesPerSession));

    return 'Good morning superstar! It is exercise time! Let us get our body '
        'moving and feel amazing! We have $_exercisesPerSession fun exercises. '
        '${_presentNextExercise()}';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    if (_waitingForReady) {
      _waitingForReady = false;
      return await _doExercise();
    }

    if (_exercisesCompleted >= _exercisesPerSession) {
      return await end();
    }

    return _presentNextExercise();
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_exercisesCompleted == 0) {
      return 'No worries! We can exercise later. Moving our body is important!';
    }
    _score += 20;
    return 'What a workout! You did $_exercisesCompleted exercises and earned '
        '$_score points! Your body is energized and ready for an amazing day! '
        'Great job, champion!';
  }

  String _presentNextExercise() {
    if (_currentSelectedPos >= _selectedExercises.length) {
      return 'You did every exercise! You are a fitness champion!';
    }

    _currentIndex = _selectedExercises[_currentSelectedPos];
    _currentSelectedPos++;
    _waitingForReady = true;

    final exercise = _exercises[_currentIndex];
    final number = _exercisesCompleted + 1;
    return 'Exercise $number of $_exercisesPerSession: ${exercise.name}! '
        '${exercise.instruction} Ready?';
  }

  Future<String> _doExercise() async {
    final exercise = _exercises[_currentIndex];
    _exercisesCompleted++;
    _score += 15;

    // Car dance if connected and compatible
    final car = _car;
    if (car != null && car.connected && exercise.canUseCar) {
      _carDanceAlong();
    }

    final remaining = _exercisesPerSession - _exercisesCompleted;

    if (remaining <= 0) {
      return '${exercise.countText} That was our last exercise! You did it!';
    }

    return '${exercise.countText} $remaining more to go! Keep it up!';
  }

  void _carDanceAlong() {
    final car = _car;
    if (car == null || !car.connected) return;
    Future(() async {
      try {
        const d = Duration(milliseconds: 300);
        await car.spinLeft(speed: 150, duration: d);
        await car.spinRight(speed: 150, duration: d);
        await car.spinLeft(speed: 150, duration: d);
        await car.spinRight(speed: 150, duration: d);
        await car.stop();
      } catch (e) {
        debugPrint('[MorningEnergizer] Car dance error: $e');
      }
    });
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop', 'quit', 'exit', "i'm done", 'no more',
      'finish', 'end', 'enough',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
