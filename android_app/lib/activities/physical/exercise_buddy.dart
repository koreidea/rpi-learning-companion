import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../bluetooth/car_chassis.dart';
import '../activity_base.dart';

/// A single exercise in the routine.
class _Exercise {
  final String name;
  final String instruction;
  final String countText; // Text to speak while counting
  final int duration; // Approximate seconds
  final bool canUseCar;

  const _Exercise({
    required this.name,
    required this.instruction,
    required this.countText,
    required this.duration,
    this.canUseCar = false,
  });
}

/// Exercise Buddy: physical activity with fun movement exercises.
///
/// Teaches: physical coordination, counting, following instructions, body awareness.
///
/// Flow:
/// 1. Bot introduces exercise time.
/// 2. Bot gives an exercise instruction (jumping jacks, hopping, etc.).
/// 3. Bot counts along with the child.
/// 4. Bot celebrates and moves to next exercise.
/// 5. 4-5 exercises total, about 3 minutes.
///
/// If car is connected, car dances along during some exercises.
class ExerciseBuddy extends Activity {
  final CarChassis? _car;
  final Random _rng = Random();

  bool _active = false;
  int _exercisesCompleted = 0;
  int _score = 0;
  int _currentIndex = -1;
  bool _waitingForReady = false;
  final List<int> _usedExercises = [];

  static const int _totalExercises = 5;

  static const List<_Exercise> _exercises = [
    _Exercise(
      name: 'Jumping Jacks',
      instruction: "Let's do 5 jumping jacks together! Spread your arms and legs wide, then jump! Ready?",
      countText: "Jump! 1! 2! 3! 4! 5! Great job!",
      duration: 10,
      canUseCar: true,
    ),
    _Exercise(
      name: 'Hop on One Foot',
      instruction: "Now let's hop on one foot! Can you hop like a bunny rabbit? Ready?",
      countText: "Hop! Hop! Hop! Hop! Hop! Amazing hopping!",
      duration: 8,
    ),
    _Exercise(
      name: 'Touch Your Toes',
      instruction: "Time to stretch! Bend down and try to touch your toes. Stretch like a giraffe bending down to drink water! Ready?",
      countText: "Reach down, down, down! Touch those toes! Hold it! 1, 2, 3! And stand up! Great stretching!",
      duration: 8,
    ),
    _Exercise(
      name: 'Spin Around',
      instruction: "Let's spin around like a helicopter! But not too fast, we don't want to get dizzy! Ready?",
      countText: "Spin! Round and round! Wheee! 1, 2, 3! Now stop! Are you dizzy?",
      duration: 6,
      canUseCar: true,
    ),
    _Exercise(
      name: 'Freeze Dance',
      instruction: "Time for freeze dance! Dance around the room, and when I say freeze, you stop like a statue! Ready?",
      countText: "Dance dance dance! Move your arms! Shake your body! And, FREEZE! Are you frozen like a statue? Great job! You can move again!",
      duration: 12,
      canUseCar: true,
    ),
    _Exercise(
      name: 'Simon Says',
      instruction: "Let's play a quick Simon Says! I will tell you what to do. Ready?",
      countText: "Simon says, clap your hands! Simon says, wave hello! Simon says, touch your nose! "
          "Simon says, stand on one foot! Now jump! Oops, Simon did not say jump! Just kidding, you did great!",
      duration: 15,
    ),
    _Exercise(
      name: 'Bear Walk',
      instruction: "Can you walk like a bear? Get on your hands and feet and walk forward! Bears are big and strong! Ready?",
      countText: "Walk like a bear! Left paw, right paw! Growl! You are a big strong bear! Great bear walking!",
      duration: 10,
    ),
    _Exercise(
      name: 'Frog Jumps',
      instruction: "Time to jump like a frog! Squat down low and jump forward! Ribbit ribbit! Ready?",
      countText: "Jump! Ribbit! Jump! Ribbit! Jump! Ribbit! 1, 2, 3! What a great frog you are!",
      duration: 8,
      canUseCar: true,
    ),
    _Exercise(
      name: 'Cat Stretch',
      instruction: "Let's stretch like a cat! Get on your hands and knees, arch your back up high like a scared cat, then push your tummy down. Ready?",
      countText: "Arch up! Like a cat! Hold it! Now push your tummy down and look up! Stretch! And relax! Purrrr! Great stretching!",
      duration: 10,
    ),
    _Exercise(
      name: 'Star Jumps',
      instruction: "Let's do star jumps! Jump up and spread your arms and legs out wide like a star! Ready?",
      countText: "Jump! Star! Jump! Star! 1! 2! 3! 4! 5! You are a shining star!",
      duration: 10,
      canUseCar: true,
    ),
  ];

  ExerciseBuddy({CarChassis? car}) : _car = car;

  // -- Activity metadata --

  @override
  String get id => 'physical_exercise';

  @override
  String get name => 'Exercise Buddy';

  @override
  String get category => 'physical';

  @override
  String get description =>
      'Fun exercises and movement games to get your body moving.';

  @override
  List<String> get skills => ['physical coordination', 'counting', 'following instructions', 'body awareness'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    return '$_exercisesCompleted exercises completed. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _exercisesCompleted = 0;
    _score = 0;
    _usedExercises.clear();
    _waitingForReady = false;
    _active = true;

    debugPrint('[ExerciseBuddy] Started');

    // Present first exercise
    return "Exercise time! Let's get our body moving and have fun! "
        "${_presentNextExercise()}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    if (_waitingForReady) {
      // Child said they're ready
      _waitingForReady = false;
      return await _doExercise();
    }

    // After an exercise, present the next one
    if (_exercisesCompleted >= _totalExercises) {
      return await end();
    }

    return _presentNextExercise();
  }

  @override
  Future<String> end() async {
    _active = false;
    debugPrint('[ExerciseBuddy] Ended, exercises=$_exercisesCompleted, score=$_score');

    if (_exercisesCompleted == 0) {
      return "Okay, we'll exercise another time! Moving our body is so important!";
    }

    _score += 20; // Completion bonus
    return "Amazing workout! You did $_exercisesCompleted "
        "exercise${_exercisesCompleted > 1 ? 's' : ''}! "
        "Your body is strong and healthy! Score: $_score points! "
        "Great job staying active!";
  }

  // -- Exercise flow --

  String _presentNextExercise() {
    final available = <int>[];
    for (int i = 0; i < _exercises.length; i++) {
      if (!_usedExercises.contains(i)) {
        available.add(i);
      }
    }

    if (available.isEmpty) {
      return "You did every exercise! You are a fitness champion!";
    }

    _currentIndex = available[_rng.nextInt(available.length)];
    _usedExercises.add(_currentIndex);
    _waitingForReady = true;

    return _exercises[_currentIndex].instruction;
  }

  Future<String> _doExercise() async {
    final exercise = _exercises[_currentIndex];
    _exercisesCompleted++;
    _score += 15;

    // If car is connected and exercise supports it, make car dance along
    final car = _car;
    if (car != null && car.connected && exercise.canUseCar) {
      // Fire and forget — car wiggles while bot speaks
      _carDanceAlong();
    }

    final remaining = _totalExercises - _exercisesCompleted;

    if (remaining <= 0) {
      return "${exercise.countText} That was our last exercise!";
    }

    return "${exercise.countText} $remaining more to go! Are you ready for the next one?";
  }

  /// Make the car do a little wiggle during the exercise (non-blocking).
  void _carDanceAlong() {
    final car = _car;
    if (car == null || !car.connected) return;

    // Run async without awaiting — let it dance in background
    Future(() async {
      try {
        const d = Duration(milliseconds: 300);
        await car.spinLeft(speed: 150, duration: d);
        await car.spinRight(speed: 150, duration: d);
        await car.spinLeft(speed: 150, duration: d);
        await car.spinRight(speed: 150, duration: d);
        await car.stop();
      } catch (e) {
        debugPrint('[ExerciseBuddy] Car dance error: $e');
      }
    });
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop the game', 'quit', 'exit', "i'm done", 'no more',
      'stop playing', 'end the game', 'finish', 'stop',
      'बंद करो', 'खेल बंद', 'रुको',
      'ఆపు', 'ఆట ఆపు', 'ఆగు',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
