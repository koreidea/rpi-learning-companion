import 'package:flutter/foundation.dart';

import '../../bluetooth/car_chassis.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// States the loop game can be in.
enum _Phase {
  /// Waiting for the child to give a move for simple repetition.
  simpleLoop,

  /// The child gave a move; bot repeated it 3 times. Now ask for a square.
  squareChallenge,

  /// Waiting for the child's answer to the square challenge.
  waitingForSquare,

  /// Square done; present the triangle challenge.
  triangleChallenge,

  /// Waiting for the child's answer to the triangle challenge.
  waitingForTriangle,

  /// All challenges done.
  done,
}

/// Loop/Pattern Game: teaches the concept of loops (repetition) through
/// car movements.
///
/// Teaches: loops, patterns, repetition, counting, geometry basics.
///
/// Flow:
/// 1. Bot explains loops. Child gives a move; bot does it 3 times.
/// 2. Bot challenges: "Make the car go in a square!" (forward + right, x4)
/// 3. Bot challenges: "Make a triangle!" (forward + turn ~120 degrees, x3)
/// 4. Summary and congratulations.
class LoopGame extends Activity {
  final CarChassis? _car;

  _Phase _phase = _Phase.done;
  bool _active = false;
  int _score = 0;
  int _challengesDone = 0;

  LoopGame({CarChassis? car}) : _car = car;

  // ── Metadata ──

  @override
  String get id => 'coding_loop';

  @override
  String get name => 'Loop and Pattern Game';

  @override
  String get category => 'coding';

  @override
  String get description =>
      'Learn about loops by making the car repeat movements in patterns.';

  @override
  List<String> get skills => ['loops', 'patterns', 'counting', 'geometry'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  SkillId? get skillId => SkillId.designThinking;

  @override
  Map<String, List<String>> get voiceTriggers => const {
    'en': ['loop game', 'pattern game', 'repeat game', 'teach me loops', 'loop the car'],
    'hi': ['लूप गेम', 'पैटर्न गेम', 'दोहराओ'],
    'te': ['లూప్ గేమ్', 'పాటర్న్ గేమ్', 'మళ్ళీ చెయ్యి'],
  };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_challengesDone == 0) return 'Just getting started.';
    return '$_challengesDone challenges done. Score: $_score.';
  }

  // ── Lifecycle ──

  @override
  Future<String> start() async {
    _active = true;
    _phase = _Phase.simpleLoop;
    _score = 0;
    _challengesDone = 0;
    debugPrint('[LoopGame] Started');

    return "Let's learn about loops! "
        "A loop means doing something again and again. "
        "Tell me a move, and I'll do it 3 times! "
        "You can say go forward, go back, turn left, or turn right.";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    switch (_phase) {
      case _Phase.simpleLoop:
        return await _handleSimpleLoop(lower);
      case _Phase.squareChallenge:
        // This phase is transitional; we present the challenge and move on
        _phase = _Phase.waitingForSquare;
        return _presentSquareChallenge();
      case _Phase.waitingForSquare:
        return await _handleSquareAnswer(lower);
      case _Phase.triangleChallenge:
        _phase = _Phase.waitingForTriangle;
        return _presentTriangleChallenge();
      case _Phase.waitingForTriangle:
        return await _handleTriangleAnswer(lower);
      case _Phase.done:
        _active = false;
        return null;
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    _phase = _Phase.done;
    debugPrint('[LoopGame] Ended, score=$_score');

    if (_challengesDone == 0) {
      return "Okay, we'll play the loop game another time!";
    }

    return "Great job! You completed $_challengesDone loop challenges "
        "and scored $_score points. "
        "Now you know that a loop means doing something over and over!";
  }

  // ── Simple loop ──

  Future<String> _handleSimpleLoop(String lower) async {
    final move = _parseMove(lower);
    if (move == null) {
      return "I didn't catch that. Try saying go forward, go back, "
          "turn left, or turn right.";
    }

    _challengesDone++;
    _score += 1;

    final hasCar = _car?.connected == true;
    final moveName = _moveName(move);

    // Execute on car if available
    if (hasCar) {
      await _executeMove(move);
      await Future.delayed(const Duration(milliseconds: 300));
      await _executeMove(move);
      await Future.delayed(const Duration(milliseconds: 300));
      await _executeMove(move);
    }

    // Transition to square challenge
    _phase = _Phase.squareChallenge;

    return "$moveName, $moveName, $moveName! "
        "3 times! That's a loop! "
        "Now here's a challenge. "
        "${_presentSquareChallenge()}";
  }

  // ── Square challenge ──

  String _presentSquareChallenge() {
    _phase = _Phase.waitingForSquare;
    return "Can you make the car go in a square? "
        "A square has 4 sides. "
        "Hint: you need 2 moves repeated 4 times. "
        "What 2 moves should I loop?";
  }

  Future<String> _handleSquareAnswer(String lower) async {
    // Accept: forward + right, forward + left, or similar
    final hasForward = lower.contains('forward') ||
        lower.contains('straight') ||
        lower.contains('ahead') ||
        lower.contains('आगे') ||
        lower.contains('ముందుకు');
    final hasTurn = lower.contains('right') ||
        lower.contains('left') ||
        lower.contains('turn') ||
        lower.contains('बाएं') ||
        lower.contains('दाएं') ||
        lower.contains('ఎడమ') ||
        lower.contains('కుడి');

    if (lower.contains('hint') || lower.contains('help')) {
      return "Think about a square. You walk along one side, then turn a corner. "
          "What 2 moves make a corner? Try forward and then a turn!";
    }

    if (hasForward && hasTurn) {
      _score += 3;
      _challengesDone++;

      final hasCar = _car?.connected == true;
      if (hasCar) {
        await _executeSquareOnCar();
      }

      final turnDir = lower.contains('left') ? 'left' : 'right';

      // Transition to triangle
      _phase = _Phase.triangleChallenge;

      return "That's right! Forward then turn $turnDir, "
          "forward then turn $turnDir, "
          "forward then turn $turnDir, "
          "forward then turn $turnDir! "
          "That's a square! You used a loop of 2 steps, 4 times! "
          "${_presentTriangleChallenge()}";
    }

    // Partial credit or wrong
    if (hasForward || hasTurn) {
      return "You're close! A square needs both moving and turning. "
          "Try saying forward then turn right. "
          "Those 2 moves repeated 4 times make a square!";
    }

    return "Hmm, think about a square shape. "
        "To make each side, you go forward. "
        "At each corner, you turn. "
        "What 2 moves should I repeat?";
  }

  Future<void> _executeSquareOnCar() async {
    final car = _car!;
    const moveDuration = Duration(milliseconds: 700);
    const turnDuration = Duration(milliseconds: 500);
    const speed = 200;

    for (int i = 0; i < 4; i++) {
      await car.forward(speed: speed, duration: moveDuration);
      await Future.delayed(const Duration(milliseconds: 200));
      await car.spinRight(speed: speed, duration: turnDuration);
      await Future.delayed(const Duration(milliseconds: 200));
    }
    await car.stop();
  }

  // ── Triangle challenge ──

  String _presentTriangleChallenge() {
    _phase = _Phase.waitingForTriangle;
    return "Now a harder one! Can you make a triangle? "
        "A triangle has 3 sides. "
        "What 2 moves should I loop, and how many times?";
  }

  Future<String> _handleTriangleAnswer(String lower) async {
    final hasForward = lower.contains('forward') ||
        lower.contains('straight') ||
        lower.contains('ahead');
    final hasTurn = lower.contains('right') ||
        lower.contains('left') ||
        lower.contains('turn');
    final hasThree = lower.contains('3') || lower.contains('three');

    if (lower.contains('hint') || lower.contains('help')) {
      return "A triangle has 3 sides and 3 corners. "
          "So you need to go forward and turn, 3 times! "
          "The turn is bigger than a square's turn.";
    }

    // Accept reasonable answers: forward + turn, mentioned 3 times
    if (hasForward && hasTurn) {
      _score += 5;
      _challengesDone++;

      final hasCar = _car?.connected == true;
      if (hasCar) {
        await _executeTriangleOnCar();
      }

      _active = false;
      _phase = _Phase.done;

      final bonus = hasThree ? " And you knew it was 3 times! Extra smart!" : "";

      return "Yes! Forward then turn, 3 times makes a triangle! "
          "The turn is bigger because a triangle's corners are wider.$bonus "
          "You really understand loops now! "
          "Final score: $_score points. You're a loop master!";
    }

    if (hasThree) {
      return "Yes, 3 times! But what 2 moves should I repeat? "
          "Think about what makes each side and each corner.";
    }

    return "A triangle is like a square but with 3 sides instead of 4. "
        "You still need to go forward and turn, "
        "but how many times? A triangle has how many sides?";
  }

  Future<void> _executeTriangleOnCar() async {
    final car = _car!;
    // Triangle needs ~120 degree turns. With the car's spin, we estimate
    // a longer turn duration.
    const moveDuration = Duration(milliseconds: 700);
    const turnDuration = Duration(milliseconds: 700); // wider turn
    const speed = 200;

    for (int i = 0; i < 3; i++) {
      await car.forward(speed: speed, duration: moveDuration);
      await Future.delayed(const Duration(milliseconds: 200));
      await car.spinRight(speed: speed, duration: turnDuration);
      await Future.delayed(const Duration(milliseconds: 200));
    }
    await car.stop();
  }

  // ── Helpers ──

  String? _parseMove(String lower) {
    if (lower.contains('forward') ||
        lower.contains('straight') ||
        lower.contains('ahead') ||
        lower.contains('आगे') ||
        lower.contains('ముందుకు')) {
      return 'forward';
    }
    if (lower.contains('back') ||
        lower.contains('reverse') ||
        lower.contains('पीछे') ||
        lower.contains('వెనక్కు')) {
      return 'backward';
    }
    if (lower.contains('left') ||
        lower.contains('बाएं') ||
        lower.contains('ఎడమ')) {
      return 'left';
    }
    if (lower.contains('right') ||
        lower.contains('दाएं') ||
        lower.contains('కుడి')) {
      return 'right';
    }
    return null;
  }

  String _moveName(String move) {
    switch (move) {
      case 'forward':
        return 'Go forward';
      case 'backward':
        return 'Go back';
      case 'left':
        return 'Turn left';
      case 'right':
        return 'Turn right';
      default:
        return move;
    }
  }

  Future<void> _executeMove(String move) async {
    if (_car?.connected != true) return;
    const speed = 200;
    const moveDuration = Duration(milliseconds: 600);
    const turnDuration = Duration(milliseconds: 400);

    switch (move) {
      case 'forward':
        await _car!.forward(speed: speed, duration: moveDuration);
      case 'backward':
        await _car!.backward(speed: speed, duration: moveDuration);
      case 'left':
        await _car!.spinLeft(speed: speed, duration: turnDuration);
      case 'right':
        await _car!.spinRight(speed: speed, duration: turnDuration);
    }
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop the game', 'quit', 'exit', 'i\'m done', 'no more',
      'stop playing', 'end the game', 'finish',
      'बंद करो', 'खेल बंद',
      'ఆపు', 'ఆట ఆపు',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
