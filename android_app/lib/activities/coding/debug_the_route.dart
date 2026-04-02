import 'dart:math';

import 'package:flutter/foundation.dart';

import '../activity_base.dart';

/// A single debugging scenario: a destination, the wrong directions,
/// which step is wrong, and what the correct replacement is.
class _Scenario {
  final String destination;
  final List<String> steps;
  final int wrongStepIndex; // 0-based
  final String wrongStep;
  final String correctStep;
  final String hint;

  const _Scenario({
    required this.destination,
    required this.steps,
    required this.wrongStepIndex,
    required this.wrongStep,
    required this.correctStep,
    required this.hint,
  });
}

/// Debug the Route activity: the bot describes wrong directions and the
/// child must identify and fix the incorrect step.
///
/// Teaches: debugging, logical reasoning, spatial thinking, problem solving.
///
/// This is a voice-only activity (no car movement needed). The bot
/// describes a route with a mistake, the child fixes it.
class DebugTheRoute extends Activity {
  static final List<_Scenario> _allScenarios = [
    const _Scenario(
      destination: 'the door',
      steps: ['go forward', 'turn left', 'go forward'],
      wrongStepIndex: 1,
      wrongStep: 'turn left',
      correctStep: 'right',
      hint: 'I keep bumping into the wall on my left side.',
    ),
    const _Scenario(
      destination: 'the kitchen',
      steps: ['go forward', 'go forward', 'turn right'],
      wrongStepIndex: 2,
      wrongStep: 'turn right',
      correctStep: 'left',
      hint: 'I end up in the bedroom instead of the kitchen.',
    ),
    const _Scenario(
      destination: 'the toy box',
      steps: ['turn right', 'go forward', 'go backward'],
      wrongStepIndex: 2,
      wrongStep: 'go backward',
      correctStep: 'forward',
      hint: 'I can see the toy box but I keep moving away from it.',
    ),
    const _Scenario(
      destination: 'the garden',
      steps: ['go forward', 'turn right', 'go forward', 'turn left'],
      wrongStepIndex: 1,
      wrongStep: 'turn right',
      correctStep: 'left',
      hint: 'I end up at the neighbor\'s house instead of the garden.',
    ),
    const _Scenario(
      destination: 'the bookshelf',
      steps: ['turn left', 'go forward', 'turn right', 'go forward'],
      wrongStepIndex: 0,
      wrongStep: 'turn left',
      correctStep: 'right',
      hint: 'I go the wrong way from the very beginning.',
    ),
  ];

  final Random _random = Random();
  final List<_Scenario> _scenarios = [];
  int _currentIndex = -1;
  int _score = 0;
  int _totalAttempts = 0;
  bool _active = false;
  bool _waitingForFix = false;

  // ── Activity metadata ──

  @override
  String get id => 'coding_debug';

  @override
  String get name => 'Debug the Route';

  @override
  String get category => 'coding';

  @override
  String get description =>
      'Find and fix the wrong step in a set of directions.';

  @override
  List<String> get skills =>
      ['debugging', 'logical reasoning', 'spatial thinking'];

  @override
  int get minAge => 4;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_totalAttempts == 0) return 'No puzzles solved yet.';
    return '$_score out of $_totalAttempts puzzles solved.';
  }

  // ── Lifecycle ──

  @override
  Future<String> start() async {
    // Shuffle and pick 3 scenarios
    _scenarios.clear();
    final shuffled = List<_Scenario>.from(_allScenarios)..shuffle(_random);
    _scenarios.addAll(shuffled.take(3));
    _currentIndex = -1;
    _score = 0;
    _totalAttempts = 0;
    _active = true;
    _waitingForFix = false;
    debugPrint('[DebugTheRoute] Started with ${_scenarios.length} scenarios');

    return "Let's play the debugging game! "
        "I'll tell you my directions to get somewhere, but one step is wrong. "
        "You need to find the mistake and fix it! Ready? "
        "${_presentNextScenario()}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    // Check for quit
    if (_isQuitTrigger(lower)) {
      return await end();
    }

    // If we haven't presented a scenario yet, something is off
    if (_currentIndex < 0 || _currentIndex >= _scenarios.length) {
      _active = false;
      return await end();
    }

    final scenario = _scenarios[_currentIndex];

    if (!_waitingForFix) {
      // We just presented the scenario, child should identify the problem.
      // Accept any attempt and move to the fix phase.
      _waitingForFix = true;

      // Check if the child already gave the fix (e.g., "turn right")
      if (_isCorrectFix(lower, scenario)) {
        return _handleCorrectFix(scenario);
      }

      // Check if they asked for a hint
      if (lower.contains('hint') || lower.contains('help')) {
        return "Here's a hint: ${scenario.hint} "
            "Which step should I change?";
      }

      // They might have said the step number or described the wrong step
      return "Good thinking! So what should step ${scenario.wrongStepIndex + 1} be instead? "
          "Should I go forward, go back, turn left, or turn right?";
    }

    // Waiting for the fix
    if (_isCorrectFix(lower, scenario)) {
      return _handleCorrectFix(scenario);
    }

    // Check for hint request
    if (lower.contains('hint') || lower.contains('help')) {
      return "Here's a hint: ${scenario.hint} "
          "Try changing step ${scenario.wrongStepIndex + 1}. "
          "Should I go forward, go back, turn left, or turn right?";
    }

    // Wrong answer -- encourage and give another chance
    return "Hmm, not quite. Let me repeat the directions: "
        "${_describeSteps(scenario)}. "
        "${scenario.hint} "
        "What should step ${scenario.wrongStepIndex + 1} be instead?";
  }

  @override
  Future<String> end() async {
    _active = false;
    _waitingForFix = false;
    debugPrint('[DebugTheRoute] Ended, score=$_score/$_totalAttempts');

    if (_totalAttempts == 0) {
      return "Okay, we'll play the debugging game another time!";
    }

    if (_score == _totalAttempts) {
      return "Amazing! You fixed all $_score bugs! "
          "You're a super debugger!";
    }

    return "Great job! You fixed $_score out of $_totalAttempts bugs. "
        "Keep practicing and you'll be an expert debugger!";
  }

  // ── Internal ──

  String _presentNextScenario() {
    _currentIndex++;
    _waitingForFix = false;

    if (_currentIndex >= _scenarios.length) {
      // All scenarios done
      _active = false;
      return _buildCompletionMessage();
    }

    final scenario = _scenarios[_currentIndex];
    _totalAttempts++;

    return "I want to reach ${scenario.destination}. "
        "Here are my directions: ${_describeSteps(scenario)}. "
        "But ${scenario.hint} "
        "Can you find the mistake and fix it?";
  }

  String _describeSteps(_Scenario scenario) {
    final parts = <String>[];
    for (int i = 0; i < scenario.steps.length; i++) {
      parts.add('step ${i + 1} ${scenario.steps[i]}');
    }
    return parts.join(', ');
  }

  bool _isCorrectFix(String lower, _Scenario scenario) {
    final correct = scenario.correctStep.toLowerCase();
    // Accept various phrasings of the correct direction
    if (correct == 'right') {
      return lower.contains('right') && !lower.contains('left');
    } else if (correct == 'left') {
      return lower.contains('left') && !lower.contains('right');
    } else if (correct == 'forward') {
      return lower.contains('forward') ||
          lower.contains('ahead') ||
          lower.contains('straight');
    } else if (correct == 'backward' || correct == 'back') {
      return lower.contains('back') || lower.contains('reverse');
    }
    return lower.contains(correct);
  }

  String _handleCorrectFix(_Scenario scenario) {
    _score++;
    _waitingForFix = false;

    // Build the corrected steps
    final corrected = List<String>.from(scenario.steps);
    if (scenario.correctStep == 'right') {
      corrected[scenario.wrongStepIndex] = 'turn right';
    } else if (scenario.correctStep == 'left') {
      corrected[scenario.wrongStepIndex] = 'turn left';
    } else if (scenario.correctStep == 'forward') {
      corrected[scenario.wrongStepIndex] = 'go forward';
    } else {
      corrected[scenario.wrongStepIndex] = 'go ${scenario.correctStep}';
    }

    final correctedRoute = corrected.join(', then ');

    // Check if there are more scenarios
    if (_currentIndex + 1 >= _scenarios.length) {
      _active = false;
      return "Yes! Let me try: $correctedRoute. "
          "I made it to ${scenario.destination}! You fixed the bug! "
          "${_buildCompletionMessage()}";
    }

    return "Yes! Let me try: $correctedRoute. "
        "I made it to ${scenario.destination}! You're a great debugger! "
        "${_presentNextScenario()}";
  }

  String _buildCompletionMessage() {
    if (_score == _totalAttempts && _totalAttempts > 0) {
      return "Wow, you fixed all $_score bugs! You're a debugging superstar!";
    }
    if (_score > 0) {
      return "You fixed $_score out of $_totalAttempts bugs. Great debugging!";
    }
    return "Good try! Debugging takes practice. Let's play again soon!";
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
