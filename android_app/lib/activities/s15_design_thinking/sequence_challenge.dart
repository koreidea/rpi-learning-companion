import 'package:flutter/foundation.dart';

import '../../bluetooth/car_chassis.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Parsed movement step for the sequence program.
class _Step {
  final String command; // forward, backward, left, right, spin, dance
  final String displayName;

  const _Step(this.command, this.displayName);
}

/// Command Sequencing activity: the child builds a "program" of car
/// movements, then the bot executes them all in order.
///
/// Teaches: sequencing, planning, cause-and-effect, basic programming.
///
/// Flow:
/// 1. Bot introduces the game and asks for steps.
/// 2. Child says commands one at a time (or several at once).
/// 3. Bot confirms each step and asks for more.
/// 4. Child says "go" / "run" / "start" to execute.
/// 5. Bot runs the program on the car (or describes it if no car).
/// 6. Bot summarizes and congratulates.
class SequenceChallenge extends Activity {
  final CarChassis? _car;

  final List<_Step> _steps = [];
  bool _active = false;
  int _score = 0;

  SequenceChallenge({CarChassis? car}) : _car = car;

  // ── Activity metadata ──

  @override
  String get id => 'coding_sequence';

  @override
  String get name => 'Command Sequence';

  @override
  String get category => 'coding';

  @override
  String get description =>
      'Give the car a sequence of commands, then watch it run your program.';

  @override
  List<String> get skills => ['sequencing', 'planning', 'logical thinking'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  SkillId? get skillId => SkillId.designThinking;

  @override
  Map<String, List<String>> get voiceTriggers => const {
    'en': ['coding game', 'program the car', 'teach me coding', 'sequence game', 'command game', 'code the car', 'programming game'],
    'hi': ['कोडिंग गेम', 'कोडिंग सिखाओ', 'गाड़ी को प्रोग्राम करो'],
    'te': ['కోడింగ్ గేమ్', 'కోడింగ్ నేర్పించు', 'కారు ప్రోగ్రామ్'],
  };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_steps.isEmpty) return 'No steps yet.';
    return '${_steps.length} steps in your program. Score: $_score.';
  }

  // ── Lifecycle ──

  @override
  Future<String> start() async {
    _steps.clear();
    _score = 0;
    _active = true;
    debugPrint('[SequenceChallenge] Started');

    final hasCar = _car?.connected == true;
    if (hasCar) {
      return "Let's play the coding game! "
          "Tell me steps for the car. You can say things like: "
          "go forward, turn left, turn right, or go back. "
          "When you're done, say Go! and I'll run your program!";
    } else {
      return "Let's play the coding game! "
          "Tell me steps like go forward, turn left, turn right, or go back. "
          "When you're done, say Go! and I'll tell you what happens!";
    }
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    // Check for stop/quit
    if (_isQuitTrigger(lower)) {
      return await end();
    }

    // Check for "go" / "run" / "start" to execute
    if (_isExecuteTrigger(lower)) {
      return await _execute();
    }

    // Check for "clear" / "reset" to start over
    if (_isClearTrigger(lower)) {
      _steps.clear();
      return "Okay, I cleared your program! Start telling me new steps.";
    }

    // Try to parse one or more commands from the utterance
    final parsed = _parseCommands(lower);

    if (parsed.isEmpty) {
      // Could not understand -- be encouraging
      return "Hmm, I didn't catch that. "
          "Try saying go forward, turn left, turn right, or go back. "
          "You have ${_steps.length} steps so far.";
    }

    // Add all parsed steps
    for (final step in parsed) {
      _steps.add(step);
    }

    // Build confirmation
    if (parsed.length == 1) {
      return "Step ${_steps.length}: ${parsed.first.displayName}. Got it! "
          "What's next? Or say Go to run your program!";
    } else {
      final names = parsed.map((s) => s.displayName).join(', then ');
      return "Nice! I added ${parsed.length} steps: $names. "
          "That's ${_steps.length} steps total! "
          "Keep going or say Go to run your program!";
    }
  }

  @override
  Future<String> end() async {
    final stepCount = _steps.length;
    _active = false;
    _steps.clear();
    debugPrint('[SequenceChallenge] Ended, score=$_score');

    if (stepCount > 0) {
      return "Great job! Your program had $stepCount steps. "
          "You're becoming a real coder!";
    }
    return "Okay, we'll play the coding game another time!";
  }

  // ── Execution ──

  Future<String?> _execute() async {
    if (_steps.isEmpty) {
      return "Your program is empty! Tell me some steps first, "
          "like go forward or turn left.";
    }

    final stepCount = _steps.length;
    _score = stepCount; // Simple scoring: 1 point per step

    final hasCar = _car?.connected == true;

    if (hasCar) {
      // Actually drive the car
      final intro = "Running your program! 3, 2, 1, go!";
      // We return the intro first; the car commands run, then we
      // build the completion message.
      // Since processResponse is a single turn, we execute inline.
      debugPrint('[SequenceChallenge] Executing $_steps.length steps on car');
      await _executeOnCar();
      _active = false;
      _steps.clear();
      return "$intro "
          "Done! Your program had $stepCount steps. Amazing coding!";
    } else {
      // Describe what would happen
      final description = _steps.map((s) => s.displayName).join(', then ');
      _active = false;
      _steps.clear();
      return "Running your program! 3, 2, 1, go! "
          "$description. "
          "Done! Your program had $stepCount steps. You're a great coder!";
    }
  }

  Future<void> _executeOnCar() async {
    const moveDuration = Duration(milliseconds: 800);
    const turnDuration = Duration(milliseconds: 500);
    const speed = 200;

    for (final step in _steps) {
      switch (step.command) {
        case 'forward':
          await _car!.forward(speed: speed, duration: moveDuration);
        case 'backward':
          await _car!.backward(speed: speed, duration: moveDuration);
        case 'left':
          await _car!.spinLeft(speed: speed, duration: turnDuration);
        case 'right':
          await _car!.spinRight(speed: speed, duration: turnDuration);
        case 'spin':
          await _car!.spinRight(speed: 220, duration: const Duration(milliseconds: 1000));
        case 'dance':
          await _car!.dance();
      }
      // Small pause between steps
      await Future.delayed(const Duration(milliseconds: 200));
    }
    await _car!.stop();
  }

  // ── Parsing helpers ──

  List<_Step> _parseCommands(String lower) {
    final results = <_Step>[];

    // Order matters: check longer phrases first to avoid partial matches
    final patterns = <String, _Step>{
      // Forward
      'go forward': const _Step('forward', 'go forward'),
      'move forward': const _Step('forward', 'go forward'),
      'forward': const _Step('forward', 'go forward'),
      'go straight': const _Step('forward', 'go forward'),
      'straight': const _Step('forward', 'go forward'),
      'ahead': const _Step('forward', 'go forward'),
      // Hindi
      'आगे': const _Step('forward', 'go forward'),
      // Telugu
      'ముందుకు': const _Step('forward', 'go forward'),

      // Backward
      'go back': const _Step('backward', 'go back'),
      'go backward': const _Step('backward', 'go back'),
      'move back': const _Step('backward', 'go back'),
      'backward': const _Step('backward', 'go back'),
      'reverse': const _Step('backward', 'go back'),
      'back': const _Step('backward', 'go back'),
      // Hindi
      'पीछे': const _Step('backward', 'go back'),
      // Telugu
      'వెనక్కు': const _Step('backward', 'go back'),

      // Left
      'turn left': const _Step('left', 'turn left'),
      'go left': const _Step('left', 'turn left'),
      'left': const _Step('left', 'turn left'),
      // Hindi
      'बाएं': const _Step('left', 'turn left'),
      // Telugu
      'ఎడమ': const _Step('left', 'turn left'),

      // Right
      'turn right': const _Step('right', 'turn right'),
      'go right': const _Step('right', 'turn right'),
      'right': const _Step('right', 'turn right'),
      // Hindi
      'दाएं': const _Step('right', 'turn right'),
      // Telugu
      'కుడి': const _Step('right', 'turn right'),

      // Spin
      'spin': const _Step('spin', 'spin around'),
      'turn around': const _Step('spin', 'spin around'),
      'rotate': const _Step('spin', 'spin around'),
      // Hindi
      'घूमो': const _Step('spin', 'spin around'),
      // Telugu
      'తిరుగు': const _Step('spin', 'spin around'),

      // Dance
      'dance': const _Step('dance', 'dance'),
      // Hindi
      'नाचो': const _Step('dance', 'dance'),
      // Telugu
      'డ్యాన్స్': const _Step('dance', 'dance'),
    };

    // We scan through the text and greedily match the longest pattern.
    // To handle "go forward then turn left then go back" style utterances,
    // we split on common conjunctions first.
    final segments = lower
        .replaceAll(RegExp(r'\b(then|and|next|after that|,)\b'), '|')
        .split('|')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty);

    for (final segment in segments) {
      bool matched = false;
      // Try longer patterns first
      final sortedKeys = patterns.keys.toList()
        ..sort((a, b) => b.length.compareTo(a.length));
      for (final pattern in sortedKeys) {
        if (segment.contains(pattern)) {
          results.add(patterns[pattern]!);
          matched = true;
          break;
        }
      }
      // If no match in this segment, skip (child may have said filler words)
      if (!matched && segment.length > 2) {
        // Try the full lower string for single-command utterances
        // (only if we haven't found anything yet)
      }
    }

    // Fallback: if segments didn't work, try the whole string
    if (results.isEmpty) {
      final sortedKeys = patterns.keys.toList()
        ..sort((a, b) => b.length.compareTo(a.length));
      for (final pattern in sortedKeys) {
        if (lower.contains(pattern)) {
          results.add(patterns[pattern]!);
          break;
        }
      }
    }

    return results;
  }

  bool _isExecuteTrigger(String lower) {
    const triggers = [
      'go', 'run', 'start', 'execute', 'do it', 'run it',
      'run my program', 'run the program', 'let\'s go',
      // Hindi
      'चलाओ', 'शुरू करो', 'चलो',
      // Telugu
      'నడుపు', 'మొదలు పెట్టు', 'పడదాం',
    ];
    // "go" is tricky because "go forward" starts with "go". Only match
    // if "go" is the entire utterance or is not followed by a direction.
    if (lower == 'go' || lower == 'go!' || lower == 'go go go') return true;
    for (final t in triggers) {
      if (t == 'go') continue; // handled above
      if (lower.contains(t)) return true;
    }
    return false;
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop the game', 'quit', 'exit', 'i\'m done', 'no more',
      'stop playing', 'end the game', 'finish',
      // Hindi
      'बंद करो', 'खेल बंद',
      // Telugu
      'ఆపు', 'ఆట ఆపు',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  bool _isClearTrigger(String lower) {
    const triggers = [
      'clear', 'reset', 'start over', 'erase', 'delete all',
      'clear the program', 'new program',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
