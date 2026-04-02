import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../activity_types.dart';

/// Build It Together: a cooperative puzzle game where two players each
/// receive different clues and must share information to solve a challenge.
///
/// The bot gives each player (or the bot plays one role if only one child
/// is present) a piece of the puzzle. They must communicate and collaborate
/// to find the answer.
class BuildItTogether extends GroupActivity {
  final Random _random = Random();

  bool _active = false;
  List<String> _participants = [];
  String? _currentParticipant;
  int _scenarioIndex = 0;
  int _phase = 0;
  // Phase 0: intro + assign roles
  // Phase 1: player 1 info shared
  // Phase 2: player 2 info shared
  // Phase 3: solving together
  // Phase 4: wrap up
  bool _soloMode = false;

  final List<int> _usedScenarioIndices = [];

  @override
  String get id => 'collaboration_build_it_together';

  @override
  String get name => 'Build It Together';

  @override
  String get category => 'collaboration';

  @override
  String get description =>
      'Work together to solve puzzles by sharing clues with a partner!';

  @override
  List<String> get skills =>
      ['collaboration', 'communication', 'problem solving'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 12;

  @override
  SkillId? get skillId => SkillId.collaboration;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'build it together',
          'team puzzle',
          'work together',
          'partner game',
          'collaboration game',
        ],
        'hi': ['साथ में बनाओ', 'टीम पहेली', 'मिलकर खेलो'],
        'te': ['కలిసి చేద్దాం', 'టీమ్ పజిల్', 'జట్టు ఆట'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  List<String> get participants => List.unmodifiable(_participants);

  @override
  String? get currentParticipant => _currentParticipant;

  @override
  Future<void> setParticipants(List<String> names) async {
    _participants = List.from(names);
    _soloMode = _participants.length < 2;
    debugPrint(
        '[BuildItTogether] Participants: $_participants, solo=$_soloMode');
  }

  @override
  Future<void> nextTurn() async {
    if (_participants.isEmpty) return;
    final currentIndex = _currentParticipant != null
        ? _participants.indexOf(_currentParticipant!)
        : -1;
    final nextIndex = (currentIndex + 1) % _participants.length;
    _currentParticipant = _participants[nextIndex];
  }

  @override
  Map<String, String> getPerParticipantFeedback() {
    final feedback = <String, String>{};
    for (final name in _participants) {
      feedback[name] =
          'Great teamwork, $name! You shared your clues and helped solve the puzzle!';
    }
    return feedback;
  }

  @override
  Future<String> start() async {
    _active = true;
    _phase = 0;

    if (_participants.isEmpty) {
      _participants = ['You'];
      _soloMode = true;
    }

    // Pick a scenario
    if (_usedScenarioIndices.length >= _scenarios.length) {
      _usedScenarioIndices.clear();
    }
    int index;
    do {
      index = _random.nextInt(_scenarios.length);
    } while (_usedScenarioIndices.contains(index));
    _usedScenarioIndices.add(index);
    _scenarioIndex = index;

    final scenario = _scenarios[_scenarioIndex];
    _phase = 1;

    if (_soloMode) {
      _currentParticipant = _participants[0];
      return "Let's play Build It Together! I will be your partner. "
          "${scenario.intro} I have one clue and you have the other. "
          "Here is your clue: ${scenario.clueA} "
          "And my clue is: ${scenario.clueB} "
          "Now, putting our clues together, ${scenario.question}";
    }

    final p1 = _participants[0];
    final p2 = _participants.length > 1 ? _participants[1] : 'Player 2';
    _currentParticipant = p1;

    return "Let's play Build It Together! ${scenario.intro} "
        "$p1, cover $p2's ears for a moment! Here is your secret clue: "
        "${scenario.clueA} Got it? Now $p2, your turn! "
        "$p1, cover your ears! $p2's clue is: ${scenario.clueB} "
        "Now share your clues with each other and figure out: ${scenario.question}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    final scenario = _scenarios[_scenarioIndex];

    switch (_phase) {
      case 1:
        // Check if they got the answer
        _phase = 2;
        if (_checkAnswer(lower, scenario.answer)) {
          _active = false;
          return "That is correct! ${scenario.explanation} "
              "Amazing teamwork! You put your clues together perfectly! "
              "${_buildEndSummary()}";
        }
        return "Hmm, not quite! Remember to combine both clues. "
            "Your clue was about: ${scenario.hintA}. "
            "${_soloMode ? 'My' : 'Your partner\'s'} clue was about: ${scenario.hintB}. "
            "Try again! ${scenario.question}";

      case 2:
        _active = false;
        if (_checkAnswer(lower, scenario.answer)) {
          return "Yes, that is it! ${scenario.explanation} "
              "Wonderful teamwork! ${_buildEndSummary()}";
        }
        return "The answer was: ${scenario.answer}! ${scenario.explanation} "
            "That was a tricky one. You will get the next one! ${_buildEndSummary()}";

      default:
        return "Put your clues together and tell me what you think!";
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary => 'Collaboration puzzle in progress.';

  // -- Internal --

  bool _checkAnswer(String guess, String answer) {
    final g = guess.toLowerCase();
    final words = answer.toLowerCase().split(RegExp(r'\s+'));
    // Check if the guess contains the key words of the answer
    int matched = 0;
    for (final word in words) {
      if (word.length > 2 && g.contains(word)) matched++;
    }
    return matched >= (words.length * 0.5).ceil();
  }

  String _buildEndSummary() {
    return "Great job working together! Collaboration means sharing what you "
        "know and listening to others. You did that brilliantly!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<_Scenario> _scenarios = [
    _Scenario(
      intro: 'We are treasure hunters looking for hidden treasure!',
      clueA: 'The treasure is near something that has water and fish.',
      clueB: 'The treasure is under something with a red roof.',
      hintA: 'water and fish',
      hintB: 'a red roof',
      question: 'Where is the treasure hidden?',
      answer: 'under the red-roofed house near the pond',
      explanation:
          'The treasure was under the red-roofed house next to the pond! '
          'One clue told us about the water, the other about the red roof.',
    ),
    _Scenario(
      intro: 'We are making a secret recipe!',
      clueA: 'The recipe needs something yellow that monkeys love.',
      clueB: 'The recipe needs something white that comes from a cow.',
      hintA: 'something yellow from monkeys',
      hintB: 'something white from cows',
      question: 'What two ingredients do we need?',
      answer: 'banana and milk',
      explanation:
          'We need a banana and milk to make a banana milkshake! '
          'Each clue gave us one ingredient.',
    ),
    _Scenario(
      intro: 'We are trying to identify a mystery animal!',
      clueA: 'The animal is very big and lives in Africa and India.',
      clueB: 'The animal has a long nose it uses to drink water.',
      hintA: 'big and lives in Africa and India',
      hintB: 'long nose for drinking',
      question: 'What animal are we thinking of?',
      answer: 'elephant',
      explanation:
          'It is an elephant! One clue told us where it lives, '
          'the other described its trunk.',
    ),
    _Scenario(
      intro: 'We are building a bridge across a river!',
      clueA: 'We need to use something strong that comes from trees.',
      clueB: 'We need something that holds pieces together, like a metal stick.',
      hintA: 'something from trees',
      hintB: 'metal that holds things together',
      question: 'What materials do we use to build the bridge?',
      answer: 'wood and nails',
      explanation:
          'We use wood from trees and nails to hold the pieces together! '
          'Each builder brought one material.',
    ),
    _Scenario(
      intro: 'We are putting together a story!',
      clueA:
          'The story begins with a little boy who found a magic lamp in a cave.',
      clueB:
          'The story ends with the boy using his last wish to make everyone happy.',
      hintA: 'how the story begins',
      hintB: 'how the story ends',
      question:
          'What do you think happened in the middle of the story?',
      answer: 'the boy made wishes from the lamp',
      explanation:
          'The boy found a magic lamp and made wishes! Your job was to '
          'imagine the exciting middle part of the story.',
    ),
  ];
}

/// A collaborative puzzle scenario with split clues.
class _Scenario {
  final String intro;
  final String clueA;
  final String clueB;
  final String hintA;
  final String hintB;
  final String question;
  final String answer;
  final String explanation;

  const _Scenario({
    required this.intro,
    required this.clueA,
    required this.clueB,
    required this.hintA,
    required this.hintB,
    required this.question,
    required this.answer,
    required this.explanation,
  });
}
