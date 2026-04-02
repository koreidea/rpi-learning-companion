import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Plot Twist: mid-activity rule changes that teach adaptability and
/// flexible thinking.
///
/// The bot starts a simple challenge (trivia, description, or counting task).
/// After 2 rounds under normal rules, a "plot twist" changes the rules.
/// The child must adapt to the new constraint while continuing the challenge.
/// Celebrates flexibility and creative problem-solving.
class PlotTwist extends Activity {
  final Random _random = Random();

  bool _active = false;
  int _round = 0;
  static const int _totalRounds = 5;
  int _score = 0;
  int _scenarioIndex = 0;
  bool _twistApplied = false;
  _Phase _phase = _Phase.idle;

  final List<int> _usedScenarioIndices = [];

  @override
  String get id => 'adaptability_plot_twist';

  @override
  String get name => 'Plot Twist!';

  @override
  String get category => 'adaptability';

  @override
  String get description =>
      'Adapt to surprise rule changes and keep going!';

  @override
  List<String> get skills =>
      ['adaptability', 'flexible thinking', 'resilience', 'creativity'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 12;

  @override
  SkillId? get skillId => SkillId.adaptability;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'plot twist',
          'twist game',
          'surprise game',
          'rule change',
          'adaptability game',
        ],
        'hi': ['प्लॉट ट्विस्ट', 'सरप्राइज गेम', 'नियम बदलो'],
        'te': ['ప్లాట్ ట్విస్ట్', 'ఆశ్చర్యం ఆట'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _round = 0;
    _score = 0;
    _twistApplied = false;

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
    _phase = _Phase.normalRound;

    debugPrint('[PlotTwist] Starting scenario: ${scenario.title}');

    return "Let's play Plot Twist! Here is the game: ${scenario.instruction} "
        "Ready? Here is your first challenge. ${scenario.challenges[0]}";
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
    _round++;
    _score += 10;

    // After round 2, apply the twist
    if (_round == 2 && !_twistApplied) {
      _twistApplied = true;
      _phase = _Phase.twistRound;

      final ack = _acknowledgements[_random.nextInt(_acknowledgements.length)];
      return "$ack Wait! Plot twist! ${scenario.twist} "
          "Now try this one with the new rule. "
          "${_getChallengeForRound(scenario, _round)}";
    }

    if (_round >= _totalRounds) {
      _active = false;
      _score += 20; // Bonus for completing
      final ack = _acknowledgements[_random.nextInt(_acknowledgements.length)];
      return "$ack You handled that change so well! "
          "Being flexible is a superpower! ${_buildEndSummary()}";
    }

    final ack = _acknowledgements[_random.nextInt(_acknowledgements.length)];
    final nextChallenge = _getChallengeForRound(scenario, _round);
    return "$ack $nextChallenge";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_round == 0) return 'No rounds played yet.';
    return 'Round $_round of $_totalRounds. Score: $_score. '
        'Twist: ${_twistApplied ? "applied" : "coming soon"}!';
  }

  // -- Internal --

  String _getChallengeForRound(_PlotTwistScenario scenario, int round) {
    if (round < scenario.challenges.length) {
      return scenario.challenges[round];
    }
    // Recycle challenges if we run out
    return scenario.challenges[round % scenario.challenges.length];
  }

  String _buildEndSummary() {
    if (_round == 0) {
      return "Thanks for trying Plot Twist! Come back for more surprises!";
    }
    return "You completed $_round rounds and scored $_score points! "
        "You adapted to the plot twist like a champion! "
        "Remember, when things change, the best thing to do is stay flexible "
        "and keep trying. You are great at that!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<String> _acknowledgements = [
    'Nice!',
    'Great job!',
    'Well done!',
    'Awesome!',
    'Brilliant!',
  ];

  static const List<_PlotTwistScenario> _scenarios = [
    _PlotTwistScenario(
      title: 'Describe It',
      instruction:
          'I will name something and you describe it as well as you can.',
      twist:
          'New rule: you cannot use the word "the" in your description!',
      challenges: [
        'Describe an elephant!',
        'Describe a rainbow!',
        'Describe your favorite food!',
        'Describe a school!',
        'Describe your best friend!',
      ],
    ),
    _PlotTwistScenario(
      title: 'Word Challenge',
      instruction:
          'I will give you a category and you name as many things in it as you can.',
      twist:
          'New rule: every answer must start with the same letter as your name!',
      challenges: [
        'Name things you find in a kitchen!',
        'Name different animals!',
        'Name things that are red!',
        'Name things you can wear!',
        'Name things that make sound!',
      ],
    ),
    _PlotTwistScenario(
      title: 'Quick Count',
      instruction:
          'I will ask counting questions and you answer as fast as you can!',
      twist:
          'New rule: you have to clap your hands before saying the answer!',
      challenges: [
        'How many legs does a dog have?',
        'How many wheels does a car have?',
        'How many days are in a week?',
        'How many fingers do you have?',
        'How many colors in a rainbow?',
      ],
    ),
    _PlotTwistScenario(
      title: 'Explain It',
      instruction:
          'I will name something and you explain it using only 3 words!',
      twist:
          'New rule: now explain using only actions and sounds, no real words!',
      challenges: [
        'Explain a cat!',
        'Explain rain!',
        'Explain dancing!',
        'Explain eating food!',
        'Explain a airplane!',
      ],
    ),
    _PlotTwistScenario(
      title: 'Opposite Day',
      instruction:
          'I will say something and you say the opposite!',
      twist:
          'New rule: now you have to say the opposite AND use it in a sentence!',
      challenges: [
        'What is the opposite of big?',
        'What is the opposite of hot?',
        'What is the opposite of happy?',
        'What is the opposite of fast?',
        'What is the opposite of loud?',
      ],
    ),
  ];
}

enum _Phase {
  idle,
  normalRound,
  twistRound,
}

/// A plot twist scenario with challenges and a mid-game rule change.
class _PlotTwistScenario {
  final String title;
  final String instruction;
  final String twist;
  final List<String> challenges;

  const _PlotTwistScenario({
    required this.title,
    required this.instruction,
    required this.twist,
    required this.challenges,
  });
}
