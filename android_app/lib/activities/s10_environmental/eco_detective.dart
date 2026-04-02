import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// A single eco-detective mission.
class _EcoMission {
  final String challenge;
  final String hint;
  final String funFact;
  final List<String> acceptableAnswers;

  const _EcoMission({
    required this.challenge,
    required this.hint,
    required this.funFact,
    required this.acceptableAnswers,
  });
}

/// Eco Detective: teaching environmental awareness through fun detective missions.
///
/// Teaches: environmental awareness, observation, responsibility, vocabulary.
///
/// Flow:
/// 1. Bot assigns a detective mission (find something in the room).
/// 2. Child finds and names the object.
/// 3. Bot validates, shares a fun fact, and gives the next mission.
/// 4. 4-5 missions total.
class EcoDetective extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _missionsCompleted = 0;
  int _score = 0;
  int _currentMissionIndex = -1;
  final List<int> _usedMissions = [];

  static const int _totalMissions = 5;

  static const List<_EcoMission> _missions = [
    _EcoMission(
      challenge: "Your mission: find something in your room that uses electricity!",
      hint: "Look for something that you can turn on and off, like a light or a fan.",
      funFact: "Great find! Things that use electricity need energy. "
          "When we turn them off when we don't need them, we save energy! "
          "That helps our planet. Mission complete!",
      acceptableAnswers: [
        'light', 'lamp', 'fan', 'tv', 'television', 'phone', 'tablet',
        'computer', 'charger', 'clock', 'radio', 'ac', 'air conditioner',
        'heater', 'fridge', 'refrigerator',
      ],
    ),
    _EcoMission(
      challenge: "Next mission: find something that is made of plastic!",
      hint: "Plastic things are usually smooth and light. Look for a bottle, a toy, or a box.",
      funFact: "Good detective work! Plastic takes a very very long time to go away, "
          "hundreds of years! That is why we should try to use less plastic "
          "and recycle it when we can. Mission complete!",
      acceptableAnswers: [
        'bottle', 'toy', 'cup', 'box', 'pen', 'container', 'bag',
        'wrapper', 'bucket', 'comb', 'brush', 'remote', 'straw',
        'plate', 'spoon',
      ],
    ),
    _EcoMission(
      challenge: "New mission: find something that is made of wood! Wood comes from trees!",
      hint: "Wooden things feel hard and sometimes you can see lines in them. "
          "Look for a table, a chair, or a door.",
      funFact: "Wonderful! Wood comes from trees. Trees are amazing because they "
          "give us clean air to breathe and homes for birds and animals. "
          "We should plant more trees! Mission complete!",
      acceptableAnswers: [
        'table', 'chair', 'door', 'desk', 'pencil', 'shelf', 'bed',
        'cupboard', 'wardrobe', 'floor', 'window', 'frame', 'stick',
        'block', 'toy',
      ],
    ),
    _EcoMission(
      challenge: "Detective mission: find something that uses water!",
      hint: "Think about what you use water with. A tap, a glass, a plant pot?",
      funFact: "Great thinking! Water is so precious. We need water to drink, "
          "to cook, and to keep clean. Plants and animals need water too. "
          "We should never waste water! Mission complete!",
      acceptableAnswers: [
        'tap', 'sink', 'glass', 'bottle', 'cup', 'plant', 'flower',
        'bucket', 'washing machine', 'shower', 'bathtub', 'hose',
        'water', 'jug',
      ],
    ),
    _EcoMission(
      challenge: "Next mission: find something that you can recycle!",
      hint: "Things made of paper, cardboard, glass bottles, or metal cans can be recycled.",
      funFact: "Excellent detective work! When we recycle, old things get turned into "
          "new things instead of becoming garbage. Paper, cardboard, glass, and "
          "metal cans can all be recycled. That is so cool! Mission complete!",
      acceptableAnswers: [
        'paper', 'newspaper', 'box', 'cardboard', 'bottle', 'can',
        'jar', 'magazine', 'book', 'notebook', 'carton', 'tin',
        'envelope', 'bag',
      ],
    ),
    _EcoMission(
      challenge: "Special mission: find something that came from a plant!",
      hint: "Many things come from plants. Cotton clothes, wooden furniture, "
          "fruits, paper, even chocolate!",
      funFact: "Amazing! So many things come from plants. Our food, our clothes, "
          "paper, and even medicine. Plants are like nature's factory! "
          "That is why we should take care of trees and plants. Mission complete!",
      acceptableAnswers: [
        'shirt', 'clothes', 'cotton', 'paper', 'fruit', 'apple',
        'banana', 'flower', 'leaf', 'wood', 'food', 'bread',
        'rice', 'vegetable', 'rubber', 'book',
      ],
    ),
  ];

  EcoDetective();

  // -- Activity metadata --

  @override
  String get id => 'world_eco_detective';

  @override
  String get name => 'Eco Detective';

  @override
  String get category => 'world';

  @override
  String get description =>
      'Be an eco detective and learn about our environment through fun missions.';

  @override
  List<String> get skills => ['environmental awareness', 'observation', 'responsibility'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  SkillId? get skillId => SkillId.environmental;

  @override
  Map<String, List<String>> get voiceTriggers => const {
    'en': ['eco detective', 'environment game', 'nature detective', 'save the planet', 'eco game', 'recycle game'],
    'hi': ['पर्यावरण खेल', 'रीसायकल खेल'],
    'te': ['పర్యావరణ ఆట', 'రీసైకిల్ ఆట'],
  };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    return '$_missionsCompleted missions completed. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _missionsCompleted = 0;
    _score = 0;
    _usedMissions.clear();
    _active = true;

    debugPrint('[EcoDetective] Started');

    final mission = _nextMission();
    if (mission == null) return await end();

    return "You are an eco detective today! Your job is to find things around you "
        "and learn how they connect to our planet. ${mission.challenge}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    // Check if they need a hint
    if (_needsHint(lower)) {
      final mission = _missions[_currentMissionIndex];
      return "Here is a hint! ${mission.hint}";
    }

    final mission = _missions[_currentMissionIndex];

    // Accept any reasonable answer — children might describe things differently
    final isValid = _isValidAnswer(lower, mission);

    if (isValid) {
      _missionsCompleted++;
      _score += 15;

      final feedback = mission.funFact;

      // Check if we should continue
      if (_missionsCompleted >= _totalMissions) {
        _active = false;
        _score += 20; // Bonus for completing all missions
        return "$feedback You completed all your missions! "
            "You are a super eco detective! You earned $_score points!";
      }

      // Next mission
      final nextMission = _nextMission();
      if (nextMission == null) {
        return await end();
      }

      return "$feedback ${nextMission.challenge}";
    }

    // If answer doesn't match, be encouraging and give a hint
    _score += 5; // Points for trying
    return "That is interesting! But let me give you a hint. ${mission.hint}";
  }

  @override
  Future<String> end() async {
    _active = false;
    debugPrint('[EcoDetective] Ended, missions=$_missionsCompleted, score=$_score');

    if (_missionsCompleted == 0) {
      return "Okay, we'll be eco detectives another time! "
          "Remember, taking care of our planet is important!";
    }

    return "Great job, eco detective! You completed $_missionsCompleted "
        "mission${_missionsCompleted > 1 ? 's' : ''} and earned $_score points! "
        "You are helping to take care of our planet!";
  }

  // -- Helpers --

  _EcoMission? _nextMission() {
    final available = <int>[];
    for (int i = 0; i < _missions.length; i++) {
      if (!_usedMissions.contains(i)) {
        available.add(i);
      }
    }
    if (available.isEmpty) return null;

    _currentMissionIndex = available[_rng.nextInt(available.length)];
    _usedMissions.add(_currentMissionIndex);
    return _missions[_currentMissionIndex];
  }

  bool _isValidAnswer(String lower, _EcoMission mission) {
    // Check against acceptable answers
    for (final answer in mission.acceptableAnswers) {
      if (lower.contains(answer)) return true;
    }
    // Be generous — if the child said something with 4+ characters,
    // accept it as a valid find (they might describe things differently)
    final words = lower.split(RegExp(r'\s+'));
    for (final word in words) {
      if (word.length >= 4 &&
          !_filler.contains(word) &&
          !_isQuitTrigger(word) &&
          !_needsHint(word)) {
        return true;
      }
    }
    return false;
  }

  static const Set<String> _filler = {
    'this', 'that', 'here', 'there', 'found', 'it is',
    'look', 'have', 'with', 'what', 'about', 'think',
  };

  bool _needsHint(String lower) {
    const triggers = [
      'hint', 'help', 'clue', "i don't know", 'what',
      'tell me', 'which one', 'not sure',
      'मदद', 'बताओ',
      'సహాయం', 'చెప్పు',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop the game', 'quit', 'exit', "i'm done", 'no more',
      'stop playing', 'end the game', 'finish',
      'बंद करो', 'खेल बंद',
      'ఆపు', 'ఆట ఆపు',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
