import 'dart:math';

import '../activity_base.dart';

/// Classification game where the child identifies which item does not belong.
///
/// The bot names 4 items and the child picks the odd one out. After each
/// answer the bot explains WHY the item does not belong, teaching
/// classification reasoning. Difficulty progresses from obvious categories
/// to subtle differences.
class OddOneOut extends Activity {
  final Random _random = Random();

  bool _active = false;
  int _roundsPlayed = 0;
  int _correctAnswers = 0;
  int _maxRounds = 6;

  // Current round state
  _OddOneOutSet? _currentSet;
  bool _waitingForAnswer = false;
  bool _waitingForReady = false;
  bool _waitingForPlayAgain = false;

  int _difficultyLevel = 0; // 0=easy, 1=medium, 2=hard
  final List<int> _usedSetIndices = [];

  @override
  String get id => 'thinking_odd_one_out';

  @override
  String get name => 'Odd One Out';

  @override
  String get category => 'thinking';

  @override
  String get description =>
      'Find which item does not belong in a group. Teaches classification and reasoning.';

  @override
  List<String> get skills => ['classification', 'critical thinking', 'reasoning'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _roundsPlayed = 0;
    _correctAnswers = 0;
    _difficultyLevel = 0;
    _usedSetIndices.clear();
    _waitingForReady = true;
    _waitingForAnswer = false;
    _waitingForPlayAgain = false;

    return "Let's play Odd One Out! I will say four things and you tell me "
        "which one does not belong. Ready?";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    if (_waitingForReady) {
      _waitingForReady = false;
      return _presentNewSet();
    }

    if (_waitingForPlayAgain) {
      _waitingForPlayAgain = false;
      if (_containsNo(lower)) {
        _active = false;
        return _buildEndSummary();
      }
      return _presentNewSet();
    }

    if (_waitingForAnswer && _currentSet != null) {
      return _processAnswer(lower);
    }

    return "I did not quite hear that. Which one does not belong?";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_roundsPlayed == 0) return 'No rounds played yet.';
    return 'Got $_correctAnswers out of $_roundsPlayed correct.';
  }

  // -- Internal --

  String _presentNewSet() {
    final sets = _getSetsForDifficulty(_difficultyLevel);

    _OddOneOutSet? chosen;
    for (int i = 0; i < sets.length; i++) {
      final globalIndex = _difficultyLevel * 100 + i;
      if (!_usedSetIndices.contains(globalIndex)) {
        chosen = sets[i];
        _usedSetIndices.add(globalIndex);
        break;
      }
    }

    if (chosen == null) {
      _difficultyLevel = (_difficultyLevel + 1).clamp(0, 2);
      _usedSetIndices.clear();
      final fallback = _getSetsForDifficulty(_difficultyLevel);
      chosen = fallback[_random.nextInt(fallback.length)];
    }

    _currentSet = chosen;
    _waitingForAnswer = true;

    // Shuffle the display order so the odd one is not always last
    final items = List<String>.from(chosen.items);
    items.shuffle(_random);

    final itemList = '${items[0]}, ${items[1]}, ${items[2]}, and ${items[3]}';
    return "Which one does not belong? $itemList.";
  }

  String _processAnswer(String answer) {
    final set = _currentSet!;
    _waitingForAnswer = false;
    _roundsPlayed++;

    final isCorrect = _matchesOddOne(answer, set.oddOne, set.oddOneAliases);

    // Progress difficulty every 2 correct answers
    if (isCorrect) {
      _correctAnswers++;
      if (_correctAnswers % 2 == 0 && _difficultyLevel < 2) {
        _difficultyLevel++;
      }
    }

    final explanation = set.explanation;

    if (_roundsPlayed >= _maxRounds) {
      _active = false;
      if (isCorrect) {
        return "That's right! ${set.oddOne} does not belong because $explanation. "
            "Wonderful thinking! ${_buildEndSummary()}";
      } else {
        return "Not quite! The answer is ${set.oddOne}. It does not belong because "
            "$explanation. Good try though! ${_buildEndSummary()}";
      }
    }

    _waitingForPlayAgain = true;

    if (isCorrect) {
      return "That's right! ${set.oddOne} does not belong because $explanation. "
          "You are so smart! Ready for the next one?";
    } else {
      return "Not quite! The answer is ${set.oddOne}. It does not belong because "
          "$explanation. But great try! Want to do another one?";
    }
  }

  bool _matchesOddOne(String answer, String oddOne, List<String> aliases) {
    final a = answer.toLowerCase().trim();
    if (a.contains(oddOne.toLowerCase())) return true;
    for (final alias in aliases) {
      if (a.contains(alias.toLowerCase())) return true;
    }
    return false;
  }

  String _buildEndSummary() {
    if (_roundsPlayed == 0) {
      return "Thanks for playing Odd One Out! Come back anytime!";
    }
    return "You got $_correctAnswers out of $_roundsPlayed right! "
        "${_correctAnswers == _roundsPlayed ? "Perfect score! You are amazing!" : "Great job! You are really learning to sort things!"}";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }

  bool _containsNo(String text) {
    const noWords = ['no', 'nah', 'nope', "don't want", 'not now'];
    for (final w in noWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }

  // -- Data --

  List<_OddOneOutSet> _getSetsForDifficulty(int level) {
    switch (level) {
      case 0:
        return _easySets;
      case 1:
        return _mediumSets;
      case 2:
        return _hardSets;
      default:
        return _easySets;
    }
  }

  static final List<_OddOneOutSet> _easySets = [
    _OddOneOutSet(
      items: ['apple', 'banana', 'car', 'mango'],
      oddOne: 'car',
      oddOneAliases: [],
      explanation: 'the others are all fruits but a car is a vehicle',
    ),
    _OddOneOutSet(
      items: ['dog', 'cat', 'chair', 'rabbit'],
      oddOne: 'chair',
      oddOneAliases: [],
      explanation: 'the others are all animals but a chair is furniture',
    ),
    _OddOneOutSet(
      items: ['red', 'blue', 'pizza', 'green'],
      oddOne: 'pizza',
      oddOneAliases: [],
      explanation: 'the others are all colors but pizza is food',
    ),
    _OddOneOutSet(
      items: ['shirt', 'pants', 'hat', 'football'],
      oddOne: 'football',
      oddOneAliases: ['ball', 'foot ball'],
      explanation: 'the others are all clothes you wear but a football is a toy',
    ),
    _OddOneOutSet(
      items: ['table', 'bed', 'lion', 'sofa'],
      oddOne: 'lion',
      oddOneAliases: [],
      explanation: 'the others are all furniture but a lion is an animal',
    ),
    _OddOneOutSet(
      items: ['milk', 'juice', 'book', 'water'],
      oddOne: 'book',
      oddOneAliases: [],
      explanation: 'the others are all things you drink but a book is for reading',
    ),
  ];

  static final List<_OddOneOutSet> _mediumSets = [
    _OddOneOutSet(
      items: ['eagle', 'butterfly', 'sparrow', 'turtle'],
      oddOne: 'turtle',
      oddOneAliases: [],
      explanation: 'the others can all fly but a turtle walks on the ground',
    ),
    _OddOneOutSet(
      items: ['ice cream', 'hot chocolate', 'popsicle', 'frozen yogurt'],
      oddOne: 'hot chocolate',
      oddOneAliases: ['chocolate', 'hot'],
      explanation: 'the others are all cold treats but hot chocolate is a warm drink',
    ),
    _OddOneOutSet(
      items: ['piano', 'guitar', 'drum', 'pencil'],
      oddOne: 'pencil',
      oddOneAliases: [],
      explanation: 'the others are all musical instruments but a pencil is for writing',
    ),
    _OddOneOutSet(
      items: ['carrot', 'potato', 'onion', 'strawberry'],
      oddOne: 'strawberry',
      oddOneAliases: ['berry'],
      explanation: 'the others all grow underground but a strawberry grows above the ground and it is a fruit',
    ),
    _OddOneOutSet(
      items: ['rain', 'snow', 'sunshine', 'telephone'],
      oddOne: 'telephone',
      oddOneAliases: ['phone'],
      explanation: 'the others are all types of weather but a telephone is something you talk on',
    ),
    _OddOneOutSet(
      items: ['doctor', 'teacher', 'pilot', 'banana'],
      oddOne: 'banana',
      oddOneAliases: [],
      explanation: 'the others are all jobs that people do but a banana is a fruit',
    ),
  ];

  static final List<_OddOneOutSet> _hardSets = [
    _OddOneOutSet(
      items: ['car', 'bicycle', 'scooter', 'horse'],
      oddOne: 'horse',
      oddOneAliases: [],
      explanation: 'the others all have wheels but a horse has legs',
    ),
    _OddOneOutSet(
      items: ['ball', 'orange', 'coin', 'box'],
      oddOne: 'box',
      oddOneAliases: [],
      explanation: 'the others are all round but a box has corners and flat sides',
    ),
    _OddOneOutSet(
      items: ['fish', 'whale', 'dolphin', 'crab'],
      oddOne: 'crab',
      oddOneAliases: [],
      explanation: 'the others all swim through the water but a crab walks sideways on the bottom',
    ),
    _OddOneOutSet(
      items: ['night', 'cave', 'shadow', 'lamp'],
      oddOne: 'lamp',
      oddOneAliases: ['light'],
      explanation: 'the others are all dark things but a lamp gives light',
    ),
    _OddOneOutSet(
      items: ['river', 'ocean', 'lake', 'desert'],
      oddOne: 'desert',
      oddOneAliases: [],
      explanation: 'the others all have lots of water but a desert is very dry',
    ),
    _OddOneOutSet(
      items: ['scissors', 'knife', 'pillow', 'needle'],
      oddOne: 'pillow',
      oddOneAliases: [],
      explanation: 'the others are all sharp things but a pillow is soft and fluffy',
    ),
  ];
}

class _OddOneOutSet {
  final List<String> items;
  final String oddOne;
  final List<String> oddOneAliases;
  final String explanation;

  const _OddOneOutSet({
    required this.items,
    required this.oddOne,
    required this.oddOneAliases,
    required this.explanation,
  });
}
