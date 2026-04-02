import 'dart:math';

import '../activity_base.dart';

/// A riddle/mystery game that teaches deductive reasoning.
///
/// The bot gives clues one at a time and the child guesses what it is.
/// Fewer clues needed = higher score. Riddles are grouped by difficulty
/// matching the child's age range.
class MysterySolver extends Activity {
  final Random _random = Random();

  bool _active = false;
  int _roundsPlayed = 0;
  int _totalScore = 0;
  int _bestScore = 0;

  // Current riddle state
  _Riddle? _currentRiddle;
  int _clueIndex = 0;
  int _maxRounds = 5;
  bool _waitingForGuess = false;
  bool _waitingForReady = false;
  bool _waitingForPlayAgain = false;

  // Difficulty tracking — start easy and progress
  int _difficultyLevel = 0; // 0=easy, 1=medium, 2=hard
  final List<int> _usedRiddleIndices = [];

  @override
  String get id => 'thinking_mystery';

  @override
  String get name => 'Mystery Solver';

  @override
  String get category => 'thinking';

  @override
  String get description =>
      'Solve riddles by listening to clues. Fewer clues means a higher score!';

  @override
  List<String> get skills => ['deductive reasoning', 'critical thinking', 'vocabulary'];

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
    _totalScore = 0;
    _bestScore = 0;
    _difficultyLevel = 0;
    _usedRiddleIndices.clear();
    _waitingForReady = true;
    _waitingForGuess = false;
    _waitingForPlayAgain = false;

    return "Let's play Mystery Solver! I'm going to think of something and "
        "give you clues. You try to guess what it is! The fewer clues you "
        "need, the higher your score. Are you ready?";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    // Check for quit signals
    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    // Waiting for initial "ready" or "yes"
    if (_waitingForReady) {
      _waitingForReady = false;
      return _startNewRiddle();
    }

    // Waiting for "yes/no" to play again
    if (_waitingForPlayAgain) {
      _waitingForPlayAgain = false;
      if (_containsNo(lower)) {
        _active = false;
        return _buildEndSummary();
      }
      return _startNewRiddle();
    }

    // Waiting for a guess
    if (_waitingForGuess && _currentRiddle != null) {
      return _processGuess(lower);
    }

    // Fallback
    return "Hmm, I did not quite catch that. Let me give you a clue! "
        "${_currentRiddle?.clues[_clueIndex] ?? 'Are you ready to play?'}";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_roundsPlayed == 0) return 'No mysteries solved yet.';
    return 'Solved $_roundsPlayed mysteries. Best score: $_bestScore stars. '
        'Total score: $_totalScore stars.';
  }

  // -- Internal helpers --

  String _startNewRiddle() {
    final riddles = _getRiddlesForDifficulty(_difficultyLevel);

    // Find an unused riddle
    _Riddle? chosen;
    for (int i = 0; i < riddles.length; i++) {
      final globalIndex = _difficultyLevel * 100 + i;
      if (!_usedRiddleIndices.contains(globalIndex)) {
        chosen = riddles[i];
        _usedRiddleIndices.add(globalIndex);
        break;
      }
    }

    // If all used at this difficulty, try next difficulty or reset
    if (chosen == null) {
      _difficultyLevel = (_difficultyLevel + 1).clamp(0, 2);
      _usedRiddleIndices.clear();
      final fallback = _getRiddlesForDifficulty(_difficultyLevel);
      chosen = fallback[_random.nextInt(fallback.length)];
    }

    _currentRiddle = chosen;
    _clueIndex = 0;
    _waitingForGuess = true;

    return "Okay, I'm thinking of something! Here is your first clue. "
        "${chosen.clues[0]} What do you think it is?";
  }

  String _processGuess(String guess) {
    final riddle = _currentRiddle!;
    final isCorrect = _isCorrectGuess(guess, riddle.answer, riddle.acceptableAnswers);

    if (isCorrect) {
      final cluesUsed = _clueIndex + 1;
      final score = _calculateScore(cluesUsed, riddle.clues.length);
      _totalScore += score;
      if (score > _bestScore) _bestScore = score;
      _roundsPlayed++;
      _waitingForGuess = false;
      _waitingForPlayAgain = true;

      // Progress difficulty every 2 correct answers
      if (_roundsPlayed % 2 == 0 && _difficultyLevel < 2) {
        _difficultyLevel++;
      }

      // Check if we've reached max rounds
      if (_roundsPlayed >= _maxRounds) {
        _active = false;
        return "Yes! It's ${riddle.displayAnswer}! Amazing job! You solved it "
            "in $cluesUsed ${cluesUsed == 1 ? 'clue' : 'clues'}! That's $score "
            "${score == 1 ? 'star' : 'stars'}! ${_buildEndSummary()}";
      }

      return "Yes! It's ${riddle.displayAnswer}! Great job! You solved it "
          "in $cluesUsed ${cluesUsed == 1 ? 'clue' : 'clues'}! That's $score "
          "${score == 1 ? 'star' : 'stars'}! Want to try another mystery?";
    }

    // Wrong guess — give next clue if available
    _clueIndex++;
    if (_clueIndex < riddle.clues.length) {
      return "Good guess, but that's not it! Here is another clue. "
          "${riddle.clues[_clueIndex]} What do you think it is now?";
    }

    // Out of clues — reveal answer
    _roundsPlayed++;
    _waitingForGuess = false;
    _waitingForPlayAgain = true;

    if (_roundsPlayed >= _maxRounds) {
      _active = false;
      return "The answer was ${riddle.displayAnswer}! That was a tricky one. "
          "${_buildEndSummary()}";
    }

    return "The answer was ${riddle.displayAnswer}! That was a tricky one. "
        "Don't worry, you'll get the next one! Want to try another mystery?";
  }

  bool _isCorrectGuess(String guess, String answer, List<String> acceptable) {
    final g = guess.toLowerCase().trim();
    final a = answer.toLowerCase().trim();

    if (g.contains(a) || a.contains(g)) return true;
    for (final alt in acceptable) {
      if (g.contains(alt.toLowerCase())) return true;
    }
    return false;
  }

  int _calculateScore(int cluesUsed, int totalClues) {
    // Fewer clues = more stars (max 3)
    if (cluesUsed == 1) return 3;
    if (cluesUsed == 2) return 2;
    return 1;
  }

  String _buildEndSummary() {
    if (_roundsPlayed == 0) {
      return "Thanks for playing Mystery Solver! Come back anytime!";
    }
    return "You solved $_roundsPlayed ${_roundsPlayed == 1 ? 'mystery' : 'mysteries'} "
        "and earned $_totalScore ${_totalScore == 1 ? 'star' : 'stars'}! "
        "Your best was $_bestScore ${_bestScore == 1 ? 'star' : 'stars'} on a "
        "single mystery! You're a great detective!";
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

  // -- Riddle data --

  List<_Riddle> _getRiddlesForDifficulty(int level) {
    switch (level) {
      case 0:
        return _easyRiddles;
      case 1:
        return _mediumRiddles;
      case 2:
        return _hardRiddles;
      default:
        return _easyRiddles;
    }
  }

  static final List<_Riddle> _easyRiddles = [
    _Riddle(
      answer: 'cat',
      displayAnswer: 'a cat',
      acceptableAnswers: ['cat', 'kitty', 'kitten'],
      clues: [
        'It has four legs.',
        'It says meow.',
        'It likes to chase mice.',
      ],
    ),
    _Riddle(
      answer: 'dog',
      displayAnswer: 'a dog',
      acceptableAnswers: ['dog', 'puppy', 'doggy'],
      clues: [
        'It has four legs and a tail.',
        'It says woof woof.',
        'It loves to fetch sticks.',
      ],
    ),
    _Riddle(
      answer: 'banana',
      displayAnswer: 'a banana',
      acceptableAnswers: ['banana'],
      clues: [
        'It is a fruit.',
        'It is yellow.',
        'Monkeys love to eat it.',
      ],
    ),
    _Riddle(
      answer: 'sun',
      displayAnswer: 'the sun',
      acceptableAnswers: ['sun'],
      clues: [
        'You can see it in the sky.',
        'It is very bright and yellow.',
        'It comes out during the day and goes away at night.',
      ],
    ),
    _Riddle(
      answer: 'elephant',
      displayAnswer: 'an elephant',
      acceptableAnswers: ['elephant'],
      clues: [
        'It is very very big.',
        'It is gray.',
        'It has a long trunk and big ears.',
      ],
    ),
    _Riddle(
      answer: 'fish',
      displayAnswer: 'a fish',
      acceptableAnswers: ['fish', 'fishy'],
      clues: [
        'It lives in water.',
        'It can swim but cannot walk.',
        'It has fins and scales.',
      ],
    ),
  ];

  static final List<_Riddle> _mediumRiddles = [
    _Riddle(
      answer: 'umbrella',
      displayAnswer: 'an umbrella',
      acceptableAnswers: ['umbrella'],
      clues: [
        'You use it when it rains.',
        'It opens up wide above your head.',
        'It keeps you dry.',
        'It can fold up small when you do not need it.',
      ],
    ),
    _Riddle(
      answer: 'bicycle',
      displayAnswer: 'a bicycle',
      acceptableAnswers: ['bicycle', 'bike', 'cycle'],
      clues: [
        'It has two wheels.',
        'You sit on it and use your legs.',
        'It has pedals that go round and round.',
        'You can ride it to the park.',
      ],
    ),
    _Riddle(
      answer: 'moon',
      displayAnswer: 'the moon',
      acceptableAnswers: ['moon'],
      clues: [
        'You can see it in the sky.',
        'It comes out at night.',
        'Sometimes it is round, sometimes it looks like a banana.',
        'It shines white in the dark sky.',
      ],
    ),
    _Riddle(
      answer: 'tree',
      displayAnswer: 'a tree',
      acceptableAnswers: ['tree'],
      clues: [
        'It is alive but does not walk.',
        'It is green and very tall.',
        'Birds like to sit on it.',
        'It has leaves and branches.',
      ],
    ),
    _Riddle(
      answer: 'ice cream',
      displayAnswer: 'ice cream',
      acceptableAnswers: ['ice cream', 'icecream'],
      clues: [
        'It is a yummy treat.',
        'It is very cold.',
        'It comes in many flavors like chocolate and vanilla.',
        'It melts if you do not eat it fast!',
      ],
    ),
    _Riddle(
      answer: 'train',
      displayAnswer: 'a train',
      acceptableAnswers: ['train'],
      clues: [
        'It is very long.',
        'It goes choo choo.',
        'It moves on tracks.',
        'Many people ride inside it to go places.',
      ],
    ),
  ];

  static final List<_Riddle> _hardRiddles = [
    _Riddle(
      answer: 'doctor',
      displayAnswer: 'a doctor',
      acceptableAnswers: ['doctor', 'dr'],
      clues: [
        'This person helps people every day.',
        'They wear a white coat.',
        'They use a stethoscope to listen to your heart.',
        'You visit them when you are sick.',
        'They work in a hospital.',
      ],
    ),
    _Riddle(
      answer: 'rainbow',
      displayAnswer: 'a rainbow',
      acceptableAnswers: ['rainbow'],
      clues: [
        'You can see it in the sky but you cannot touch it.',
        'It has many beautiful colors.',
        'It appears after the rain stops.',
        'It looks like a big colorful arch.',
        'It has seven colors.',
      ],
    ),
    _Riddle(
      answer: 'clock',
      displayAnswer: 'a clock',
      acceptableAnswers: ['clock', 'watch'],
      clues: [
        'You can find it on the wall or on your wrist.',
        'It has numbers on it.',
        'It goes tick tock tick tock.',
        'It has hands that move in a circle.',
        'It tells you the time.',
      ],
    ),
    _Riddle(
      answer: 'library',
      displayAnswer: 'a library',
      acceptableAnswers: ['library'],
      clues: [
        'It is a special place you can visit.',
        'You have to be quiet there.',
        'It is full of books.',
        'You can borrow things and bring them back later.',
        'People go there to read and learn.',
      ],
    ),
    _Riddle(
      answer: 'shadow',
      displayAnswer: 'a shadow',
      acceptableAnswers: ['shadow'],
      clues: [
        'Everyone has one but you cannot pick it up.',
        'It is always dark.',
        'It follows you everywhere outside.',
        'It changes size during the day.',
        'The sun makes it appear behind you.',
      ],
    ),
  ];
}

class _Riddle {
  final String answer;
  final String displayAnswer;
  final List<String> acceptableAnswers;
  final List<String> clues;

  const _Riddle({
    required this.answer,
    required this.displayAnswer,
    required this.acceptableAnswers,
    required this.clues,
  });
}
