import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// A single math problem with story context.
class _MathProblem {
  final String question;
  final int answer;
  final String answerWord;
  final String celebration;

  const _MathProblem({
    required this.question,
    required this.answer,
    required this.answerWord,
    required this.celebration,
  });
}

/// Number Friends: age-appropriate math through stories and counting games.
///
/// Teaches: counting, addition, subtraction, number sense, problem solving.
///
/// Flow:
/// 1. Bot presents a story-based math problem.
/// 2. Child answers.
/// 3. Bot confirms (right/wrong with encouragement).
/// 4. Progressive difficulty based on age and performance.
/// 5. 6-8 problems per session.
class NumberFriends extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _problemsSolved = 0;
  int _correctAnswers = 0;
  int _score = 0;
  int _currentLevel = 1; // 1, 2, or 3
  int _consecutiveCorrect = 0;
  int _consecutiveWrong = 0;
  _MathProblem? _currentProblem;

  static const int _maxProblems = 8;

  /// Age hint: 3-4 = level 1, 4-5 = level 2, 5-6 = level 3.
  final int _startingAge;

  NumberFriends({int startingAge = 4}) : _startingAge = startingAge;

  // -- Activity metadata --

  @override
  String get id => 'math_numbers';

  @override
  String get name => 'Number Friends';

  @override
  String get category => 'math';

  @override
  String get description =>
      'Fun counting and math with story problems about animals, toys, and more.';

  @override
  List<String> get skills => ['counting', 'addition', 'subtraction', 'problem solving'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  SkillId? get skillId => SkillId.scientificThinking;

  @override
  Map<String, List<String>> get voiceTriggers => const {
    'en': ['math game', 'counting game', 'number game', 'teach me math', 'play with numbers', 'addition game'],
    'hi': ['गणित खेल', 'गिनती खेल', 'नंबर खेल'],
    'te': ['లెక్కల ఆట', 'సంఖ్య ఆట', 'గణితం ఆట'],
  };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_problemsSolved == 0) return 'No problems yet.';
    return '$_correctAnswers out of $_problemsSolved correct. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _problemsSolved = 0;
    _correctAnswers = 0;
    _score = 0;
    _consecutiveCorrect = 0;
    _consecutiveWrong = 0;
    _active = true;

    // Set starting level based on age
    _currentLevel = _startingAge <= 3 ? 1 : (_startingAge <= 4 ? 1 : 2);

    debugPrint('[NumberFriends] Started at level $_currentLevel');

    _currentProblem = _generateProblem();
    return "Let's play with numbers! I will tell you a little story and you "
        "tell me the answer. Ready? Here we go! ${_currentProblem!.question}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    // Extract number from the child's response
    final childAnswer = _extractNumber(lower);

    if (childAnswer == null) {
      return "Hmm, I did not catch the number. Can you say just the number? "
          "${_currentProblem!.question}";
    }

    _problemsSolved++;
    final correct = childAnswer == _currentProblem!.answer;

    if (correct) {
      _correctAnswers++;
      _consecutiveCorrect++;
      _consecutiveWrong = 0;
      _score += 10 + (_currentLevel * 5); // Higher level = more points

      // Adjust difficulty
      if (_consecutiveCorrect >= 2 && _currentLevel < 3) {
        _currentLevel++;
        _consecutiveCorrect = 0;
        debugPrint('[NumberFriends] Level up to $_currentLevel');
      }

      // Check if done
      if (_problemsSolved >= _maxProblems) {
        return "${_currentProblem!.celebration} ${await end()}";
      }

      _currentProblem = _generateProblem();
      return "${_currentProblem!.celebration} Here is the next one. "
          "${_currentProblem!.question}";

    } else {
      _consecutiveWrong++;
      _consecutiveCorrect = 0;
      _score += 3; // Points for trying

      // Adjust difficulty
      if (_consecutiveWrong >= 2 && _currentLevel > 1) {
        _currentLevel--;
        _consecutiveWrong = 0;
        debugPrint('[NumberFriends] Level down to $_currentLevel');
      }

      final answer = _currentProblem!.answer;
      final answerWord = _currentProblem!.answerWord;

      // Check if done
      if (_problemsSolved >= _maxProblems) {
        return "The answer is $answer, $answerWord! That was a tricky one. ${await end()}";
      }

      _currentProblem = _generateProblem();
      return "Almost! The answer is $answer, $answerWord. "
          "Great try! Let's do another one. ${_currentProblem!.question}";
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    debugPrint('[NumberFriends] Ended, correct=$_correctAnswers/$_problemsSolved, score=$_score');

    if (_problemsSolved == 0) {
      return "Okay, we'll play with numbers another time!";
    }

    _score += _correctAnswers * 5; // Bonus for total correct

    if (_correctAnswers == _problemsSolved) {
      return "Wow, you got all $_correctAnswers right! You are a math superstar! "
          "Score: $_score points!";
    }

    return "Great job! You got $_correctAnswers out of $_problemsSolved right. "
        "You are getting better at numbers! Score: $_score points!";
  }

  // -- Problem generation --

  _MathProblem _generateProblem() {
    switch (_currentLevel) {
      case 1:
        return _generateLevel1();
      case 2:
        return _generateLevel2();
      case 3:
        return _generateLevel3();
      default:
        return _generateLevel1();
    }
  }

  /// Level 1 (age 3-4): counting 1-10, adding within 5.
  _MathProblem _generateLevel1() {
    final type = _rng.nextInt(3);

    if (type == 0) {
      // Simple counting
      final count = _rng.nextInt(5) + 1; // 1-5
      final object = _randomObject();
      return _MathProblem(
        question: "I see $count ${_plural(object, count)}. How many ${_plural(object, 2)} is that?",
        answer: count,
        answerWord: _numberWord(count),
        celebration: _randomCelebration(),
      );
    } else if (type == 1) {
      // Addition within 5
      final a = _rng.nextInt(3) + 1; // 1-3
      final b = _rng.nextInt(3) + 1; // 1-3
      final sum = a + b;
      final object = _randomObject();
      return _MathProblem(
        question: "You have $a ${_plural(object, a)}. Your friend gives you $b more. "
            "How many ${_plural(object, 2)} do you have now?",
        answer: sum,
        answerWord: _numberWord(sum),
        celebration: "That's right! $a plus $b is $sum! ${_randomCelebration()}",
      );
    } else {
      // Which is more?
      final a = _rng.nextInt(5) + 1;
      int b;
      do {
        b = _rng.nextInt(5) + 1;
      } while (b == a);
      final answer = a > b ? a : b;
      return _MathProblem(
        question: "Which is more, $a or $b?",
        answer: answer,
        answerWord: _numberWord(answer),
        celebration: "Yes! $answer is more! ${_randomCelebration()}",
      );
    }
  }

  /// Level 2 (age 4-5): adding within 10, simple subtraction.
  _MathProblem _generateLevel2() {
    final type = _rng.nextInt(2);

    if (type == 0) {
      // Addition within 10
      final a = _rng.nextInt(5) + 1; // 1-5
      final b = _rng.nextInt(5) + 1; // 1-5
      final sum = a + b;
      final object = _randomObject();
      return _MathProblem(
        question: "You have $a ${_plural(object, a)} and your mom gives you $b more. "
            "How many ${_plural(object, 2)} do you have altogether?",
        answer: sum,
        answerWord: _numberWord(sum),
        celebration: "That's right! $a plus $b is $sum! ${_randomCelebration()}",
      );
    } else {
      // Subtraction within 5
      final total = _rng.nextInt(4) + 2; // 2-5
      final take = _rng.nextInt(total - 1) + 1; // 1 to total-1
      final remain = total - take;
      final object = _randomObject();
      return _MathProblem(
        question: "You have $total ${_plural(object, total)}. You eat $take. "
            "How many ${_plural(object, 2)} are left?",
        answer: remain,
        answerWord: _numberWord(remain),
        celebration: "Yes! $total minus $take is $remain! ${_randomCelebration()}",
      );
    }
  }

  /// Level 3 (age 5-6): adding within 20, subtraction within 10.
  _MathProblem _generateLevel3() {
    final type = _rng.nextInt(2);

    if (type == 0) {
      // Addition within 20
      final a = _rng.nextInt(10) + 1; // 1-10
      final b = _rng.nextInt(10) + 1; // 1-10
      final sum = a + b;
      final object = _randomObject();
      return _MathProblem(
        question: "There are $a ${_plural(object, a)} on one table and $b ${_plural(object, b)} "
            "on another table. How many ${_plural(object, 2)} are there in total?",
        answer: sum,
        answerWord: _numberWord(sum),
        celebration: "Excellent! $a plus $b is $sum! ${_randomCelebration()}",
      );
    } else {
      // Subtraction within 10
      final total = _rng.nextInt(8) + 3; // 3-10
      final take = _rng.nextInt(total - 1) + 1;
      final remain = total - take;
      final object = _randomObject();
      return _MathProblem(
        question: "You have $total ${_plural(object, total)}. You give $take to your friend. "
            "How many ${_plural(object, 2)} do you have left?",
        answer: remain,
        answerWord: _numberWord(remain),
        celebration: "That's right! $total minus $take is $remain! ${_randomCelebration()}",
      );
    }
  }

  // -- Helpers --

  int? _extractNumber(String text) {
    // Try to find a digit
    final digitMatch = RegExp(r'\d+').firstMatch(text);
    if (digitMatch != null) {
      return int.tryParse(digitMatch.group(0)!);
    }

    // Try word numbers
    const wordToNum = <String, int>{
      'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
      'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
      'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
      'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
      'eighteen': 18, 'nineteen': 19, 'twenty': 20,
      // Hindi
      'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4,
      'पांच': 5, 'छह': 6, 'सात': 7, 'आठ': 8, 'नौ': 9, 'दस': 10,
      // Telugu
      'సున్నా': 0, 'ఒకటి': 1, 'రెండు': 2, 'మూడు': 3, 'నాలుగు': 4,
      'ఐదు': 5, 'ఆరు': 6, 'ఏడు': 7, 'ఎనిమిది': 8, 'తొమ్మిది': 9, 'పది': 10,
    };

    for (final entry in wordToNum.entries) {
      if (text.contains(entry.key)) return entry.value;
    }

    return null;
  }

  String _numberWord(int n) {
    const words = [
      'zero', 'one', 'two', 'three', 'four', 'five',
      'six', 'seven', 'eight', 'nine', 'ten',
      'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
      'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
    ];
    if (n >= 0 && n < words.length) return words[n];
    return n.toString();
  }

  String _randomObject() {
    const objects = [
      'apple', 'cookie', 'star', 'toy', 'ball',
      'banana', 'puppy', 'kitten', 'bird', 'fish',
      'flower', 'candy', 'crayon', 'sticker', 'block',
    ];
    return objects[_rng.nextInt(objects.length)];
  }

  String _plural(String noun, int count) {
    if (count == 1) return noun;
    // Simple pluralization
    if (noun.endsWith('y')) {
      return '${noun.substring(0, noun.length - 1)}ies';
    }
    if (noun.endsWith('sh') || noun.endsWith('ch') || noun.endsWith('x')) {
      return '${noun}es';
    }
    return '${noun}s';
  }

  String _randomCelebration() {
    const celebrations = [
      "You are amazing!",
      "Great job!",
      "You are so smart!",
      "Wonderful!",
      "You are a math whiz!",
      "Fantastic!",
      "Super counting!",
      "Brilliant!",
    ];
    return celebrations[_rng.nextInt(celebrations.length)];
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
