import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Real or Fake: the bot presents claims and the child judges whether they
/// are true or false. After each answer, the bot explains why and teaches
/// source-checking skills.
///
/// Covers common misconceptions and fun facts. 8 items per session with
/// score tracking. Teaches critical evaluation of information.
class RealOrFake extends Activity {
  final Random _random = Random();

  bool _active = false;
  int _round = 0;
  static const int _maxRounds = 8;
  int _score = 0;
  int _currentClaimIndex = 0;
  _Phase _phase = _Phase.idle;

  final List<int> _claimOrder = [];

  @override
  String get id => 'digital_citizenship_real_or_fake';

  @override
  String get name => 'Real or Fake?';

  @override
  String get category => 'digital_citizenship';

  @override
  String get description =>
      'Can you tell which facts are real and which are fake?';

  @override
  List<String> get skills =>
      ['digital citizenship', 'critical thinking', 'information literacy'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.digitalCitizenship;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'real or fake',
          'true or false',
          'fact check',
          'fact game',
          'is it real',
        ],
        'hi': ['सच या झूठ', 'असली या नकली', 'फैक्ट चेक'],
        'te': ['నిజమా కాదా', 'నిజం లేదా అబద్ధం'],
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
    _phase = _Phase.presenting;

    // Shuffle claims
    _claimOrder.clear();
    _claimOrder.addAll(List.generate(_claims.length, (i) => i));
    _claimOrder.shuffle(_random);
    _currentClaimIndex = 0;

    debugPrint('[RealOrFake] Started');

    return "Welcome to Real or Fake! I am going to tell you something, and "
        "you tell me if it is REAL or FAKE. Think carefully before you answer! "
        "Here is the first one. ${_presentClaim()}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    if (_phase == _Phase.presenting) {
      return _evaluateAnswer(lower);
    }

    if (_phase == _Phase.explained) {
      _round++;
      if (_round >= _maxRounds) {
        _active = false;
        return _buildEndSummary();
      }
      _phase = _Phase.presenting;
      _currentClaimIndex++;
      return "Next one! ${_presentClaim()}";
    }

    return "Is it real or fake? What do you think?";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_round == 0) return 'No claims checked yet.';
    return 'Checked $_round claims. Score: $_score/$_round.';
  }

  // -- Internal --

  String _presentClaim() {
    if (_currentClaimIndex >= _claimOrder.length) {
      _currentClaimIndex = 0;
    }
    final claim = _claims[_claimOrder[_currentClaimIndex]];
    return "${claim.statement} Real or fake?";
  }

  String _evaluateAnswer(String lower) {
    final claim = _claims[_claimOrder[_currentClaimIndex]];
    final saidReal = lower.contains('real') ||
        lower.contains('true') ||
        lower.contains('yes');
    final saidFake = lower.contains('fake') ||
        lower.contains('false') ||
        lower.contains('no') ||
        lower.contains('lie');

    if (!saidReal && !saidFake) {
      return "Is it real or fake? Say real if you think it is true, "
          "or fake if you think it is made up!";
    }

    final childSaysReal = saidReal && !saidFake;
    final isCorrect = childSaysReal == claim.isReal;

    _phase = _Phase.explained;

    if (isCorrect) {
      _score++;
      return "That is correct! ${claim.isReal ? 'It is real!' : 'It is fake!'} "
          "${claim.explanation} Great thinking! "
          "Score: $_score out of ${_round + 1}. Ready for the next one?";
    }

    return "Not quite! ${claim.isReal ? 'This one is actually real!' : 'This one is actually fake!'} "
        "${claim.explanation} That is okay, now you know! "
        "Score: $_score out of ${_round + 1}. Ready for the next one?";
  }

  String _buildEndSummary() {
    if (_round == 0) {
      return "Thanks for trying Real or Fake! Come back to test your fact-checking skills!";
    }

    String verdict;
    if (_score == _round) {
      verdict = "Perfect score! You are a fact-checking superstar!";
    } else if (_score >= _round * 0.7) {
      verdict = "Great job! You have strong fact-checking skills!";
    } else if (_score >= _round * 0.5) {
      verdict = "Good try! Keep questioning things and you will get even better!";
    } else {
      verdict = "That was tricky! The important thing is to always check before you believe.";
    }

    return "You checked $_round claims and got $_score right! $verdict "
        "Remember: just because something sounds true does not mean it is. "
        "Always ask: where did this information come from?";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<_Claim> _claims = [
    _Claim(
      statement: 'Elephants can fly.',
      isReal: false,
      explanation:
          'Elephants are the largest land animals and are way too heavy to fly. '
          'Dumbo is just a cartoon!',
    ),
    _Claim(
      statement: 'Honey never expires. Archaeologists found 3000-year-old honey that was still good.',
      isReal: true,
      explanation:
          'Honey has natural preservatives. Scientists found edible honey in '
          'ancient Egyptian tombs! Its low moisture and high acidity prevent bacteria.',
    ),
    _Claim(
      statement: 'Humans use only 10 percent of their brain.',
      isReal: false,
      explanation:
          'This is a popular myth! Brain scans show we use all parts of our '
          'brain. Different parts are active at different times.',
    ),
    _Claim(
      statement: 'Octopuses have 3 hearts.',
      isReal: true,
      explanation:
          'Octopuses really do have 3 hearts! Two pump blood to the gills '
          'and one pumps it to the rest of the body.',
    ),
    _Claim(
      statement: 'Eating carrots gives you night vision.',
      isReal: false,
      explanation:
          'This myth started during World War 2! The British spread this story '
          'to hide the fact that they had invented radar. Carrots are healthy, '
          'but they do not give you super vision.',
    ),
    _Claim(
      statement: 'The Great Wall of China is visible from space.',
      isReal: false,
      explanation:
          'Astronauts have confirmed that the Great Wall is not visible from '
          'space with the naked eye. It is long but too narrow to see from that far.',
    ),
    _Claim(
      statement: 'Bananas are berries, but strawberries are not.',
      isReal: true,
      explanation:
          'In botany, a berry comes from a single flower with one ovary. '
          'Bananas qualify, but strawberries do not. Science names are funny!',
    ),
    _Claim(
      statement: 'Lightning never strikes the same place twice.',
      isReal: false,
      explanation:
          'Lightning absolutely can strike the same place multiple times! '
          'Tall buildings like the Empire State Building get hit about 20 times a year.',
    ),
    _Claim(
      statement: 'A group of flamingos is called a flamboyance.',
      isReal: true,
      explanation:
          'Yes, a group of flamingos really is called a flamboyance! '
          'With their bright pink color, the name fits perfectly.',
    ),
    _Claim(
      statement: 'Goldfish have a memory of only 3 seconds.',
      isReal: false,
      explanation:
          'Studies have shown goldfish can remember things for months! '
          'They can even learn to navigate mazes. The 3-second myth is not true.',
    ),
    _Claim(
      statement: 'Venus is the hottest planet in our solar system, even hotter than Mercury.',
      isReal: true,
      explanation:
          'Even though Mercury is closer to the Sun, Venus is hotter because '
          'its thick atmosphere traps heat. It reaches about 465 degrees Celsius!',
    ),
    _Claim(
      statement: 'Cracking your knuckles causes arthritis.',
      isReal: false,
      explanation:
          'Scientists have studied this and found no link between knuckle '
          'cracking and arthritis. The sound comes from gas bubbles popping in your joints.',
    ),
  ];
}

enum _Phase {
  idle,
  presenting,
  explained,
}

/// A factual claim with its truth value and explanation.
class _Claim {
  final String statement;
  final bool isReal;
  final String explanation;

  const _Claim({
    required this.statement,
    required this.isReal,
    required this.explanation,
  });
}
