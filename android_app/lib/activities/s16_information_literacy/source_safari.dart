import 'dart:math';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// A pair of claims where one is more trustworthy than the other.
class _SourcePair {
  final String claimA;
  final String sourceA;
  final String claimB;
  final String sourceB;

  /// 'a' or 'b' — which is more trustworthy.
  final String betterSource;
  final String explanation;

  const _SourcePair({
    required this.claimA,
    required this.sourceA,
    required this.claimB,
    required this.sourceB,
    required this.betterSource,
    required this.explanation,
  });
}

/// Source Safari: two versions of a "fact" are presented and the child
/// decides which source to trust.
///
/// Teaches: source evaluation, authority, evidence awareness,
/// information literacy, critical thinking.
///
/// Flow:
/// 1. Bot presents two claims from different sources.
/// 2. Child picks which one they trust more.
/// 3. Bot explains why one source is more reliable.
/// 4. Teaches concepts: authority, expertise, evidence, peer review.
/// 5. 5 pairs per session, score tracked.
class SourceSafari extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _pairsCompleted = 0;
  int _correctChoices = 0;
  int _score = 0;
  int _maxPairs = 5;

  /// 0=present pair, 1=evaluate choice
  int _phase = 0;
  _SourcePair? _currentPair;
  final List<int> _usedIndices = [];

  static const List<_SourcePair> _pairs = [
    _SourcePair(
      claimA: 'Dinosaurs disappeared about 66 million years ago.',
      sourceA: 'A science encyclopedia reviewed by paleontologists',
      claimB: 'Dinosaurs disappeared about 100 million years ago.',
      sourceB: 'A random blog post with no references',
      betterSource: 'a',
      explanation: 'The encyclopedia was reviewed by experts called paleontologists '
          'who study dinosaurs. The blog post has no references and no expert '
          'checked it. Experts and references make a source more trustworthy!',
    ),
    _SourcePair(
      claimA: 'Vaccines are safe and protect us from diseases.',
      sourceA: 'The World Health Organization, a group of doctors',
      claimB: 'Vaccines are dangerous and should be avoided.',
      sourceB: 'A social media post by someone with no medical training',
      betterSource: 'a',
      explanation: 'The World Health Organization has thousands of doctors and '
          'scientists who study vaccines carefully. A social media post from '
          'someone with no medical training is not a reliable source for '
          'health information. Always trust experts on health topics!',
    ),
    _SourcePair(
      claimA: 'The Earth is round, like a ball.',
      sourceA: 'NASA, the space agency that has photos from space',
      claimB: 'The Earth is flat, like a pancake.',
      sourceB: 'A video by someone on YouTube with no scientific training',
      betterSource: 'a',
      explanation: 'NASA has actual photographs of Earth from space showing it '
          'is round. They have thousands of scientists and real evidence. '
          'A YouTube video without evidence is just opinion, not fact!',
    ),
    _SourcePair(
      claimA: 'Humans landed on the moon in 1969.',
      sourceA: 'NASA records, confirmed by scientists worldwide',
      claimB: 'The moon landing was fake, filmed in a studio.',
      sourceB: 'A conspiracy website with no evidence',
      betterSource: 'a',
      explanation: 'The moon landing was confirmed by scientists from many '
          'countries, even countries that were competing with America. '
          'There are moon rocks, photos, and reflectors on the moon. '
          'A website claiming it is fake provides no real evidence.',
    ),
    _SourcePair(
      claimA: 'Drinking water is important for health, about 8 glasses a day.',
      sourceA: 'A children\'s health book written by a doctor',
      claimB: 'You only need to drink water when you feel thirsty.',
      sourceB: 'Your friend who heard it from another friend',
      betterSource: 'a',
      explanation: 'A health book written by a doctor uses medical research. '
          'Information passed from friend to friend often gets changed, '
          'like a game of telephone. Books by experts are more reliable!',
    ),
    _SourcePair(
      claimA: 'Sugar in large amounts is not good for your teeth.',
      sourceA: 'A dentist who treats teeth every day',
      claimB: 'Sugar is fine for your teeth as long as you brush.',
      sourceB: 'A candy company advertisement',
      betterSource: 'a',
      explanation: 'A dentist has studied teeth for years and sees what sugar '
          'does. A candy company wants to sell candy, so they might not tell '
          'you the full truth. Be careful of information from people who '
          'are trying to sell you something!',
    ),
    _SourcePair(
      claimA: 'Polar bears are endangered because ice is melting.',
      sourceA: 'National Geographic magazine, researched by wildlife experts',
      claimB: 'Polar bears are doing great and their numbers are increasing.',
      sourceB: 'A comment on social media with no sources cited',
      betterSource: 'a',
      explanation: 'National Geographic has reporters and scientists who visit '
          'the Arctic and study polar bears. A social media comment with no '
          'sources is just someone\'s opinion, not researched fact!',
    ),
    _SourcePair(
      claimA: 'Exercise helps your brain work better and makes you smarter.',
      sourceA: 'A university research study published in a science journal',
      claimB: 'Exercise only helps your body, not your brain.',
      sourceB: 'Something someone said at a party',
      betterSource: 'a',
      explanation: 'University research studies are tested and reviewed by other '
          'scientists before they are published. This is called peer review. '
          'Something heard at a party has not been checked by anyone!',
    ),
  ];

  SourceSafari();

  @override
  String get id => 'info_source_safari';

  @override
  String get name => 'Source Safari';

  @override
  String get category => 'information';

  @override
  String get description =>
      'Two claims, two sources. Which one do you trust? Learn to spot reliable information.';

  @override
  List<String> get skills => [
        'source evaluation',
        'critical thinking',
        'information literacy',
        'evidence awareness',
      ];

  @override
  int get minAge => 6;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.informationLiteracy;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'source safari',
          'which is true',
          'spot the fake',
          'source game',
          'fact check',
        ],
        'hi': [
          'सोर्स खेल',
          'कौन सा सच है',
          'फैक्ट चेक',
        ],
        'te': [
          'సోర్స్ ఆట',
          'ఏది నిజం',
          'ఫ్యాక్ట్ చెక్',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_pairsCompleted == 0) return 'No pairs evaluated yet.';
    return '$_correctChoices correct out of $_pairsCompleted. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _pairsCompleted = 0;
    _correctChoices = 0;
    _score = 0;
    _usedIndices.clear();

    return 'Welcome to Source Safari! I will show you two different claims '
        'from two different sources. You decide which source you trust more. '
        'Ready? ${_presentNextPair()}';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      return await end();
    }

    if (_phase == 0) {
      return _evaluateChoice(lower);
    }

    // After explanation, present next pair or end
    if (_pairsCompleted >= _maxPairs) {
      return await end();
    }

    return _presentNextPair();
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_pairsCompleted == 0) {
      return 'Come back to practice spotting reliable sources!';
    }
    return 'Source Safari complete! You got $_correctChoices out of '
        '$_pairsCompleted correct! Score: $_score points! '
        'Remember: check who said it, look for evidence, and '
        'trust experts over random people!';
  }

  String _presentNextPair() {
    final available = <int>[];
    for (int i = 0; i < _pairs.length; i++) {
      if (!_usedIndices.contains(i)) available.add(i);
    }
    if (available.isEmpty) {
      _active = false;
      return 'You evaluated all the source pairs! Amazing! $_correctChoices '
          'correct out of $_pairsCompleted!';
    }

    final idx = available[_rng.nextInt(available.length)];
    _usedIndices.add(idx);
    _currentPair = _pairs[idx];
    _phase = 0;

    return 'Here are two claims. '
        'Claim A: ${_currentPair!.claimA} This comes from ${_currentPair!.sourceA}. '
        'Claim B: ${_currentPair!.claimB} This comes from ${_currentPair!.sourceB}. '
        'Which source do you trust more, A or B?';
  }

  String _evaluateChoice(String lower) {
    _pairsCompleted++;
    _phase = 1;

    final pair = _currentPair!;
    bool choseA = lower.contains('a') || lower.contains('first') || lower.contains('one');
    bool choseB = lower.contains('b') || lower.contains('second') || lower.contains('two');

    // If neither is clear, try to detect from content
    if (!choseA && !choseB) {
      choseA = true; // Default to A
    }

    final childChoice = choseA ? 'a' : 'b';
    final isCorrect = childChoice == pair.betterSource;

    if (isCorrect) {
      _correctChoices++;
      _score += 15;
      return 'Correct! Great detective work! ${pair.explanation} '
          '${_pairsCompleted < _maxPairs ? 'Ready for the next one?' : ''}';
    } else {
      _score += 5; // Participation points
      return 'Good try! Actually, the better source here is '
          '${pair.betterSource == 'a' ? 'Claim A' : 'Claim B'}. '
          '${pair.explanation} '
          '${_pairsCompleted < _maxPairs ? 'Let us try another one!' : ''}';
    }
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
