import 'dart:math';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// A fact the bot can share with a modeled citation.
class _CitedFact {
  final String fact;
  final String source;

  const _CitedFact({required this.fact, required this.source});
}

/// Citation Station: practice citing sources for facts in a fun
/// fact-exchange game.
///
/// Teaches: source attribution, information reliability awareness,
/// communication skills, information literacy.
///
/// Flow:
/// 1. Bot shares a fact and models citation: "I learned this from a science book!"
/// 2. Asks child to share a fact and tell where they learned it.
/// 3. If child does not know: "That is okay! Next time, try to remember."
/// 4. 5 fact exchanges per session.
class CitationStation extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _factsExchanged = 0;
  int _citedFacts = 0;
  int _score = 0;
  int _maxFacts = 5;

  /// 0=bot shares fact, 1=child shares fact, 2=evaluate citation
  int _phase = 0;
  final List<int> _usedIndices = [];

  static const List<_CitedFact> _botFacts = [
    _CitedFact(fact: 'Octopuses have three hearts and blue blood!', source: 'a marine biology book'),
    _CitedFact(fact: 'Honey never goes bad. Archaeologists found 3000-year-old honey that was still edible!', source: 'a National Geographic article'),
    _CitedFact(fact: 'The shortest war in history lasted only 38 minutes, between Britain and Zanzibar.', source: 'a world history encyclopedia'),
    _CitedFact(fact: 'Bananas are slightly radioactive because they contain potassium.', source: 'a science textbook'),
    _CitedFact(fact: 'A group of flamingos is called a flamboyance!', source: 'a wildlife documentary'),
    _CitedFact(fact: 'The Eiffel Tower grows about 6 inches taller in summer because metal expands in heat.', source: 'an engineering magazine'),
    _CitedFact(fact: 'Dolphins sleep with one eye open because half their brain stays awake.', source: 'a marine science journal'),
    _CitedFact(fact: 'There are more stars in the universe than grains of sand on all Earth\'s beaches.', source: 'a NASA website'),
    _CitedFact(fact: 'Cows have best friends and get stressed when separated from them.', source: 'an animal behavior research study'),
    _CitedFact(fact: 'The inventor of the Pringles can is buried in a Pringles can!', source: 'a fun facts book'),
  ];

  CitationStation();

  @override
  String get id => 'info_citation_station';

  @override
  String get name => 'Citation Station';

  @override
  String get category => 'information';

  @override
  String get description =>
      'Exchange fun facts and practice saying where you learned them.';

  @override
  List<String> get skills => [
        'source attribution',
        'information reliability',
        'communication',
        'information literacy',
      ];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.informationLiteracy;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'citation station',
          'fact exchange',
          'where did you learn that',
          'cite your source',
          'fun facts game',
        ],
        'hi': [
          'तथ्य खेल',
          'कहाँ से सीखा',
          'स्रोत बताओ',
        ],
        'te': [
          'వాస్తవాల ఆట',
          'ఎక్కడ నేర్చుకున్నావు',
          'సోర్స్ చెప్పు',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_factsExchanged == 0) return 'No facts exchanged yet.';
    return '$_factsExchanged facts exchanged, $_citedFacts cited properly. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _factsExchanged = 0;
    _citedFacts = 0;
    _score = 0;
    _phase = 0;
    _usedIndices.clear();

    return 'Welcome to Citation Station! We are going to exchange fun facts, '
        'and the key rule is: always try to say where you learned it! '
        'Watch how I do it. ${_shareBotFact()}';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      return await end();
    }

    switch (_phase) {
      case 0:
        // After bot shared a fact, child's turn
        _phase = 1;
        return 'Cool, right? Now it is your turn! Tell me a fun fact you know. '
            'And try to tell me where you learned it!';

      case 1:
        // Child shared a fact, evaluate citation
        _phase = 2;
        _factsExchanged++;
        return _evaluateCitation(lower);

      case 2:
        // After evaluation, share next bot fact or end
        if (_factsExchanged >= _maxFacts) {
          return await end();
        }
        _phase = 0;
        return _shareBotFact();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_factsExchanged == 0) {
      return 'Come back to exchange fun facts anytime!';
    }
    return 'We exchanged $_factsExchanged facts! You cited the source '
        'for $_citedFacts of them. Score: $_score points! '
        'Remember, whenever you share a fact, try to say where you learned it. '
        'It helps people trust what you say!';
  }

  String _shareBotFact() {
    final available = <int>[];
    for (int i = 0; i < _botFacts.length; i++) {
      if (!_usedIndices.contains(i)) available.add(i);
    }
    if (available.isEmpty) {
      _usedIndices.clear();
      for (int i = 0; i < _botFacts.length; i++) available.add(i);
    }

    final idx = available[_rng.nextInt(available.length)];
    _usedIndices.add(idx);
    final fact = _botFacts[idx];

    return 'Here is my fact: ${fact.fact} I learned this from ${fact.source}! '
        'See how I said where I learned it?';
  }

  String _evaluateCitation(String lower) {
    // Check if child mentioned a source
    const sourceIndicators = [
      'book', 'read', 'teacher', 'school', 'mom', 'dad', 'parent',
      'magazine', 'tv', 'video', 'youtube', 'website', 'internet',
      'documentary', 'heard', 'told me', 'learned', 'class', 'library',
      'newspaper', 'article', 'science', 'movie', 'show',
    ];

    final hasCitation = sourceIndicators.any((s) => lower.contains(s));

    if (hasCitation) {
      _citedFacts++;
      _score += 15;
      return 'Excellent! You shared a fun fact AND told me where you learned it! '
          'That makes your fact much more trustworthy. Great citation!';
    } else {
      _score += 5;
      return 'That is a cool fact! But I notice you did not say where you '
          'learned it. That is okay! Next time, try to remember whether you '
          'read it in a book, heard it from a teacher, or saw it on TV. '
          'It helps people know if the information is reliable.';
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
