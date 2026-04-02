import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// A common "fact" that may or may not be true.
class _ScienceMyth {
  final String claim;
  final bool isTrue;
  final String explanation;

  const _ScienceMyth({
    required this.claim,
    required this.isTrue,
    required this.explanation,
  });
}

/// ScienceMythBuster: investigate common "facts" and determine truth.
///
/// Teaches: critical thinking, evidence evaluation, skepticism,
/// scientific literacy.
///
/// Flow:
/// 1. Bot presents a common claim or "fact."
/// 2. Child guesses: true or false?
/// 3. Bot reveals the answer and explains using LLM.
/// 4. Discusses how to verify claims.
/// 5. 3-4 myths per session.
class ScienceMythBuster extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _mythsInvestigated = 0;
  int _correctGuesses = 0;
  int _score = 0;

  /// 0=present myth, 1=reveal and explain, 2=next or end
  int _phase = 0;
  _ScienceMyth? _currentMyth;
  final List<int> _usedIndices = [];
  final List<Map<String, String>> _history = [];

  static const int _maxMyths = 4;

  static const List<_ScienceMyth> _myths = [
    _ScienceMyth(
      claim: 'Goldfish have a memory of only 3 seconds.',
      isTrue: false,
      explanation: 'Goldfish can actually remember things for months! '
          'Scientists have trained goldfish to push levers and navigate '
          'mazes. Their memory is much better than people think.',
    ),
    _ScienceMyth(
      claim: 'Lightning never strikes the same place twice.',
      isTrue: false,
      explanation: 'Lightning strikes the same place many times! Tall '
          'buildings like the Empire State Building get struck about '
          '25 times every year. Lightning likes to hit the tallest thing around.',
    ),
    _ScienceMyth(
      claim: 'Honey never goes bad. It can last for thousands of years.',
      isTrue: true,
      explanation: 'Honey found in ancient Egyptian tombs was still good '
          'to eat after 3000 years! Honey has very little water and is '
          'slightly acidic, which stops bacteria from growing in it.',
    ),
    _ScienceMyth(
      claim: 'We only use 10 percent of our brains.',
      isTrue: false,
      explanation: 'We actually use all parts of our brain! Brain scans '
          'show that over a day, every area of the brain is active. '
          'Different parts work at different times depending on what we are doing.',
    ),
    _ScienceMyth(
      claim: 'Octopuses have three hearts.',
      isTrue: true,
      explanation: 'Octopuses really do have three hearts! Two hearts '
          'pump blood to the gills to get oxygen, and one heart pumps '
          'blood to the rest of the body. When they swim, the main heart '
          'actually stops, which is why they prefer crawling!',
    ),
    _ScienceMyth(
      claim: 'The Great Wall of China can be seen from space.',
      isTrue: false,
      explanation: 'Even though it is very long, the Great Wall is not wide '
          'enough to see from space with just your eyes. Astronauts have '
          'confirmed this. You would need binoculars or a camera to spot it.',
    ),
    _ScienceMyth(
      claim: 'Bananas grow on trees.',
      isTrue: false,
      explanation: 'Banana plants look like trees, but they are actually '
          'the world\'s largest herbs! Their trunk is not made of wood, '
          'it is made of tightly packed leaves.',
    ),
    _ScienceMyth(
      claim: 'Sound travels faster in water than in air.',
      isTrue: true,
      explanation: 'Sound travels about 4 times faster in water than in air! '
          'That is because water molecules are packed closer together, so the '
          'vibrations can pass from one molecule to the next more quickly.',
    ),
    _ScienceMyth(
      claim: 'Chameleons change color to blend in with their surroundings.',
      isTrue: false,
      explanation: 'Chameleons mostly change color to communicate and '
          'control their temperature, not to hide! They change color based '
          'on their mood, to talk to other chameleons, or to warm up and cool down.',
    ),
    _ScienceMyth(
      claim: 'A year on Venus is shorter than a day on Venus.',
      isTrue: true,
      explanation: 'Venus spins so incredibly slowly that one full day on '
          'Venus takes 243 Earth days. But it only takes 225 Earth days to '
          'go around the Sun. So a Venus day is actually longer than a Venus year!',
    ),
  ];

  ScienceMythBuster({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'science_myth_buster';

  @override
  String get name => 'Science Myth Buster';

  @override
  String get category => 'science';

  @override
  String get description =>
      'Investigate common "facts" and discover the truth behind them.';

  @override
  List<String> get skills => [
        'critical thinking',
        'evidence evaluation',
        'skepticism',
        'scientific literacy',
      ];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.scientificThinking;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'myth buster',
          'science myths',
          'true or false science',
          'fact or fiction',
          'is it true',
        ],
        'hi': [
          'मिथ बस्टर',
          'सच या झूठ',
          'विज्ञान सच',
        ],
        'te': [
          'మిత్ బస్టర్',
          'నిజమా అబద్ధమా',
          'సైన్స్ నిజం',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_mythsInvestigated == 0) return 'No myths investigated yet.';
    return '$_correctGuesses out of $_mythsInvestigated correct. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _mythsInvestigated = 0;
    _correctGuesses = 0;
    _score = 0;
    _phase = 0;
    _usedIndices.clear();
    _history.clear();

    return 'Welcome to Science Myth Buster! I am going to tell you some '
        'things that people often say. Your job is to figure out: is it '
        'true, or is it a myth? Ready? ${_presentNewMyth()}';
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
        // Child guessed true or false
        final guessedTrue = _guessedTrue(lower);
        final guessedFalse = _guessedFalse(lower);

        if (!guessedTrue && !guessedFalse) {
          return 'Is it true or is it a myth? Just say true or false!';
        }

        _mythsInvestigated++;
        _phase = 1;
        final myth = _currentMyth!;
        final correct =
            (guessedTrue && myth.isTrue) || (guessedFalse && !myth.isTrue);

        if (correct) {
          _correctGuesses++;
          _score += 15;
          _history.add({'role': 'user', 'content': childSaid});
          final explanation = await _getLlmResponse(
            childSaid,
            'The child correctly guessed that "${myth.claim}" is '
            '${myth.isTrue ? "true" : "a myth"}. Celebrate their correct '
            'answer! Then explain why: ${myth.explanation} '
            'Make it fun and memorable. 2-3 sentences.',
          );
          return 'You got it! $explanation '
              'Would you like to bust another myth?';
        } else {
          _score += 5;
          _history.add({'role': 'user', 'content': childSaid});
          final explanation = await _getLlmResponse(
            childSaid,
            'The child guessed wrong about "${myth.claim}". It is actually '
            '${myth.isTrue ? "true" : "a myth"}. Gently correct them '
            'without making them feel bad. Explain: ${myth.explanation} '
            'Make it interesting. 2-3 sentences.',
          );
          return 'Tricky one! $explanation '
              'Would you like to try another?';
        }

      case 1:
        // Next myth or end
        if (_isAffirmative(lower) && _mythsInvestigated < _maxMyths) {
          _history.clear();
          return _presentNewMyth();
        }
        return await end();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_mythsInvestigated == 0) {
      return 'Come back to bust some myths anytime! Stay curious '
          'and always question what you hear.';
    }
    return 'Great myth busting! You investigated $_mythsInvestigated '
        'claim${_mythsInvestigated != 1 ? 's' : ''} and got '
        '$_correctGuesses right. Score: $_score points! '
        'Remember, a good scientist always checks the facts!';
  }

  String _presentNewMyth() {
    final available = <int>[];
    for (int i = 0; i < _myths.length; i++) {
      if (!_usedIndices.contains(i)) available.add(i);
    }
    if (available.isEmpty) {
      _usedIndices.clear();
      for (int i = 0; i < _myths.length; i++) {
        available.add(i);
      }
    }

    final idx = available[_rng.nextInt(available.length)];
    _usedIndices.add(idx);
    _currentMyth = _myths[idx];
    _phase = 0;

    return 'People say: ${_currentMyth!.claim} '
        'Do you think this is true, or is it a myth?';
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt =
        'You are Kore, a science educator for kids aged 5-14. You help '
        'children evaluate common claims and discover scientific truth. '
        '$guidance '
        'Rules: Never make the child feel bad for guessing wrong. '
        'Make explanations memorable and fun. Connect to things kids know. '
        'No markdown, no bullets, no emojis. Speak naturally.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._history,
    ];

    try {
      final provider = _llmRouter.getProvider();
      if (onSpeakSentence != null) {
        return await _streamWithTts(provider, messages);
      }
      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }
      final response = buffer.toString().trim();
      if (response.isNotEmpty) {
        _history.add({'role': 'assistant', 'content': response});
        return response;
      }
      return _currentMyth!.explanation;
    } catch (e) {
      return _currentMyth!.explanation;
    }
  }

  Future<String> _streamWithTts(
    LlmProvider provider,
    List<Map<String, String>> messages,
  ) async {
    _sentenceBuffer.reset();
    final fullResponse = <String>[];
    try {
      await for (final token in provider.stream(messages)) {
        final sentence = _sentenceBuffer.feed(token);
        if (sentence != null) {
          fullResponse.add(sentence);
          await onSpeakSentence!(sentence);
        }
      }
      final remaining = _sentenceBuffer.flush();
      if (remaining != null) {
        fullResponse.add(remaining);
        await onSpeakSentence!(remaining);
      }
      final result = fullResponse.join(' ');
      if (result.isNotEmpty) {
        _history.add({'role': 'assistant', 'content': result});
      }
      return result.isNotEmpty ? result : _currentMyth!.explanation;
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return _currentMyth!.explanation;
    }
  }

  bool _guessedTrue(String text) {
    const words = ['true', 'real', 'right', 'correct', 'fact', 'believe',
      'yes', 'it is true', 'i think so', 'सच', 'నిజం'];
    return words.any((w) => text.contains(w));
  }

  bool _guessedFalse(String text) {
    const words = ['false', 'fake', 'wrong', 'myth', 'not true', 'lie',
      'no way', 'nope', 'fiction', 'झूठ', 'अबद్ధం'];
    return words.any((w) => text.contains(w));
  }

  bool _isAffirmative(String text) {
    const words = ['yes', 'yeah', 'sure', 'okay', 'another', 'more',
      'next', 'one more', 'हाँ', 'और', 'అవును', 'ఇంకొకటి'];
    return words.any((w) => text.contains(w));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough',
      'no', 'no thanks', 'no more'];
    return quitWords.any((w) => text.contains(w));
  }
}
