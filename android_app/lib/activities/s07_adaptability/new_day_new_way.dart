import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_provider.dart';
import '../../core/llm/llm_router.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Callback type for streaming TTS -- receives a complete sentence to speak.
typedef SpeakSentenceCallback = Future<void> Function(String sentence);

/// New Day New Way: explain a concept using an unrelated metaphor.
///
/// The bot assigns a concept and a metaphor frame, and the child must
/// explain the concept through that metaphor. For example, "Explain how
/// a school works, but pretend it is a restaurant." The LLM engages with
/// the child's creative metaphorical explanation.
class NewDayNewWay extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  final Random _random = Random();

  bool _active = false;
  int _challengesCompleted = 0;
  static const int _maxChallenges = 3;
  int _followUp = 0;
  static const int _maxFollowUps = 2;
  _MetaphorPair? _currentPair;
  bool _waitingForPlayAgain = false;

  final List<Map<String, String>> _conversationHistory = [];
  final List<int> _usedPairIndices = [];

  NewDayNewWay({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'adaptability_new_day_new_way';

  @override
  String get name => 'New Day New Way';

  @override
  String get category => 'adaptability';

  @override
  String get description =>
      'Explain things in unexpected ways using creative metaphors!';

  @override
  List<String> get skills =>
      ['adaptability', 'creative thinking', 'flexible reasoning'];

  @override
  int get minAge => 6;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.adaptability;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'new day new way',
          'metaphor game',
          'explain differently',
          'creative explain',
        ],
        'hi': ['नया तरीका', 'अलग तरीके से समझाओ'],
        'te': ['కొత్త దారి', 'వేరే విధంగా చెప్పు'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _challengesCompleted = 0;
    _followUp = 0;
    _conversationHistory.clear();

    return _presentNewChallenge();
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    if (_waitingForPlayAgain) {
      _waitingForPlayAgain = false;
      if (_containsNo(lower)) {
        _active = false;
        return _buildEndSummary();
      }
      return _presentNewChallenge();
    }

    _followUp++;
    _conversationHistory.add({'role': 'user', 'content': childSaid});

    if (_followUp >= _maxFollowUps) {
      _challengesCompleted++;
      _followUp = 0;
      _conversationHistory.clear();

      final response = await _getLlmResponse(childSaid, isWrapUp: true);

      if (_challengesCompleted >= _maxChallenges) {
        _active = false;
        return "$response ${_buildEndSummary()}";
      }

      _waitingForPlayAgain = true;
      return "$response You are so creative! Want to try another one?";
    }

    final response = await _getLlmResponse(childSaid, isWrapUp: false);
    _conversationHistory.add({'role': 'assistant', 'content': response});
    return response;
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_challengesCompleted == 0) return 'No metaphor challenges completed yet.';
    return 'Completed $_challengesCompleted creative metaphor challenges.';
  }

  // -- Internal --

  String _presentNewChallenge() {
    if (_usedPairIndices.length >= _pairs.length) {
      _usedPairIndices.clear();
    }
    int index;
    do {
      index = _random.nextInt(_pairs.length);
    } while (_usedPairIndices.contains(index));
    _usedPairIndices.add(index);

    _currentPair = _pairs[index];
    _followUp = 0;
    _conversationHistory.clear();

    return "Here is a fun challenge! ${_currentPair!.prompt} "
        "${_currentPair!.hint} Go ahead, tell me your version!";
  }

  Future<String> _getLlmResponse(
    String childSaid, {
    required bool isWrapUp,
  }) async {
    final pair = _currentPair!;
    final wrapGuidance = isWrapUp
        ? 'Wrap up by celebrating how creative the child\'s metaphor was. '
            'Mention one specific comparison they made that was clever. '
            'Do not ask more questions.'
        : 'Build on the child\'s metaphor with enthusiasm. Add your own '
            'fun comparison using their framework. Then ask a follow-up: '
            '"And what about [another aspect]? What would that be?"';

    final systemPrompt =
        'You are Buddy, a creative thinking coach for children. '
        'The child was asked: "${pair.prompt}" '
        '$wrapGuidance '
        'Rules: 2-3 sentences. Be excited about their creativity. '
        'Stay within their metaphor framework. '
        'Do not use markdown, bullet points, or emojis. Speak naturally.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._conversationHistory,
    ];

    if (_conversationHistory.isEmpty ||
        _conversationHistory.last['content'] != childSaid) {
      messages.add({'role': 'user', 'content': childSaid});
    }

    try {
      final provider = _llmRouter.getProvider();

      if (onSpeakSentence != null) {
        return await _streamWithTts(provider, messages);
      }

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }
      final result = buffer.toString().trim();
      if (result.isNotEmpty) return result;
    } catch (e) {
      debugPrint('[NewDayNewWay] LLM error: $e');
    }

    if (isWrapUp) {
      return "What a creative way to explain that! You thought about it in "
          "a totally new way, and that is what adaptable thinkers do!";
    }
    return "I love that comparison! Can you tell me more about how that works "
        "in your version?";
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

      return fullResponse.isNotEmpty
          ? fullResponse.join(' ')
          : 'That is so creative!';
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return 'That is so creative!';
    }
  }

  String _buildEndSummary() {
    if (_challengesCompleted == 0) {
      return "Thanks for trying New Day New Way! Come back for more creative thinking!";
    }
    return "You completed $_challengesCompleted creative metaphor "
        "${_challengesCompleted == 1 ? 'challenge' : 'challenges'}! "
        "Thinking about things in new ways is what makes you a flexible, "
        "adaptable thinker. That is an amazing skill!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  bool _containsNo(String text) {
    const noWords = ['no', 'nah', 'nope', "don't want", 'not now'];
    return noWords.any((w) => text.contains(w));
  }

  static const List<_MetaphorPair> _pairs = [
    _MetaphorPair(
      prompt: 'Explain how a school works, but pretend it is a restaurant!',
      hint: 'Teachers are chefs, students are customers, lessons are dishes on the menu.',
    ),
    _MetaphorPair(
      prompt: 'Explain how rain happens, but like it is a sports match!',
      hint: 'The clouds are the teams, the rain drops are the players, the sun is the referee.',
    ),
    _MetaphorPair(
      prompt: 'Explain how a library works, but pretend it is a hospital!',
      hint: 'Books are patients, librarians are doctors, reading is the medicine.',
    ),
    _MetaphorPair(
      prompt: 'Explain how a garden grows, but like it is a music concert!',
      hint: 'Seeds are musicians, water is the audience, flowers are the songs.',
    ),
    _MetaphorPair(
      prompt: 'Explain how a car works, but pretend it is a human body!',
      hint: 'The engine is the heart, fuel is food, wheels are legs, the horn is the voice.',
    ),
    _MetaphorPair(
      prompt: 'Explain how a computer works, but like it is a kitchen!',
      hint: 'The processor is the cook, memory is the fridge, programs are recipes.',
    ),
    _MetaphorPair(
      prompt: 'Explain how the solar system works, but like it is a playground!',
      hint: 'The sun is the teacher on duty, planets are kids playing different games.',
    ),
    _MetaphorPair(
      prompt: 'Explain how a post office works, but pretend it is a beehive!',
      hint: 'Letters are honey, postmen are worker bees, addresses are flowers.',
    ),
    _MetaphorPair(
      prompt: 'Explain how a train works, but like it is a caterpillar!',
      hint: 'Each carriage is a body segment, the engine is the head, tracks are the branch.',
    ),
    _MetaphorPair(
      prompt: 'Explain how sleep works, but pretend it is a phone charging!',
      hint: 'Your bed is the charger, dreams are software updates, waking up is reaching 100 percent.',
    ),
  ];
}

/// A concept-metaphor pair for creative explanation.
class _MetaphorPair {
  final String prompt;
  final String hint;

  const _MetaphorPair({
    required this.prompt,
    required this.hint,
  });
}
