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

/// Friendly debate activity where the bot takes one side and the child argues
/// the other.
///
/// Teaches: critical thinking, argumentation, respectful disagreement, and
/// perspective-taking. The bot presents one argument per turn, the child
/// responds, and the bot counters. After 3-4 rounds, the bot summarizes
/// the child's best points.
class DebateBuddy extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  final Random _random = Random();

  bool _active = false;
  int _round = 0;
  int _maxRounds = 4;
  String? _currentTopic;
  String? _botSide;
  bool _waitingForReady = false;
  bool _waitingForChoice = false;

  final List<Map<String, String>> _debateHistory = [];
  final List<String> _childBestPoints = [];
  final List<int> _usedTopicIndices = [];

  DebateBuddy({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'critical_thinking_debate_buddy';

  @override
  String get name => 'Debate Buddy';

  @override
  String get category => 'thinking';

  @override
  String get description =>
      'Have a friendly debate and practice making strong arguments.';

  @override
  List<String> get skills =>
      ['argumentation', 'critical thinking', 'perspective taking'];

  @override
  int get minAge => 7;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.criticalThinking;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'debate',
          'debate game',
          'let us debate',
          'debate buddy',
          'argue',
          'argument game',
        ],
        'hi': ['बहस', 'बहस करो', 'डिबेट'],
        'te': ['వాదన', 'డిబేట్', 'వాదించు'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _round = 0;
    _debateHistory.clear();
    _childBestPoints.clear();
    _waitingForReady = false;
    _waitingForChoice = true;

    debugPrint('[DebateBuddy] Started');

    return "Welcome to Debate Buddy! I will pick a fun topic, take one side, "
        "and you argue the other. There are no wrong answers, just great "
        "thinking! Here are some topics. Which sounds fun? "
        "One, cats are smarter than dogs. Two, summer is better than winter. "
        "Three, books are better than movies. Or just say surprise me!";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    if (_waitingForChoice) {
      _waitingForChoice = false;
      return _selectTopic(lower);
    }

    if (_waitingForReady) {
      _waitingForReady = false;
      return await _presentBotArgument();
    }

    // Child is making their argument
    _round++;
    _debateHistory.add({'role': 'user', 'content': childSaid});

    if (_round >= _maxRounds) {
      _active = false;
      final response = await _getLlmResponse(childSaid, isLastRound: true);
      return "$response ${_buildEndSummary()}";
    }

    final response = await _getLlmResponse(childSaid, isLastRound: false);
    _debateHistory.add({'role': 'assistant', 'content': response});
    return response;
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_round == 0) return 'No debate rounds yet.';
    return 'Debated $_round rounds on: $_currentTopic.';
  }

  // -- Internal --

  String _selectTopic(String lower) {
    // Check if child picked a number or keyword
    int topicIndex;
    if (lower.contains('one') ||
        lower.contains('1') ||
        lower.contains('cat')) {
      topicIndex = 0;
    } else if (lower.contains('two') ||
        lower.contains('2') ||
        lower.contains('summer')) {
      topicIndex = 1;
    } else if (lower.contains('three') ||
        lower.contains('3') ||
        lower.contains('book')) {
      topicIndex = 2;
    } else {
      // Pick a random topic
      topicIndex = _random.nextInt(_topics.length);
    }

    final topic = _topics[topicIndex];
    _currentTopic = topic.statement;
    _botSide = topic.botPosition;
    _usedTopicIndices.add(topicIndex);
    _waitingForReady = true;

    return "Great choice! Our debate topic is: ${topic.statement}. "
        "I will argue that ${topic.botPosition}. You argue the opposite! "
        "Are you ready? I will go first.";
  }

  Future<String> _presentBotArgument() async {
    final systemPrompt =
        'You are Buddy, a friendly debate partner for children aged 7-14. '
        'The debate topic is: "$_currentTopic". '
        'You are arguing: "$_botSide". '
        'Present ONE clear argument in 2-3 sentences. Be fun and convincing '
        'but not aggressive. End by inviting the child to respond. '
        'Do not use markdown, bullet points, or emojis. Speak naturally.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      {
        'role': 'user',
        'content': 'Present your opening argument.',
      },
    ];

    try {
      final provider = _llmRouter.getProvider();

      if (onSpeakSentence != null) {
        final response = await _streamWithTts(provider, messages);
        _debateHistory.add({'role': 'assistant', 'content': response});
        return "$response Now it is your turn! What do you think?";
      }

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }
      final result = buffer.toString().trim();
      if (result.isNotEmpty) {
        _debateHistory.add({'role': 'assistant', 'content': result});
        return "$result Now it is your turn! What do you think?";
      }
    } catch (e) {
      debugPrint('[DebateBuddy] LLM error: $e');
    }

    // Fallback
    final fallback = "I think $_botSide, and here is why. It makes life more "
        "interesting and fun! Now it is your turn to argue the other side!";
    _debateHistory.add({'role': 'assistant', 'content': fallback});
    return fallback;
  }

  Future<String> _getLlmResponse(
    String childSaid, {
    required bool isLastRound,
  }) async {
    final roundGuidance = isLastRound
        ? 'This is the final round. Acknowledge the child\'s best argument. '
            'Summarize one strong point they made. Say something like "You '
            'really convinced me about..." Do not ask a follow-up.'
        : 'Counter the child\'s argument with ONE new point. Acknowledge '
            'something good they said first. Keep it fun and respectful. '
            'End by challenging them to respond.';

    final systemPrompt =
        'You are Buddy, a friendly debate partner for children aged 7-14. '
        'Debate topic: "$_currentTopic". Your position: "$_botSide". '
        '$roundGuidance '
        'Rules: 2-3 sentences maximum. Be encouraging. Never be mean or '
        'dismissive. Do not use markdown, bullet points, or emojis. '
        'Speak naturally.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._debateHistory,
    ];

    if (_debateHistory.isEmpty ||
        _debateHistory.last['content'] != childSaid) {
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
      debugPrint('[DebateBuddy] LLM error: $e');
    }

    if (isLastRound) {
      return "Great debate! You made some really strong points.";
    }
    return "That is an interesting point! But I still think $_botSide. "
        "Can you convince me otherwise?";
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
          : 'That is a good point!';
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return 'That is a good point!';
    }
  }

  String _buildEndSummary() {
    if (_round == 0) {
      return "Thanks for stopping by Debate Buddy! Come back anytime for a fun debate!";
    }
    return "Great debate on $_currentTopic! We went $_round rounds. "
        "You made some really strong arguments. The most important thing in a "
        "debate is thinking carefully about both sides. You did that brilliantly!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<_DebateTopic> _topics = [
    _DebateTopic(
      statement: 'Cats are smarter than dogs',
      botPosition: 'cats are smarter because they are independent thinkers',
    ),
    _DebateTopic(
      statement: 'Summer is better than winter',
      botPosition: 'summer is better because of longer days and outdoor fun',
    ),
    _DebateTopic(
      statement: 'Books are better than movies',
      botPosition: 'books are better because they let your imagination run free',
    ),
    _DebateTopic(
      statement: 'School uniforms should be required',
      botPosition: 'uniforms are good because they make everyone equal',
    ),
    _DebateTopic(
      statement: 'Homework should be banned',
      botPosition: 'homework helps you practice and remember what you learned',
    ),
    _DebateTopic(
      statement: 'Robots will replace teachers',
      botPosition: 'robots can teach because they never get tired and know everything',
    ),
    _DebateTopic(
      statement: 'Vegetables taste better than sweets',
      botPosition: 'vegetables are better because they give you real energy and come in so many flavors',
    ),
    _DebateTopic(
      statement: 'Playing outside is better than video games',
      botPosition: 'playing outside is better because you get fresh air and exercise',
    ),
  ];
}

/// A debate topic with the bot's assigned position.
class _DebateTopic {
  final String statement;
  final String botPosition;

  const _DebateTopic({
    required this.statement,
    required this.botPosition,
  });
}
