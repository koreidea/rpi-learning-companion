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

/// Future City: design a sustainable city together with the bot.
///
/// The bot guides the child through 6-7 urban planning decisions about
/// energy, transport, parks, buildings, waste, and water. The LLM responds
/// to each decision and builds on the city plan. At the end, the bot
/// summarizes the complete sustainable city they designed.
class FutureCity extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  bool _active = false;
  int _decisionIndex = 0;
  final List<String> _decisions = [];
  final List<Map<String, String>> _conversationHistory = [];

  FutureCity({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'environmental_future_city';

  @override
  String get name => 'Future City';

  @override
  String get category => 'environmental';

  @override
  String get description =>
      'Design a sustainable city of the future together!';

  @override
  List<String> get skills =>
      ['environmental awareness', 'planning', 'creative thinking', 'systems thinking'];

  @override
  int get minAge => 6;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.environmental;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'future city',
          'design a city',
          'build a city',
          'sustainable city',
          'green city',
        ],
        'hi': ['भविष्य का शहर', 'शहर बनाओ', 'हरा शहर'],
        'te': ['భవిష్యత్ నగరం', 'నగరం నిర్మించు', 'పచ్చ నగరం'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _decisionIndex = 0;
    _decisions.clear();
    _conversationHistory.clear();

    debugPrint('[FutureCity] Started');

    return "Welcome to Future City! You are the chief architect of a brand "
        "new city. Your mission: make it the most sustainable, eco-friendly "
        "city in the world! I will ask you about different parts of the city "
        "and you decide how to design them. Ready? "
        "${_questions[0]}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    _decisions.add(childSaid);
    _conversationHistory.add({'role': 'user', 'content': childSaid});

    // Get LLM response about their decision
    final response = await _getLlmResponse(childSaid);
    _conversationHistory.add({'role': 'assistant', 'content': response});

    _decisionIndex++;
    if (_decisionIndex >= _questions.length) {
      _active = false;
      return "$response ${await _generateCitySummary()}";
    }

    return "$response Next decision! ${_questions[_decisionIndex]}";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    return 'City design: $_decisionIndex of ${_questions.length} decisions made.';
  }

  // -- Internal --

  Future<String> _getLlmResponse(String childSaid) async {
    final questionTopic = _decisionIndex < _topics.length
        ? _topics[_decisionIndex]
        : 'city design';

    final systemPrompt =
        'You are Buddy, an eco-city design coach for children. '
        'The child is designing a sustainable city. '
        'Current topic: "$questionTopic". '
        'The child decided: "$childSaid". '
        'Respond in 2-3 sentences. Comment on the environmental impact of '
        'their choice. If it is eco-friendly, celebrate it. If not, gently '
        'suggest a greener alternative. Add one fun fact about sustainable '
        'cities. Do not use markdown, bullet points, or emojis.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._conversationHistory,
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
      final result = buffer.toString().trim();
      if (result.isNotEmpty) return result;
    } catch (e) {
      debugPrint('[FutureCity] LLM error: $e');
    }

    return "Interesting choice! That will make our city unique.";
  }

  Future<String> _generateCitySummary() async {
    final decisionsText = _decisions
        .asMap()
        .entries
        .map((e) => '${_topics[e.key]}: ${e.value}')
        .join('. ');

    final systemPrompt =
        'You are Buddy, summarizing a child\'s sustainable city design. '
        'Their decisions: "$decisionsText". '
        'Give an exciting 3-4 sentence summary of their city. Name the city '
        'something fun. Highlight the most eco-friendly features. '
        'End with encouragement. Do not use markdown or emojis.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      {'role': 'user', 'content': 'Summarize my city!'},
    ];

    try {
      final provider = _llmRouter.getProvider();
      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }
      final result = buffer.toString().trim();
      if (result.isNotEmpty) return result;
    } catch (e) {
      debugPrint('[FutureCity] Summary LLM error: $e');
    }

    return "Your city has amazing sustainable features! You thought about "
        "energy, transport, nature, and more. You would make a fantastic "
        "city planner!";
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
          : 'That is an interesting design choice!';
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return 'That is an interesting design choice!';
    }
  }

  String _buildEndSummary() {
    if (_decisionIndex == 0) {
      return "Thanks for trying Future City! Come back to design your dream city!";
    }
    return "You made $_decisionIndex design decisions for your sustainable city! "
        "Every choice you made helps create a better future. "
        "The world needs city planners who think about the planet like you do!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<String> _topics = [
    'energy source',
    'transportation',
    'parks and green spaces',
    'buildings and homes',
    'waste management',
    'water system',
    'food production',
  ];

  static const List<String> _questions = [
    'First, how will your city get its energy? Solar panels, wind turbines, '
        'hydropower, or something else?',
    'How will people get around your city? Cars, buses, bicycles, trains, '
        'or something futuristic?',
    'Where should we put the parks and green spaces? Near homes, near schools, '
        'in the center, or everywhere?',
    'What should the buildings be made of? Should they have green roofs with '
        'plants on top? Should they be tall or short?',
    'How will your city handle garbage and waste? Recycling, composting, '
        'or something creative?',
    'Where will the city get clean water? Rivers, rainwater collection, '
        'water recycling, or something else?',
    'How will your city grow food? Community gardens, vertical farms, '
        'rooftop gardens, or nearby farms?',
  ];
}
