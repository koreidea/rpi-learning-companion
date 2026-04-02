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

/// Eco Audit: analyze daily routine for environmental impact.
///
/// The bot asks about the child's daily habits (shower length, plastic use,
/// electricity, transport) and provides fun facts and gentle suggestions.
/// An eco-score is calculated at the end based on their responses.
class EcoAudit extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  bool _active = false;
  int _questionIndex = 0;
  int _ecoScore = 0;
  static const int _maxScore = 50;

  final List<Map<String, String>> _responses = [];

  EcoAudit({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'environmental_eco_audit';

  @override
  String get name => 'Eco Audit';

  @override
  String get category => 'environmental';

  @override
  String get description =>
      'Check how eco-friendly your daily routine is!';

  @override
  List<String> get skills =>
      ['environmental awareness', 'self-reflection', 'responsibility'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.environmental;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'eco audit',
          'environment check',
          'green check',
          'eco score',
          'how green am i',
        ],
        'hi': ['इको ऑडिट', 'पर्यावरण जांच', 'हरा स्कोर'],
        'te': ['ఇకో ఆడిట్', 'పర్యావరణ తనిఖీ'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _questionIndex = 0;
    _ecoScore = 0;
    _responses.clear();

    return "Let's do an Eco Audit! I am going to ask you about your daily "
        "habits and we will see how eco-friendly you are. There are no wrong "
        "answers, just honest ones! Here is the first question. "
        "${_questions[0].question}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    // Evaluate current question
    final question = _questions[_questionIndex];
    final points = _evaluateAnswer(lower, question);
    _ecoScore += points;
    _responses.add({
      'question': question.topic,
      'answer': childSaid,
      'points': points.toString(),
    });

    // Get LLM response with fun fact
    final feedback = await _getLlmFeedback(childSaid, question, points);

    _questionIndex++;
    if (_questionIndex >= _questions.length) {
      _active = false;
      return "$feedback ${_buildEndSummary()}";
    }

    return "$feedback Next question! ${_questions[_questionIndex].question}";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    return 'Eco score: $_ecoScore/$_maxScore. Questions: $_questionIndex/${_questions.length}.';
  }

  // -- Internal --

  int _evaluateAnswer(String lower, _EcoQuestion question) {
    // Award points based on eco-friendly keywords
    int points = 5; // Base for answering
    for (final goodWord in question.ecoFriendlyKeywords) {
      if (lower.contains(goodWord)) {
        points = 10;
        break;
      }
    }
    return points;
  }

  Future<String> _getLlmFeedback(
    String childSaid,
    _EcoQuestion question,
    int points,
  ) async {
    final systemPrompt =
        'You are Buddy, an environmental awareness coach for children. '
        'The child answered a question about "${question.topic}". '
        'Their answer: "$childSaid". '
        'They scored $points out of 10 eco points. '
        'Give a brief response: acknowledge their answer, share ONE fun '
        'environmental fact related to ${question.topic}, and if they could '
        'be more eco-friendly, give ONE gentle suggestion. '
        'Rules: 2-3 sentences. Be positive and encouraging. '
        'Do not use markdown, bullet points, or emojis.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      {'role': 'user', 'content': childSaid},
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
      debugPrint('[EcoAudit] LLM error: $e');
    }

    return question.fallbackFact;
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
          : 'Interesting! Let me tell you a fun eco fact.';
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return 'Interesting! Let me tell you a fun eco fact.';
    }
  }

  String _buildEndSummary() {
    String rating;
    if (_ecoScore >= 40) {
      rating = "Eco Champion! You are doing amazing things for the planet!";
    } else if (_ecoScore >= 25) {
      rating = "Eco Explorer! You are on the right track with some great habits.";
    } else {
      rating = "Eco Beginner! Every small change makes a difference, and you are learning!";
    }

    return "Your eco score is $_ecoScore out of $_maxScore! $rating "
        "Remember, even small actions like turning off lights, using less "
        "plastic, and taking shorter showers make a big difference for "
        "our planet. Keep being green!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<_EcoQuestion> _questions = [
    _EcoQuestion(
      topic: 'water usage',
      question: 'How long is your shower usually? Quick, medium, or long?',
      ecoFriendlyKeywords: ['short', 'quick', 'fast', 'five', '5', 'bucket'],
      fallbackFact:
          'A 5-minute shower uses about 40 liters of water. '
          'Cutting it by 2 minutes saves enough water for a plant for a week!',
    ),
    _EcoQuestion(
      topic: 'plastic usage',
      question: 'Did you use any plastic today? Like bags, bottles, or straws?',
      ecoFriendlyKeywords: ['no', 'steel', 'cloth', 'reusable', 'metal', 'glass'],
      fallbackFact:
          'A single plastic bag takes up to 500 years to break down! '
          'Using cloth bags is a simple way to help.',
    ),
    _EcoQuestion(
      topic: 'electricity',
      question: 'Do you turn off lights and fans when you leave a room?',
      ecoFriendlyKeywords: ['yes', 'always', 'turn off', 'switch off'],
      fallbackFact:
          'Turning off one light bulb for 8 hours saves enough electricity '
          'to charge your tablet 10 times!',
    ),
    _EcoQuestion(
      topic: 'transport',
      question: 'How do you get to school? Walk, cycle, bus, or car?',
      ecoFriendlyKeywords: ['walk', 'cycle', 'bicycle', 'bus', 'train'],
      fallbackFact:
          'Walking or cycling to school produces zero pollution! '
          'Even taking a bus is much better than a car because one bus '
          'replaces 40 cars on the road.',
    ),
    _EcoQuestion(
      topic: 'food waste',
      question: 'Do you finish all the food on your plate, or is there leftover?',
      ecoFriendlyKeywords: ['finish', 'all', 'everything', 'no waste', 'clean'],
      fallbackFact:
          'About one third of all food produced in the world is wasted! '
          'Taking only what you can eat is a simple way to fight food waste.',
    ),
  ];
}

/// An eco audit question with evaluation criteria.
class _EcoQuestion {
  final String topic;
  final String question;
  final List<String> ecoFriendlyKeywords;
  final String fallbackFact;

  const _EcoQuestion({
    required this.topic,
    required this.question,
    required this.ecoFriendlyKeywords,
    required this.fallbackFact,
  });
}
