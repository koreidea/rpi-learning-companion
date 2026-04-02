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

/// What Would You Do: leadership dilemma scenarios that explore decision-making,
/// responsibility, and ethical reasoning.
///
/// The bot presents a scenario where the child is in a leadership position.
/// The LLM explores the child's reasoning, asks probing follow-ups, and
/// introduces relevant leadership concepts like fairness, responsibility,
/// and courage.
class WhatWouldYouDo extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  final Random _random = Random();

  bool _active = false;
  int _scenariosCompleted = 0;
  int _followUp = 0;
  static const int _maxFollowUps = 3;
  static const int _maxScenarios = 2;
  String? _currentScenario;
  bool _waitingForReady = false;
  bool _waitingForPlayAgain = false;

  final List<Map<String, String>> _conversationHistory = [];
  final List<int> _usedScenarioIndices = [];

  WhatWouldYouDo({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'leadership_what_would_you_do';

  @override
  String get name => 'What Would You Do?';

  @override
  String get category => 'leadership';

  @override
  String get description =>
      'Face real leadership dilemmas and explore your decision-making.';

  @override
  List<String> get skills =>
      ['leadership', 'decision making', 'ethical reasoning', 'empathy'];

  @override
  int get minAge => 7;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.leadership;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'what would you do',
          'leadership game',
          'leader game',
          'dilemma game',
          'decision game',
        ],
        'hi': ['तुम क्या करोगे', 'लीडर खेल', 'निर्णय खेल'],
        'te': ['ఏమి చేస్తావు', 'లీడర్ ఆట', 'నిర్ణయం ఆట'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _scenariosCompleted = 0;
    _followUp = 0;
    _conversationHistory.clear();
    _waitingForReady = true;
    _waitingForPlayAgain = false;

    return "Welcome to What Would You Do! I am going to put you in a tough "
        "situation where you are the leader. There is no single right answer. "
        "I just want to hear how YOU would handle it. Ready?";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    if (_waitingForReady) {
      _waitingForReady = false;
      return _presentNewScenario();
    }

    if (_waitingForPlayAgain) {
      _waitingForPlayAgain = false;
      if (_containsNo(lower)) {
        _active = false;
        return _buildEndSummary();
      }
      return _presentNewScenario();
    }

    // Process the child's decision
    _followUp++;
    _conversationHistory.add({'role': 'user', 'content': childSaid});

    if (_followUp >= _maxFollowUps) {
      _scenariosCompleted++;
      _followUp = 0;
      _conversationHistory.clear();

      final response = await _getLlmResponse(childSaid, isWrapUp: true);

      if (_scenariosCompleted >= _maxScenarios) {
        _active = false;
        return "$response ${_buildEndSummary()}";
      }

      _waitingForPlayAgain = true;
      return "$response That was great leadership thinking! "
          "Want to try another scenario?";
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
    if (_scenariosCompleted == 0) return 'No leadership scenarios explored yet.';
    return 'Explored $_scenariosCompleted leadership scenarios.';
  }

  // -- Internal --

  String _presentNewScenario() {
    if (_usedScenarioIndices.length >= _scenarios.length) {
      _usedScenarioIndices.clear();
    }
    int index;
    do {
      index = _random.nextInt(_scenarios.length);
    } while (_usedScenarioIndices.contains(index));
    _usedScenarioIndices.add(index);

    _currentScenario = _scenarios[index];
    _followUp = 0;
    _conversationHistory.clear();

    return "Here is the situation. $_currentScenario What would you do?";
  }

  Future<String> _getLlmResponse(
    String childSaid, {
    required bool isWrapUp,
  }) async {
    final wrapGuidance = isWrapUp
        ? 'This is the wrap-up. Praise the child\'s reasoning. Mention ONE '
            'leadership quality they showed (like fairness, courage, empathy, '
            'or responsibility). Do not ask another question.'
        : 'Ask a thoughtful follow-up that deepens the child\'s thinking. '
            'For example: "How would the other person feel?" or "What if it '
            'does not work?" Keep it to ONE question.';

    final systemPrompt =
        'You are Buddy, a leadership coach for children aged 7-14. '
        'The child is exploring a leadership scenario: "$_currentScenario" '
        '$wrapGuidance '
        'Rules: 2-3 sentences max. Be warm and respectful. Never say their '
        'answer is wrong. Explore their reasoning. Do not use markdown, '
        'bullet points, or emojis. Speak naturally.';

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
      debugPrint('[WhatWouldYouDo] LLM error: $e');
    }

    if (isWrapUp) {
      return "That shows real leadership thinking! You considered how your "
          "decision would affect others, and that is what great leaders do.";
    }
    return "Interesting! And how do you think that would make the other people feel?";
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
          : 'That is interesting thinking!';
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return 'That is interesting thinking!';
    }
  }

  String _buildEndSummary() {
    if (_scenariosCompleted == 0) {
      return "Thanks for trying What Would You Do! Come back to practice leadership!";
    }
    return "You explored $_scenariosCompleted leadership "
        "${_scenariosCompleted == 1 ? 'scenario' : 'scenarios'}! "
        "A good leader thinks about how their decisions affect everyone. "
        "You showed great thinking today!";
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

  static const List<String> _scenarios = [
    'You are the captain of a cricket team. Your best batsman is also your '
        'worst fielder. The final match is tomorrow. Do you keep him in '
        'the team, move him to a different position, or talk to him about it?',
    'You are the class monitor. Your best friend is copying answers during '
        'a test. The teacher asks you if anyone is cheating. What do you do?',
    'You are leading a school project. One team member is not doing their '
        'work, and the deadline is tomorrow. How do you handle it?',
    'You have to choose between two friends for the last spot on the quiz '
        'team. One is your closer friend, but the other is better at quizzes. '
        'Who do you pick and why?',
    'Your whole team wants to do a dance performance for the school show, '
        'but you think a skit would be much better. You are the team leader. '
        'What do you do?',
    'A younger student is being picked on by some older kids during lunch. '
        'You are the only senior student nearby. What do you do?',
  ];
}
