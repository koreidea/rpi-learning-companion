import 'dart:math';

import '../activity_base.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../audio/sentence_buffer.dart';

/// Callback type for streaming TTS — receives a complete sentence to speak.
typedef SpeakSentenceCallback = Future<void> Function(String sentence);

/// Open-ended hypothetical thinking game using LLM for contextual responses.
///
/// Poses "what would happen if" questions and uses the LLM to build on
/// the child's imaginative answers. Each scenario goes 3-4 rounds of
/// follow-up before moving to a new "what if" prompt.
class WhatIfGame extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS. When set, each sentence from the
  /// LLM is passed to this callback as soon as it is ready, instead of
  /// buffering the full response. This allows low-latency TTS during
  /// LLM streaming.
  SpeakSentenceCallback? onSpeakSentence;

  final Random _random = Random();

  bool _active = false;
  int _scenariosCompleted = 0;
  int _currentFollowUps = 0;
  int _maxFollowUpsPerScenario = 3;
  int _maxScenarios = 3;

  String? _currentScenario;
  bool _waitingForReady = false;
  bool _waitingForAnswer = false;
  bool _waitingForPlayAgain = false;

  final List<int> _usedScenarioIndices = [];

  // Conversation context for the current scenario
  final List<Map<String, String>> _scenarioHistory = [];

  WhatIfGame({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'thinking_what_if';

  @override
  String get name => 'What If Game';

  @override
  String get category => 'thinking';

  @override
  String get description =>
      'Imagine wild what-if scenarios and explore creative ideas together.';

  @override
  List<String> get skills => ['imagination', 'creative thinking', 'verbal expression'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _scenariosCompleted = 0;
    _currentFollowUps = 0;
    _usedScenarioIndices.clear();
    _scenarioHistory.clear();
    _waitingForReady = true;
    _waitingForAnswer = false;
    _waitingForPlayAgain = false;

    return "Let's play the What If game! I will ask a fun question and you "
        "tell me what you think would happen. There are no wrong answers! "
        "Are you ready?";
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

    if (_waitingForAnswer && _currentScenario != null) {
      return await _processAnswer(childSaid);
    }

    return "Tell me what you think would happen!";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_scenariosCompleted == 0) return 'No scenarios explored yet.';
    return 'Explored $_scenariosCompleted what-if scenarios.';
  }

  // -- Internal --

  String _presentNewScenario() {
    // Pick an unused scenario
    String? chosen;
    for (int i = 0; i < _scenarios.length; i++) {
      if (!_usedScenarioIndices.contains(i)) {
        chosen = _scenarios[i];
        _usedScenarioIndices.add(i);
        break;
      }
    }

    if (chosen == null) {
      _usedScenarioIndices.clear();
      final idx = _random.nextInt(_scenarios.length);
      chosen = _scenarios[idx];
      _usedScenarioIndices.add(idx);
    }

    _currentScenario = chosen;
    _currentFollowUps = 0;
    _scenarioHistory.clear();
    _waitingForAnswer = true;

    return "Okay, here is a fun one! What would happen if $chosen";
  }

  Future<String> _processAnswer(String childSaid) async {
    _currentFollowUps++;

    // Add child's response to scenario history
    _scenarioHistory.add({'role': 'user', 'content': childSaid});

    // Check if we should move to a new scenario
    if (_currentFollowUps >= _maxFollowUpsPerScenario) {
      _scenariosCompleted++;
      _waitingForAnswer = false;

      if (_scenariosCompleted >= _maxScenarios) {
        _active = false;
        final response = await _getLlmResponse(childSaid, isLastRound: true);
        return "$response That was so much fun! ${_buildEndSummary()}";
      }

      _waitingForPlayAgain = true;
      final response = await _getLlmResponse(childSaid, isLastRound: true);
      return "$response I love your ideas! Want to try another what-if question?";
    }

    // Continue the conversation — get LLM follow-up
    final response = await _getLlmResponse(childSaid, isLastRound: false);
    _scenarioHistory.add({'role': 'assistant', 'content': response});
    return response;
  }

  Future<String> _getLlmResponse(String childSaid, {required bool isLastRound}) async {
    final systemPrompt = _buildSystemPrompt(isLastRound: isLastRound);
    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
    ];

    // Add scenario context
    if (_currentScenario != null) {
      messages.add({
        'role': 'assistant',
        'content': 'What would happen if ${_currentScenario!}',
      });
    }

    // Add conversation history
    messages.addAll(_scenarioHistory);

    // If the last entry is not the child's latest message, add it
    if (_scenarioHistory.isEmpty ||
        _scenarioHistory.last['content'] != childSaid) {
      messages.add({'role': 'user', 'content': childSaid});
    }

    try {
      final provider = _llmRouter.getProvider();

      // Stream tokens through sentence buffer for low-latency TTS
      if (onSpeakSentence != null) {
        return await _streamWithTts(provider, messages);
      }

      // Non-streaming fallback: collect full response
      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }

      final response = buffer.toString().trim();
      if (response.isEmpty) {
        return _fallbackResponse(isLastRound);
      }
      return response;
    } catch (e) {
      return _fallbackResponse(isLastRound);
    }
  }

  /// Stream LLM tokens, sending each complete sentence to TTS immediately.
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
      return result.isNotEmpty ? result : _fallbackResponse(false);
    } catch (e) {
      // If we already spoke some sentences, return what we have
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return _fallbackResponse(false);
    }
  }

  String _buildSystemPrompt({required bool isLastRound}) {
    final roundGuidance = isLastRound
        ? 'This is the last round for this scenario. Celebrate the child\'s '
            'imagination and summarize one fun idea they shared. Do not ask '
            'a follow-up question.'
        : 'Build on the child\'s answer with enthusiasm. Add a small fun '
            'detail to their idea and then ask a follow-up question to keep '
            'them thinking.';

    return 'You are Buddy, a creative play companion for a 3-6 year old child. '
        'You are playing the What If game. $roundGuidance '
        'Rules: Use simple short sentences. Be very encouraging and excited '
        'about their ideas. Never say an idea is wrong. Keep your response '
        'to 2-3 sentences maximum. Do not use any markdown, bullet points, '
        'or numbered lists. Speak naturally as if talking out loud.';
  }

  String _fallbackResponse(bool isLastRound) {
    if (isLastRound) {
      return "Wow, what a cool idea! You have such a wonderful imagination!";
    }
    const responses = [
      "Wow, that is such a cool idea! And then what do you think would happen next?",
      "I love that! That would be so funny! What else might happen?",
      "That is amazing! You have such a great imagination! Tell me more!",
      "Oh wow, I never thought of that! What would you do if that really happened?",
    ];
    return responses[_random.nextInt(responses.length)];
  }

  String _buildEndSummary() {
    if (_scenariosCompleted == 0) {
      return "Thanks for playing the What If game! Come back anytime to "
          "imagine more fun things!";
    }
    return "We explored $_scenariosCompleted fun what-if "
        "${_scenariosCompleted == 1 ? 'scenario' : 'scenarios'}! "
        "You have an amazing imagination!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }

  bool _containsNo(String text) {
    const noWords = ['no', 'nah', 'nope', "don't want", 'not now'];
    for (final w in noWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }

  static const List<String> _scenarios = [
    'it rained candy instead of water?',
    'animals could talk just like people?',
    'you could fly like a bird?',
    'you were as tall as a building?',
    'the ocean was made of juice?',
    'toys came alive at night when you were sleeping?',
    'you could be invisible?',
    'there was no gravity and everything floated?',
    'you could talk to plants and flowers?',
    'you had a pet dragon?',
    'you could breathe underwater like a fish?',
    'everything you drew came to life?',
    'you could shrink down to the size of an ant?',
    'you could travel back to the time of dinosaurs?',
    'clouds were made of cotton candy?',
  ];
}
