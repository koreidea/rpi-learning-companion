import 'package:flutter/foundation.dart';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_provider.dart';
import '../../core/llm/llm_router.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Callback type for streaming TTS -- receives a complete sentence to speak.
typedef SpeakSentenceCallback = Future<void> Function(String sentence);

/// Tell Me About Your Day: a daily conversational check-in that helps
/// children practice narrative structure naturally.
///
/// The bot asks open-ended questions about the child's day, uses the LLM
/// to generate thoughtful follow-up questions, and wraps up with a warm
/// summary. Targets 4-5 conversational turns over about 3 minutes.
class TellMeAboutDay extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  bool _active = false;
  int _turn = 0;
  static const int _maxTurns = 5;

  final List<Map<String, String>> _conversationHistory = [];

  TellMeAboutDay({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'communication_tell_me_about_day';

  @override
  String get name => 'Tell Me About Your Day';

  @override
  String get category => 'communication';

  @override
  String get description =>
      'Share your day and practice telling stories with details.';

  @override
  List<String> get skills =>
      ['narrative', 'communication', 'self-expression', 'vocabulary'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.communication;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'tell me about my day',
          'how was my day',
          'about my day',
          'daily chat',
          'talk about my day',
        ],
        'hi': ['मेरा दिन कैसा था', 'आज क्या हुआ', 'दिन के बारे में'],
        'te': ['నా రోజు ఎలా గడిచింది', 'ఈ రోజు ఏం జరిగింది'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _turn = 0;
    _conversationHistory.clear();

    debugPrint('[TellMeAboutDay] Started');

    return "Hey there! I would love to hear about your day. "
        "How was your day today? What was the best part?";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return await _generateSummary();
    }

    _turn++;
    _conversationHistory.add({'role': 'user', 'content': childSaid});

    if (_turn >= _maxTurns) {
      _active = false;
      final summary = await _generateSummary();
      return summary;
    }

    final response = await _generateFollowUp(childSaid);
    _conversationHistory.add({'role': 'assistant', 'content': response});
    return response;
  }

  @override
  Future<String> end() async {
    _active = false;
    return await _generateSummary();
  }

  @override
  String get progressSummary {
    if (_turn == 0) return 'No conversation yet.';
    return 'Chatted for $_turn turns about the day.';
  }

  // -- LLM generation --

  Future<String> _generateFollowUp(String childSaid) async {
    final systemPrompt =
        'You are Buddy, a warm and friendly companion having a daily '
        'conversation with a child about their day. '
        'Ask a follow-up question that helps them practice telling stories '
        'with details. Be genuinely interested and curious. '
        'If they mention something interesting, ask about it specifically. '
        'If they seem quiet, offer gentle prompts like "Did anything funny happen?" '
        'Keep your response to 1-2 sentences. Always end with a question. '
        'Do not use markdown, bullet points, or emojis. Speak naturally.';

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
      debugPrint('[TellMeAboutDay] LLM error: $e');
    }

    // Fallback follow-ups
    const fallbacks = [
      'That sounds interesting! Can you tell me more about that?',
      'Oh wow! And what happened after that?',
      'That is so cool! Who were you with?',
      'I love hearing about that! Did anything funny happen today?',
    ];
    return fallbacks[_turn % fallbacks.length];
  }

  Future<String> _generateSummary() async {
    if (_conversationHistory.isEmpty) {
      return "It was nice chatting! Come back tomorrow to tell me about your day!";
    }

    final systemPrompt =
        'You are Buddy, a warm companion for children. '
        'Summarize the conversation about the child\'s day in 2-3 sentences. '
        'Mention the most interesting part they shared. Be warm and positive. '
        'End with encouragement. Do not use markdown or emojis.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._conversationHistory,
      {
        'role': 'user',
        'content': 'Please summarize our conversation about my day.',
      },
    ];

    try {
      final provider = _llmRouter.getProvider();
      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }
      final result = buffer.toString().trim();
      if (result.isNotEmpty) {
        return "$result Thanks for sharing your day with me!";
      }
    } catch (e) {
      debugPrint('[TellMeAboutDay] Summary LLM error: $e');
    }

    return "Sounds like you had a great day! I loved hearing about it. "
        "Thanks for sharing with me! See you tomorrow!";
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
          : 'Tell me more about that!';
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return 'Tell me more about that!';
    }
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
      'that is all', "that's it", 'nothing else',
    ];
    return quitWords.any((w) => text.contains(w));
  }
}
