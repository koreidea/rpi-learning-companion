import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// CuriosityHour: child-directed learning where the child chooses the topic.
///
/// Teaches: self-directed learning, curiosity, questioning, research skills.
///
/// Flow:
/// 1. Bot asks: "What do YOU want to learn about today?"
/// 2. Child picks a topic.
/// 3. Bot uses LLM to share interesting facts and ask guiding questions.
/// 4. Encourages deeper exploration with follow-up questions.
/// 5. 3-4 rounds of exploration per topic.
class CuriosityHour extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _topicsExplored = 0;
  int _roundsInTopic = 0;
  int _score = 0;
  String _currentTopic = '';

  /// 0=ask topic, 1=exploring, 2=ask to go deeper or new topic
  int _phase = 0;
  final List<Map<String, String>> _history = [];

  static const int _maxRoundsPerTopic = 4;

  CuriosityHour({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'curiosity_hour';

  @override
  String get name => 'Curiosity Hour';

  @override
  String get category => 'self-direction';

  @override
  String get description =>
      'Choose what YOU want to learn about and explore it together.';

  @override
  List<String> get skills =>
      ['self-directed learning', 'curiosity', 'questioning', 'research'];

  @override
  int get minAge => 4;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.selfDirection;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'curiosity hour',
          'i want to learn',
          'teach me something',
          'explore a topic',
          'what can i learn',
        ],
        'hi': [
          'जिज्ञासा का समय',
          'मुझे सिखाओ',
          'कुछ नया सीखना है',
        ],
        'te': [
          'ఆసక్తి సమయం',
          'నాకు నేర్పించు',
          'ఏదైనా నేర్చుకోవాలి',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_topicsExplored == 0 && _currentTopic.isEmpty) {
      return 'No topics explored yet.';
    }
    return 'Explored $_topicsExplored topic${_topicsExplored != 1 ? 's' : ''}. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _topicsExplored = 0;
    _roundsInTopic = 0;
    _score = 0;
    _currentTopic = '';
    _phase = 0;
    _history.clear();

    return 'Welcome to Curiosity Hour! This is YOUR time to learn about '
        'whatever interests you. It could be space, animals, volcanoes, '
        'robots, anything at all! What do you want to learn about today?';
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
        // Child chose a topic
        _currentTopic = childSaid.trim();
        _roundsInTopic = 0;
        _history.clear();
        _score += 10;
        _phase = 1;

        return await _getLlmResponse(
          childSaid,
          'The child wants to learn about: $_currentTopic. '
          'Share 2-3 fascinating, age-appropriate facts about this topic. '
          'Then ask a thought-provoking question that encourages the child '
          'to think deeper about it. Keep it fun and conversational. '
          '3-4 sentences maximum.',
        );

      case 1:
        // Exploring the topic
        _roundsInTopic++;
        _score += 10;
        _history.add({'role': 'user', 'content': childSaid});

        if (_roundsInTopic >= _maxRoundsPerTopic) {
          _phase = 2;
          _topicsExplored++;
          final response = await _getLlmResponse(
            childSaid,
            'The child has been exploring "$_currentTopic" for a while. '
            'Respond to what they said, share one last cool fact, and '
            'wrap up this topic with encouragement. 2-3 sentences.',
          );
          return '$response Would you like to explore a different topic, '
              'or are you done for now?';
        }

        return await _getLlmResponse(
          childSaid,
          'The child is learning about "$_currentTopic" and said: '
          '"$childSaid". Build on their response. Share another interesting '
          'fact or connection. Ask a follow-up question that goes deeper. '
          '3-4 sentences maximum.',
        );

      case 2:
        // After finishing a topic, ask for new one or end
        if (_wantsAnother(lower)) {
          _phase = 0;
          _currentTopic = '';
          _history.clear();
          return 'Wonderful! What else would you like to learn about?';
        }
        return await end();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_topicsExplored == 0 && _roundsInTopic == 0) {
      return 'Come back anytime you are curious about something! '
          'There is always something new to discover.';
    }

    if (_currentTopic.isNotEmpty && _roundsInTopic > 0) {
      _topicsExplored++;
    }

    return 'What a great learning session! You explored '
        '$_topicsExplored topic${_topicsExplored != 1 ? 's' : ''} today. '
        'Your curiosity is your superpower! Score: $_score points. '
        'Come back whenever you want to learn more!';
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt =
        'You are Kore, a curious and encouraging learning buddy for kids '
        'aged 4-14. You help children explore topics they are curious about. '
        '$guidance '
        'Rules: Keep language age-appropriate. Be enthusiastic about their '
        'curiosity. Make connections to things kids find interesting. '
        'No markdown, no bullets, no emojis. Speak naturally.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._history,
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
      final response = buffer.toString().trim();
      if (response.isNotEmpty) {
        _history.add({'role': 'assistant', 'content': response});
        return response;
      }
      return _fallback();
    } catch (e) {
      return _fallback();
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
      return result.isNotEmpty ? result : _fallback();
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return _fallback();
    }
  }

  String _fallback() {
    const fallbacks = [
      'That is really interesting! Tell me more about what you are thinking.',
      'Great question! What else would you like to know about this?',
      'I love your curiosity! What part of this topic interests you most?',
    ];
    return fallbacks[_rng.nextInt(fallbacks.length)];
  }

  bool _wantsAnother(String lower) {
    const triggers = ['yes', 'yeah', 'sure', 'okay', 'another', 'more',
      'new topic', 'different', 'something else',
      'हाँ', 'और', 'अवुनु', 'అవును', 'ఇంకొకటి'];
    return triggers.any((t) => lower.contains(t));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough',
      'no thanks', 'no more', 'bye'];
    return quitWords.any((w) => text.contains(w));
  }
}
