import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// Ask Better Questions: teaches kids HOW to find answers instead of
/// just giving answers directly.
///
/// Teaches: research skills, critical thinking, information literacy,
/// self-directed learning.
///
/// Flow:
/// 1. Bot invites the child to ask any question.
/// 2. Instead of answering, teaches search strategies.
/// 3. "What words would you search for?"
/// 4. "Where would you look? Book? Website? Expert?"
/// 5. "If two sources disagree, how would you decide?"
/// 6. 2-3 questions per session.
class AskBetterQuestions extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  bool _active = false;
  int _questionsExplored = 0;
  int _score = 0;
  int _maxQuestions = 3;

  /// 0=ask question, 1=search words, 2=where to look, 3=source conflict, 4=next or end
  int _phase = 0;
  final List<Map<String, String>> _history = [];

  AskBetterQuestions({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'info_ask_better_questions';

  @override
  String get name => 'Ask Better Questions';

  @override
  String get category => 'information';

  @override
  String get description =>
      'Learn HOW to find answers instead of just being given answers.';

  @override
  List<String> get skills => [
        'research skills',
        'critical thinking',
        'information literacy',
        'self-directed learning',
      ];

  @override
  int get minAge => 6;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.informationLiteracy;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'ask better questions',
          'how to find answers',
          'research game',
          'information game',
          'teach me to research',
        ],
        'hi': [
          'बेहतर सवाल',
          'जवाब कैसे खोजें',
          'रिसर्च खेल',
        ],
        'te': [
          'మంచి ప్రశ్నలు',
          'సమాధానాలు ఎలా కనుగొనాలి',
          'రీసెర్చ్ ఆట',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_questionsExplored == 0) return 'No questions explored yet.';
    return 'Explored $_questionsExplored questions. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _questionsExplored = 0;
    _score = 0;
    _phase = 0;
    _history.clear();

    return 'Welcome to Ask Better Questions! Today, instead of just giving '
        'you answers, I am going to teach you how to find answers yourself. '
        'That is a superpower that will help you your whole life! '
        'Ask me any question you are curious about.';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      return await end();
    }

    _history.add({'role': 'user', 'content': childSaid});

    switch (_phase) {
      case 0:
        // Child asked a question
        _phase = 1;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child asked a question. Do NOT answer it directly. '
            'Instead say: Great question! But instead of just telling you, '
            'let me teach you how to find this yourself. '
            'Ask: If you were going to search for this on the internet or in a '
            'library, what words or phrases would you search for? '
            'Keep it to 2-3 sentences.');

      case 1:
        // Search words
        _phase = 2;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child suggested search words. Validate their choices. '
            'Maybe suggest one better keyword. '
            'Then ask: Where would you look for the answer? A book? A website? '
            'Ask a teacher or expert? Which would give the most reliable answer? '
            'Keep it to 2-3 sentences.');

      case 2:
        // Where to look
        _phase = 3;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child identified where to look. Good thinking! '
            'Now ask: What if you found two different answers from two different '
            'places? How would you decide which one to trust? '
            'Hint: Think about who wrote it and whether they are experts. '
            'Keep it to 2-3 sentences.');

      case 3:
        // Source conflict resolution
        _phase = 4;
        _score += 15;
        _questionsExplored++;

        if (_questionsExplored >= _maxQuestions) {
          _active = false;
          final wrap = await _getLlmResponse(childSaid,
              'Celebrate their critical thinking. Now briefly give the actual '
              'answer to their original question as a reward. '
              'Then say they now have research superpowers! '
              'Keep it to 3-4 sentences.');
          return '$wrap ${_buildEndSummary()}';
        }

        final wrap = await _getLlmResponse(childSaid,
            'Briefly praise their thinking about source reliability. '
            'Then briefly give a hint about the actual answer as a reward. '
            'Keep it to 2 sentences.');
        _history.clear();
        _phase = 0;
        return '$wrap Great research skills! Ask me another question.';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt = 'You are Kore, an information literacy coach for kids aged 6-14. '
        'You teach kids how to find reliable information themselves. '
        '$guidance '
        'Rules: Do not just give answers. Teach the process. '
        'No markdown. Speak naturally. Be encouraging.';

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
      if (result.isNotEmpty) _history.add({'role': 'assistant', 'content': result});
      return result.isNotEmpty ? result : _fallback();
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return _fallback();
    }
  }

  String _fallback() {
    return 'Good thinking! Finding reliable answers is one of the most '
        'important skills you can learn.';
  }

  String _buildEndSummary() {
    if (_questionsExplored == 0) {
      return 'Come back to practice your research superpowers!';
    }
    return 'You explored $_questionsExplored questions and learned how to '
        'find answers like a researcher! Score: $_score points! '
        'Remember: search smart, check your sources, and stay curious!';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
