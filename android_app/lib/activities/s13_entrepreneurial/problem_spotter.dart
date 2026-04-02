import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// Problem Spotter: guide kids through identifying real-world problems
/// and brainstorming solutions, teaching the first step of entrepreneurship.
///
/// Teaches: observation, problem identification, empathy, creative solutions,
/// entrepreneurial thinking.
///
/// Flow:
/// 1. Bot asks the child about something that annoyed them today.
/// 2. Child shares a problem.
/// 3. Bot guides through: who has it -> why -> possible solutions -> best one.
/// 4. Celebrates: "You just did entrepreneurship step 1!"
/// 5. One problem deep-dive per session.
class ProblemSpotter extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  bool _active = false;
  int _score = 0;

  /// 0=intro, 1=who has it, 2=why, 3=solutions, 4=best one, 5=celebrate
  int _phase = 0;
  final List<Map<String, String>> _history = [];

  ProblemSpotter({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'entrepreneurial_problem_spotter';

  @override
  String get name => 'Problem Spotter';

  @override
  String get category => 'entrepreneurial';

  @override
  String get description =>
      'Spot real-world problems and brainstorm solutions like an entrepreneur.';

  @override
  List<String> get skills => [
        'observation',
        'problem identification',
        'creative solutions',
        'entrepreneurial thinking',
      ];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.entrepreneurial;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'problem spotter',
          'spot a problem',
          'find a problem',
          'entrepreneur game',
          'business idea',
        ],
        'hi': [
          'समस्या खोजो',
          'उद्यमी खेल',
          'बिज़नेस आइडिया',
        ],
        'te': [
          'సమస్య కనుగొను',
          'వ్యాపార ఆలోచన',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    return 'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _score = 0;
    _phase = 0;
    _history.clear();

    return 'Welcome to Problem Spotter! Every great invention started because '
        'someone noticed a problem and said, I can fix this! '
        'Tell me, what is one thing that annoyed you or bothered you today? '
        'It can be anything, big or small.';
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
        _phase = 1;
        _score += 10;
        return await _getLlmGuide(childSaid,
            'The child identified a problem. Acknowledge it warmly. '
            'Then ask: How many other kids or people do you think have this same problem? '
            'Keep it to 2-3 sentences. Be encouraging.');

      case 1:
        _phase = 2;
        _score += 10;
        return await _getLlmGuide(childSaid,
            'The child answered about who else has this problem. Validate their thinking. '
            'Now ask: Why do you think this problem happens? What causes it? '
            'Keep it to 2-3 sentences.');

      case 2:
        _phase = 3;
        _score += 10;
        return await _getLlmGuide(childSaid,
            'The child explained why the problem happens. Great thinking! '
            'Now ask: Can you think of 3 different ways to fix this problem? '
            'They can be silly, creative, or practical. There are no bad ideas! '
            'Keep it to 2-3 sentences.');

      case 3:
        _phase = 4;
        _score += 10;
        return await _getLlmGuide(childSaid,
            'The child suggested solutions. Praise each one. '
            'Now ask: Which one do you think is the best? Which one could actually work? '
            'Keep it to 2-3 sentences.');

      case 4:
        _phase = 5;
        _score += 20;
        _active = false;
        return await _getLlmGuide(childSaid,
            'The child picked their best solution. Celebrate enthusiastically! '
            'Tell them: You just completed the first steps of entrepreneurship! '
            'Step 1: Spot a problem. Step 2: Understand who has it. Step 3: Figure out why. '
            'Step 4: Brainstorm solutions. Step 5: Pick the best one. '
            'Real inventors and business people do exactly this! '
            'Give a final encouraging message about their potential. '
            'Keep it to 4-5 sentences.');

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_score == 0) {
      return 'Come back anytime to spot problems and become an entrepreneur!';
    }
    return 'Great problem-spotting session! You earned $_score points! '
        'Remember, every problem is an opportunity waiting to be solved!';
  }

  Future<String> _getLlmGuide(String childSaid, String guidance) async {
    final systemPrompt = 'You are Kore, an encouraging entrepreneurship coach for kids aged 5-14. '
        'You are guiding a child through problem identification and solution brainstorming. '
        '$guidance '
        'Rules: No markdown, no bullets. Speak naturally. Be very encouraging. '
        'Use simple language appropriate for a child.';

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
    return 'That is really great thinking! You are looking at the world like '
        'an entrepreneur. Keep going, what else comes to mind?';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
