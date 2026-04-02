import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// A moral dilemma with no single right answer.
class _Dilemma {
  final String situation;
  final String followUp;

  const _Dilemma({required this.situation, required this.followUp});
}

/// Ethics Engine: daily moral dilemmas that explore reasoning without
/// judging right or wrong.
///
/// Teaches: ethical reasoning, empathy, perspective-taking, moral thinking,
/// critical evaluation.
///
/// Flow:
/// 1. Bot presents a moral dilemma.
/// 2. Child makes a choice.
/// 3. Bot asks why and explores reasoning using LLM.
/// 4. Bot asks: "What would happen if everyone made that choice?"
/// 5. Celebrates thoughtful reasoning.
///
/// 2-3 dilemmas per session.
class EthicsEngine extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _dilemmasDiscussed = 0;
  int _score = 0;
  int _maxDilemmas = 2;

  /// 0=present dilemma, 1=ask why, 2=everyone question, 3=next or end
  int _phase = 0;
  _Dilemma? _currentDilemma;
  final List<int> _usedIndices = [];
  final List<Map<String, String>> _history = [];

  static const List<_Dilemma> _dilemmas = [
    _Dilemma(
      situation: 'Your friend cheated on a test. The teacher asks if you saw anything. What do you do?',
      followUp: 'Would it change your answer if the friend was your best friend?',
    ),
    _Dilemma(
      situation: 'You find 100 rupees on the ground. Nobody is around. What do you do?',
      followUp: 'What if it was 1000 rupees? Would that change your decision?',
    ),
    _Dilemma(
      situation: 'Your friend tells you a secret, but the secret could hurt someone else. What do you do?',
      followUp: 'Is it ever okay to break a promise to protect someone?',
    ),
    _Dilemma(
      situation: 'Everyone is making fun of the new kid at school. Your friends want you to join in. What do you do?',
      followUp: 'How do you think the new kid feels? What could you do to help?',
    ),
    _Dilemma(
      situation: 'You broke a vase while playing inside the house. Nobody saw it happen. What do you do?',
      followUp: 'How would you feel if you told the truth? How would you feel if you kept it a secret?',
    ),
    _Dilemma(
      situation: 'Your sibling took your toy without asking, and now it is broken. What do you do?',
      followUp: 'Is there a way to solve this that makes both of you feel better?',
    ),
    _Dilemma(
      situation: 'A shopkeeper gave you extra change by mistake. What do you do?',
      followUp: 'What if the shopkeeper was very rich? What if they were very poor? Does it matter?',
    ),
    _Dilemma(
      situation: 'You see someone littering in the park. What do you do?',
      followUp: 'Is it your responsibility to say something, or is it not your business?',
    ),
    _Dilemma(
      situation: 'Your friend asks you to copy your homework. They will get in trouble if they do not have it. What do you do?',
      followUp: 'How could you help your friend without letting them copy?',
    ),
    _Dilemma(
      situation: 'You promised to play with one friend, but another friend invites you to something more fun. What do you do?',
      followUp: 'How important is keeping a promise?',
    ),
  ];

  EthicsEngine({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'ethics_engine';

  @override
  String get name => 'Ethics Engine';

  @override
  String get category => 'ethics';

  @override
  String get description =>
      'Explore moral dilemmas and practice ethical thinking.';

  @override
  List<String> get skills => [
        'ethical reasoning',
        'empathy',
        'perspective-taking',
        'moral thinking',
      ];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.ethics;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'ethics game',
          'moral dilemma',
          'right or wrong',
          'what should i do',
          'ethics engine',
        ],
        'hi': [
          'नैतिकता खेल',
          'सही या गलत',
          'क्या करना चाहिए',
        ],
        'te': [
          'నీతి ఆట',
          'సరైనది ఏమిటి',
          'ఏం చేయాలి',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_dilemmasDiscussed == 0) return 'No dilemmas discussed yet.';
    return 'Discussed $_dilemmasDiscussed dilemmas. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _dilemmasDiscussed = 0;
    _score = 0;
    _phase = 0;
    _usedIndices.clear();
    _history.clear();

    return 'Welcome to the Ethics Engine! I am going to tell you about some '
        'tricky situations. There is no single right answer, and I will not '
        'judge you. I just want to hear your thinking! '
        '${_presentNewDilemma()}';
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
        // Child made their choice, ask why
        _phase = 1;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child made a choice about a moral dilemma. '
            'Do not judge their choice. Acknowledge it thoughtfully. '
            'Then ask: Why did you choose that? What made you decide? '
            'Keep it to 2-3 sentences.');

      case 1:
        // Child explained why, ask the "everyone" question
        _phase = 2;
        _score += 10;
        final dilemma = _currentDilemma!;
        return await _getLlmResponse(childSaid,
            'The child explained their reasoning. Validate their thinking. '
            'Then ask: ${dilemma.followUp} Also ask: What would happen if '
            'everyone in the world made the same choice? '
            'Keep it to 2-3 sentences.');

      case 2:
        // Child answered the broader question, wrap up this dilemma
        _phase = 3;
        _score += 10;
        _dilemmasDiscussed++;

        if (_dilemmasDiscussed >= _maxDilemmas) {
          _active = false;
          final wrap = await _getLlmResponse(childSaid,
              'The child gave their final thoughts. Celebrate their thoughtful '
              'reasoning. Tell them they are a great thinker. '
              'Keep it to 2 sentences.');
          return '$wrap ${_buildEndSummary()}';
        }

        final wrap = await _getLlmResponse(childSaid,
            'The child gave their final thoughts. Briefly celebrate their thinking. '
            'Keep it to 1-2 sentences.');
        return '$wrap Here is another one! ${_presentNewDilemma()}';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  String _presentNewDilemma() {
    final available = <int>[];
    for (int i = 0; i < _dilemmas.length; i++) {
      if (!_usedIndices.contains(i)) available.add(i);
    }

    if (available.isEmpty) {
      _usedIndices.clear();
      for (int i = 0; i < _dilemmas.length; i++) {
        available.add(i);
      }
    }

    final idx = available[_rng.nextInt(available.length)];
    _usedIndices.add(idx);
    _currentDilemma = _dilemmas[idx];
    _history.clear();
    _phase = 0;

    return 'Here is the situation. ${_currentDilemma!.situation} '
        'What would you do?';
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt = 'You are Kore, a thoughtful ethics coach for kids aged 5-14. '
        'You explore moral dilemmas without judging. There is no right or wrong answer. '
        '$guidance '
        'Rules: Never say the child is wrong. Never lecture. Ask open questions. '
        'No markdown, no bullets. Speak naturally.';

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
    return 'That is really thoughtful! There is no perfect answer to this, '
        'and the fact that you are thinking about it shows great character.';
  }

  String _buildEndSummary() {
    if (_dilemmasDiscussed == 0) {
      return 'Come back anytime to explore more ethical dilemmas!';
    }
    return 'We discussed $_dilemmasDiscussed '
        'dilemma${_dilemmasDiscussed > 1 ? 's' : ''} today! '
        'Score: $_score points! You have a great sense of right and wrong. '
        'Keep thinking about how your choices affect others!';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
