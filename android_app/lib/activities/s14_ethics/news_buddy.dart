import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// A simplified news scenario for age-appropriate discussion.
class _NewsScenario {
  final String headline;
  final String details;

  const _NewsScenario({required this.headline, required this.details});
}

/// News Buddy: age-appropriate current event discussions.
///
/// Teaches: critical thinking, civic awareness, perspective-taking,
/// discussion skills, empathy.
///
/// Flow:
/// 1. Bot presents a simplified positive news scenario.
/// 2. Asks: Is this fair? Who benefits? What would you do?
/// 3. LLM drives discussion based on child's responses.
/// 4. 2 scenarios per session.
class NewsBuddy extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _scenariosDiscussed = 0;
  int _score = 0;
  int _maxScenarios = 2;

  /// 0=present scenario, 1=discuss, 2=wrap + next or end
  int _phase = 0;
  int _discussionRounds = 0;
  _NewsScenario? _current;
  final List<int> _usedIndices = [];
  final List<Map<String, String>> _history = [];

  static const List<_NewsScenario> _scenarios = [
    _NewsScenario(
      headline: 'A city planted 10,000 trees to fight pollution',
      details: 'A big city decided to plant 10,000 trees along its roads and in parks. They said it will clean the air, give shade, and make the city more beautiful. Volunteers of all ages helped plant the trees.',
    ),
    _NewsScenario(
      headline: 'Students invented a way to clean ocean plastic',
      details: 'A group of school students designed a special net that catches plastic waste in the ocean without hurting fish. Their teacher helped them build it, and now a company wants to make more of them!',
    ),
    _NewsScenario(
      headline: 'A law was passed to give free lunch to all school children',
      details: 'The government decided that every child in school should get a free healthy lunch every day. The meals include rice, vegetables, fruit, and milk. This way, no child goes hungry while learning.',
    ),
    _NewsScenario(
      headline: 'A village built a library using donated books',
      details: 'People from a big city donated thousands of books to a village that had no library. The villagers came together and built a beautiful library. Now children there can read any book they want for free!',
    ),
    _NewsScenario(
      headline: 'A child raised money to build a playground for kids with disabilities',
      details: 'An 8-year-old girl noticed that her friend in a wheelchair could not play on the regular playground. She raised money by selling lemonade and drawings. Now they are building a playground where everyone can play!',
    ),
    _NewsScenario(
      headline: 'Solar panels on school roofs now power the whole school',
      details: 'A school put solar panels on its roof and now gets all its electricity from the sun! The money they save on electricity bills goes to buying new books and computers for students.',
    ),
    _NewsScenario(
      headline: 'A farmer found a way to grow food using less water',
      details: 'A farmer invented a new way to water crops using tiny drops instead of flooding fields. It uses 70 percent less water and the crops actually grow better! Other farmers are learning the technique.',
    ),
    _NewsScenario(
      headline: 'A neighborhood banned cars for one day and let kids play in the streets',
      details: 'A neighborhood decided to close the roads to cars for one day every month so kids could play safely in the streets. They had games, art, and music. Parents and grandparents joined in too!',
    ),
  ];

  NewsBuddy({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'ethics_news_buddy';

  @override
  String get name => 'News Buddy';

  @override
  String get category => 'ethics';

  @override
  String get description =>
      'Discuss age-appropriate news stories and think about fairness.';

  @override
  List<String> get skills => [
        'critical thinking',
        'civic awareness',
        'perspective-taking',
        'discussion skills',
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
          'news buddy',
          'tell me news',
          'what is happening in the world',
          'news game',
          'discuss news',
        ],
        'hi': [
          'समाचार दोस्त',
          'खबर बताओ',
          'दुनिया में क्या हो रहा है',
        ],
        'te': [
          'వార్తల స్నేహితుడు',
          'వార్తలు చెప్పు',
          'ప్రపంచంలో ఏం జరుగుతోంది',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_scenariosDiscussed == 0) return 'No stories discussed yet.';
    return 'Discussed $_scenariosDiscussed stories. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _scenariosDiscussed = 0;
    _score = 0;
    _usedIndices.clear();

    return 'Welcome to News Buddy! I have some interesting stories about '
        'things happening in the world. Let us talk about them together! '
        '${_presentNewScenario()}';
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
        // Child gave initial reaction, discuss deeper
        _phase = 1;
        _discussionRounds = 0;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child shared their initial reaction to the news story. '
            'Validate their thinking. Ask a follow-up question: Is this fair? '
            'Who benefits the most? Who might not like this? '
            'Keep it to 2-3 sentences.');

      case 1:
        _discussionRounds++;
        _score += 10;
        if (_discussionRounds >= 2) {
          _phase = 2;
          _scenariosDiscussed++;

          if (_scenariosDiscussed >= _maxScenarios) {
            _active = false;
            final wrap = await _getLlmResponse(childSaid,
                'Wrap up the discussion. Celebrate the child\'s thinking. '
                'Summarize one key insight they shared. Keep it to 2 sentences.');
            return '$wrap ${_buildEndSummary()}';
          }

          final wrap = await _getLlmResponse(childSaid,
              'Briefly wrap up this story discussion. Keep it to 1 sentence.');
          return '$wrap Let us look at another story! ${_presentNewScenario()}';
        }

        return await _getLlmResponse(childSaid,
            'Continue the discussion. Ask: What would you do if you were '
            'in charge? How would you make it even better? '
            'Keep it to 2-3 sentences.');

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  String _presentNewScenario() {
    final available = <int>[];
    for (int i = 0; i < _scenarios.length; i++) {
      if (!_usedIndices.contains(i)) available.add(i);
    }
    if (available.isEmpty) {
      _usedIndices.clear();
      for (int i = 0; i < _scenarios.length; i++) available.add(i);
    }

    final idx = available[_rng.nextInt(available.length)];
    _usedIndices.add(idx);
    _current = _scenarios[idx];
    _history.clear();
    _phase = 0;
    _discussionRounds = 0;

    return 'Here is today\'s story: ${_current!.headline}. '
        '${_current!.details} What do you think about this?';
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt = 'You are Kore, a thoughtful news discussion partner for kids aged 5-14. '
        'You discuss age-appropriate positive news stories to develop civic thinking. '
        '$guidance '
        'Rules: Be encouraging. Ask open questions. No markdown. Speak naturally.';

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
    return 'That is a really interesting point! What else do you think about it?';
  }

  String _buildEndSummary() {
    if (_scenariosDiscussed == 0) {
      return 'Come back to discuss more stories about the world!';
    }
    return 'We discussed $_scenariosDiscussed '
        'stor${_scenariosDiscussed > 1 ? 'ies' : 'y'} today! '
        'Score: $_score points! You are a great thinker who cares about '
        'the world!';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
