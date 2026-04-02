import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// KoreNewsChannel: kid creates a "news show" with bot as co-anchor.
///
/// Teaches: storytelling, public speaking, summarization, media literacy.
///
/// Flow:
/// 1. Bot introduces the news show and assigns roles.
/// 2. Child picks a news topic (real or imaginary).
/// 3. Bot plays co-anchor, adds context, asks follow-up questions.
/// 4. Child delivers their news report.
/// 5. 2-3 news stories per "broadcast."
class KoreNewsChannel extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _storiesCovered = 0;
  int _score = 0;

  /// 0=pick topic, 1=deliver report, 2=follow-up, 3=next or end
  int _phase = 0;
  String _currentTopic = '';
  final List<Map<String, String>> _history = [];

  static const int _maxStories = 3;

  static const List<String> _topicSuggestions = [
    'a dog that learned to ride a skateboard',
    'a new flavor of ice cream invented by kids',
    'a robot that helps with homework',
    'a volcano that erupted candy',
    'a school where recess lasts all day',
    'a cat that became mayor of a town',
    'a garden that grows pizza',
    'kids who discovered a new planet',
  ];

  KoreNewsChannel({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'kore_news_channel';

  @override
  String get name => 'Kore News Channel';

  @override
  String get category => 'media';

  @override
  String get description =>
      'Be a news anchor and create your own news broadcast.';

  @override
  List<String> get skills =>
      ['storytelling', 'public speaking', 'summarization', 'media literacy'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.mediaCreation;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'news channel',
          'news show',
          'be a reporter',
          'news anchor',
          'kore news',
          'breaking news',
        ],
        'hi': [
          'समाचार चैनल',
          'न्यूज़ शो',
          'रिपोर्टर बनो',
        ],
        'te': [
          'న్యూస్ ఛానల్',
          'వార్తలు చెప్పు',
          'రిపోర్టర్ అవ్వు',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_storiesCovered == 0) return 'No stories covered yet.';
    return 'Covered $_storiesCovered news '
        'stor${_storiesCovered != 1 ? 'ies' : 'y'}. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _storiesCovered = 0;
    _score = 0;
    _phase = 0;
    _history.clear();

    final suggestion =
        _topicSuggestions[_rng.nextInt(_topicSuggestions.length)];

    return 'Good evening, and welcome to the Kore News Channel! '
        'I am your co-anchor, Kore, and YOU are the lead reporter tonight. '
        'What story would you like to report on? It can be real or totally '
        'made up! For example, how about $suggestion? '
        'What is your first story?';
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
        // Child picked a topic
        _currentTopic = childSaid.trim();
        _phase = 1;
        _score += 10;
        return await _getLlmResponse(
          childSaid,
          'The child chose a news topic: "$_currentTopic". '
          'As their co-anchor, react with excitement. Set the scene like a '
          'real news broadcast. Say something like: "Breaking news!" Then '
          'ask the child to give us more details as the lead reporter. '
          'Keep it to 2-3 sentences. Be playful and professional.',
        );

      case 1:
        // Child delivered their report
        _phase = 2;
        _score += 15;
        return await _getLlmResponse(
          childSaid,
          'The child just delivered their news report about "$_currentTopic". '
          'As co-anchor, react like a real news partner: comment on what '
          'they said, add a fun detail, and ask one follow-up question a '
          'real reporter would ask (who, what, where, when, why, or how). '
          'Keep it to 2-3 sentences.',
        );

      case 2:
        // Child answered follow-up
        _storiesCovered++;
        _score += 10;
        _phase = 3;

        if (_storiesCovered >= _maxStories) {
          _active = false;
          final wrap = await _getLlmResponse(
            childSaid,
            'The child finished answering. Wrap up this story and the whole '
            'broadcast. Thank the lead reporter for excellent journalism. '
            'Sign off like a real news show. 2-3 sentences.',
          );
          return '$wrap $_buildEndSummary()';
        }

        final wrap = await _getLlmResponse(
          childSaid,
          'The child answered the follow-up question. Briefly wrap up this '
          'story with a fun comment. Then say: we have another story coming '
          'up. 1-2 sentences.',
        );
        _phase = 0;
        _currentTopic = '';
        _history.clear();
        return '$wrap What is our next story, lead reporter?';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  String _buildEndSummary() {
    if (_storiesCovered == 0) {
      return 'This has been the Kore News Channel. Come back anytime '
          'to report the news!';
    }
    return 'That is a wrap for tonight on the Kore News Channel! '
        'Our lead reporter covered $_storiesCovered '
        'stor${_storiesCovered != 1 ? 'ies' : 'y'} tonight. '
        'Score: $_score points! Great journalism!';
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt =
        'You are Kore, a friendly and professional co-anchor on a kids news '
        'show. The child is the lead reporter. You play along with whatever '
        'news story they choose, real or imaginary. '
        '$guidance '
        'Rules: Stay in character as a news co-anchor. Be encouraging and '
        'treat the child like a real colleague. Never correct their stories. '
        'No markdown, no bullets, no emojis. Speak naturally like a TV anchor.';

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
    return 'What an incredible story! Tell me more, lead reporter. '
        'Our viewers want to hear all the details!';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough',
      'end the show', 'sign off'];
    return quitWords.any((w) => text.contains(w));
  }
}
