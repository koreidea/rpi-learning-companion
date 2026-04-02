import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// PodcastProducer: record a mini-podcast episode together.
///
/// Teaches: storytelling, interviewing, structured communication, creativity.
///
/// Flow:
/// 1. Bot introduces the podcast and helps child pick a theme.
/// 2. Bot plays host/guest depending on child's preference.
/// 3. Interview-style conversation with 3-4 questions.
/// 4. Wrap up with a podcast sign-off.
class PodcastProducer extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _questionsAsked = 0;
  int _score = 0;
  String _podcastTheme = '';

  /// 0=pick theme, 1=intro, 2=interview, 3=wrap-up
  int _phase = 0;
  final List<Map<String, String>> _history = [];

  static const int _maxQuestions = 4;

  static const List<String> _themeSuggestions = [
    'your favorite things in the whole world',
    'what it would be like to live on Mars',
    'the funniest thing that ever happened to you',
    'if you could have any superpower',
    'what school would be like if kids were in charge',
    'the best adventure you can imagine',
    'your dream invention',
    'animals that should be able to talk',
  ];

  PodcastProducer({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'podcast_producer';

  @override
  String get name => 'Podcast Producer';

  @override
  String get category => 'media';

  @override
  String get description =>
      'Record a mini-podcast episode and practice storytelling.';

  @override
  List<String> get skills =>
      ['storytelling', 'interviewing', 'communication', 'creativity'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.mediaCreation;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'podcast',
          'make a podcast',
          'record a show',
          'podcast producer',
          'start a podcast',
        ],
        'hi': [
          'पॉडकास्ट',
          'पॉडकास्ट बनाओ',
          'शो रिकॉर्ड करो',
        ],
        'te': [
          'పాడ్కాస్ట్',
          'పాడ్కాస్ట్ చేయి',
          'షో రికార్డ్ చేయి',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_questionsAsked == 0) return 'No podcast recorded yet.';
    return 'Podcast on "$_podcastTheme". $_questionsAsked questions discussed. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _questionsAsked = 0;
    _score = 0;
    _podcastTheme = '';
    _phase = 0;
    _history.clear();

    final suggestion =
        _themeSuggestions[_rng.nextInt(_themeSuggestions.length)];

    return 'Welcome to the Podcast Studio! Today we are going to make our '
        'very own podcast episode. I will be the host and you are my special '
        'guest! What should our episode be about? Maybe we could talk about '
        '$suggestion? What topic excites you?';
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
        // Child picked a theme
        _podcastTheme = childSaid.trim();
        _phase = 1;
        _score += 10;
        return await _getLlmResponse(
          childSaid,
          'The child chose the podcast topic: "$_podcastTheme". '
          'Do the podcast intro: welcome listeners, introduce the topic, '
          'introduce the child as your special guest. Then ask your first '
          'interview question about the topic. Be warm and professional '
          'like a real podcast host. 3-4 sentences.',
        );

      case 1:
        // After intro, move to interview phase
        _phase = 2;
        _questionsAsked++;
        _score += 10;
        return await _getLlmResponse(
          childSaid,
          'The child answered your first podcast question. React positively '
          'to what they said, add a brief comment, and ask your next '
          'question that digs deeper into "$_podcastTheme". '
          '2-3 sentences.',
        );

      case 2:
        // Interview questions
        _questionsAsked++;
        _score += 10;

        if (_questionsAsked >= _maxQuestions) {
          _phase = 3;
          final wrap = await _getLlmResponse(
            childSaid,
            'The child gave their final answer. React to it warmly. '
            'Then wrap up the podcast: thank your guest, tell listeners '
            'what a great episode this was. Sign off the show. '
            '2-3 sentences.',
          );
          _active = false;
          return '$wrap ${_buildEndSummary()}';
        }

        return await _getLlmResponse(
          childSaid,
          'The child answered a podcast question about "$_podcastTheme". '
          'React to their answer with genuine interest. Then ask the next '
          'interview question. Make it creative and fun. '
          'Question ${_questionsAsked + 1} of $_maxQuestions. '
          '2-3 sentences.',
        );

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
    if (_questionsAsked == 0) {
      return 'Come back to the Podcast Studio anytime! We will make an '
          'awesome episode together.';
    }
    return 'That is a wrap on our podcast episode '
        '${_podcastTheme.isNotEmpty ? 'about "$_podcastTheme"' : ''}! '
        'You answered $_questionsAsked questions like a pro. '
        'Score: $_score points! You are a natural podcaster!';
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt =
        'You are Kore, a friendly and enthusiastic podcast host for kids. '
        'The child is your special guest. You are recording a podcast '
        'episode about "$_podcastTheme". '
        '$guidance '
        'Rules: Stay in character as a podcast host. Be genuinely curious '
        'about the child\'s answers. Never dismiss what they say. '
        'No markdown, no bullets, no emojis. Speak naturally.';

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
    return 'That is such a great answer! Our listeners are going to love '
        'this episode. Tell me more!';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough',
      'end the podcast', 'wrap up'];
    return quitWords.any((w) => text.contains(w));
  }
}
