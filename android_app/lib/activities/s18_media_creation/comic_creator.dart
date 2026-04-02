import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// ComicCreator: dictate a comic story panel by panel.
///
/// Teaches: sequential storytelling, visual thinking, narrative structure,
/// creative expression.
///
/// Flow:
/// 1. Bot helps child create a comic story panel by panel.
/// 2. Child describes what happens in each panel.
/// 3. Bot narrates the panel with dramatic flair and asks for the next.
/// 4. After 4-6 panels, wrap up the comic with a dramatic ending.
class ComicCreator extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _panelCount = 0;
  int _score = 0;
  String _comicTitle = '';
  String _mainCharacter = '';

  /// 0=pick title, 1=pick character, 2=create panels, 3=ending
  int _phase = 0;
  final List<Map<String, String>> _history = [];
  final List<String> _panelDescriptions = [];

  static const int _maxPanels = 6;

  ComicCreator({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'comic_creator';

  @override
  String get name => 'Comic Creator';

  @override
  String get category => 'media';

  @override
  String get description =>
      'Create your own comic story panel by panel with voice.';

  @override
  List<String> get skills => [
        'sequential storytelling',
        'visual thinking',
        'narrative structure',
        'creative expression',
      ];

  @override
  int get minAge => 4;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.mediaCreation;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'comic creator',
          'make a comic',
          'comic book',
          'create a comic',
          'comic story',
        ],
        'hi': [
          'कॉमिक बनाओ',
          'कॉमिक कहानी',
          'कॉमिक बुक',
        ],
        'te': [
          'కామిక్ చేయి',
          'కామిక్ కథ',
          'కామిక్ బుక్',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_panelCount == 0) return 'No comic panels created yet.';
    return 'Comic: "$_comicTitle". $_panelCount panels created. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _panelCount = 0;
    _score = 0;
    _comicTitle = '';
    _mainCharacter = '';
    _phase = 0;
    _history.clear();
    _panelDescriptions.clear();

    return 'Welcome to the Comic Creator! We are going to make a comic book '
        'together, one panel at a time. You tell me what happens, and I will '
        'narrate it with dramatic flair! First, what should the title of '
        'your comic be?';
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
        // Child chose a title
        _comicTitle = childSaid.trim();
        _phase = 1;
        _score += 10;
        return 'Great title! "$_comicTitle" sounds like an amazing comic! '
            'Now, who is the main character? Give me a name and tell me '
            'a little about them.';

      case 1:
        // Child described the main character
        _mainCharacter = childSaid.trim();
        _phase = 2;
        _score += 10;
        return await _getLlmResponse(
          childSaid,
          'The child is creating a comic called "$_comicTitle" with main '
          'character: "$_mainCharacter". Narrate the opening of Panel 1 in '
          'a dramatic comic-book style. Set the scene briefly. Then ask: '
          'What happens first? What do we see in Panel 1? '
          '2-3 sentences.',
        );

      case 2:
        // Creating panels
        _panelCount++;
        _panelDescriptions.add(childSaid.trim());
        _score += 15;

        if (_panelCount >= _maxPanels) {
          _phase = 3;
          final wrap = await _getLlmResponse(
            childSaid,
            'The child described the final panel ($_panelCount of $_maxPanels) '
            'of their comic "$_comicTitle". Narrate this panel dramatically. '
            'Then deliver an epic comic-book ending. Celebrate their comic. '
            '3-4 sentences.',
          );
          _active = false;
          return '$wrap ${_buildEndSummary()}';
        }

        return await _getLlmResponse(
          childSaid,
          'The child described Panel $_panelCount of their comic '
          '"$_comicTitle": "$childSaid". Narrate this panel in dramatic '
          'comic-book style with sound effects like pow, whoosh, or zoom. '
          'Then ask: What happens next in Panel ${_panelCount + 1}? '
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
    if (_panelCount == 0) {
      return 'Come back to the Comic Creator anytime! We will make '
          'an awesome comic together.';
    }
    return 'Your comic "$_comicTitle" has $_panelCount '
        'panel${_panelCount != 1 ? 's' : ''}! '
        'You are a real comic book artist! Score: $_score points! '
        'You can draw these panels to bring your story to life!';
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt =
        'You are Kore, an enthusiastic comic book narrator helping a child '
        'create their own comic story panel by panel. The comic is called '
        '"$_comicTitle" and the main character is "$_mainCharacter". '
        '$guidance '
        'Rules: Use dramatic narration like a comic book. Add sound effects. '
        'Build on what the child says, never contradict them. '
        'Keep the story exciting and age-appropriate. '
        'No markdown, no bullets, no emojis. Speak naturally with energy.';

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
    return 'What an exciting panel! The story is getting really good. '
        'What happens next?';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough',
      'end the comic'];
    return quitWords.any((w) => text.contains(w));
  }
}
