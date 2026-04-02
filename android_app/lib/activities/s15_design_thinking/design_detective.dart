import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../../vision/camera_manager.dart';
import '../../vision/vision_describer.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// Design Detective: find and improve poorly designed everyday objects.
///
/// Teaches: observation, empathy, design thinking, problem-solving,
/// user-centered design.
///
/// Flow:
/// 1. Bot asks child to look around and find something poorly designed.
/// 2. If camera is available, child shows the object.
/// 3. Bot discusses: What is wrong? Who uses it? How would you fix it?
/// 4. LLM drives the design critique conversation.
/// 5. 2-3 objects per session.
class DesignDetective extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  final CameraManager? _camera;
  final VisionDescriber? _visionDescriber;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _objectsAnalyzed = 0;
  int _score = 0;
  int _maxObjects = 3;

  /// 0=find object, 1=what's wrong, 2=who uses it, 3=fix it, 4=next or end
  int _phase = 0;
  final List<Map<String, String>> _history = [];

  DesignDetective({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    CameraManager? camera,
    VisionDescriber? visionDescriber,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer(),
        _camera = camera,
        _visionDescriber = visionDescriber;

  @override
  String get id => 'design_detective';

  @override
  String get name => 'Design Detective';

  @override
  String get category => 'design';

  @override
  String get description =>
      'Find everyday objects with bad design and figure out how to improve them.';

  @override
  List<String> get skills => [
        'observation',
        'empathy',
        'design thinking',
        'problem-solving',
      ];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.designThinking;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'design detective',
          'bad design',
          'fix the design',
          'design game',
          'improve things',
        ],
        'hi': [
          'डिज़ाइन जासूस',
          'डिज़ाइन सुधारो',
          'डिज़ाइन खेल',
        ],
        'te': [
          'డిజైన్ డిటెక్టివ్',
          'డిజైన్ బాగు చేయి',
          'డిజైన్ ఆట',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_objectsAnalyzed == 0) return 'No objects analyzed yet.';
    return 'Analyzed $_objectsAnalyzed objects. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _objectsAnalyzed = 0;
    _score = 0;
    _phase = 0;
    _history.clear();

    final hasCamera = _camera != null;
    final cameraHint = hasCamera
        ? 'You can show me the object using the camera, or just describe it!'
        : 'Describe something in your house that annoys you because of bad design.';

    return 'Welcome, Design Detective! Great designers notice things that '
        'could be better. Look around your room or house. Find one thing that '
        'is poorly designed or hard to use. $cameraHint';
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
        // Child identified an object
        _phase = 1;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child identified a poorly designed object. Acknowledge their observation. '
            'Ask: What exactly is wrong with it? What makes it hard to use or annoying? '
            'Keep it to 2-3 sentences.');

      case 1:
        // What is wrong with it
        _phase = 2;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child explained what is wrong with the design. Good detective work! '
            'Now ask: Who uses this thing? A kid? An adult? An elderly person? '
            'Would the problem be worse for some people than others? '
            'Keep it to 2-3 sentences.');

      case 2:
        // Who uses it
        _phase = 3;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child thought about who uses the object. Now the fun part! '
            'Ask: If you could redesign this from scratch, how would you make '
            'it better? Think about the person using it. What would make their '
            'life easier? Give them 1 small hint or starting idea. '
            'Keep it to 2-3 sentences.');

      case 3:
        // How to fix it
        _phase = 4;
        _score += 15;
        _objectsAnalyzed++;

        if (_objectsAnalyzed >= _maxObjects) {
          _active = false;
          final wrap = await _getLlmResponse(childSaid,
              'The child designed a solution! Celebrate their design thinking. '
              'Tell them they went through the design process: '
              'observe, empathize, ideate! Keep it to 2-3 sentences.');
          return '$wrap ${_buildEndSummary()}';
        }

        final wrap = await _getLlmResponse(childSaid,
            'Great design solution! Briefly praise it. Keep it to 1-2 sentences.');
        _history.clear();
        _phase = 0;
        return '$wrap Now find another object to improve! Look around, what '
            'else could be designed better?';

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
    final systemPrompt = 'You are Kore, a design thinking coach for kids aged 5-14. '
        'You teach kids to notice bad design and think of improvements. '
        '$guidance '
        'Rules: Be encouraging. No markdown. Speak naturally.';

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
    return 'That is a great observation! Good designers always notice these things. '
        'How would you make it better?';
  }

  String _buildEndSummary() {
    if (_objectsAnalyzed == 0) {
      return 'Come back anytime to practice your design detective skills!';
    }
    return 'You analyzed $_objectsAnalyzed '
        'object${_objectsAnalyzed > 1 ? 's' : ''} and came up with '
        'improvements for each one! Score: $_score points! '
        'You think like a real designer!';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
