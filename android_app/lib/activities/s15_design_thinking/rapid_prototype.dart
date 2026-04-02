import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../../vision/camera_manager.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// A design challenge prompt.
class _DesignChallenge {
  final String challenge;
  final String userHint;

  const _DesignChallenge({required this.challenge, required this.userHint});
}

/// Rapid Prototype: guided design challenges using the 5-step design
/// thinking process (Empathize, Define, Ideate, Prototype, Test).
///
/// Teaches: design thinking, empathy, creative problem-solving,
/// prototyping, testing mindset.
///
/// Flow:
/// 1. Bot presents a design challenge.
/// 2. Empathize: Who will use it?
/// 3. Define: What is the main problem?
/// 4. Ideate: What are 3 possible solutions?
/// 5. Prototype: Draw your best idea and show me (camera optional).
/// 6. Test: How would you test if it works?
class RapidPrototype extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  final CameraManager? _camera;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _score = 0;

  /// 0=challenge presented, 1=empathize, 2=define, 3=ideate, 4=prototype, 5=test, 6=done
  int _phase = 0;
  _DesignChallenge? _currentChallenge;
  final List<Map<String, String>> _history = [];

  static const List<_DesignChallenge> _challenges = [
    _DesignChallenge(
      challenge: 'Design a water bottle for your grandmother who has arthritis and cannot grip well.',
      userHint: 'Think about what makes gripping hard for her.',
    ),
    _DesignChallenge(
      challenge: 'Design a school bag that never hurts your back, no matter how many books are inside.',
      userHint: 'Think about weight distribution and comfort.',
    ),
    _DesignChallenge(
      challenge: 'Design a lunchbox that keeps food hot and cold at the same time, for different items.',
      userHint: 'Think about separate compartments and insulation.',
    ),
    _DesignChallenge(
      challenge: 'Design a toy that a child who cannot see can enjoy just as much as a child who can.',
      userHint: 'Think about touch, sound, and smell.',
    ),
    _DesignChallenge(
      challenge: 'Design a way for kids to safely cross a busy road near school.',
      userHint: 'Think about visibility, signals, and barriers.',
    ),
    _DesignChallenge(
      challenge: 'Design a pencil that never needs sharpening and never runs out.',
      userHint: 'Think about what material the tip could be made of.',
    ),
    _DesignChallenge(
      challenge: 'Design an umbrella that works even in very strong wind.',
      userHint: 'Think about shape and flexibility.',
    ),
    _DesignChallenge(
      challenge: 'Design a homework helper tool that does not give you the answers but helps you learn.',
      userHint: 'Think about hints, explanations, and practice.',
    ),
  ];

  RapidPrototype({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    CameraManager? camera,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer(),
        _camera = camera;

  @override
  String get id => 'design_rapid_prototype';

  @override
  String get name => 'Rapid Prototype';

  @override
  String get category => 'design';

  @override
  String get description =>
      'Solve a design challenge using the 5-step design thinking process.';

  @override
  List<String> get skills => [
        'design thinking',
        'empathy',
        'creative problem-solving',
        'prototyping',
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
          'rapid prototype',
          'design challenge',
          'design thinking',
          'design something',
          'prototype game',
        ],
        'hi': [
          'डिज़ाइन चैलेंज',
          'प्रोटोटाइप बनाओ',
          'डिज़ाइन सोच',
        ],
        'te': [
          'డిజైన్ ఛాలెంజ్',
          'ప్రోటోటైప్',
          'డిజైన్ ఆలోచన',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    final stepNames = ['Empathize', 'Define', 'Ideate', 'Prototype', 'Test'];
    final step = _phase.clamp(0, 4);
    return 'Design step: ${step < 5 ? stepNames[step] : 'Complete'}. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _score = 0;
    _phase = 0;
    _history.clear();

    _currentChallenge = _challenges[_rng.nextInt(_challenges.length)];

    return 'Welcome to Rapid Prototype! Today we are going to use the '
        '5 steps of design thinking: Empathize, Define, Ideate, Prototype, '
        'and Test. Here is your challenge! '
        '${_currentChallenge!.challenge} '
        'Step 1, Empathize: Who will use this? Tell me about the person.';
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
        // Empathize response, move to Define
        _phase = 1;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child described who will use the product (Empathize step). '
            'Validate their empathy. Then ask Step 2, Define: '
            'What is the main problem this person faces? '
            'Hint: ${_currentChallenge!.userHint} '
            'Keep it to 2-3 sentences.');

      case 1:
        // Define response, move to Ideate
        _phase = 2;
        _score += 10;
        return await _getLlmResponse(childSaid,
            'The child defined the main problem (Define step). Great clarity! '
            'Now Step 3, Ideate: Can you think of 3 different possible '
            'solutions? They can be wild, creative, or practical! '
            'Keep it to 2-3 sentences.');

      case 2:
        // Ideate response, move to Prototype
        _phase = 3;
        _score += 15;
        final cameraNote = _camera != null
            ? 'If you want, draw your idea on paper and show me with the camera! Or just describe it in detail.'
            : 'Describe your design in detail. What does it look like? What is it made of?';
        return await _getLlmResponse(childSaid,
            'The child brainstormed solutions (Ideate step). Praise their creativity. '
            'Now Step 4, Prototype: Pick your best idea and describe or draw it! '
            '$cameraNote '
            'Keep it to 2-3 sentences.');

      case 3:
        // Prototype response, move to Test
        _phase = 4;
        _score += 15;
        return await _getLlmResponse(childSaid,
            'The child described their prototype (Prototype step). '
            'Acknowledge the design with enthusiasm. '
            'Now Step 5, Test: How would you test if this works? '
            'Who would you give it to? What would you ask them? '
            'What could go wrong? '
            'Keep it to 2-3 sentences.');

      case 4:
        // Test response, wrap up
        _phase = 5;
        _score += 20;
        _active = false;
        return await _getLlmResponse(childSaid,
            'The child described how they would test their design (Test step). '
            'Celebrate completing all 5 steps of design thinking! '
            'Recap: they empathized with the user, defined the problem, '
            'brainstormed ideas, created a prototype, and planned testing. '
            'That is exactly what professional designers do! '
            'Keep it to 3-4 sentences. Be very enthusiastic.');

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_phase == 0) {
      return 'Come back anytime for a design challenge!';
    }
    return 'You completed ${_phase} of 5 design thinking steps! '
        'Score: $_score points! Remember the steps: Empathize, Define, '
        'Ideate, Prototype, Test. You can solve any design problem with them!';
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt = 'You are Kore, a design thinking coach for kids aged 5-14. '
        'You guide kids through the 5-step design thinking process. '
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
    return 'Great thinking! You are approaching this like a real designer. '
        'Keep going!';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
