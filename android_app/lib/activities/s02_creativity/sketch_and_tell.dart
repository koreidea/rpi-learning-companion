import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../../vision/camera_manager.dart';
import '../../vision/vision_describer.dart';
import '../activity_base.dart';

/// Creative drawing activity that uses the camera to see the child's artwork.
///
/// The bot gives a creative prompt, the child draws, and says "done" or "look"
/// when ready. The camera captures the drawing, the vision API describes it,
/// and the bot asks imaginative follow-up questions. If no camera is available
/// the child describes their drawing verbally instead.
///
/// Each session has 3 rounds with different creative prompts.
class SketchAndTell extends Activity {
  final CameraManager? _cameraManager;
  final VisionDescriber? _visionDescriber;
  final String _apiKey;

  final Random _random = Random();

  bool _active = false;
  int _round = 0;
  static const int _maxRounds = 3;
  int _followUp = 0;
  static const int _maxFollowUps = 2;
  _Phase _phase = _Phase.idle;
  String? _currentPrompt;
  String? _drawingDescription;

  final List<int> _usedPromptIndices = [];

  SketchAndTell({
    CameraManager? camera,
    VisionDescriber? visionDescriber,
    String apiKey = '',
  })  : _cameraManager = camera,
        _visionDescriber = visionDescriber,
        _apiKey = apiKey;

  @override
  String get id => 'creativity_sketch_and_tell';

  @override
  String get name => 'Sketch and Tell';

  @override
  String get category => 'creativity';

  @override
  String get description =>
      'Draw something creative and tell me all about it!';

  @override
  List<String> get skills =>
      ['creativity', 'imagination', 'verbal expression', 'visual thinking'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 10;

  @override
  SkillId? get skillId => SkillId.creativity;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'sketch and tell',
          'drawing game',
          'draw something',
          'let me draw',
          'drawing time',
        ],
        'hi': ['चित्र बनाओ', 'ड्राइंग खेल', 'ड्राइंग'],
        'te': ['బొమ్మ గీయి', 'డ్రాయింగ్ ఆట', 'చిత్రం'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _round = 0;
    _followUp = 0;
    _usedPromptIndices.clear();
    _phase = _Phase.givingPrompt;

    return _presentNewPrompt();
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    switch (_phase) {
      case _Phase.idle:
        return "Draw something and say done when you are ready!";

      case _Phase.givingPrompt:
        // Child acknowledged the prompt, waiting for drawing
        _phase = _Phase.waitingForDrawing;
        return "Take your time! Say done or look when your drawing is ready.";

      case _Phase.waitingForDrawing:
        if (_containsDone(lower)) {
          return await _captureDrawing();
        }
        return "Still drawing? Take your time! Say done when you are ready.";

      case _Phase.askingFollowUps:
        return _handleFollowUp(childSaid);

      case _Phase.waitingForNext:
        if (_containsNo(lower)) {
          _active = false;
          return _buildEndSummary();
        }
        return _presentNewPrompt();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_round == 0) return 'No drawings yet.';
    return 'Created $_round ${_round == 1 ? 'drawing' : 'drawings'}.';
  }

  // -- Internal --

  String _presentNewPrompt() {
    // Pick an unused prompt
    int index;
    if (_usedPromptIndices.length >= _prompts.length) {
      _usedPromptIndices.clear();
    }
    do {
      index = _random.nextInt(_prompts.length);
    } while (_usedPromptIndices.contains(index));
    _usedPromptIndices.add(index);

    _currentPrompt = _prompts[index];
    _phase = _Phase.givingPrompt;
    _followUp = 0;

    return "Here is your creative challenge! $_currentPrompt "
        "Take your time drawing and say done when you are ready for me to look!";
  }

  Future<String> _captureDrawing() async {
    // Try camera-based capture
    if (_cameraManager != null && _visionDescriber != null && _apiKey.isNotEmpty) {
      try {
        final cameraReady = await _cameraManager!.init();
        if (cameraReady) {
          final frame = await _cameraManager!.captureFrame();
          if (frame != null && frame.isNotEmpty) {
            _drawingDescription = await _visionDescriber!.describe(
              imageBytes: frame,
              apiKey: _apiKey,
              userPrompt:
                  'A child drew something based on the prompt: "$_currentPrompt". '
                  'Describe what you see in their drawing in a fun, encouraging way. '
                  'Keep it to 1-2 sentences. Be specific about what you see.',
            );

            _phase = _Phase.askingFollowUps;
            return "Wow, let me see! $_drawingDescription "
                "${_followUpQuestions[0]}";
          }
        }
      } catch (e) {
        debugPrint('[SketchAndTell] Camera error: $e');
      }
    }

    // Fallback: ask child to describe verbally
    _phase = _Phase.askingFollowUps;
    _drawingDescription = null;
    return "I wish I could see your drawing! Can you describe it to me? "
        "What did you draw?";
  }

  String _handleFollowUp(String childSaid) {
    _followUp++;

    if (_followUp >= _maxFollowUps) {
      _round++;

      if (_round >= _maxRounds) {
        _active = false;
        final ack = _acknowledgements[_random.nextInt(_acknowledgements.length)];
        return "$ack What an amazing artist you are! ${_buildEndSummary()}";
      }

      _phase = _Phase.waitingForNext;
      final ack = _acknowledgements[_random.nextInt(_acknowledgements.length)];
      return "$ack I love your creativity! Want to draw something else?";
    }

    final ack = _acknowledgements[_random.nextInt(_acknowledgements.length)];
    final question = _followUpQuestions[
        _followUp < _followUpQuestions.length ? _followUp : 0];
    return "$ack $question";
  }

  String _buildEndSummary() {
    if (_round == 0) {
      return "Thanks for playing Sketch and Tell! Come back anytime to draw!";
    }
    return "You created $_round amazing ${_round == 1 ? 'drawing' : 'drawings'}! "
        "You have such a wonderful imagination! Keep drawing and creating!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  bool _containsDone(String text) {
    const doneWords = ['done', 'look', 'ready', 'finished', 'see', 'check'];
    return doneWords.any((w) => text.contains(w));
  }

  bool _containsNo(String text) {
    const noWords = ['no', 'nah', 'nope', "don't want", 'not now'];
    return noWords.any((w) => text.contains(w));
  }

  static const List<String> _prompts = [
    'Draw something that can fly but is not a bird!',
    'Draw the silliest animal you can imagine!',
    'Draw a house where a dragon would live!',
    'Draw what you think lives at the bottom of the ocean!',
    'Draw a magical vehicle that can go anywhere!',
    'Draw your dream playground!',
    'Draw a friendly monster!',
    'Draw what the inside of a cloud looks like!',
    'Draw a machine that makes your favorite food!',
    'Draw an animal wearing a funny hat!',
    'Draw what a rainbow tastes like!',
    'Draw a superhero who saves animals!',
  ];

  static const List<String> _followUpQuestions = [
    'What special powers does it have?',
    'Where does it live? Tell me about its home!',
    'Does it have a name? What is it called?',
    'What is its favorite thing to do?',
  ];

  static const List<String> _acknowledgements = [
    'That is so creative!',
    'Wow, what an imagination you have!',
    'I love that idea!',
    'That is really cool!',
    'How wonderful!',
    'You are such a great artist!',
  ];
}

enum _Phase {
  idle,
  givingPrompt,
  waitingForDrawing,
  askingFollowUps,
  waitingForNext,
}
