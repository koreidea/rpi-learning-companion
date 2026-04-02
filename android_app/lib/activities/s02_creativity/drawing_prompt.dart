import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../vision/camera_manager.dart';
import '../../vision/vision_describer.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Drawing prompt with optional visual feedback from the camera.
class _DrawingChallenge {
  final String prompt;
  final String encouragement;

  const _DrawingChallenge({required this.prompt, required this.encouragement});
}

/// Drawing Challenge: bot gives a drawing prompt, child draws and optionally
/// shows their work via camera for feedback.
///
/// Teaches: creativity, fine motor skills, visual expression, confidence.
///
/// Flow:
/// 1. Bot gives a drawing prompt.
/// 2. Child draws (takes their time).
/// 3. Child says "done" / "look" / "finished" etc.
/// 4. If camera available, captures and describes the drawing.
/// 5. Bot gives enthusiastic feedback.
class DrawingPrompt extends Activity {
  final CameraManager? _camera;
  final VisionDescriber? _visionDescriber;
  final String _apiKey;
  final Random _rng = Random();

  bool _active = false;
  int _drawingsCompleted = 0;
  int _score = 0;
  bool _waitingForDrawing = false;
  String _currentPrompt = '';

  static const List<_DrawingChallenge> _challenges = [
    _DrawingChallenge(
      prompt: "Can you draw a house with a big tree and a bright sun? Take your time!",
      encouragement: "I love houses and trees!",
    ),
    _DrawingChallenge(
      prompt: "Can you draw your family? Show me everyone! Take your time!",
      encouragement: "Family drawings are the best!",
    ),
    _DrawingChallenge(
      prompt: "Can you draw your favorite animal? Make it big and colorful! Take your time!",
      encouragement: "I love animals!",
    ),
    _DrawingChallenge(
      prompt: "Can you draw a beautiful rainbow with lots of colors? Take your time!",
      encouragement: "Rainbows are so pretty!",
    ),
    _DrawingChallenge(
      prompt: "Can you draw a flower garden with butterflies? Take your time!",
      encouragement: "Gardens are so beautiful!",
    ),
    _DrawingChallenge(
      prompt: "Can you draw a spaceship flying to the moon? Take your time!",
      encouragement: "Space adventures are so exciting!",
    ),
    _DrawingChallenge(
      prompt: "Can you draw a fish swimming in the ocean? Take your time!",
      encouragement: "I love the ocean!",
    ),
    _DrawingChallenge(
      prompt: "Can you draw a birthday cake with candles? Take your time!",
      encouragement: "Birthday cakes are so yummy!",
    ),
  ];

  DrawingPrompt({
    CameraManager? camera,
    VisionDescriber? visionDescriber,
    String apiKey = '',
  })  : _camera = camera,
        _visionDescriber = visionDescriber,
        _apiKey = apiKey;

  // -- Activity metadata --

  @override
  String get id => 'creativity_drawing';

  @override
  String get name => 'Drawing Challenge';

  @override
  String get category => 'creativity';

  @override
  String get description =>
      'Get a fun drawing prompt and show your artwork to the bot.';

  @override
  List<String> get skills => ['creativity', 'fine motor skills', 'visual expression'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  SkillId? get skillId => SkillId.creativity;

  @override
  Map<String, List<String>> get voiceTriggers => const {
    'en': ['drawing game', 'let us draw', 'draw something', 'drawing time', 'drawing challenge'],
    'hi': ['ड्राइंग खेल', 'चित्र बनाओ', 'ड्रॉ करो'],
    'te': ['బొమ్మ వేయి', 'డ్రాయింగ్ ఆట', 'బొమ్మ ఆట'],
  };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    return '$_drawingsCompleted drawings completed. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _drawingsCompleted = 0;
    _score = 0;
    _active = true;
    _waitingForDrawing = true;

    final challenge = _challenges[_rng.nextInt(_challenges.length)];
    _currentPrompt = challenge.prompt;
    debugPrint('[DrawingPrompt] Started with prompt: $_currentPrompt');

    return "Drawing time! ${challenge.encouragement} ${challenge.prompt} "
        "Tell me when you are done!";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    if (_waitingForDrawing) {
      // Check if child says they're done
      if (_isDoneTrigger(lower)) {
        return await _handleDrawingDone();
      }

      // Child is still drawing — encourage them
      return "Take your time! I am excited to see your drawing. Tell me when you are done!";
    }

    // After viewing a drawing, ask if they want another
    if (_wantsAnother(lower)) {
      return _nextChallenge();
    }

    // Default: wrap up
    return await end();
  }

  @override
  Future<String> end() async {
    _active = false;
    _waitingForDrawing = false;
    debugPrint('[DrawingPrompt] Ended, drawings=$_drawingsCompleted, score=$_score');

    if (_drawingsCompleted == 0) {
      return "Okay, we'll draw another time! I bet you're an amazing artist!";
    }

    return "You made $_drawingsCompleted beautiful drawing${_drawingsCompleted > 1 ? 's' : ''}! "
        "You are such a talented artist! Keep drawing every day!";
  }

  // -- Drawing completion --

  Future<String> _handleDrawingDone() async {
    _drawingsCompleted++;
    _score += 15;
    _waitingForDrawing = false;

    // Try to use camera + vision to describe the drawing
    final cam = _camera;
    final vision = _visionDescriber;
    if (cam != null && vision != null && _apiKey.isNotEmpty) {
      try {
        final cameraReady = await cam.init();
        if (cameraReady) {
          final frame = await cam.captureFrame();
          if (frame != null && frame.isNotEmpty) {
            final description = await vision.describe(
              imageBytes: frame,
              apiKey: _apiKey,
              userPrompt: 'The child just drew a picture. '
                  'The drawing prompt was: "$_currentPrompt". '
                  'Describe what you see in their drawing in a very encouraging way. '
                  'Even if the drawing is messy, find something wonderful about it.',
            );
            _score += 5; // Bonus for showing the drawing

            return "$description Would you like to draw something else?";
          }
        }
      } catch (e) {
        debugPrint('[DrawingPrompt] Vision error: $e');
      }
    }

    // Fallback without camera
    return "I bet your drawing is beautiful! I love it! "
        "Would you like to draw something else?";
  }

  String _nextChallenge() {
    _waitingForDrawing = true;
    final challenge = _challenges[_rng.nextInt(_challenges.length)];
    _currentPrompt = challenge.prompt;
    return "Great! Here is your next drawing challenge. ${challenge.encouragement} "
        "${challenge.prompt} Tell me when you are done!";
  }

  // -- Trigger helpers --

  bool _isDoneTrigger(String lower) {
    const triggers = [
      'done', "i'm done", 'finished', 'look', 'i did it',
      'ready', 'see', 'look at this', 'i drew it', 'all done',
      'show you', 'here it is', 'ta da',
      // Hindi
      'हो गया', 'देखो', 'बन गया',
      // Telugu
      'అయిపోయింది', 'చూడు', 'రెడీ',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  bool _wantsAnother(String lower) {
    const triggers = [
      'yes', 'yeah', 'sure', 'okay', 'another', 'more',
      'one more', 'again', 'next',
      // Hindi
      'हाँ', 'और', 'एक और',
      // Telugu
      'అవును', 'ఇంకొకటి', 'మరొకటి',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop the game', 'quit', 'exit', "i'm done playing", 'no more',
      'stop playing', 'end the game', 'finish', 'no', 'no thanks',
      'बंद करो', 'खेल बंद', 'नहीं',
      'ఆపు', 'ఆట ఆపు', 'వద్దు',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
