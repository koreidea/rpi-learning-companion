import 'dart:async';

import 'package:flutter/foundation.dart';

import '../../core/llm/llm_router.dart';
import '../../vision/camera_manager.dart';
import '../../vision/vision_describer.dart';
import '../activity_base.dart';

/// Fun science fact for a category of natural objects.
class _NatureFact {
  final String category;
  final List<String> keywords;
  final String fact;
  final String bigWord;
  final String bigWordPronunciation;

  const _NatureFact({
    required this.category,
    required this.keywords,
    required this.fact,
    required this.bigWord,
    required this.bigWordPronunciation,
  });
}

/// Nature Explorer: use the camera to identify objects and teach science facts.
///
/// Teaches: scientific observation, vocabulary, curiosity, nature appreciation.
///
/// Flow:
/// 1. Bot asks child to show something from nature.
/// 2. Camera captures, vision API identifies it.
/// 3. Bot shares a fun science fact about what it sees.
/// 4. Bot teaches a big science word.
/// 5. Repeat 3-4 times.
class NatureExplorer extends Activity {
  final CameraManager? _camera;
  final VisionDescriber? _visionDescriber;
  final LlmRouter _llmRouter;
  final String _apiKey;

  bool _active = false;
  int _observationsCompleted = 0;
  int _score = 0;
  bool _waitingForShow = true;

  static const List<_NatureFact> _facts = [
    _NatureFact(
      category: 'leaf',
      keywords: ['leaf', 'leaves', 'plant', 'green'],
      fact: 'Leaves are like tiny food factories! They use sunlight to make food for the tree. '
          'And guess what? They also make the air we breathe!',
      bigWord: 'photosynthesis',
      bigWordPronunciation: 'photo-sin-the-sis',
    ),
    _NatureFact(
      category: 'flower',
      keywords: ['flower', 'petal', 'rose', 'sunflower', 'daisy', 'blossom'],
      fact: 'Flowers have bright colors and sweet smells to attract bees and butterflies. '
          'The bees help the flower make seeds so new flowers can grow!',
      bigWord: 'pollination',
      bigWordPronunciation: 'pollin-ay-shun',
    ),
    _NatureFact(
      category: 'rock',
      keywords: ['rock', 'stone', 'pebble', 'mineral'],
      fact: 'Some rocks are millions and millions of years old! '
          'That is older than the dinosaurs! Rocks can be smooth from water or rough from volcanoes.',
      bigWord: 'geology',
      bigWordPronunciation: 'jee-ol-oh-jee',
    ),
    _NatureFact(
      category: 'water',
      keywords: ['water', 'drop', 'rain', 'puddle', 'stream', 'river', 'lake'],
      fact: 'Water is amazing! It can be a liquid in your glass, ice in the freezer, '
          'or steam from a hot pot. It keeps changing form!',
      bigWord: 'evaporation',
      bigWordPronunciation: 'ee-vap-or-ay-shun',
    ),
    _NatureFact(
      category: 'soil',
      keywords: ['soil', 'dirt', 'mud', 'ground', 'earth', 'worm'],
      fact: 'Soil is not just dirt! It is full of tiny creatures. A handful of soil has more '
          'living things in it than all the people on Earth! Plants need soil to grow.',
      bigWord: 'ecosystem',
      bigWordPronunciation: 'ee-co-sis-tem',
    ),
    _NatureFact(
      category: 'sky',
      keywords: ['sky', 'cloud', 'sun', 'blue', 'weather'],
      fact: 'The sky looks blue because sunlight bounces off tiny pieces of air! '
          'Clouds are made of millions of teeny tiny water drops floating together.',
      bigWord: 'atmosphere',
      bigWordPronunciation: 'at-mos-fear',
    ),
    _NatureFact(
      category: 'tree',
      keywords: ['tree', 'bark', 'trunk', 'branch', 'wood'],
      fact: 'Trees can live for hundreds of years! You can count a tree\'s age by counting '
          'the rings inside its trunk. Trees give us shade, fruit, and clean air.',
      bigWord: 'dendrology',
      bigWordPronunciation: 'den-drol-oh-jee',
    ),
    _NatureFact(
      category: 'insect',
      keywords: ['bug', 'insect', 'ant', 'beetle', 'butterfly', 'caterpillar', 'spider', 'bee'],
      fact: 'Insects are the most common animals on Earth! Ants can carry things that are '
          '50 times heavier than themselves. That is like you carrying a car!',
      bigWord: 'entomology',
      bigWordPronunciation: 'en-toe-mol-oh-jee',
    ),
  ];

  NatureExplorer({
    CameraManager? camera,
    VisionDescriber? visionDescriber,
    required LlmRouter llmRouter,
    String apiKey = '',
  })  : _camera = camera,
        _visionDescriber = visionDescriber,
        _llmRouter = llmRouter,
        _apiKey = apiKey;

  // -- Activity metadata --

  @override
  String get id => 'science_nature';

  @override
  String get name => 'Nature Explorer';

  @override
  String get category => 'science';

  @override
  String get description =>
      'Show things from nature to the camera and learn cool science facts.';

  @override
  List<String> get skills => ['scientific observation', 'vocabulary', 'curiosity'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    return '$_observationsCompleted observations made. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _observationsCompleted = 0;
    _score = 0;
    _waitingForShow = true;
    _active = true;

    debugPrint('[NatureExplorer] Started');

    return "Let's be nature scientists today! Can you show me something from nature? "
        "A leaf, a flower, a rock, anything from the world around you! "
        "Hold it up and say look or show me!";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    if (_waitingForShow) {
      // Check if child wants to show something
      if (_isShowTrigger(lower)) {
        return await _handleObservation(childSaid);
      }

      // Check if child describes what they have (no camera needed)
      final fact = _matchFactFromText(lower);
      if (fact != null) {
        return _presentFact(fact, null);
      }

      return "Hold up what you want to show me and say look! "
          "Or just tell me what you found!";
    }

    // After an observation, ask if they want to find more
    if (_wantsAnother(lower)) {
      _waitingForShow = true;
      return "Great! Find something else from nature and show me!";
    }

    return await end();
  }

  @override
  Future<String> end() async {
    _active = false;
    debugPrint('[NatureExplorer] Ended, observations=$_observationsCompleted, score=$_score');

    if (_observationsCompleted == 0) {
      return "Okay, we'll explore nature another time! "
          "There are so many amazing things to discover!";
    }

    return "What a great nature scientist you are! You observed $_observationsCompleted "
        "thing${_observationsCompleted > 1 ? 's' : ''} and learned new science words! "
        "Score: $_score points!";
  }

  // -- Observation handling --

  Future<String> _handleObservation(String childSaid) async {
    _waitingForShow = false;

    // Try camera + vision
    String? visionDescription;
    final cam = _camera;
    final vision = _visionDescriber;
    if (cam != null && vision != null && _apiKey.isNotEmpty) {
      try {
        final cameraReady = await cam.init();
        if (cameraReady) {
          final frame = await cam.captureFrame();
          if (frame != null && frame.isNotEmpty) {
            visionDescription = await vision.describe(
              imageBytes: frame,
              apiKey: _apiKey,
              userPrompt: 'The child is showing me something from nature. '
                  'What natural object do you see? Name it simply.',
            );
          }
        }
      } catch (e) {
        debugPrint('[NatureExplorer] Vision error: $e');
      }
    }

    // Try to match a fact
    final searchText = '${childSaid.toLowerCase()} ${visionDescription?.toLowerCase() ?? ''}';
    final fact = _matchFactFromText(searchText);

    if (fact != null) {
      return _presentFact(fact, visionDescription);
    }

    // Use LLM for unknown objects
    if (visionDescription != null && visionDescription.isNotEmpty) {
      return await _generateFactWithLlm(visionDescription);
    }

    // Fallback
    _observationsCompleted++;
    _score += 10;
    return "That is very interesting! Nature is full of amazing things. "
        "Would you like to show me something else?";
  }

  String _presentFact(_NatureFact fact, String? visionDescription) {
    _observationsCompleted++;
    _score += 15;
    _waitingForShow = false;

    final intro = visionDescription != null
        ? "I can see it! "
        : "Oh, a ${fact.category}! ";

    return "$intro${fact.fact} "
        "The science word for this is ${fact.bigWord}. "
        "Can you say ${fact.bigWordPronunciation}? "
        "Would you like to show me something else?";
  }

  _NatureFact? _matchFactFromText(String text) {
    for (final fact in _facts) {
      for (final keyword in fact.keywords) {
        if (text.contains(keyword)) return fact;
      }
    }
    return null;
  }

  Future<String> _generateFactWithLlm(String visionDescription) async {
    _observationsCompleted++;
    _score += 15;

    try {
      final provider = _llmRouter.getProvider();
      final instruction =
          'You are a nature scientist talking to a 3-6 year old child. '
          'The child showed you something and the camera sees: "$visionDescription". '
          'Share a fun science fact about what you see in 2-3 simple sentences. '
          'Teach one big science word and spell it out phonetically. '
          'Be enthusiastic! Do not use emojis or markdown.';

      final messages = [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': 'Tell me about what you see'},
      ];

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }

      final result = buffer.toString().trim();
      if (result.isNotEmpty) {
        return "$result Would you like to show me something else?";
      }
    } catch (e) {
      debugPrint('[NatureExplorer] LLM error: $e');
    }

    return "That is very cool! Nature is full of wonders. "
        "Would you like to show me something else?";
  }

  // -- Trigger helpers --

  bool _isShowTrigger(String lower) {
    const triggers = [
      'look', 'see', 'show', 'here', 'found', 'got',
      'this', 'check', 'watch',
      'देखो', 'यह', 'दिखाओ',
      'చూడు', 'ఇది', 'చూపిస్తాను',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  bool _wantsAnother(String lower) {
    const triggers = [
      'yes', 'yeah', 'sure', 'okay', 'another', 'more',
      'one more', 'again', 'next',
      'हाँ', 'और',
      'అవును', 'ఇంకొకటి',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop the game', 'quit', 'exit', "i'm done", 'no more',
      'stop playing', 'end the game', 'finish', 'no', 'no thanks',
      'बंद करो', 'खेल बंद', 'नहीं',
      'ఆపు', 'ఆట ఆపు', 'వద్దు',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
