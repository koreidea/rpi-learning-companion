import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../../vision/camera_manager.dart';
import '../../vision/vision_describer.dart';
import '../activity_base.dart';

/// Nature Narrator: point the camera at nature (plant, bug, sky, rock) and
/// hear a mini-story or fun fact about it.
///
/// Uses the vision API to identify natural objects, then tells an interesting
/// fact or mini-story. Falls back to built-in facts for common items if
/// camera is unavailable. 3 objects per session.
class NatureNarrator extends Activity {
  final CameraManager? _cameraManager;
  final VisionDescriber? _visionDescriber;
  final String _apiKey;
  final Random _random = Random();

  bool _active = false;
  int _objectsExplored = 0;
  static const int _maxObjects = 3;
  _Phase _phase = _Phase.idle;
  String? _identifiedObject;

  NatureNarrator({
    CameraManager? camera,
    VisionDescriber? visionDescriber,
    String apiKey = '',
  })  : _cameraManager = camera,
        _visionDescriber = visionDescriber,
        _apiKey = apiKey;

  @override
  String get id => 'environmental_nature_narrator';

  @override
  String get name => 'Nature Narrator';

  @override
  String get category => 'environmental';

  @override
  String get description =>
      'Point the camera at nature and hear amazing facts and stories!';

  @override
  List<String> get skills =>
      ['environmental awareness', 'curiosity', 'observation'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 12;

  @override
  SkillId? get skillId => SkillId.environmental;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'nature narrator',
          'what is this plant',
          'nature facts',
          'tell me about nature',
          'nature game',
        ],
        'hi': ['प्रकृति', 'ये क्या है', 'पेड़ पौधे', 'प्रकृति की कहानी'],
        'te': ['ప్రకృతి', 'ఇది ఏమిటి', 'చెట్లు పువ్వులు'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _objectsExplored = 0;
    _phase = _Phase.waitingForObject;

    return "Welcome to Nature Narrator! Point the camera at something from "
        "nature, like a plant, a tree, a flower, or even a bug! "
        "Or just tell me what you see. Say ready when you are looking at something!";
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
        _phase = _Phase.waitingForObject;
        return "Point at something from nature and say ready!";

      case _Phase.waitingForObject:
        return await _identifyObject(childSaid);

      case _Phase.telling:
        _objectsExplored++;
        if (_objectsExplored >= _maxObjects) {
          _active = false;
          return "Wonderful question! ${_buildEndSummary()}";
        }
        _phase = _Phase.waitingForNext;
        return "Great observation! Want to explore another natural object?";

      case _Phase.waitingForNext:
        if (_containsNo(lower)) {
          _active = false;
          return _buildEndSummary();
        }
        _phase = _Phase.waitingForObject;
        return "Point the camera at something else from nature and say ready!";
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    return 'Explored $_objectsExplored natural objects.';
  }

  // -- Internal --

  Future<String> _identifyObject(String childSaid) async {
    // Try camera-based identification
    if (_cameraManager != null &&
        _visionDescriber != null &&
        _apiKey.isNotEmpty) {
      try {
        final cameraReady = await _cameraManager!.init();
        if (cameraReady) {
          final frame = await _cameraManager!.captureFrame();
          if (frame != null && frame.isNotEmpty) {
            final description = await _visionDescriber!.describe(
              imageBytes: frame,
              apiKey: _apiKey,
              userPrompt:
                  'Identify the natural object (plant, tree, flower, insect, '
                  'rock, sky, cloud, etc.) that the child is pointing at. '
                  'Then tell a fun, interesting fact or mini-story about it '
                  'in 3-4 sentences. Make it exciting for a child. '
                  'If it is an Indian species, mention its local significance. '
                  'Do not use emojis or markdown.',
            );

            _identifiedObject = description;
            _phase = _Phase.telling;
            return "$description Do you want to know anything else about it?";
          }
        }
      } catch (e) {
        debugPrint('[NatureNarrator] Camera error: $e');
      }
    }

    // Fallback: check child's verbal description
    _phase = _Phase.telling;
    final fact = _findFactForDescription(childSaid.toLowerCase());
    _identifiedObject = childSaid;
    return "$fact Do you want to know anything else about it?";
  }

  String _findFactForDescription(String description) {
    for (final entry in _builtInFacts.entries) {
      for (final keyword in entry.value.keywords) {
        if (description.contains(keyword)) {
          return entry.value.fact;
        }
      }
    }

    return "That is really interesting! Nature is full of amazing things. "
        "Did you know that there are over 8 million species of living things "
        "on our planet? Every single one has a special job in nature!";
  }

  String _buildEndSummary() {
    if (_objectsExplored == 0) {
      return "Thanks for trying Nature Narrator! Come back to explore the natural world!";
    }
    return "You explored $_objectsExplored natural "
        "${_objectsExplored == 1 ? 'object' : 'objects'}! "
        "Nature is all around us, full of stories and surprises. "
        "Keep being curious about the world!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  bool _containsNo(String text) {
    const noWords = ['no', 'nah', 'nope', "don't want", 'not now'];
    return noWords.any((w) => text.contains(w));
  }

  static final Map<String, _NatureFact> _builtInFacts = {
    'neem': _NatureFact(
      keywords: ['neem'],
      fact:
          'That looks like a neem tree! Ancient Indians called it the village '
          'pharmacy. Every part of the neem tree, the leaves, bark, and seeds, '
          'has been used for medicine for thousands of years. Even today, neem '
          'is used in toothpaste and skin care!',
    ),
    'banyan': _NatureFact(
      keywords: ['banyan', 'big tree', 'hanging roots'],
      fact:
          'That is a banyan tree! It is the national tree of India. Banyan '
          'trees grow roots from their branches that hang down and become new '
          'trunks. One banyan tree in Kolkata has over 3,000 aerial roots and '
          'looks like a whole forest!',
    ),
    'ant': _NatureFact(
      keywords: ['ant', 'ants'],
      fact:
          'Those are ants! Did you know that ants can carry objects 50 times '
          'their own body weight? That is like you carrying a car! Ants live '
          'in colonies that can have millions of members, and they communicate '
          'by leaving a chemical trail.',
    ),
    'butterfly': _NatureFact(
      keywords: ['butterfly', 'butterflies'],
      fact:
          'Beautiful butterfly! A butterfly starts its life as a tiny egg, '
          'becomes a caterpillar, then wraps itself in a chrysalis, and finally '
          'emerges as a butterfly. This amazing transformation is called '
          'metamorphosis. Some butterflies migrate thousands of kilometers!',
    ),
    'flower': _NatureFact(
      keywords: ['flower', 'rose', 'marigold', 'jasmine', 'lotus'],
      fact:
          'What a lovely flower! Flowers are not just pretty. They help plants '
          'make seeds by attracting bees and butterflies with their colors and '
          'smell. The lotus is India\'s national flower and can grow in muddy '
          'water but still bloom beautifully!',
    ),
    'cloud': _NatureFact(
      keywords: ['cloud', 'clouds', 'sky'],
      fact:
          'Look at those clouds! Clouds are made of billions of tiny water '
          'droplets floating in the air. A single fluffy cloud can weigh as '
          'much as 100 elephants! Different shapes of clouds tell us different '
          'things about the weather.',
    ),
    'rock': _NatureFact(
      keywords: ['rock', 'stone', 'pebble'],
      fact:
          'Interesting rock! Some rocks are millions of years old. Rocks are '
          'made of minerals, and there are three main types: igneous from '
          'volcanoes, sedimentary from layers of sand and mud, and metamorphic '
          'which are changed by heat and pressure deep underground!',
    ),
    'grass': _NatureFact(
      keywords: ['grass', 'lawn', 'green'],
      fact:
          'That is grass! Grass is one of the most important plants on Earth. '
          'It prevents soil from washing away, provides food for many animals, '
          'and produces oxygen for us to breathe. There are over 12,000 '
          'different species of grass in the world!',
    ),
  };
}

enum _Phase {
  idle,
  waitingForObject,
  telling,
  waitingForNext,
}

/// A built-in nature fact with keyword matching.
class _NatureFact {
  final List<String> keywords;
  final String fact;

  const _NatureFact({required this.keywords, required this.fact});
}
