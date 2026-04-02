import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Pronunciation Coach: the bot reads a passage, the child repeats it, and
/// the bot compares the STT transcription against the original to give
/// gentle feedback.
///
/// Focuses on clarity, confidence, and expression rather than accent.
/// Passages are grouped by age band. Each session has 3 rounds.
class PronunciationCoach extends Activity {
  final Random _random = Random();

  bool _active = false;
  int _round = 0;
  static const int _maxRounds = 3;
  int _score = 0;
  _Phase _phase = _Phase.idle;
  String? _currentPassage;
  AgeBand _ageBand;

  final List<int> _usedPassageIndices = [];

  PronunciationCoach({AgeBand ageBand = AgeBand.nursery}) : _ageBand = ageBand;

  /// Update the target age band (affects which passages are used).
  set ageBand(AgeBand value) => _ageBand = value;

  @override
  String get id => 'communication_pronunciation_coach';

  @override
  String get name => 'Pronunciation Coach';

  @override
  String get category => 'communication';

  @override
  String get description =>
      'Listen, repeat, and practice speaking clearly and confidently.';

  @override
  List<String> get skills =>
      ['pronunciation', 'confidence', 'listening', 'speaking'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.communication;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'pronunciation coach',
          'practice speaking',
          'speaking practice',
          'say it again',
          'repeat after me',
        ],
        'hi': ['बोलने की प्रैक्टिस', 'उच्चारण', 'मेरे बाद बोलो'],
        'te': ['ఉచ్చారణ ప్రాక్టీస్', 'నా తర్వాత చెప్పు'],
      };

  @override
  AgeBand get targetAgeBand => _ageBand;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _round = 0;
    _score = 0;
    _usedPassageIndices.clear();
    _phase = _Phase.readingPassage;

    return _presentNewPassage();
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
        return "Say ready to hear the next passage!";

      case _Phase.readingPassage:
        // Child heard the passage, now read back
        _phase = _Phase.waitingForRepeat;
        return "Now it is your turn! Repeat what I just said. Take your time!";

      case _Phase.waitingForRepeat:
        return _evaluateRepeat(childSaid);

      case _Phase.waitingForNext:
        if (_containsNo(lower)) {
          _active = false;
          return _buildEndSummary();
        }
        return _presentNewPassage();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_round == 0) return 'No passages practiced yet.';
    return 'Practiced $_round passages. Score: $_score.';
  }

  // -- Internal --

  String _presentNewPassage() {
    final passages = _getPassagesForBand(_ageBand);

    if (_usedPassageIndices.length >= passages.length) {
      _usedPassageIndices.clear();
    }
    int index;
    do {
      index = _random.nextInt(passages.length);
    } while (_usedPassageIndices.contains(index));
    _usedPassageIndices.add(index);

    _currentPassage = passages[index];
    _phase = _Phase.readingPassage;

    return "Listen carefully and then repeat after me. $_currentPassage";
  }

  String _evaluateRepeat(String childSaid) {
    _round++;
    final original = _currentPassage!.toLowerCase().trim();
    final repeated = childSaid.toLowerCase().trim();

    // Simple word-level comparison
    final originalWords = original
        .replaceAll(RegExp(r'[^a-z\s]'), '')
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty)
        .toList();
    final repeatedWords = repeated
        .replaceAll(RegExp(r'[^a-z\s]'), '')
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty)
        .toList();

    int matched = 0;
    for (final word in originalWords) {
      if (repeatedWords.contains(word)) matched++;
    }

    final accuracy = originalWords.isNotEmpty
        ? (matched / originalWords.length * 100).round()
        : 0;

    String feedback;
    if (accuracy >= 80) {
      _score += 3;
      feedback = "Excellent! You said almost every word perfectly! "
          "Your speaking is clear and confident.";
    } else if (accuracy >= 50) {
      _score += 2;
      feedback = "Good job! You got most of the words right. "
          "Try speaking a bit louder and slower next time for even better results.";
    } else {
      _score += 1;
      feedback = "Nice try! Some of the words were tricky. That is totally "
          "okay. The more you practice, the easier it gets. "
          "Try speaking slowly and clearly.";
    }

    if (_round >= _maxRounds) {
      _active = false;
      return "$feedback ${_buildEndSummary()}";
    }

    _phase = _Phase.waitingForNext;
    return "$feedback Want to try another passage?";
  }

  String _buildEndSummary() {
    if (_round == 0) {
      return "Thanks for trying Pronunciation Coach! Come back to practice!";
    }
    return "You practiced $_round ${_round == 1 ? 'passage' : 'passages'} "
        "and scored $_score points! Great speaking practice. "
        "Remember, speaking clearly is a skill that gets better every time you try!";
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

  List<String> _getPassagesForBand(AgeBand band) {
    switch (band) {
      case AgeBand.nursery:
        return _nurseryPassages;
      case AgeBand.junior:
        return _juniorPassages;
      case AgeBand.senior:
        return _seniorPassages;
    }
  }

  static const List<String> _nurseryPassages = [
    'The big red bus goes down the road. Beep beep!',
    'A little cat sat on a mat. The cat is fat and happy.',
    'I see a bird. The bird can fly. Fly, bird, fly!',
    'One fish, two fish. Red fish, blue fish.',
    'The sun is hot. The rain is wet. I like the sun the best.',
    'My dog is big. My dog is brown. My dog likes to run around.',
  ];

  static const List<String> _juniorPassages = [
    'The stars twinkle in the dark sky like tiny diamonds scattered across a velvet blanket.',
    'Butterflies start their lives as caterpillars and then transform into beautiful creatures with colorful wings.',
    'The library is a magical place where you can travel to any world just by opening a book.',
    'Scientists discovered that dolphins can talk to each other using clicks and whistles.',
    'The tallest mountain in the world is Mount Everest. It stands over eight thousand meters high.',
    'Honey bees visit over two million flowers to make just one pound of honey.',
  ];

  static const List<String> _seniorPassages = [
    'In the heart of the Amazon rainforest, scientists discovered a species of frog that glows in the dark.',
    'The human brain contains about one hundred billion neurons, each connecting to thousands of others.',
    'Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into food and oxygen.',
    'The International Space Station orbits Earth at a speed of over twenty-seven thousand kilometers per hour.',
    'Ancient civilizations like the Indus Valley had sophisticated drainage systems thousands of years before modern plumbing.',
    'Artificial intelligence is transforming how we work, learn, and communicate with each other.',
  ];
}

enum _Phase {
  idle,
  readingPassage,
  waitingForRepeat,
  waitingForNext,
}
