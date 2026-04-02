import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../activity_types.dart';

/// Compliment Circle: an appreciation round after group activities.
///
/// Each participant says one thing they liked about another participant's
/// contribution. The bot models the behavior first, then invites each
/// participant. If only one child is present, the bot asks them to share
/// something they are proud of.
class ComplimentCircle extends GroupActivity {
  final Random _random = Random();

  bool _active = false;
  List<String> _participants = [];
  String? _currentParticipant;
  int _currentIndex = -1;
  int _complimentsGiven = 0;
  bool _soloMode = false;
  bool _botModeled = false;

  @override
  String get id => 'collaboration_compliment_circle';

  @override
  String get name => 'Compliment Circle';

  @override
  String get category => 'collaboration';

  @override
  String get description =>
      'Share what you liked about each other after working together!';

  @override
  List<String> get skills =>
      ['appreciation', 'empathy', 'communication', 'teamwork'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 12;

  @override
  SkillId? get skillId => SkillId.collaboration;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'compliment circle',
          'say something nice',
          'appreciation round',
          'compliment game',
        ],
        'hi': ['तारीफ', 'कुछ अच्छा बोलो', 'सराहना'],
        'te': ['మెచ్చుకో', 'ఏదైనా మంచి చెప్పు'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  List<String> get participants => List.unmodifiable(_participants);

  @override
  String? get currentParticipant => _currentParticipant;

  @override
  Future<void> setParticipants(List<String> names) async {
    _participants = List.from(names);
    _soloMode = _participants.length < 2;
    debugPrint(
        '[ComplimentCircle] Participants: $_participants, solo=$_soloMode');
  }

  @override
  Future<void> nextTurn() async {
    if (_participants.isEmpty) return;
    _currentIndex = (_currentIndex + 1) % _participants.length;
    _currentParticipant = _participants[_currentIndex];
  }

  @override
  Map<String, String> getPerParticipantFeedback() {
    final feedback = <String, String>{};
    for (final name in _participants) {
      feedback[name] = 'Thank you for sharing kind words, $name!';
    }
    return feedback;
  }

  @override
  Future<String> start() async {
    _active = true;
    _complimentsGiven = 0;
    _botModeled = false;
    _currentIndex = -1;

    if (_participants.isEmpty) {
      _participants = ['You'];
      _soloMode = true;
    }

    debugPrint('[ComplimentCircle] Starting, solo=$_soloMode');

    if (_soloMode) {
      _botModeled = true;
      _currentParticipant = _participants[0];
      return "Let's do a compliment circle! I will go first. "
          "I really liked how you tried your best and shared creative ideas today! "
          "Now your turn. Tell me one thing you are proud of from today!";
    }

    // Multi-player: bot models first
    _botModeled = true;
    await nextTurn();
    final firstPerson = _currentParticipant ?? _participants[0];
    return "Let's do a compliment circle! I will go first. "
        "I really liked how $firstPerson came up with such creative ideas! "
        "Now, ${_getNextComplimenter()}, say one thing you liked about "
        "${_getComplimentTarget()}'s contribution!";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    _complimentsGiven++;

    if (_soloMode) {
      _active = false;
      return "That is wonderful! Being proud of yourself is important. "
          "Remember, you are amazing just the way you are! ${_buildEndSummary()}";
    }

    // Move to next person
    await nextTurn();

    if (_complimentsGiven >= _participants.length) {
      _active = false;
      return "Beautiful! Everyone shared something kind. "
          "${_buildEndSummary()}";
    }

    return "${_acknowledgements[_random.nextInt(_acknowledgements.length)]} "
        "Now, $_currentParticipant, say one thing you liked about "
        "${_getComplimentTarget()}'s contribution!";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    return '$_complimentsGiven compliments shared!';
  }

  // -- Internal --

  String _getNextComplimenter() {
    if (_participants.length < 2) return _participants.isNotEmpty ? _participants[0] : 'Friend';
    final nextIdx = (_currentIndex + 1) % _participants.length;
    return _participants[nextIdx];
  }

  String _getComplimentTarget() {
    if (_participants.length < 2) return 'yourself';
    // The person being complimented is someone other than the current person
    final targetIdx = (_currentIndex) % _participants.length;
    return _participants[targetIdx];
  }

  String _buildEndSummary() {
    return "What a wonderful compliment circle! Saying nice things about "
        "others makes everyone feel good, including you! "
        "Keep spreading kindness!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<String> _acknowledgements = [
    'That was such a kind thing to say!',
    'How lovely! That made everyone smile!',
    'What a beautiful compliment!',
    'That is so thoughtful!',
    'Wonderful words!',
  ];
}
