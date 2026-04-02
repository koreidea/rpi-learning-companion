import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../activity_types.dart';

/// Story Relay: round-robin collaborative storytelling where each participant
/// adds one sentence at a time.
///
/// The bot starts the story, then each participant (including the bot if
/// solo mode) takes turns adding to the narrative. After each participant
/// has 2-3 turns, the bot wraps up with a fun ending.
class StoryRelay extends GroupActivity {
  final Random _random = Random();

  bool _active = false;
  List<String> _participants = [];
  String? _currentParticipant;
  int _currentParticipantIndex = -1;
  int _totalTurns = 0;
  int _turnsPerParticipant = 0;
  static const int _maxTurnsPerPerson = 3;
  bool _soloMode = false;

  final List<String> _storyParts = [];

  @override
  String get id => 'collaboration_story_relay';

  @override
  String get name => 'Story Relay';

  @override
  String get category => 'collaboration';

  @override
  String get description =>
      'Take turns adding to a story! Round-robin storytelling fun.';

  @override
  List<String> get skills =>
      ['collaboration', 'creativity', 'listening', 'narrative'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 12;

  @override
  SkillId? get skillId => SkillId.collaboration;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'story relay',
          'relay story',
          'group story',
          'take turns story',
          'round robin story',
        ],
        'hi': ['कहानी रिले', 'बारी बारी कहानी', 'समूह कहानी'],
        'te': ['కథ రిలే', 'వంతుల కథ', 'సమూహ కథ'],
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
    debugPrint('[StoryRelay] Participants: $_participants, solo=$_soloMode');
  }

  @override
  Future<void> nextTurn() async {
    if (_participants.isEmpty) return;
    _currentParticipantIndex =
        (_currentParticipantIndex + 1) % _participants.length;
    _currentParticipant = _participants[_currentParticipantIndex];
  }

  @override
  Map<String, String> getPerParticipantFeedback() {
    final feedback = <String, String>{};
    for (final name in _participants) {
      feedback[name] = 'Awesome storytelling, $name! '
          'You added some really creative parts to our story!';
    }
    return feedback;
  }

  @override
  Future<String> start() async {
    _active = true;
    _totalTurns = 0;
    _turnsPerParticipant = 0;
    _storyParts.clear();
    _currentParticipantIndex = -1;

    if (_participants.isEmpty) {
      _participants = ['You'];
      _soloMode = true;
    }

    // Pick a story opening
    final opening = _openings[_random.nextInt(_openings.length)];
    _storyParts.add(opening);

    // Set first participant
    await nextTurn();

    debugPrint('[StoryRelay] Starting with ${_participants.length} participants');

    if (_soloMode) {
      return "Let's create a story together, taking turns! I will start. "
          "$opening Now it is your turn! Add one sentence to continue the story.";
    }

    return "Let's create a story together, taking turns! I will start. "
        "$opening $_currentParticipant, you are up first! "
        "Add one sentence to continue the story.";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    // Record the participant's contribution
    _storyParts.add(childSaid);
    _totalTurns++;

    // In solo mode, bot alternates with the child
    if (_soloMode) {
      _turnsPerParticipant++;
      if (_turnsPerParticipant >= _maxTurnsPerPerson) {
        _active = false;
        return _wrapUpStory();
      }

      // Bot adds a sentence
      final botAddition = _generateBotSentence();
      _storyParts.add(botAddition);
      return "$botAddition Your turn again! What happens next?";
    }

    // Multi-player mode
    await nextTurn();

    // Check if everyone has had enough turns
    final turnsNeeded = _participants.length * _maxTurnsPerPerson;
    if (_totalTurns >= turnsNeeded) {
      _active = false;
      return _wrapUpStory();
    }

    final ack = _acknowledgements[_random.nextInt(_acknowledgements.length)];
    return "$ack $_currentParticipant, your turn! Continue the story!";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_totalTurns == 0) return 'Story relay not started yet.';
    return '$_totalTurns turns taken in the story.';
  }

  // -- Internal --

  String _generateBotSentence() {
    const additions = [
      'And then something magical happened right before their eyes!',
      'Suddenly, they heard a mysterious sound coming from behind the trees.',
      'A friendly creature appeared and waved hello to everyone.',
      'The sky turned the most beautiful shade of purple.',
      'They found a secret door that nobody had noticed before.',
      'A gentle breeze carried the sweetest music they had ever heard.',
      'And to everyone\'s surprise, flowers started blooming everywhere.',
    ];
    return additions[_random.nextInt(additions.length)];
  }

  String _wrapUpStory() {
    const endings = [
      'And from that day on, they remembered the adventure with big smiles.',
      'And so, the most wonderful day came to a happy end.',
      'And they lived happily ever after, always ready for the next adventure!',
    ];
    final ending = endings[_random.nextInt(endings.length)];
    _storyParts.add(ending);

    return "$ending What an incredible story we created together! "
        "${_buildEndSummary()}";
  }

  String _buildEndSummary() {
    if (_totalTurns == 0) {
      return "Thanks for joining Story Relay! Come back to tell stories together!";
    }
    return "We built a story with $_totalTurns turns! "
        "Every person added their own creative touch. "
        "That is what makes teamwork so special!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<String> _openings = [
    'Once upon a time, in a magical forest, a little fox found a glowing stone.',
    'Long ago, on a floating island above the clouds, there lived a tiny dragon.',
    'One rainy afternoon, two friends discovered a door in the back of their cupboard.',
    'In a village where animals could talk, the oldest owl called a meeting.',
    'Deep under the ocean, a young mermaid found a bottle with a message inside.',
    'On the tallest mountain, a brave explorer saw a rainbow touching the ground.',
  ];

  static const List<String> _acknowledgements = [
    'I love that!',
    'What a great addition!',
    'Oh, how exciting!',
    'That was wonderful!',
    'Brilliant!',
  ];
}
