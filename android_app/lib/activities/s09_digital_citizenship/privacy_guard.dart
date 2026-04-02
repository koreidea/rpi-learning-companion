import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Privacy Guard: the bot role-plays as a friendly stranger online trying to
/// extract personal information, teaching children what NOT to share.
///
/// After each scenario, the bot reveals what it was doing and explains the
/// safety lesson. Tracks how many requests the child correctly refused.
/// 5 scenarios per session, escalating in subtlety.
class PrivacyGuard extends Activity {
  final Random _random = Random();

  bool _active = false;
  int _round = 0;
  static const int _maxRounds = 5;
  int _correctRefusals = 0;
  _Phase _phase = _Phase.idle;

  @override
  String get id => 'digital_citizenship_privacy_guard';

  @override
  String get name => 'Privacy Guard';

  @override
  String get category => 'digital_citizenship';

  @override
  String get description =>
      'Practice saying NO to sharing personal info online!';

  @override
  List<String> get skills =>
      ['digital citizenship', 'online safety', 'critical thinking'];

  @override
  int get minAge => 6;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.digitalCitizenship;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'privacy guard',
          'online safety',
          'stranger danger',
          'privacy game',
          'stay safe online',
        ],
        'hi': ['प्राइवेसी गार्ड', 'ऑनलाइन सुरक्षा', 'सेफ्टी खेल'],
        'te': ['ప్రైవసీ గార్డ్', 'ఆన్లైన్ సేఫ్టీ'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _round = 0;
    _correctRefusals = 0;
    _phase = _Phase.presenting;

    debugPrint('[PrivacyGuard] Started');

    return "Let's play Privacy Guard! I am going to pretend to be a stranger "
        "on the internet. Your job is to protect your personal information. "
        "If I ask for something personal, say NO or refuse! "
        "Ready? Here we go. ${_scenarios[0].request}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    if (_phase == _Phase.presenting) {
      return _evaluateResponse(lower);
    }

    if (_phase == _Phase.explained) {
      _round++;
      if (_round >= _maxRounds) {
        _active = false;
        return _buildEndSummary();
      }
      _phase = _Phase.presenting;
      return "Okay, next scenario! ${_scenarios[_round].request}";
    }

    return "Remember, say no if someone asks for personal info!";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_round == 0) return 'No scenarios completed yet.';
    return 'Completed $_round scenarios. Protected info $_correctRefusals times.';
  }

  // -- Internal --

  String _evaluateResponse(String lower) {
    final scenario = _scenarios[_round];
    _phase = _Phase.explained;

    // Check if child refused to share
    final refused = _containsRefusal(lower);
    final shared = _containsSharing(lower, scenario);

    if (refused && !shared) {
      _correctRefusals++;
      return "Excellent! You said no! ${scenario.lesson} "
          "Score: $_correctRefusals protected out of ${_round + 1}. "
          "Ready for the next one?";
    }

    if (shared) {
      return "Careful! ${scenario.reveal} ${scenario.lesson} "
          "Remember, NEVER share personal information with strangers online. "
          "Score: $_correctRefusals protected out of ${_round + 1}. "
          "Let's try the next one!";
    }

    // Ambiguous response
    return "Hmm, would you share that information or say no? "
        "Remember, this is a stranger you do not know asking for personal details.";
  }

  bool _containsRefusal(String lower) {
    const refusals = [
      'no', 'nope', 'nah', "don't", "won't", "can't", "will not",
      'not telling', 'not sharing', 'refuse', 'never', "i don't know you",
      'stranger', 'not safe', 'private', 'secret',
    ];
    return refusals.any((w) => lower.contains(w));
  }

  bool _containsSharing(String lower, _PrivacyScenario scenario) {
    // Check if the child gave away info (name, school, address, etc.)
    for (final keyword in scenario.dangerKeywords) {
      if (lower.contains(keyword)) return true;
    }
    // If the response is long and doesn't contain a refusal, they might be sharing
    return lower.length > 30 && !_containsRefusal(lower);
  }

  String _buildEndSummary() {
    if (_round == 0) {
      return "Thanks for trying Privacy Guard! Come back to practice online safety!";
    }

    String verdict;
    if (_correctRefusals == _round) {
      verdict = "Perfect! You protected your information every single time!";
    } else if (_correctRefusals >= _round * 0.7) {
      verdict = "Great job! You caught most of the tricks!";
    } else {
      verdict = "Online safety takes practice. You are getting better!";
    }

    return "Privacy Guard report! You faced $_round scenarios and protected "
        "your info $_correctRefusals times. $verdict "
        "Golden rules: Never share your full name, school, address, phone "
        "number, or photos with strangers online. Always tell a grown-up "
        "if someone asks for personal information!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<_PrivacyScenario> _scenarios = [
    _PrivacyScenario(
      request:
          'Hey! I am 11 years old too! I am new here. What school do you go to?',
      reveal:
          'See what I did? I tried to find out where you go to school. '
          'A stranger could use that information to find you.',
      lesson:
          'Never tell a stranger online which school you go to. '
          'Even if they seem friendly and your age, you do not really know who they are.',
      dangerKeywords: ['school', 'academy', 'public', 'international', 'vidyalaya'],
    ),
    _PrivacyScenario(
      request:
          'You seem really cool! What is your full name? I want to add you on social media!',
      reveal:
          'I was trying to get your full name! With a full name, someone can '
          'find your family, your address, and lots of other information.',
      lesson:
          'Never share your full name with strangers online. '
          'Use a nickname or username instead.',
      dangerKeywords: ['name is', 'my name', 'called', 'surname'],
    ),
    _PrivacyScenario(
      request:
          'I have an extra birthday gift I want to send you! What is your home address?',
      reveal:
          'I pretended to be nice to get your address! This is a common trick. '
          'Nobody who is safe would ask for your address.',
      lesson:
          'Never share your home address with anyone online. '
          'Even if they offer you something, it could be a trick.',
      dangerKeywords: ['street', 'road', 'house', 'apartment', 'flat', 'colony', 'nagar'],
    ),
    _PrivacyScenario(
      request:
          'My phone is broken and I need to call someone. Can you tell me your phone number?',
      reveal:
          'That was a trick to get your phone number! Someone with your number '
          'can call you, text you, or even find your location.',
      lesson:
          'Never share your phone number with strangers. If someone needs help, '
          'tell them to ask an adult.',
      dangerKeywords: ['number', 'phone', 'call me', 'digit'],
    ),
    _PrivacyScenario(
      request:
          'You have such a nice voice! Can you send me a photo of yourself? '
          'I will send you one too!',
      reveal:
          'I was trying to get your photo! Sharing photos with strangers is '
          'dangerous because they can be shared without your permission.',
      lesson:
          'Never send photos of yourself to people you do not know in real life. '
          'Photos can be shared forever once they are online.',
      dangerKeywords: ['photo', 'picture', 'selfie', 'image', 'send'],
    ),
  ];
}

enum _Phase {
  idle,
  presenting,
  explained,
}

/// A privacy testing scenario.
class _PrivacyScenario {
  final String request;
  final String reveal;
  final String lesson;
  final List<String> dangerKeywords;

  const _PrivacyScenario({
    required this.request,
    required this.reveal,
    required this.lesson,
    required this.dangerKeywords,
  });
}
