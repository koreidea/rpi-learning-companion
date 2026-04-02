import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Screen Time Coach: teaches the 20-20-20 rule and guides a quick screen
/// break with eye exercises and posture awareness.
///
/// A short 2-minute activity that walks the child through looking at
/// something far away, discusses why screen breaks matter, and covers
/// topics like blue light and posture.
class ScreenTimeCoach extends Activity {
  bool _active = false;
  int _phase = 0;
  // Phase 0: intro
  // Phase 1: 20-20-20 exercise
  // Phase 2: posture check
  // Phase 3: fun facts
  // Phase 4: wrap up

  @override
  String get id => 'digital_citizenship_screen_time_coach';

  @override
  String get name => 'Screen Time Coach';

  @override
  String get category => 'digital_citizenship';

  @override
  String get description =>
      'Take a healthy screen break with the 20-20-20 rule!';

  @override
  List<String> get skills =>
      ['digital citizenship', 'health awareness', 'self-care'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.digitalCitizenship;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'screen time coach',
          'screen break',
          'eye break',
          'take a break',
          'twenty twenty twenty',
        ],
        'hi': ['स्क्रीन ब्रेक', 'आंखों का ब्रेक', 'ब्रेक लो'],
        'te': ['స్క్రీన్ బ్రేక్', 'కళ్ళ బ్రేక్', 'విరామం తీసుకో'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _phase = 0;

    debugPrint('[ScreenTimeCoach] Started');

    return "Time for a screen break! Did you know there is a cool rule "
        "called the 20-20-20 rule? Every 20 minutes, you should look at "
        "something 20 feet away for 20 seconds. It keeps your eyes healthy! "
        "Let's try it right now. Are you ready?";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    _phase++;

    switch (_phase) {
      case 1:
        return "Great! Look at something far away from you. Maybe a wall, "
            "a window, or something across the room. Keep looking at it. "
            "I will count to 20 for you. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "
            "11, 12, 13, 14, 15, 16, 17, 18, 19, 20! "
            "How do your eyes feel? A bit better?";

      case 2:
        return "Now let's check your posture! Sit up straight. "
            "Are your feet flat on the floor? Is the screen at eye level? "
            "Roll your shoulders back. Good! Sitting properly prevents "
            "back pain and headaches. How does that feel?";

      case 3:
        return "Here is a fun fact! Screens give off something called blue "
            "light. Too much blue light can make it hard to fall asleep at "
            "night. That is why it is good to stop using screens at least "
            "one hour before bedtime. Did you know that?";

      case 4:
        _active = false;
        return "Great screen break! Remember the 20-20-20 rule: every 20 "
            "minutes, look at something 20 feet away for 20 seconds. "
            "Your eyes will thank you! ${_buildEndSummary()}";

      default:
        _active = false;
        return _buildEndSummary();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    return 'Screen break: phase $_phase of 4.';
  }

  // -- Internal --

  String _buildEndSummary() {
    return "Screen break complete! You took care of your eyes, checked your "
        "posture, and learned about blue light. Taking breaks is a sign "
        "of being a smart screen user!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }
}
