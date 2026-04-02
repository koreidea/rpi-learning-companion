import 'dart:math';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// A daily observation prompt with a hypothesis template.
class _ObservationPrompt {
  final String observation;
  final String hypothesisStarter;
  final String testIdea;

  const _ObservationPrompt({
    required this.observation,
    required this.hypothesisStarter,
    required this.testIdea,
  });
}

/// HypothesisOfDay: daily observation and hypothesis practice.
///
/// Teaches: observation, hypothesis formation, scientific reasoning,
/// prediction, cause and effect.
///
/// Flow:
/// 1. Bot presents a daily observation or phenomenon.
/// 2. Asks child to make a hypothesis (what do you think and why?).
/// 3. Discusses the child's thinking and shares the explanation.
/// 4. Suggests a simple way to test the idea.
class HypothesisOfDay extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _hypothesesMade = 0;
  int _score = 0;

  /// 0=present observation, 1=child hypothesizes, 2=discuss and test,
  /// 3=next or end
  int _phase = 0;
  _ObservationPrompt? _currentPrompt;
  final List<int> _usedIndices = [];

  static const int _maxHypotheses = 2;

  static const List<_ObservationPrompt> _prompts = [
    _ObservationPrompt(
      observation: 'When you blow on hot soup, it cools down faster. '
          'Why do you think blowing makes food cool down?',
      hypothesisStarter: 'I think blowing makes it cool because',
      testIdea: 'Try this: put two cups of warm water on the table. '
          'Blow on one but not the other. After a minute, feel both. '
          'Which one is cooler?',
    ),
    _ObservationPrompt(
      observation: 'When it rains, worms come out of the ground. '
          'Why do you think worms come out when it rains?',
      hypothesisStarter: 'I think worms come out because',
      testIdea: 'Next time it rains, go outside and count how many worms '
          'you can see. Then check a dry day. Do you see the same number?',
    ),
    _ObservationPrompt(
      observation: 'A ball rolls faster down a steep ramp than a gentle one. '
          'Why do you think steeper ramps make things go faster?',
      hypothesisStarter: 'I think it goes faster because',
      testIdea: 'Stack some books to make two ramps, one tall and one short. '
          'Roll a ball down each one. Which one reaches the bottom first?',
    ),
    _ObservationPrompt(
      observation: 'Plants grow toward the window, even if you turn them around. '
          'Why do you think plants bend toward the light?',
      hypothesisStarter: 'I think plants bend toward light because',
      testIdea: 'Put a plant near a window. Turn it around every day. '
          'Watch what happens over a week. Does it keep turning back?',
    ),
    _ObservationPrompt(
      observation: 'When you put salt on ice, the ice melts faster. '
          'Why do you think salt makes ice melt?',
      hypothesisStarter: 'I think salt makes ice melt because',
      testIdea: 'Get two ice cubes. Put salt on one and leave the other alone. '
          'Watch and see which one melts first!',
    ),
    _ObservationPrompt(
      observation: 'The moon looks different every night. Sometimes it is round, '
          'sometimes it is a thin sliver. Why do you think the moon changes shape?',
      hypothesisStarter: 'I think the moon changes because',
      testIdea: 'Draw the moon every night for a week. Can you see a pattern? '
          'A flashlight and a ball in a dark room can show you how it works!',
    ),
    _ObservationPrompt(
      observation: 'When you mix red and blue paint, you get purple. '
          'Why do you think colors mix to make new colors?',
      hypothesisStarter: 'I think colors mix because',
      testIdea: 'Try mixing different colors of paint or food coloring. '
          'Can you predict what color you will get before mixing?',
    ),
    _ObservationPrompt(
      observation: 'Bubbles are always round, never square or triangle shaped. '
          'Why do you think bubbles are always round?',
      hypothesisStarter: 'I think bubbles are round because',
      testIdea: 'Try making bubbles with different shaped wands. '
          'Does a square wand make square bubbles? Try it and find out!',
    ),
    _ObservationPrompt(
      observation: 'When you spin around and then stop, the room seems to '
          'keep spinning. Why do you think that happens?',
      hypothesisStarter: 'I think the room seems to spin because',
      testIdea: 'Try spinning slowly for 5 seconds, then stop. '
          'Now try 10 seconds. Does the room spin longer after more spinning?',
    ),
    _ObservationPrompt(
      observation: 'Metal spoons get hot when you put them in hot water, '
          'but wooden spoons do not. Why do you think that happens?',
      hypothesisStarter: 'I think metal gets hot because',
      testIdea: 'Put a metal spoon and a plastic spoon in warm water. '
          'After a minute, feel the handles. Which one is warmer?',
    ),
  ];

  HypothesisOfDay();

  @override
  String get id => 'hypothesis_of_day';

  @override
  String get name => 'Hypothesis of the Day';

  @override
  String get category => 'science';

  @override
  String get description =>
      'Make observations about the world and form scientific hypotheses.';

  @override
  List<String> get skills => [
        'observation',
        'hypothesis formation',
        'scientific reasoning',
        'prediction',
      ];

  @override
  int get minAge => 4;

  @override
  int get maxAge => 12;

  @override
  SkillId? get skillId => SkillId.scientificThinking;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'hypothesis of the day',
          'science question',
          'why does that happen',
          'hypothesis game',
          'daily science',
        ],
        'hi': [
          'आज का अनुमान',
          'विज्ञान सवाल',
          'ऐसा क्यों होता है',
        ],
        'te': [
          'ఈ రోజు ఊహ',
          'సైన్స్ ప్రశ్న',
          'ఇది ఎందుకు జరుగుతుంది',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_hypothesesMade == 0) return 'No hypotheses made yet.';
    return 'Made $_hypothesesMade '
        'hypothesis${_hypothesesMade != 1 ? 'es' : ''}. '
        'Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _hypothesesMade = 0;
    _score = 0;
    _phase = 0;
    _usedIndices.clear();

    return 'Welcome to Hypothesis of the Day! Scientists start by observing '
        'things and then asking: why does that happen? Today, you are the '
        'scientist! ${_presentNewObservation()}';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      return await end();
    }

    switch (_phase) {
      case 0:
        // Child made their hypothesis
        _phase = 1;
        _score += 15;
        final prompt = _currentPrompt!;
        return 'That is a really good hypothesis! Scientists think about '
            'things just like you did. Here is something cool: '
            '${prompt.testIdea} Would you like to try another observation?';

      case 1:
        // Child responded to test idea
        _hypothesesMade++;

        if (_isAffirmative(lower) && _hypothesesMade < _maxHypotheses) {
          return _presentNewObservation();
        }

        return await end();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_hypothesesMade == 0) {
      return 'Come back tomorrow for a new observation to investigate! '
          'Scientists never stop asking questions.';
    }
    return 'Great science thinking today! You made $_hypothesesMade '
        'hypothesis${_hypothesesMade != 1 ? 'es' : ''}. '
        'Score: $_score points! Remember, every great discovery started '
        'with someone asking why.';
  }

  String _presentNewObservation() {
    final available = <int>[];
    for (int i = 0; i < _prompts.length; i++) {
      if (!_usedIndices.contains(i)) available.add(i);
    }

    if (available.isEmpty) {
      _usedIndices.clear();
      for (int i = 0; i < _prompts.length; i++) {
        available.add(i);
      }
    }

    final idx = available[_rng.nextInt(available.length)];
    _usedIndices.add(idx);
    _currentPrompt = _prompts[idx];
    _phase = 0;

    return 'Here is your observation. ${_currentPrompt!.observation} '
        'What is your hypothesis? ${_currentPrompt!.hypothesisStarter}...';
  }

  bool _isAffirmative(String text) {
    const words = ['yes', 'yeah', 'sure', 'okay', 'ok', 'another', 'more',
      'next', 'one more', 'हाँ', 'और', 'అవును', 'ఇంకొకటి'];
    return words.any((w) => text.contains(w));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough',
      'no', 'no thanks', 'no more'];
    return quitWords.any((w) => text.contains(w));
  }
}
