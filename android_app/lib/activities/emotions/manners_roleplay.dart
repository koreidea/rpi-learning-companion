import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../core/llm/llm_router.dart';
import '../activity_base.dart';

/// A roleplay scenario teaching social skills.
class _Scenario {
  final String situation;
  final String prompt;
  final String skill;
  final List<String> goodResponses;
  final String praise;

  const _Scenario({
    required this.situation,
    required this.prompt,
    required this.skill,
    required this.goodResponses,
    required this.praise,
  });
}

/// Manners Roleplay: role-play scenarios teaching manners and social skills.
///
/// Teaches: social skills, empathy, politeness, communication.
///
/// Flow:
/// 1. Bot presents a scenario.
/// 2. Child responds.
/// 3. Bot evaluates and praises (using LLM for nuanced responses).
/// 4. Bot asks follow-up ("what would you do next?").
/// 5. Move to next scenario.
class MannersRoleplay extends Activity {
  final LlmRouter _llmRouter;
  final Random _rng = Random();

  bool _active = false;
  int _scenariosCompleted = 0;
  int _score = 0;
  int _phase = 0; // 0=present, 1=first response, 2=follow-up, 3=transition
  _Scenario? _currentScenario;
  final List<int> _usedScenarios = [];
  String _childFirstResponse = '';

  static const int _maxScenarios = 4;

  static const List<_Scenario> _scenarios = [
    _Scenario(
      situation: 'Friend fell down',
      prompt: "Let's practice being kind! Pretend your friend fell down and hurt their knee. "
          "What would you say to them?",
      skill: 'empathy',
      goodResponses: [
        'okay', 'are you okay', 'hurt', 'help', 'sorry', 'alright',
        'fine', 'get up', 'cry', 'better',
      ],
      praise: "That is so kind! Asking if someone is okay shows you care about them.",
    ),
    _Scenario(
      situation: 'Receiving a gift',
      prompt: "Pretend someone gives you a birthday present! A big box with a ribbon! "
          "What would you say?",
      skill: 'gratitude',
      goodResponses: [
        'thank you', 'thanks', 'love it', 'wow', 'amazing',
        'so nice', 'like it', 'great',
      ],
      praise: "Wonderful! Saying thank you makes the person who gave you the gift feel so happy!",
    ),
    _Scenario(
      situation: 'Bumping into someone',
      prompt: "Oops! Pretend you were running and you accidentally bumped into someone. "
          "What would you say?",
      skill: 'apologizing',
      goodResponses: [
        'sorry', 'excuse me', 'my fault', 'did not mean', 'accident',
        'oops', 'are you okay', 'pardon',
      ],
      praise: "That is perfect! Saying sorry when we accidentally bump into someone shows we are responsible.",
    ),
    _Scenario(
      situation: 'New kid at school',
      prompt: "Pretend there is a new kid at your school and they look a little shy. "
          "What would you say to make them feel welcome?",
      skill: 'friendliness',
      goodResponses: [
        'hi', 'hello', 'name', 'play', 'friend', 'come', 'join',
        'welcome', 'sit with', 'together',
      ],
      praise: "That is so friendly! Introducing yourself makes new people feel welcome and happy!",
    ),
    _Scenario(
      situation: 'Sharing toys',
      prompt: "Pretend your little brother or sister wants to play with your favorite toy. "
          "What would you do?",
      skill: 'sharing',
      goodResponses: [
        'share', 'turn', 'play together', 'here', 'have it',
        'your turn', 'we can', 'both', 'together',
      ],
      praise: "You are so generous! Sharing is one of the kindest things we can do. It makes everyone happy!",
    ),
    _Scenario(
      situation: 'Asking for help',
      prompt: "Pretend you need help reaching something on a high shelf. "
          "How would you ask a grown-up for help?",
      skill: 'politeness',
      goodResponses: [
        'please', 'help', 'can you', 'could you', 'would you',
        'excuse me', 'reach', 'get',
      ],
      praise: "Great manners! Saying please when we ask for help is very polite. People love helping polite kids!",
    ),
    _Scenario(
      situation: 'Someone shares food',
      prompt: "Pretend your friend shares their snack with you at lunchtime. "
          "What would you say? Would you offer them some of yours too?",
      skill: 'reciprocity',
      goodResponses: [
        'thank you', 'thanks', 'want some', 'share mine', 'have some',
        'try', 'nice', 'yummy',
      ],
      praise: "That is wonderful! Saying thank you and offering to share back shows you are a great friend!",
    ),
    _Scenario(
      situation: 'Waiting for your turn',
      prompt: "Pretend you are at the playground and there is a line for the slide. "
          "What would you do?",
      skill: 'patience',
      goodResponses: [
        'wait', 'turn', 'line', 'queue', 'patient', 'after',
        'next', 'my turn', 'stand',
      ],
      praise: "Excellent! Waiting for your turn is called patience, and it shows respect for others!",
    ),
  ];

  MannersRoleplay({required LlmRouter llmRouter})
      : _llmRouter = llmRouter;

  // -- Activity metadata --

  @override
  String get id => 'emotions_manners';

  @override
  String get name => 'Manners Practice';

  @override
  String get category => 'emotions';

  @override
  String get description =>
      'Practice kindness and good manners through fun pretend scenarios.';

  @override
  List<String> get skills => ['social skills', 'empathy', 'politeness', 'communication'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    return '$_scenariosCompleted scenarios completed. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _scenariosCompleted = 0;
    _score = 0;
    _usedScenarios.clear();
    _phase = 0;
    _active = true;

    debugPrint('[MannersRoleplay] Started');

    return _presentNextScenario();
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    switch (_phase) {
      case 0:
        // Child responded to the scenario
        _childFirstResponse = childSaid;
        _phase = 1;
        return await _evaluateResponse(childSaid);

      case 1:
        // Follow-up: what would you do next?
        _phase = 2;
        _score += 5;
        return await _evaluateFollowUp(childSaid);

      case 2:
        // Transition to next scenario or end
        _scenariosCompleted++;
        _score += 10;

        if (_scenariosCompleted >= _maxScenarios) {
          return await end();
        }

        _phase = 0;
        return "Great! Let's try another one! ${_presentNextScenario()}";

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    debugPrint('[MannersRoleplay] Ended, scenarios=$_scenariosCompleted, score=$_score');

    if (_scenariosCompleted == 0) {
      return "Okay, we'll practice manners another time! "
          "Remember, being kind makes the world a better place!";
    }

    _score += 15; // Completion bonus
    return "You did an amazing job! You practiced being kind in $_scenariosCompleted "
        "different situations. You are such a thoughtful and polite person! "
        "Score: $_score points!";
  }

  // -- Scenario management --

  String _presentNextScenario() {
    final available = <int>[];
    for (int i = 0; i < _scenarios.length; i++) {
      if (!_usedScenarios.contains(i)) {
        available.add(i);
      }
    }

    if (available.isEmpty) {
      _active = false;
      return "You practiced all the scenarios! You are a manners master!";
    }

    final index = available[_rng.nextInt(available.length)];
    _usedScenarios.add(index);
    _currentScenario = _scenarios[index];

    return _currentScenario!.prompt;
  }

  Future<String> _evaluateResponse(String childSaid) async {
    final lower = childSaid.toLowerCase();
    final scenario = _currentScenario!;

    // Check if the response shows the right idea
    final showsUnderstanding = scenario.goodResponses.any((r) => lower.contains(r));

    if (showsUnderstanding) {
      _score += 15;
      return "${scenario.praise} What would you do next?";
    }

    // Use LLM for nuanced evaluation
    try {
      final provider = _llmRouter.getProvider();
      final instruction =
          'A 3-6 year old child is practicing social skills. '
          'The scenario: "${scenario.situation}". '
          'The child said: "$childSaid". '
          'The skill being practiced is: ${scenario.skill}. '
          'If the child showed any kindness or understanding, praise them warmly. '
          'If they seem confused, gently explain what a kind response would be. '
          'Keep it to 2 sentences. Do not use emojis or markdown.';

      final messages = [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': childSaid},
      ];

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }

      final result = buffer.toString().trim();
      if (result.isNotEmpty) {
        _score += 10;
        return "$result What would you do next?";
      }
    } catch (e) {
      debugPrint('[MannersRoleplay] LLM error: $e');
    }

    // Fallback
    _score += 10;
    return "That is a good try! ${scenario.praise} What would you do next?";
  }

  Future<String> _evaluateFollowUp(String childSaid) async {
    // Any follow-up answer is good — we want to encourage thinking
    try {
      final provider = _llmRouter.getProvider();
      final instruction =
          'A 3-6 year old child is practicing social skills in the scenario: '
          '"${_currentScenario!.situation}". '
          'They first said: "$_childFirstResponse". '
          'When asked what they would do next, they said: "$childSaid". '
          'Give brief warm praise for their answer in 1 sentence. '
          'Do not use emojis or markdown.';

      final messages = [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': childSaid},
      ];

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }

      final result = buffer.toString().trim();
      if (result.isNotEmpty) return result;
    } catch (e) {
      debugPrint('[MannersRoleplay] Follow-up LLM error: $e');
    }

    return "That is wonderful! You are such a kind person!";
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop the game', 'quit', 'exit', "i'm done", 'no more',
      'stop playing', 'end the game', 'finish',
      'बंद करो', 'खेल बंद',
      'ఆపు', 'ఆట ఆపు',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
