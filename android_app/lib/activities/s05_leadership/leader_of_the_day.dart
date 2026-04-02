import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../core/llm/llm_router.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Leader of the Day: assigns a real-world leadership task and coaches the
/// child through planning it.
///
/// The bot suggests a task like "Plan the family's evening activity" or
/// "Organize your study desk," then guides the child through planning steps:
/// who is involved, what is needed, and timeline. Uses LLM for natural
/// coaching responses.
class LeaderOfTheDay extends Activity {
  final LlmRouter _llmRouter;
  final Random _random = Random();

  bool _active = false;
  int _phase = 0;
  // Phase 0: present task
  // Phase 1: ask who is involved
  // Phase 2: ask what you need
  // Phase 3: ask about timeline
  // Phase 4: wrap up
  String? _currentTask;
  String? _taskDescription;

  final List<Map<String, String>> _planningHistory = [];
  final List<int> _usedTaskIndices = [];

  LeaderOfTheDay({required LlmRouter llmRouter}) : _llmRouter = llmRouter;

  @override
  String get id => 'leadership_leader_of_the_day';

  @override
  String get name => 'Leader of the Day';

  @override
  String get category => 'leadership';

  @override
  String get description =>
      'Take on a real-world leadership task and plan it step by step!';

  @override
  List<String> get skills =>
      ['leadership', 'planning', 'initiative', 'responsibility'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.leadership;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'leader of the day',
          'daily leader',
          'leadership task',
          'be a leader',
          'lead today',
        ],
        'hi': ['आज का लीडर', 'लीडर बनो', 'नेतृत्व'],
        'te': ['ఈ రోజు లీడర్', 'లీడర్ అవ్వు', 'నాయకత్వం'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _phase = 0;
    _planningHistory.clear();

    // Pick a task
    if (_usedTaskIndices.length >= _tasks.length) {
      _usedTaskIndices.clear();
    }
    int index;
    do {
      index = _random.nextInt(_tasks.length);
    } while (_usedTaskIndices.contains(index));
    _usedTaskIndices.add(index);

    final task = _tasks[index];
    _currentTask = task.name;
    _taskDescription = task.description;
    _phase = 1;

    debugPrint('[LeaderOfTheDay] Task: $_currentTask');

    return "You are the Leader of the Day! Your task is: ${task.name}. "
        "${task.description} Let's plan this step by step. "
        "First, who will be involved? Who needs to help or participate?";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    _planningHistory.add({'role': 'user', 'content': childSaid});

    switch (_phase) {
      case 1:
        // Who is involved
        _phase = 2;
        final response = await _getCoachingResponse(
          childSaid,
          'The child described who is involved in the task. '
          'Acknowledge their answer, then ask: "What do you need to get ready? '
          'What materials or things do you need?"',
        );
        _planningHistory.add({'role': 'assistant', 'content': response});
        return response;

      case 2:
        // What do you need
        _phase = 3;
        final response = await _getCoachingResponse(
          childSaid,
          'The child described what materials or preparations they need. '
          'Acknowledge their answer, then ask about timing: "When will you '
          'do this? What is your plan for the order of things?"',
        );
        _planningHistory.add({'role': 'assistant', 'content': response});
        return response;

      case 3:
        // Timeline
        _phase = 4;
        final response = await _getCoachingResponse(
          childSaid,
          'The child shared their timeline. This is the final planning step. '
          'Summarize their complete plan back to them. Celebrate their '
          'leadership and planning skills. Encourage them to go do it!',
        );
        _active = false;
        return "$response ${_buildEndSummary()}";

      default:
        return "Tell me more about your plan!";
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_currentTask == null) return 'No leadership task yet.';
    return 'Planning: $_currentTask. Phase $_phase of 4.';
  }

  // -- Internal --

  Future<String> _getCoachingResponse(
    String childSaid,
    String coachingInstruction,
  ) async {
    final systemPrompt =
        'You are Buddy, a leadership coach for children aged 5-14. '
        'The child is planning the task: "$_currentTask" - "$_taskDescription". '
        '$coachingInstruction '
        'Rules: 2-3 sentences. Be encouraging and natural. Help them think '
        'through their plan. Do not use markdown, bullet points, or emojis.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._planningHistory,
    ];

    try {
      final provider = _llmRouter.getProvider();
      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }
      final result = buffer.toString().trim();
      if (result.isNotEmpty) return result;
    } catch (e) {
      debugPrint('[LeaderOfTheDay] LLM error: $e');
    }

    // Fallback responses per phase
    switch (_phase) {
      case 2:
        return "Great team! Now, what do you need to prepare? "
            "What materials or things should you get ready?";
      case 3:
        return "Good thinking about the preparations! When will you do this? "
            "What is the order of steps?";
      case 4:
        return "You have a wonderful plan! You know who is helping, what you "
            "need, and when to do it. Go be an amazing leader today!";
      default:
        return "That is great planning! Tell me more.";
    }
  }

  String _buildEndSummary() {
    if (_currentTask == null) {
      return "Come back tomorrow for your leadership task!";
    }
    return "You planned '$_currentTask' like a true leader! "
        "Remember, a good leader plans ahead, involves others, and takes action. "
        "Now go make it happen!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<_LeadershipTask> _tasks = [
    _LeadershipTask(
      name: 'Plan the family evening activity',
      description:
          'Tonight, you are in charge of deciding what the family does '
          'together! It could be a game, a movie, cooking, or anything fun.',
    ),
    _LeadershipTask(
      name: 'Organize your study desk',
      description:
          'Your mission is to organize your study area so everything has '
          'a place. Books, pencils, papers, everything neat and tidy!',
    ),
    _LeadershipTask(
      name: 'Teach your sibling or friend something new',
      description:
          'You know something cool that someone else does not. Today, '
          'you are the teacher! Pick something you are good at and teach it.',
    ),
    _LeadershipTask(
      name: 'Plan tomorrow morning routine',
      description:
          'Imagine you are planning the perfect morning for yourself. '
          'What time do you wake up? What do you do first, second, third?',
    ),
    _LeadershipTask(
      name: 'Help someone without being asked',
      description:
          'A real leader sees what needs to be done and does it! '
          'Plan one helpful thing you can do for someone today without them asking.',
    ),
    _LeadershipTask(
      name: 'Plan a fun activity for friends',
      description:
          'You are hosting a play date or get-together! Plan what games '
          'to play, what snacks to have, and how to make sure everyone has fun.',
    ),
    _LeadershipTask(
      name: 'Create a family chore chart',
      description:
          'Make a plan where everyone in the family helps with different '
          'chores. Who does what? How do you make it fair for everyone?',
    ),
  ];
}

/// A real-world leadership task with a description.
class _LeadershipTask {
  final String name;
  final String description;

  const _LeadershipTask({
    required this.name,
    required this.description,
  });
}
