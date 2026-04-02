import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../activity_types.dart';

/// Budget Boss: weekly virtual allowance budgeting activity teaching
/// allocation across needs, wants, and savings.
///
/// The child receives 200 virtual rupees and decides how to allocate them
/// across categories. A surprise expense tests their emergency fund. Teaches
/// budgeting fundamentals in a fun, interactive way.
class BudgetBoss extends SimulationActivity {
  final Random _random = Random();

  bool _active = false;
  int _phase = 0;
  // Phase 0: intro
  // Phase 1: allocate needs
  // Phase 2: allocate wants
  // Phase 3: allocate savings
  // Phase 4: surprise event
  // Phase 5: wrap up
  Map<String, dynamic> _state = {};

  @override
  String get id => 'financial_literacy_budget_boss';

  @override
  String get name => 'Budget Boss';

  @override
  String get category => 'financial_literacy';

  @override
  String get description =>
      'Plan a weekly budget! Decide how to split money between needs, wants, and savings.';

  @override
  List<String> get skills =>
      ['financial literacy', 'budgeting', 'planning', 'math'];

  @override
  int get minAge => 6;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.financialLiteracy;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'budget boss',
          'budget game',
          'budgeting',
          'plan my money',
          'money plan',
        ],
        'hi': ['बजट बॉस', 'बजट खेल', 'पैसे का प्लान'],
        'te': ['బడ్జెట్ బాస్', 'బడ్జెట్ ఆట', 'డబ్బు ప్లాన్'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Map<String, dynamic> get simulationState => Map.unmodifiable(_state);

  @override
  Future<void> saveSimulationState() async {
    debugPrint('[BudgetBoss] State: $_state');
  }

  @override
  Future<void> loadSimulationState() async {
    _state = {
      'total': 200,
      'needs': 0,
      'wants': 0,
      'savings': 0,
      'remaining': 200,
      'surprise_cost': 0,
      'surprise_covered': false,
    };
  }

  @override
  Future<String> start() async {
    _active = true;
    _phase = 1;
    await loadSimulationState();

    return "You are the Budget Boss! You have 200 rupees for the week. "
        "You need to divide it into three piles: Needs, like school supplies "
        "and food. Wants, like toys, games, and treats. And Savings, for "
        "emergencies or something big you want later. "
        "Let's start! How much do you want to put in Needs? "
        "Remember, you have 200 rupees total.";
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
      case 1:
        // Allocate needs
        final amount = _extractNumber(lower, 0, 200, 80);
        _state['needs'] = amount;
        _state['remaining'] = 200 - amount;
        _phase = 2;
        return "You put $amount rupees in Needs. Good! "
            "You have ${_state['remaining']} rupees left. "
            "How much do you want to put in Wants?";

      case 2:
        // Allocate wants
        final remaining = _state['remaining'] as int;
        final amount = _extractNumber(lower, 0, remaining, remaining ~/ 2);
        _state['wants'] = amount;
        _state['remaining'] = remaining - amount;
        _phase = 3;

        if (_state['remaining'] == 0) {
          _state['savings'] = 0;
          _phase = 4;
          return "You put $amount rupees in Wants. But you have nothing left "
              "for Savings! Let's see what happens. ${_presentSurprise()}";
        }

        return "You put $amount rupees in Wants. "
            "You have ${_state['remaining']} rupees left. "
            "The rest goes to Savings. That means ${_state['remaining']} rupees "
            "in your savings! Does that sound good?";

      case 3:
        // Confirm savings
        _state['savings'] = _state['remaining'];
        _state['remaining'] = 0;
        _phase = 4;
        return "Your budget is set! "
            "Needs: ${_state['needs']} rupees. "
            "Wants: ${_state['wants']} rupees. "
            "Savings: ${_state['savings']} rupees. "
            "Now let's see what the week brings! ${_presentSurprise()}";

      case 4:
        // React to surprise
        _phase = 5;
        final savings = _state['savings'] as int;
        final surpriseCost = _state['surprise_cost'] as int;

        if (savings >= surpriseCost) {
          _state['surprise_covered'] = true;
          _state['savings'] = savings - surpriseCost;
          _active = false;
          return "Good news! You had $savings rupees saved, so you could "
              "cover the $surpriseCost rupee surprise. You still have "
              "${_state['savings']} rupees in savings! "
              "See how saving helps? ${_buildEndSummary()}";
        } else {
          _state['surprise_covered'] = false;
          _active = false;
          return "Uh oh! The surprise cost $surpriseCost rupees but you only "
              "had $savings rupees saved. You could not fully cover it! "
              "This is why saving is so important. Even small savings help! "
              "${_buildEndSummary()}";
        }

      default:
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
    return 'Needs: ${_state['needs']}, Wants: ${_state['wants']}, '
        'Savings: ${_state['savings']}.';
  }

  // -- Internal --

  String _presentSurprise() {
    final surprise = _surprises[_random.nextInt(_surprises.length)];
    _state['surprise_cost'] = surprise.cost;
    return "Oh no! ${surprise.description} That costs ${surprise.cost} rupees! "
        "Can your savings cover it?";
  }

  int _extractNumber(String text, int min, int max, int defaultVal) {
    final match = RegExp(r'\d+').firstMatch(text);
    if (match != null) {
      final num = int.tryParse(match.group(0)!) ?? defaultVal;
      return num.clamp(min, max);
    }
    return defaultVal;
  }

  String _buildEndSummary() {
    final needs = _state['needs'] ?? 0;
    final wants = _state['wants'] ?? 0;
    final savings = _state['savings'] ?? 0;
    final covered = _state['surprise_covered'] as bool? ?? false;

    String tip;
    if (covered) {
      tip = "Your savings saved the day! A good rule is to save at least "
          "20 percent of your money. That way surprises do not ruin your plans.";
    } else {
      tip = "Next time, try saving more! A good rule is to put at least "
          "20 percent in savings. Even 40 rupees out of 200 can make a big "
          "difference when surprises happen.";
    }

    return "Budget Boss report! You put $needs in needs, $wants in wants, "
        "and $savings in savings. $tip "
        "You are learning to manage money like a pro!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<_Surprise> _surprises = [
    _Surprise(
      description: 'Your pencil box broke and you need a new one!',
      cost: 50,
    ),
    _Surprise(
      description: 'Your friend\'s birthday is this week and you need a gift!',
      cost: 40,
    ),
    _Surprise(
      description: 'Your water bottle cracked and is leaking!',
      cost: 30,
    ),
    _Surprise(
      description: 'You need new art supplies for a school project!',
      cost: 45,
    ),
    _Surprise(
      description: 'Your school bag zipper broke!',
      cost: 60,
    ),
  ];
}

/// A surprise expense that tests the child's savings.
class _Surprise {
  final String description;
  final int cost;

  const _Surprise({required this.description, required this.cost});
}
