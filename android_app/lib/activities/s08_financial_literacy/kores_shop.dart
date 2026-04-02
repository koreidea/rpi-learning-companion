import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../activity_types.dart';

/// Kore's Shop: a virtual shop simulation teaching needs vs wants, saving,
/// and opportunity cost.
///
/// The child starts with 100 virtual rupees and makes 4-5 purchase decisions.
/// Each decision introduces concepts like delayed gratification, needs vs
/// wants, and budgeting. State is tracked in a simulation map.
class KoresShop extends SimulationActivity {
  final Random _random = Random();

  bool _active = false;
  int _round = 0;
  static const int _maxRounds = 5;
  Map<String, dynamic> _state = {};
  _Phase _phase = _Phase.idle;
  int _currentOfferIndex = 0;

  @override
  String get id => 'financial_literacy_kores_shop';

  @override
  String get name => 'Kore\'s Shop';

  @override
  String get category => 'financial_literacy';

  @override
  String get description =>
      'Run a virtual shop! Learn about money, needs, wants, and saving.';

  @override
  List<String> get skills =>
      ['financial literacy', 'decision making', 'math', 'planning'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 12;

  @override
  SkillId? get skillId => SkillId.financialLiteracy;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'shop game', 'kore shop', 'money game', 'shopping game',
          'buy things', 'virtual shop',
        ],
        'hi': ['दुकान खेल', 'पैसा खेल', 'खरीदारी'],
        'te': ['షాప్ ఆట', 'డబ్బు ఆట', 'కొనుగోలు'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Map<String, dynamic> get simulationState => Map.unmodifiable(_state);

  @override
  Future<void> saveSimulationState() async {
    // State is transient per session for this activity
    debugPrint('[KoresShop] State: $_state');
  }

  @override
  Future<void> loadSimulationState() async {
    // Start fresh each session
    _state = {
      'balance': 100,
      'items_bought': <String>[],
      'savings': 0,
      'round': 0,
    };
  }

  @override
  Future<String> start() async {
    _active = true;
    _round = 0;
    await loadSimulationState();
    _phase = _Phase.presenting;

    // Shuffle offers
    _currentOfferIndex = 0;

    debugPrint('[KoresShop] Started with balance: ${_state['balance']}');

    return "Welcome to Kore's Shop! You have 100 rupees to spend wisely. "
        "I will show you things you can buy. You decide what to get! "
        "Remember, once you spend your money, it is gone. "
        "Let's see what is in the shop today! ${_presentOffer()}";
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
        return "Would you like to buy or save?";

      case _Phase.presenting:
        return _handleDecision(lower);

      case _Phase.followUp:
        _round++;
        _state['round'] = _round;
        _phase = _Phase.presenting;

        if (_round >= _maxRounds) {
          _active = false;
          return _buildEndSummary();
        }
        return "Okay! Let's see what else is in the shop. ${_presentOffer()}";
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    final balance = _state['balance'] ?? 100;
    final items = _state['items_bought'] as List? ?? [];
    return 'Balance: $balance rupees. Items: ${items.length}. Round: $_round/$_maxRounds.';
  }

  // -- Internal --

  String _presentOffer() {
    if (_currentOfferIndex >= _offers.length) {
      _currentOfferIndex = 0;
    }
    final offer = _offers[_currentOfferIndex];
    _currentOfferIndex++;

    final balance = _state['balance'] as int;

    if (offer.isDecision) {
      return "${offer.description} You have $balance rupees left. ${offer.question}";
    }

    return "${offer.description} It costs ${offer.price} rupees. "
        "You have $balance rupees. Do you want to buy it?";
  }

  String _handleDecision(String lower) {
    final offer = _offers[(_currentOfferIndex - 1).clamp(0, _offers.length - 1)];
    final balance = _state['balance'] as int;
    final items = _state['items_bought'] as List<String>;

    final wantsToBuy = _containsYes(lower) || lower.contains('buy');
    final wantsToSave = lower.contains('save') || lower.contains('wait') || lower.contains('later');

    if (offer.isDecision) {
      _phase = _Phase.followUp;
      if (wantsToSave || lower.contains('wait')) {
        _state['savings'] = (_state['savings'] as int) + 10;
        return "${offer.savingResponse} Smart thinking! Waiting can save you money. "
            "You still have $balance rupees.";
      }
      return "${offer.buyResponse} You have $balance rupees left.";
    }

    if (wantsToBuy) {
      if (balance < offer.price) {
        _phase = _Phase.followUp;
        return "Oh no, you only have $balance rupees but this costs "
            "${offer.price} rupees. You do not have enough! "
            "This is why saving is important.";
      }

      _state['balance'] = balance - offer.price;
      items.add(offer.itemName);
      _phase = _Phase.followUp;

      final newBalance = _state['balance'] as int;
      final lesson = offer.isNeed
          ? "Good choice! ${offer.itemName} is something you really need."
          : "That is a want, not a need. Fun to have, but not necessary!";
      return "You bought ${offer.itemName} for ${offer.price} rupees! $lesson "
          "You have $newBalance rupees left.";
    }

    if (wantsToSave || _containsNo(lower)) {
      _phase = _Phase.followUp;
      return "Smart! You decided to save your money. "
          "You still have $balance rupees. "
          "Saving means you will have more choices later!";
    }

    return "Would you like to buy it or save your money?";
  }

  String _buildEndSummary() {
    final balance = _state['balance'] as int? ?? 100;
    final items = _state['items_bought'] as List? ?? [];
    final savings = _state['savings'] as int? ?? 0;

    final itemCount = items.length;
    String verdict;
    if (balance >= 50) {
      verdict = "You are a super saver! You still have lots of money left.";
    } else if (balance >= 20) {
      verdict = "You balanced spending and saving well!";
    } else {
      verdict = "You spent a lot, but you got some fun things!";
    }

    return "Shopping is done! You bought $itemCount "
        "${itemCount == 1 ? 'item' : 'items'} and have $balance rupees left. "
        "${savings > 0 ? 'You saved $savings rupees by waiting. ' : ''}"
        "$verdict "
        "Remember: think about what you NEED versus what you WANT, "
        "and always try to save a little!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  bool _containsYes(String text) {
    const yesWords = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok'];
    return yesWords.any((w) => text.contains(w));
  }

  bool _containsNo(String text) {
    const noWords = ['no', 'nah', 'nope', "don't want", 'not now', 'pass'];
    return noWords.any((w) => text.contains(w));
  }

  static const List<_ShopOffer> _offers = [
    _ShopOffer(
      itemName: 'pencil box',
      price: 20,
      isNeed: true,
      isDecision: false,
      description: 'A colorful pencil box with all the school supplies you need!',
      question: '',
      buyResponse: '',
      savingResponse: '',
    ),
    _ShopOffer(
      itemName: 'toy car',
      price: 30,
      isNeed: false,
      isDecision: false,
      description: 'A shiny red toy car that makes racing sounds!',
      question: '',
      buyResponse: '',
      savingResponse: '',
    ),
    _ShopOffer(
      itemName: '',
      price: 0,
      isNeed: false,
      isDecision: true,
      description:
          'The shopkeeper says: "This toy is 50 rupees today, but next week it will be 40 rupees."',
      question: 'Do you buy it now or wait until next week?',
      buyResponse: 'You bought it for 50 rupees. But if you had waited, it would be 40!',
      savingResponse: 'You chose to wait! Next week you can get it for 10 rupees less.',
    ),
    _ShopOffer(
      itemName: 'book of stories',
      price: 25,
      isNeed: true,
      isDecision: false,
      description: 'A wonderful book full of adventure stories!',
      question: '',
      buyResponse: '',
      savingResponse: '',
    ),
    _ShopOffer(
      itemName: 'ice cream',
      price: 15,
      isNeed: false,
      isDecision: false,
      description: 'A delicious chocolate ice cream cone!',
      question: '',
      buyResponse: '',
      savingResponse: '',
    ),
    _ShopOffer(
      itemName: 'fancy stickers',
      price: 10,
      isNeed: false,
      isDecision: false,
      description: 'A pack of sparkly stickers with animals and stars!',
      question: '',
      buyResponse: '',
      savingResponse: '',
    ),
    _ShopOffer(
      itemName: '',
      price: 0,
      isNeed: false,
      isDecision: true,
      description:
          'Your friend wants to split the cost of a big puzzle. It costs 40 rupees total, so your share would be 20 rupees.',
      question: 'Do you want to share the cost and the puzzle with your friend?',
      buyResponse:
          'You shared the cost! You paid only 20 rupees and you both get to enjoy the puzzle. Sharing can be smart!',
      savingResponse:
          'You decided to save. That is fine too! You kept your money for something else.',
    ),
    _ShopOffer(
      itemName: 'water bottle',
      price: 15,
      isNeed: true,
      isDecision: false,
      description: 'A sturdy water bottle you can carry to school every day!',
      question: '',
      buyResponse: '',
      savingResponse: '',
    ),
  ];
}

enum _Phase {
  idle,
  presenting,
  followUp,
}

/// A shop offer or decision point.
class _ShopOffer {
  final String itemName;
  final int price;
  final bool isNeed;
  final bool isDecision;
  final String description;
  final String question;
  final String buyResponse;
  final String savingResponse;

  const _ShopOffer({
    required this.itemName,
    required this.price,
    required this.isNeed,
    required this.isDecision,
    required this.description,
    required this.question,
    required this.buyResponse,
    required this.savingResponse,
  });
}
