import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../activity_types.dart';

/// Lemonade Stand: a business simulation where the child sets prices, manages
/// supply, and adjusts strategy based on simulated market conditions.
///
/// Each session runs 5 "days" where the child sets price and quantity. The
/// bot simulates weather, customer demand, and competition. The child sees
/// their revenue, costs, and profit each day and must adjust.
class LemonadeStand extends SimulationActivity {
  final Random _random = Random();

  bool _active = false;
  int _day = 0;
  static const int _maxDays = 5;
  Map<String, dynamic> _state = {};
  _Phase _phase = _Phase.idle;

  // Current day settings
  int _price = 10;
  int _cups = 20;
  String _location = 'park';

  @override
  String get id => 'financial_literacy_lemonade_stand';

  @override
  String get name => 'Lemonade Stand';

  @override
  String get category => 'financial_literacy';

  @override
  String get description =>
      'Run your own lemonade business! Set prices, make lemonade, earn profit.';

  @override
  List<String> get skills =>
      ['financial literacy', 'entrepreneurship', 'math', 'decision making'];

  @override
  int get minAge => 6;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.financialLiteracy;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'lemonade stand',
          'lemonade game',
          'business game',
          'sell lemonade',
          'run a business',
        ],
        'hi': ['नींबू पानी दुकान', 'बिजनेस खेल', 'दुकान खोलो'],
        'te': ['నిమ్మరసం అంగడి', 'బిజినెస్ ఆట'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Map<String, dynamic> get simulationState => Map.unmodifiable(_state);

  @override
  Future<void> saveSimulationState() async {
    debugPrint('[LemonadeStand] State: $_state');
  }

  @override
  Future<void> loadSimulationState() async {
    _state = {
      'total_revenue': 0,
      'total_cost': 0,
      'total_profit': 0,
      'day': 0,
      'decisions': <Map<String, dynamic>>[],
    };
  }

  @override
  Future<String> start() async {
    _active = true;
    _day = 0;
    await loadSimulationState();
    _phase = _Phase.settingPrice;

    return "Welcome to your Lemonade Stand! You are the boss! "
        "Each day, you decide how much to charge, how many cups to make, "
        "and where to sell. Let's start Day 1! "
        "How much do you want to charge per cup? You can pick between 5 and 20 rupees.";
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
        return "How much do you want to charge per cup?";

      case _Phase.settingPrice:
        _price = _extractNumber(lower, 5, 20, 10);
        _phase = _Phase.settingQuantity;
        return "Great, $_price rupees per cup! Now, how many cups do you want "
            "to make? Making each cup costs 3 rupees for ingredients. "
            "You can make between 10 and 50 cups.";

      case _Phase.settingQuantity:
        _cups = _extractNumber(lower, 10, 50, 20);
        _phase = _Phase.settingLocation;
        return "You will make $_cups cups! That will cost you ${_cups * 3} "
            "rupees for ingredients. Now, where do you want to sell? "
            "Say school, park, or market.";

      case _Phase.settingLocation:
        if (lower.contains('school')) {
          _location = 'school';
        } else if (lower.contains('market')) {
          _location = 'market';
        } else {
          _location = 'park';
        }
        _phase = _Phase.showingResults;
        return _simulateDay();

      case _Phase.showingResults:
        _day++;
        _state['day'] = _day;

        if (_day >= _maxDays) {
          _active = false;
          return _buildEndSummary();
        }

        _phase = _Phase.settingPrice;
        return "Day ${_day + 1}! Based on what happened, do you want to "
            "change your price? How much per cup this time? (5 to 20 rupees)";
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    final profit = _state['total_profit'] ?? 0;
    return 'Day $_day of $_maxDays. Total profit: $profit rupees.';
  }

  // -- Internal --

  String _simulateDay() {
    // Simulate conditions
    final weatherRoll = _random.nextInt(100);
    String weather;
    double weatherMultiplier;
    if (weatherRoll < 40) {
      weather = 'sunny';
      weatherMultiplier = 1.3;
    } else if (weatherRoll < 70) {
      weather = 'cloudy';
      weatherMultiplier = 1.0;
    } else {
      weather = 'rainy';
      weatherMultiplier = 0.5;
    }

    // Location multiplier
    double locationMultiplier;
    switch (_location) {
      case 'school':
        locationMultiplier = 1.2;
        break;
      case 'market':
        locationMultiplier = 1.0;
        break;
      default:
        locationMultiplier = 1.1;
    }

    // Price affects demand (cheaper = more customers)
    double priceMultiplier;
    if (_price <= 7) {
      priceMultiplier = 1.4;
    } else if (_price <= 12) {
      priceMultiplier = 1.0;
    } else {
      priceMultiplier = 0.6;
    }

    // Calculate customers
    final baseCustomers = 15;
    int customers = (baseCustomers *
            weatherMultiplier *
            locationMultiplier *
            priceMultiplier)
        .round();
    customers = customers.clamp(2, _cups);

    // Financials
    final revenue = customers * _price;
    final cost = _cups * 3;
    final profit = revenue - cost;
    final unsold = _cups - customers;

    // Update state
    _state['total_revenue'] = (_state['total_revenue'] as int) + revenue;
    _state['total_cost'] = (_state['total_cost'] as int) + cost;
    _state['total_profit'] = (_state['total_profit'] as int) + profit;
    (_state['decisions'] as List).add({
      'day': _day + 1,
      'price': _price,
      'cups': _cups,
      'location': _location,
      'weather': weather,
      'customers': customers,
      'revenue': revenue,
      'cost': cost,
      'profit': profit,
    });

    String weatherReport;
    switch (weather) {
      case 'sunny':
        weatherReport = 'It was a beautiful sunny day! Lots of thirsty people!';
        break;
      case 'rainy':
        weatherReport =
            'It rained today! Not many people came out to buy lemonade.';
        break;
      default:
        weatherReport = 'It was a cloudy day. A decent number of people came by.';
    }

    String profitMessage;
    if (profit > 0) {
      profitMessage = 'You made a profit of $profit rupees! Great business day!';
    } else if (profit == 0) {
      profitMessage = 'You broke even. No profit, no loss.';
    } else {
      profitMessage =
          'Oh no, you lost ${-profit} rupees today. You spent more than you earned.';
    }

    final unsoldMessage = unsold > 0
        ? ' You had $unsold cups left over that nobody bought.'
        : ' You sold every single cup!';

    return "Day ${_day + 1} results! $weatherReport "
        "$customers customers came to your $_location stand. "
        "You sold $customers cups at $_price rupees each. "
        "Revenue: $revenue rupees. Cost: $cost rupees. "
        "$profitMessage$unsoldMessage "
        "Total profit so far: ${_state['total_profit']} rupees.";
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
    final totalProfit = _state['total_profit'] as int? ?? 0;
    final totalRevenue = _state['total_revenue'] as int? ?? 0;
    final totalCost = _state['total_cost'] as int? ?? 0;

    String verdict;
    if (totalProfit > 50) {
      verdict = "You are a business genius!";
    } else if (totalProfit > 0) {
      verdict = "You made a profit! Not bad for your first business.";
    } else {
      verdict = "Running a business is tough, but you learned a lot!";
    }

    return "Your lemonade stand is closed for the season! "
        "Over $_day days: Total revenue was $totalRevenue rupees, "
        "total costs were $totalCost rupees, and your total profit was "
        "$totalProfit rupees. $verdict "
        "Remember: set fair prices, do not make too many cups you cannot sell, "
        "and pay attention to the weather!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }
}

enum _Phase {
  idle,
  settingPrice,
  settingQuantity,
  settingLocation,
  showingResults,
}
