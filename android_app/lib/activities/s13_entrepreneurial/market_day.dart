import 'dart:math';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_types.dart';

/// A product the child can sell in their virtual shop.
class _Product {
  final String name;
  final int costPrice;
  final int suggestedPrice;

  const _Product({
    required this.name,
    required this.costPrice,
    required this.suggestedPrice,
  });
}

/// A market event that affects sales.
class _MarketEvent {
  final String description;
  final double salesMultiplier;

  const _MarketEvent({
    required this.description,
    required this.salesMultiplier,
  });
}

/// Market Day: a virtual market simulation where the child runs a shop.
///
/// Teaches: pricing, supply/demand, competition, profit calculation,
/// entrepreneurial thinking, basic math.
///
/// Flow:
/// 1. Child chooses a product to sell (food, toys, or clothes).
/// 2. Child sets a price.
/// 3. Bot simulates: customers arrive based on price and events.
/// 4. Random events each round (festivals, competition, weather).
/// 5. Track revenue, expenses, profit over 5 rounds.
class MarketDay extends SimulationActivity {
  final Random _rng = Random();

  bool _active = false;
  int _round = 0;
  int _maxRounds = 5;

  Map<String, dynamic> _state = {};

  /// 0=choose product, 1=set price, 2=results, 3=next round or end
  int _phase = 0;

  _Product? _chosenProduct;
  int _currentPrice = 0;
  int _stock = 20;

  static const List<_Product> _products = [
    _Product(name: 'Lemonade', costPrice: 5, suggestedPrice: 10),
    _Product(name: 'Cookies', costPrice: 8, suggestedPrice: 15),
    _Product(name: 'Toy Cars', costPrice: 15, suggestedPrice: 25),
    _Product(name: 'Friendship Bracelets', costPrice: 3, suggestedPrice: 8),
    _Product(name: 'Stickers', costPrice: 2, suggestedPrice: 5),
  ];

  static const List<_MarketEvent> _events = [
    _MarketEvent(description: 'It is a sunny day! More customers are out shopping!', salesMultiplier: 1.5),
    _MarketEvent(description: 'It is raining! Fewer customers today.', salesMultiplier: 0.5),
    _MarketEvent(description: 'Today is festival day! Lots of customers!', salesMultiplier: 2.0),
    _MarketEvent(description: 'A new competitor opened next door! Some customers go there instead.', salesMultiplier: 0.7),
    _MarketEvent(description: 'Your supplier raised prices! Each item costs 3 more rupees to make.', salesMultiplier: 1.0),
    _MarketEvent(description: 'A famous person recommended your shop! Customers are lining up!', salesMultiplier: 1.8),
    _MarketEvent(description: 'Normal day at the market. Steady customers.', salesMultiplier: 1.0),
    _MarketEvent(description: 'School holidays! Kids are out shopping with their parents!', salesMultiplier: 1.6),
  ];

  MarketDay();

  @override
  String get id => 'entrepreneurial_market_day';

  @override
  String get name => 'Market Day';

  @override
  String get category => 'entrepreneurial';

  @override
  String get description =>
      'Run your own virtual shop! Set prices, manage stock, and make a profit.';

  @override
  List<String> get skills => [
        'pricing',
        'supply and demand',
        'profit calculation',
        'entrepreneurial thinking',
      ];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.entrepreneurial;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'market day',
          'shop game',
          'run a shop',
          'selling game',
          'market simulation',
        ],
        'hi': [
          'बाज़ार का दिन',
          'दुकान का खेल',
          'बेचने का खेल',
        ],
        'te': [
          'మార్కెట్ రోజు',
          'దుకాణం ఆట',
          'అమ్మకం ఆట',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Map<String, dynamic> get simulationState => _state;

  @override
  Future<void> saveSimulationState() async {
    // State is ephemeral for this simulation
  }

  @override
  Future<void> loadSimulationState() async {
    // Nothing to load
  }

  @override
  String get progressSummary {
    final totalProfit = _state['totalProfit'] ?? 0;
    return 'Round $_round of $_maxRounds. Total profit: $totalProfit rupees.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _round = 0;
    _phase = 0;
    _state = {
      'totalRevenue': 0,
      'totalExpenses': 0,
      'totalProfit': 0,
      'roundResults': <String>[],
    };

    final productNames = _products.map((p) => p.name).join(', ');
    return 'Welcome to Market Day! You get to run your own shop! '
        'First, choose what you want to sell. Your options are: $productNames. '
        'What would you like to sell?';
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
        return _processProductChoice(lower);

      case 1:
        return _processPrice(lower);

      case 2:
        // After seeing results, continue to next round
        if (_round >= _maxRounds) {
          return await end();
        }
        _phase = 1;
        _stock = 20;
        return 'Round ${_round + 1}! You have 20 ${_chosenProduct!.name} to sell. '
            'Each one costs you ${_chosenProduct!.costPrice} rupees to make. '
            'What price will you sell each one for?';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    final totalProfit = _state['totalProfit'] as int;
    final totalRevenue = _state['totalRevenue'] as int;

    if (_round == 0) {
      return 'Come back to Market Day anytime to run your own business!';
    }

    String verdict;
    if (totalProfit > 100) {
      verdict = 'You are a business genius! Amazing profits!';
    } else if (totalProfit > 50) {
      verdict = 'Great business skills! You made a solid profit!';
    } else if (totalProfit > 0) {
      verdict = 'You made a profit! Every business starts small.';
    } else {
      verdict = 'You lost money this time, but every entrepreneur learns from mistakes!';
    }

    return 'Market Day is over! You played $_round rounds. '
        'Total revenue: $totalRevenue rupees. '
        'Total profit: $totalProfit rupees. '
        '$verdict';
  }

  String _processProductChoice(String lower) {
    _Product? chosen;
    for (final p in _products) {
      if (lower.contains(p.name.toLowerCase())) {
        chosen = p;
        break;
      }
    }

    if (chosen == null) {
      // Try to match partial
      for (final p in _products) {
        for (final word in p.name.toLowerCase().split(' ')) {
          if (lower.contains(word) && word.length > 2) {
            chosen = p;
            break;
          }
        }
        if (chosen != null) break;
      }
    }

    if (chosen == null) {
      chosen = _products[_rng.nextInt(_products.length)];
    }

    _chosenProduct = chosen;
    _phase = 1;
    _stock = 20;

    return 'Great choice! You are selling ${chosen.name}! '
        'Each one costs you ${chosen.costPrice} rupees to make. '
        'You have 20 in stock. What price do you want to sell each one for? '
        'Hint: if you price it too high, fewer people will buy. Too low, '
        'and you will not make a profit!';
  }

  String _processPrice(String lower) {
    // Extract a number from the response
    final numberMatch = RegExp(r'\d+').firstMatch(lower);
    if (numberMatch != null) {
      _currentPrice = int.parse(numberMatch.group(0)!);
    } else {
      _currentPrice = _chosenProduct!.suggestedPrice;
    }

    // Clamp to reasonable range
    _currentPrice = _currentPrice.clamp(1, 100);

    return _simulateRound();
  }

  String _simulateRound() {
    _round++;
    final product = _chosenProduct!;
    final event = _events[_rng.nextInt(_events.length)];

    // Calculate customers based on price
    double priceRatio = product.suggestedPrice / _currentPrice;
    int baseCustomers = (10 * priceRatio).round().clamp(2, 20);

    // Apply event multiplier
    int actualCustomers = (baseCustomers * event.salesMultiplier).round();
    actualCustomers = actualCustomers.clamp(0, _stock);

    // Handle supplier price increase event
    int costPrice = product.costPrice;
    if (event.description.contains('supplier raised prices')) {
      costPrice += 3;
    }

    int revenue = actualCustomers * _currentPrice;
    int expenses = _stock * costPrice;
    int profit = revenue - expenses;

    _state['totalRevenue'] = (_state['totalRevenue'] as int) + revenue;
    _state['totalExpenses'] = (_state['totalExpenses'] as int) + expenses;
    _state['totalProfit'] = (_state['totalProfit'] as int) + profit;

    _phase = 2;

    final profitText = profit >= 0
        ? 'You made a profit of $profit rupees!'
        : 'You lost ${profit.abs()} rupees this round.';

    String nextText;
    if (_round >= _maxRounds) {
      nextText = 'That was the last round! Say anything to see your final results.';
    } else {
      nextText = '${_maxRounds - _round} rounds left! Ready for the next round?';
    }

    return 'Round $_round results! ${event.description} '
        'You sold $actualCustomers ${product.name} at $_currentPrice rupees each. '
        'Revenue: $revenue rupees. Expenses: $expenses rupees. '
        '$profitText $nextText';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
