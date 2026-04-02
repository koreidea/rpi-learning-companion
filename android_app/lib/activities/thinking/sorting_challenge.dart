import 'dart:math';

import '../activity_base.dart';

/// Sorting and classification game with progressive category rounds.
///
/// Each round presents a binary category (big/small, alive/not alive, etc.).
/// The bot names items one at a time and the child sorts them. Rounds get
/// progressively harder, moving from concrete attributes to more abstract ones.
class SortingChallenge extends Activity {
  final Random _random = Random();

  bool _active = false;
  int _totalCorrect = 0;
  int _totalAsked = 0;

  // Round tracking
  int _currentRoundIndex = 0;
  int _itemIndex = 0;
  _SortingRound? _currentRound;
  bool _waitingForReady = false;
  bool _waitingForAnswer = false;
  bool _waitingForNextRound = false;
  int _roundCorrect = 0;

  @override
  String get id => 'thinking_sorting';

  @override
  String get name => 'Sorting Challenge';

  @override
  String get category => 'thinking';

  @override
  String get description =>
      'Sort items into categories like big or small, alive or not alive.';

  @override
  List<String> get skills => ['classification', 'critical thinking', 'vocabulary'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _totalCorrect = 0;
    _totalAsked = 0;
    _currentRoundIndex = 0;
    _itemIndex = 0;
    _roundCorrect = 0;
    _waitingForReady = true;
    _waitingForAnswer = false;
    _waitingForNextRound = false;

    _currentRound = _rounds[0];

    return "Let's play the Sorting Challenge! I will name something and you "
        "tell me which group it belongs to. "
        "First up: is it ${_currentRound!.categoryA} or ${_currentRound!.categoryB}? "
        "Ready?";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    if (_waitingForReady) {
      _waitingForReady = false;
      return _presentNextItem();
    }

    if (_waitingForNextRound) {
      _waitingForNextRound = false;
      if (_containsNo(lower)) {
        _active = false;
        return _buildEndSummary();
      }
      return _startNextRound();
    }

    if (_waitingForAnswer && _currentRound != null) {
      return _processAnswer(lower);
    }

    return "Which group does it belong to?";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_totalAsked == 0) return 'No items sorted yet.';
    return 'Sorted $_totalCorrect out of $_totalAsked items correctly.';
  }

  // -- Internal --

  String _presentNextItem() {
    final round = _currentRound!;
    final item = round.items[_itemIndex];
    _waitingForAnswer = true;

    return "Is ${item.article}${item.name} ${round.categoryA} or ${round.categoryB}?";
  }

  String _processAnswer(String answer) {
    final round = _currentRound!;
    final item = round.items[_itemIndex];
    _waitingForAnswer = false;
    _totalAsked++;

    final isCorrect = _checkAnswer(answer, item.category, round);

    if (isCorrect) {
      _totalCorrect++;
      _roundCorrect++;
    }

    _itemIndex++;

    // Check if round is complete
    if (_itemIndex >= round.items.length) {
      // Summarize round
      final roundSummary = "You got $_roundCorrect out of ${round.items.length} in "
          "the ${round.categoryA} or ${round.categoryB} round!";

      _currentRoundIndex++;

      // Check if all rounds are done
      if (_currentRoundIndex >= _rounds.length) {
        _active = false;
        final feedback = isCorrect
            ? "Yes! ${item.nameWithArticle} is ${item.category}! ${item.reason}."
            : "Actually, ${item.nameWithArticle} is ${item.category}. ${item.reason}.";
        return "$feedback $roundSummary ${_buildEndSummary()}";
      }

      _waitingForNextRound = true;
      _roundCorrect = 0;
      _itemIndex = 0;
      _currentRound = _rounds[_currentRoundIndex];

      final feedback = isCorrect
          ? "Yes! ${item.nameWithArticle} is ${item.category}! ${item.reason}."
          : "Actually, ${item.nameWithArticle} is ${item.category}. ${item.reason}.";

      return "$feedback $roundSummary "
          "Now let's try a new one. Is it ${_currentRound!.categoryA} "
          "or ${_currentRound!.categoryB}? Ready?";
    }

    // More items in this round
    final feedback = isCorrect
        ? _correctFeedback(item)
        : _incorrectFeedback(item);

    _waitingForAnswer = true;
    final nextItem = round.items[_itemIndex];
    return "$feedback Next one! Is ${nextItem.article}${nextItem.name} "
        "${round.categoryA} or ${round.categoryB}?";
  }

  bool _checkAnswer(String answer, String correctCategory, _SortingRound round) {
    final a = answer.toLowerCase().trim();
    final catA = round.categoryA.toLowerCase();
    final catB = round.categoryB.toLowerCase();

    if (correctCategory.toLowerCase() == catA) {
      return a.contains(catA) && !a.contains(catB);
    } else {
      return a.contains(catB) && !a.contains(catA);
    }
  }

  String _correctFeedback(_SortItem item) {
    const praises = [
      'That is right!',
      'Yes!',
      'Correct!',
      'You got it!',
      'Exactly right!',
    ];
    final praise = praises[_random.nextInt(praises.length)];
    return "$praise ${item.nameWithArticle} is ${item.category}! ${item.reason}.";
  }

  String _incorrectFeedback(_SortItem item) {
    return "Hmm, actually ${item.nameWithArticle} is ${item.category}. "
        "${item.reason}. But good try!";
  }

  String? _startNextRound() {
    return _presentNextItem();
  }

  String _buildEndSummary() {
    if (_totalAsked == 0) {
      return "Thanks for playing the Sorting Challenge! Come back anytime!";
    }
    return "You sorted $_totalCorrect out of $_totalAsked items correctly! "
        "${_totalCorrect == _totalAsked ? "Perfect score! Amazing job!" : "Great work! You are really learning to sort things!"}";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }

  bool _containsNo(String text) {
    const noWords = ['no', 'nah', 'nope', "don't want", 'not now'];
    for (final w in noWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }

  // -- Round data --

  static final List<_SortingRound> _rounds = [
    _SortingRound(
      categoryA: 'big',
      categoryB: 'small',
      items: [
        _SortItem(name: 'elephant', article: 'an ', category: 'big', reason: 'Elephants are huge, one of the biggest animals on land'),
        _SortItem(name: 'ant', article: 'an ', category: 'small', reason: 'Ants are tiny, you can barely see them'),
        _SortItem(name: 'mountain', article: 'a ', category: 'big', reason: 'Mountains are enormous, they reach up to the sky'),
        _SortItem(name: 'button', article: 'a ', category: 'small', reason: 'Buttons are little, they fit on your shirt'),
        _SortItem(name: 'whale', article: 'a ', category: 'big', reason: 'Whales are the biggest animals in the whole ocean'),
        _SortItem(name: 'seed', article: 'a ', category: 'small', reason: 'Seeds are so tiny they fit in your hand'),
      ],
    ),
    _SortingRound(
      categoryA: 'alive',
      categoryB: 'not alive',
      items: [
        _SortItem(name: 'dog', article: 'a ', category: 'alive', reason: 'Dogs breathe, eat, and grow'),
        _SortItem(name: 'rock', article: 'a ', category: 'not alive', reason: 'Rocks do not eat or grow'),
        _SortItem(name: 'flower', article: 'a ', category: 'alive', reason: 'Flowers grow from seeds and need water and sun'),
        _SortItem(name: 'car', article: 'a ', category: 'not alive', reason: 'Cars are machines, they do not eat or grow'),
        _SortItem(name: 'bird', article: 'a ', category: 'alive', reason: 'Birds are animals, they breathe and fly'),
        _SortItem(name: 'table', article: 'a ', category: 'not alive', reason: 'Tables are made of wood but they do not grow anymore'),
      ],
    ),
    _SortingRound(
      categoryA: 'hot',
      categoryB: 'cold',
      items: [
        _SortItem(name: 'fire', article: '', category: 'hot', reason: 'Fire is very very hot, never touch it'),
        _SortItem(name: 'ice cream', article: '', category: 'cold', reason: 'Ice cream is frozen and yummy'),
        _SortItem(name: 'sun', article: 'the ', category: 'hot', reason: 'The sun is a giant ball of fire in the sky'),
        _SortItem(name: 'snow', article: '', category: 'cold', reason: 'Snow is frozen water that falls from the sky'),
        _SortItem(name: 'tea', article: '', category: 'hot', reason: 'Tea is a warm drink that grown-ups like'),
        _SortItem(name: 'penguin', article: 'a ', category: 'cold', reason: 'Penguins live where it is very cold with ice everywhere'),
      ],
    ),
    _SortingRound(
      categoryA: 'can fly',
      categoryB: 'cannot fly',
      items: [
        _SortItem(name: 'bird', article: 'a ', category: 'can fly', reason: 'Birds have wings and fly through the sky'),
        _SortItem(name: 'elephant', article: 'an ', category: 'cannot fly', reason: 'Elephants are too heavy to fly'),
        _SortItem(name: 'butterfly', article: 'a ', category: 'can fly', reason: 'Butterflies have beautiful wings to fly'),
        _SortItem(name: 'fish', article: 'a ', category: 'cannot fly', reason: 'Fish swim in water but they cannot fly'),
        _SortItem(name: 'airplane', article: 'an ', category: 'can fly', reason: 'Airplanes have big wings and engines to fly'),
        _SortItem(name: 'cat', article: 'a ', category: 'cannot fly', reason: 'Cats are great jumpers but they cannot fly'),
      ],
    ),
  ];
}

class _SortingRound {
  final String categoryA;
  final String categoryB;
  final List<_SortItem> items;

  const _SortingRound({
    required this.categoryA,
    required this.categoryB,
    required this.items,
  });
}

class _SortItem {
  final String name;
  final String article;
  final String category;
  final String reason;

  const _SortItem({
    required this.name,
    required this.article,
    required this.category,
    required this.reason,
  });

  String get nameWithArticle => '$article$name';
}
