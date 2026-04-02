import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../core/llm/llm_router.dart';
import '../activity_base.dart';

/// Topic for the invention game — what the child will invent.
class _InventionTopic {
  final String name;
  final String prompt;
  final List<String> followUpHints;

  const _InventionTopic({
    required this.name,
    required this.prompt,
    required this.followUpHints,
  });
}

/// Imagination Invention: child invents something imaginary, bot asks
/// follow-up questions about it.
///
/// Teaches: creativity, imagination, descriptive language, confidence.
///
/// Flow:
/// 1. Bot picks a topic and asks the child to invent something.
/// 2. Child describes their invention.
/// 3. Bot asks follow-up questions (name, abilities, food, etc.).
/// 4. Bot summarizes the invention enthusiastically.
class InventionTime extends Activity {
  final LlmRouter _llmRouter;
  final Random _rng = Random();

  bool _active = false;
  int _turnCount = 0;
  int _score = 0;
  static const int _maxTurns = 5;

  _InventionTopic? _currentTopic;
  final List<String> _childAnswers = [];
  String _inventionName = '';

  static const List<_InventionTopic> _topics = [
    _InventionTopic(
      name: 'a new animal',
      prompt: "Let's invent something! If you could make a brand new animal, what would it look like? "
          "Does it have wings? Wheels? A long tail? What color is it?",
      followUpHints: ['name', 'food', 'sound', 'home', 'superpower'],
    ),
    _InventionTopic(
      name: 'a new vehicle',
      prompt: "Let's invent something! If you could build any vehicle, what would it be? "
          "Does it fly? Does it go underwater? What color is it?",
      followUpHints: ['name', 'fuel', 'speed', 'passengers', 'special feature'],
    ),
    _InventionTopic(
      name: 'a new food',
      prompt: "Let's invent something! If you could create a brand new food, what would it taste like? "
          "Is it sweet? Salty? Crunchy? What does it look like?",
      followUpHints: ['name', 'color', 'when to eat', 'ingredient', 'who loves it'],
    ),
    _InventionTopic(
      name: 'a new toy',
      prompt: "Let's invent something! If you could make the coolest toy ever, what would it do? "
          "Does it move? Does it talk? Does it glow?",
      followUpHints: ['name', 'material', 'size', 'who plays with it', 'special trick'],
    ),
    _InventionTopic(
      name: 'a new planet',
      prompt: "Let's invent something! If you could create a new planet, what would it be like? "
          "What color is the sky? What grows there? Who lives there?",
      followUpHints: ['name', 'weather', 'creatures', 'food', 'fun activity'],
    ),
    _InventionTopic(
      name: 'a superhero',
      prompt: "Let's invent something! If you could create a superhero, what powers would they have? "
          "Can they fly? Are they super strong? What do they wear?",
      followUpHints: ['name', 'costume color', 'weakness', 'best friend', 'villain they fight'],
    ),
  ];

  InventionTime({required LlmRouter llmRouter})
      : _llmRouter = llmRouter;

  // -- Activity metadata --

  @override
  String get id => 'creativity_invention';

  @override
  String get name => 'Invention Time';

  @override
  String get category => 'creativity';

  @override
  String get description =>
      'Invent something imaginary and describe it to the bot.';

  @override
  List<String> get skills => ['creativity', 'imagination', 'descriptive language', 'confidence'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_childAnswers.isEmpty) return 'No invention yet.';
    return '$_turnCount details shared. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _childAnswers.clear();
    _inventionName = '';
    _turnCount = 0;
    _score = 0;
    _active = true;

    _currentTopic = _topics[_rng.nextInt(_topics.length)];
    debugPrint('[InventionTime] Started with topic: ${_currentTopic!.name}');

    return _currentTopic!.prompt;
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    _childAnswers.add(childSaid);
    _turnCount++;
    _score += 10;

    // After enough turns, wrap up
    if (_turnCount >= _maxTurns) {
      return await _wrapUp();
    }

    // Second turn: ask for a name
    if (_turnCount == 1) {
      return await _generateFollowUp(childSaid, askForName: true);
    }

    // If turn 2, capture the name
    if (_turnCount == 2) {
      _inventionName = _extractName(childSaid);
    }

    // Generate a natural follow-up question
    return await _generateFollowUp(childSaid);
  }

  @override
  Future<String> end() async {
    _active = false;
    final turns = _turnCount;
    debugPrint('[InventionTime] Ended after $turns turns, score=$_score');

    if (_childAnswers.isEmpty) {
      return "Okay, we'll invent something another time! You have such a great imagination!";
    }

    _score += 10;
    return "What a wonderful invention! You are such a creative inventor!";
  }

  // -- LLM follow-up generation --

  Future<String> _generateFollowUp(String childSaid, {bool askForName = false}) async {
    try {
      final provider = _llmRouter.getProvider();
      final context = _childAnswers.join('. ');
      final nameRef = _inventionName.isNotEmpty ? _inventionName : 'their invention';

      String instruction;
      if (askForName) {
        instruction =
            'The child is inventing ${_currentTopic!.name}. '
            'They just described it as: "$childSaid". '
            'React with excitement to what they said (1 short sentence), then ask: what would you name it? '
            'Do not use emojis or markdown. Speak naturally to a 3-6 year old.';
      } else {
        final hintIndex = (_turnCount - 1).clamp(0, _currentTopic!.followUpHints.length - 1);
        final hint = _currentTopic!.followUpHints[hintIndex];
        instruction =
            'The child is inventing ${_currentTopic!.name} called "$nameRef". '
            'Here is what they have described so far: "$context". '
            'They just said: "$childSaid". '
            'React with excitement (1 short sentence), then ask a fun follow-up question about: $hint. '
            'Do not use emojis or markdown. Speak naturally to a 3-6 year old.';
      }

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
      debugPrint('[InventionTime] LLM error: $e');
    }

    // Fallback
    if (askForName) {
      return "That sounds amazing! What would you name it?";
    }
    const fallbacks = [
      "Wow, that is so cool! What else can it do?",
      "That is awesome! What is its favorite thing?",
      "I love that! Tell me more about it!",
      "How wonderful! What makes it special?",
    ];
    return fallbacks[_turnCount % fallbacks.length];
  }

  Future<String> _wrapUp() async {
    _active = false;
    _score += 20;

    try {
      final provider = _llmRouter.getProvider();
      final context = _childAnswers.join('. ');
      final nameRef = _inventionName.isNotEmpty ? _inventionName : 'their invention';

      final instruction =
          'The child invented ${_currentTopic!.name} called "$nameRef". '
          'Here is everything they described: "$context". '
          'Give a fun, enthusiastic 2-sentence summary of their invention. '
          'End by saying they are an amazing inventor. '
          'Do not use emojis or markdown. Speak naturally to a 3-6 year old.';

      final messages = [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': 'Summarize my invention'},
      ];

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }

      final result = buffer.toString().trim();
      if (result.isNotEmpty) return result;
    } catch (e) {
      debugPrint('[InventionTime] Wrap-up LLM error: $e');
    }

    final nameRef = _inventionName.isNotEmpty ? _inventionName : 'your invention';
    return "What a wonderful creation! $nameRef is absolutely amazing! "
        "You are such a creative inventor!";
  }

  /// Try to extract a short name from the child's response.
  String _extractName(String text) {
    // Take the first few meaningful words as the name
    final words = text
        .replaceAll(RegExp(r'[^\w\s]'), '')
        .split(RegExp(r'\s+'))
        .where((w) => w.length > 1)
        .toList();

    if (words.isEmpty) return '';
    // Use up to 3 words
    return words.take(3).join(' ');
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
