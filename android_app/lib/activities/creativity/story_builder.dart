import 'dart:async';

import 'package:flutter/foundation.dart';

import '../../core/llm/llm_router.dart';
import '../activity_base.dart';

/// Collaborative Story Builder: bot and child take turns building a story.
///
/// Teaches: creativity, narrative thinking, imagination, verbal expression.
///
/// Flow:
/// 1. Bot starts with a story opening.
/// 2. Child adds to the story.
/// 3. Bot continues using LLM based on what child said.
/// 4. Back and forth for 5-6 turns each.
/// 5. Bot wraps up with a happy ending.
class StoryBuilder extends Activity {
  final LlmRouter _llmRouter;

  bool _active = false;
  int _childTurns = 0;
  int _botTurns = 0;
  int _score = 0;
  static const int _maxTurnsEach = 6;

  /// Rolling story context — what has happened so far.
  final List<String> _storyParts = [];

  /// Pre-defined story openings to keep things fresh.
  static const List<String> _openings = [
    'Once upon a time, there was a little rabbit who found a magic door in a big oak tree.',
    'One sunny morning, a brave little kitten discovered a rainbow bridge in the garden.',
    'In a land made of candy, there lived a tiny dragon who loved to sing.',
    'Deep in the forest, a friendly bear found a shiny golden key on the path.',
    'On a cloud high in the sky, there was a little star who wanted to visit the Earth.',
    'Under the sea, a baby dolphin found a treasure chest filled with colorful shells.',
  ];

  int _openingIndex = 0;

  StoryBuilder({required LlmRouter llmRouter})
      : _llmRouter = llmRouter;

  // -- Activity metadata --

  @override
  String get id => 'creativity_story';

  @override
  String get name => 'Story Builder';

  @override
  String get category => 'creativity';

  @override
  String get description =>
      'Build a story together, taking turns to add to the adventure.';

  @override
  List<String> get skills => ['creativity', 'narrative', 'imagination', 'verbal expression'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_storyParts.isEmpty) return 'No story yet.';
    return '$_childTurns turns taken. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _storyParts.clear();
    _childTurns = 0;
    _botTurns = 0;
    _score = 0;
    _active = true;

    // Rotate through openings
    final opening = _openings[_openingIndex % _openings.length];
    _openingIndex++;
    _storyParts.add(opening);
    _botTurns++;

    debugPrint('[StoryBuilder] Started with opening: ${opening.substring(0, 40)}...');

    return "Let's make a story together! I'll start, then you add the next part. "
        "$opening What happens next?";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    // Check for quit
    if (_isQuitTrigger(lower)) {
      return await end();
    }

    // Record child's contribution
    _storyParts.add(childSaid);
    _childTurns++;
    _score += 10; // Points for each contribution

    // Check if we should wrap up
    if (_childTurns >= _maxTurnsEach) {
      return await _wrapUpStory(childSaid);
    }

    // Use LLM to continue the story
    try {
      final continuation = await _generateContinuation(childSaid);
      _storyParts.add(continuation);
      _botTurns++;

      // Check if bot has also hit max turns — wrap up next time
      if (_botTurns >= _maxTurnsEach) {
        _active = false;
        _score += 20; // Bonus for completing the whole story
        return "$continuation And they all lived happily ever after! "
            "What a wonderful story we made together! You are such a great storyteller!";
      }

      return "$continuation What happens next?";
    } catch (e) {
      debugPrint('[StoryBuilder] LLM error: $e');
      // Fallback without LLM
      return _fallbackContinuation(childSaid);
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    final turns = _childTurns;
    debugPrint('[StoryBuilder] Ended after $turns child turns, score=$_score');

    if (_storyParts.isEmpty) {
      return "Okay, we'll tell a story another time!";
    }

    _score += turns * 5;
    return "What a great story we made! You added $turns parts all by yourself. "
        "You are an amazing storyteller!";
  }

  // -- LLM story generation --

  Future<String> _generateContinuation(String childSaid) async {
    final provider = _llmRouter.getProvider();

    final storySoFar = _storyParts.join(' ');
    final prompt =
        'You are co-creating a story with a 3-6 year old child. '
        'Here is the story so far: "$storySoFar" '
        'The child just added: "$childSaid" '
        'Continue the story based on what the child said. '
        'Keep your addition to 1-2 sentences. Make it fun, magical, and age-appropriate. '
        'Do not use emojis. Do not use markdown. Speak naturally as if talking to a small child. '
        '${_botTurns >= _maxTurnsEach - 2 ? "Start guiding the story toward a happy ending." : ""}';

    final messages = [
      {'role': 'system', 'content': prompt},
      {'role': 'user', 'content': childSaid},
    ];

    final buffer = StringBuffer();
    await for (final token in provider.stream(messages)) {
      buffer.write(token);
    }

    final result = buffer.toString().trim();
    if (result.isEmpty) {
      return _fallbackContinuation(childSaid);
    }
    return result;
  }

  String _fallbackContinuation(String childSaid) {
    _botTurns++;
    // Simple fallback continuations that don't need LLM
    const fallbacks = [
      'Oh wow, that is so exciting! And then something magical happened.',
      'That is amazing! Everyone was so happy about that.',
      'What a wonderful idea! And suddenly, they heard a beautiful sound.',
      'How fun! Then a friendly butterfly flew by and waved hello.',
      'That is so cool! And a little bird started singing a happy song.',
    ];
    final text = fallbacks[_botTurns % fallbacks.length];
    _storyParts.add(text);
    return "$text What happens next?";
  }

  Future<String> _wrapUpStory(String childSaid) async {
    _active = false;
    _score += 20; // Bonus for completing

    try {
      final provider = _llmRouter.getProvider();
      final storySoFar = _storyParts.join(' ');
      final prompt =
          'You are co-creating a story with a 3-6 year old child. '
          'Here is the story so far: "$storySoFar" '
          'The child just added: "$childSaid" '
          'Now wrap up the story with a happy ending in 1-2 sentences. '
          'Make it satisfying and joyful. Do not use emojis or markdown.';

      final messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': childSaid},
      ];

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }

      final ending = buffer.toString().trim();
      if (ending.isNotEmpty) {
        return "$ending What a wonderful story we made together! "
            "You are such a creative storyteller!";
      }
    } catch (e) {
      debugPrint('[StoryBuilder] Wrap-up LLM error: $e');
    }

    return "And they all lived happily ever after! "
        "What a wonderful story we made together! You are such a creative storyteller!";
  }

  // -- Helpers --

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop the game', 'quit', 'exit', "i'm done", 'no more',
      'stop playing', 'end the game', 'finish', 'stop the story',
      'end the story',
      // Hindi
      'बंद करो', 'खेल बंद',
      // Telugu
      'ఆపు', 'ఆట ఆపు',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
