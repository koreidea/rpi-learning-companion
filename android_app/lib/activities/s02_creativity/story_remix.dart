import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../core/llm/llm_router.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Story Remix: classic stories with a twist where the child decides what
/// happens next.
///
/// The bot starts a well-known fairy tale but introduces a twist midway.
/// The child takes over the narrative, and the LLM continues building on
/// their ideas. After 4-5 turns the bot wraps up with a fun conclusion.
class StoryRemix extends Activity {
  final LlmRouter _llmRouter;
  final Random _random = Random();

  bool _active = false;
  int _childTurns = 0;
  static const int _maxTurns = 5;
  int _storyIndex = 0;

  final List<String> _storyParts = [];
  final List<int> _usedStoryIndices = [];

  StoryRemix({required LlmRouter llmRouter}) : _llmRouter = llmRouter;

  @override
  String get id => 'creativity_story_remix';

  @override
  String get name => 'Story Remix';

  @override
  String get category => 'creativity';

  @override
  String get description =>
      'Classic stories with a twist! You decide what happens next.';

  @override
  List<String> get skills =>
      ['creativity', 'narrative thinking', 'imagination'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 10;

  @override
  SkillId? get skillId => SkillId.creativity;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'story remix',
          'remix a story',
          'change a story',
          'twist story',
          'different ending',
        ],
        'hi': ['कहानी बदलो', 'नई कहानी', 'कहानी रीमिक्स'],
        'te': ['కథ మార్చు', 'కొత్త కథ', 'కథ రీమిక్స్'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _childTurns = 0;
    _storyParts.clear();

    // Pick an unused story
    if (_usedStoryIndices.length >= _stories.length) {
      _usedStoryIndices.clear();
    }
    int index;
    do {
      index = _random.nextInt(_stories.length);
    } while (_usedStoryIndices.contains(index));
    _usedStoryIndices.add(index);
    _storyIndex = index;

    final story = _stories[_storyIndex];
    _storyParts.add(story.opening);
    _storyParts.add(story.twist);

    debugPrint('[StoryRemix] Starting: ${story.title}');

    return "Let's remix a story! You know ${story.title}, right? "
        "${story.opening} But here is the twist! ${story.twist} "
        "What do you think happens next?";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    _storyParts.add(childSaid);
    _childTurns++;

    if (_childTurns >= _maxTurns) {
      return await _wrapUpStory(childSaid);
    }

    try {
      final continuation = await _generateContinuation(childSaid);
      _storyParts.add(continuation);
      return "$continuation What happens next in your version?";
    } catch (e) {
      debugPrint('[StoryRemix] LLM error: $e');
      return _fallbackContinuation();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_childTurns == 0) return 'No story remixed yet.';
    return 'Remixed ${_stories[_storyIndex].title} with $_childTurns twists.';
  }

  // -- LLM generation --

  Future<String> _generateContinuation(String childSaid) async {
    final provider = _llmRouter.getProvider();
    final story = _stories[_storyIndex];
    final storySoFar = _storyParts.join(' ');

    final prompt =
        'You are co-creating a remixed version of "${story.title}" with a '
        'child. The original twist was: "${story.twist}" '
        'Story so far: "$storySoFar" '
        'The child just added: "$childSaid" '
        'Continue the remixed story in 1-2 sentences. Build on the child\'s '
        'ideas. Keep it fun, magical, and surprising. '
        'Do not use emojis or markdown. Speak naturally.';

    final messages = [
      {'role': 'system', 'content': prompt},
      {'role': 'user', 'content': childSaid},
    ];

    final buffer = StringBuffer();
    await for (final token in provider.stream(messages)) {
      buffer.write(token);
    }

    final result = buffer.toString().trim();
    return result.isNotEmpty ? result : _fallbackContinuation();
  }

  Future<String> _wrapUpStory(String childSaid) async {
    _active = false;

    try {
      final provider = _llmRouter.getProvider();
      final story = _stories[_storyIndex];
      final storySoFar = _storyParts.join(' ');

      final prompt =
          'You are wrapping up a remixed version of "${story.title}" with a '
          'child. Story so far: "$storySoFar" '
          'The child just added: "$childSaid" '
          'Wrap up with a happy, satisfying ending in 1-2 sentences. '
          'Do not use emojis or markdown.';

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
        return "$ending And that is how YOUR version of "
            "${_stories[_storyIndex].title} ends! What a creative storyteller "
            "you are! ${_buildEndSummary()}";
      }
    } catch (e) {
      debugPrint('[StoryRemix] Wrap-up error: $e');
    }

    return "And they all lived happily ever after in YOUR version of the story! "
        "What a creative storyteller you are! ${_buildEndSummary()}";
  }

  String _fallbackContinuation() {
    const fallbacks = [
      'Oh wow, that changes everything! And then something unexpected happened.',
      'That is so different from the original! Everyone was surprised.',
      'What a twist! The other characters could not believe it.',
      'That is amazing! And it made the whole story even more exciting.',
    ];
    final text = fallbacks[_random.nextInt(fallbacks.length)];
    _storyParts.add(text);
    return "$text What happens next in your version?";
  }

  String _buildEndSummary() {
    if (_childTurns == 0) {
      return "Thanks for trying Story Remix! Come back anytime to remix a classic story!";
    }
    return "You remixed ${_stories[_storyIndex].title} with $_childTurns "
        "awesome twists! Your version was even better than the original!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<_RemixStory> _stories = [
    _RemixStory(
      title: 'The Three Little Pigs',
      opening:
          'Once upon a time, three little pigs set out to build their own houses.',
      twist:
          'But in this version, there are no bricks, sticks, or straw anywhere! '
          'The pigs have to find completely new materials to build with.',
    ),
    _RemixStory(
      title: 'Goldilocks and the Three Bears',
      opening:
          'One day, Goldilocks wandered into a cottage in the forest and found three bowls of porridge.',
      twist:
          'But in this version, the bears come home and they are super friendly! '
          'They invite Goldilocks to stay for dinner and a sleepover.',
    ),
    _RemixStory(
      title: 'Jack and the Beanstalk',
      opening:
          'Jack planted magic beans and a huge beanstalk grew all the way to the clouds.',
      twist:
          'But in this version, the giant at the top is actually really nice! '
          'He is lonely and just wants a friend to play with.',
    ),
    _RemixStory(
      title: 'Cinderella',
      opening:
          'Cinderella lived with her stepmother and stepsisters. One day, an invitation to the royal ball arrived.',
      twist:
          'But in this version, Cinderella does not want to go to the ball at all! '
          'She has a much more exciting adventure planned.',
    ),
    _RemixStory(
      title: 'The Tortoise and the Hare',
      opening:
          'The hare challenged the tortoise to a race, sure that he would win easily.',
      twist:
          'But in this version, instead of racing against each other, they decide to '
          'work together as a team to beat a much bigger challenge!',
    ),
    _RemixStory(
      title: 'Little Red Riding Hood',
      opening:
          'Little Red Riding Hood set off through the forest to visit her grandmother.',
      twist:
          'But in this version, the wolf is actually scared of the forest! '
          'He asks Red Riding Hood for help finding his way home.',
    ),
  ];
}

/// A classic story with an opening and a creative twist.
class _RemixStory {
  final String title;
  final String opening;
  final String twist;

  const _RemixStory({
    required this.title,
    required this.opening,
    required this.twist,
  });
}
