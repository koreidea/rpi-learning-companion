import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_provider.dart';
import '../../core/llm/llm_router.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Callback type for streaming TTS -- receives a complete sentence to speak.
typedef SpeakSentenceCallback = Future<void> Function(String sentence);

/// Emotion Detective: stories with emotional pauses where the child
/// identifies and explores characters' feelings.
///
/// The bot tells a short story, pauses at an emotional moment, and asks
/// the child to identify the character's feelings. The LLM then explores
/// deeper emotional understanding through follow-up questions, connecting
/// the fictional situation to the child's own experiences.
class EmotionDetective extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  final Random _random = Random();

  bool _active = false;
  int _storiesCompleted = 0;
  static const int _maxStories = 3;
  int _followUp = 0;
  static const int _maxFollowUps = 2;
  _EmotionStory? _currentStory;
  bool _waitingForReady = false;
  bool _waitingForPlayAgain = false;

  final List<Map<String, String>> _conversationHistory = [];
  final List<int> _usedStoryIndices = [];

  EmotionDetective({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'emotional_intelligence_emotion_detective';

  @override
  String get name => 'Emotion Detective';

  @override
  String get category => 'emotions';

  @override
  String get description =>
      'Read emotions in stories and explore how characters feel.';

  @override
  List<String> get skills =>
      ['emotional intelligence', 'empathy', 'perspective taking'];

  @override
  int get minAge => 4;

  @override
  int get maxAge => 12;

  @override
  SkillId? get skillId => SkillId.emotionalIntelligence;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'emotion detective',
          'feeling detective',
          'emotion story',
          'how do they feel',
          'feelings game',
        ],
        'hi': ['भावना जासूस', 'भावना कहानी', 'कैसा लग रहा है'],
        'te': ['భావాల డిటెక్టివ్', 'భావాల కథ'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _storiesCompleted = 0;
    _followUp = 0;
    _conversationHistory.clear();
    _waitingForReady = true;
    _waitingForPlayAgain = false;

    return "Welcome to Emotion Detective! I am going to tell you short "
        "stories, and your job is to figure out how the characters feel. "
        "Are you ready to be a feelings detective?";
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
      return _presentNewStory();
    }

    if (_waitingForPlayAgain) {
      _waitingForPlayAgain = false;
      if (_containsNo(lower)) {
        _active = false;
        return _buildEndSummary();
      }
      return _presentNewStory();
    }

    // Process the child's emotion identification
    _followUp++;
    _conversationHistory.add({'role': 'user', 'content': childSaid});

    if (_followUp >= _maxFollowUps) {
      _storiesCompleted++;
      _followUp = 0;
      _conversationHistory.clear();

      final response = await _getLlmResponse(childSaid, isWrapUp: true);

      if (_storiesCompleted >= _maxStories) {
        _active = false;
        return "$response ${_buildEndSummary()}";
      }

      _waitingForPlayAgain = true;
      return "$response Great detective work! Want to hear another story?";
    }

    final response = await _getLlmResponse(childSaid, isWrapUp: false);
    _conversationHistory.add({'role': 'assistant', 'content': response});
    return response;
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_storiesCompleted == 0) return 'No emotion stories explored yet.';
    return 'Explored $_storiesCompleted emotion stories.';
  }

  // -- Internal --

  String _presentNewStory() {
    if (_usedStoryIndices.length >= _stories.length) {
      _usedStoryIndices.clear();
    }
    int index;
    do {
      index = _random.nextInt(_stories.length);
    } while (_usedStoryIndices.contains(index));
    _usedStoryIndices.add(index);

    _currentStory = _stories[index];
    _followUp = 0;
    _conversationHistory.clear();

    return "Here is a story for you. ${_currentStory!.narration} "
        "${_currentStory!.question}";
  }

  Future<String> _getLlmResponse(
    String childSaid, {
    required bool isWrapUp,
  }) async {
    final story = _currentStory!;
    final wrapGuidance = isWrapUp
        ? 'Wrap up by validating the child\'s emotional understanding. '
            'Connect it to their life: "Has something like that happened to you?" '
            'or "You really understand how others feel!" Do not ask more questions.'
        : 'The child identified an emotion. Validate their answer warmly. '
            'Then ask a follow-up that deepens empathy: "Why do you think they '
            'feel that way?" or "What could someone do to help them feel better?"';

    final systemPrompt =
        'You are Buddy, a warm emotional intelligence coach for children. '
        'Story: "${story.narration}" The target emotion is: "${story.emotion}". '
        '$wrapGuidance '
        'Rules: 2-3 sentences. Be warm and validating. Never say the child '
        'is wrong about an emotion. All emotion guesses are valid. '
        'Do not use markdown, bullet points, or emojis. Speak naturally.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._conversationHistory,
    ];

    if (_conversationHistory.isEmpty ||
        _conversationHistory.last['content'] != childSaid) {
      messages.add({'role': 'user', 'content': childSaid});
    }

    try {
      final provider = _llmRouter.getProvider();

      if (onSpeakSentence != null) {
        return await _streamWithTts(provider, messages);
      }

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }
      final result = buffer.toString().trim();
      if (result.isNotEmpty) return result;
    } catch (e) {
      debugPrint('[EmotionDetective] LLM error: $e');
    }

    if (isWrapUp) {
      return "You are such a great emotion detective! Understanding how "
          "others feel is one of the most important skills in the world.";
    }
    return "That is a great observation! Why do you think they might feel that way?";
  }

  Future<String> _streamWithTts(
    LlmProvider provider,
    List<Map<String, String>> messages,
  ) async {
    _sentenceBuffer.reset();
    final fullResponse = <String>[];

    try {
      await for (final token in provider.stream(messages)) {
        final sentence = _sentenceBuffer.feed(token);
        if (sentence != null) {
          fullResponse.add(sentence);
          await onSpeakSentence!(sentence);
        }
      }

      final remaining = _sentenceBuffer.flush();
      if (remaining != null) {
        fullResponse.add(remaining);
        await onSpeakSentence!(remaining);
      }

      return fullResponse.isNotEmpty
          ? fullResponse.join(' ')
          : 'That is a great observation!';
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return 'That is a great observation!';
    }
  }

  String _buildEndSummary() {
    if (_storiesCompleted == 0) {
      return "Thanks for being an Emotion Detective! Come back to explore more feelings!";
    }
    return "You explored $_storiesCompleted "
        "${_storiesCompleted == 1 ? 'story' : 'stories'} as an Emotion Detective! "
        "Understanding feelings helps us be kinder to everyone. "
        "You have a wonderful heart!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  bool _containsNo(String text) {
    const noWords = ['no', 'nah', 'nope', "don't want", 'not now'];
    return noWords.any((w) => text.contains(w));
  }

  static const List<_EmotionStory> _stories = [
    _EmotionStory(
      narration:
          'Ria was so excited about her birthday party. She invited all her '
          'friends. But on the day of the party, her best friend Meera did not '
          'come and did not even say why.',
      question: 'How do you think Ria felt when Meera did not show up?',
      emotion: 'disappointment and hurt',
    ),
    _EmotionStory(
      narration:
          'Arjun practiced his dance routine for weeks for the school show. '
          'When it was his turn, everyone clapped and cheered. The teacher '
          'said it was the best performance of the day.',
      question: 'How do you think Arjun felt when everyone cheered for him?',
      emotion: 'pride and joy',
    ),
    _EmotionStory(
      narration:
          'Priya had a beautiful new coloring set. Her classmate Ravi got '
          'the same set but also got extra glitter pens. Priya kept looking '
          'at Ravi\'s glitter pens during art class.',
      question: 'How do you think Priya felt looking at Ravi\'s extra pens?',
      emotion: 'jealousy',
    ),
    _EmotionStory(
      narration:
          'It was Kabir\'s first day at a new school. He did not know anyone. '
          'During lunch, he sat alone at a table while everyone else sat in groups.',
      question: 'How do you think Kabir felt sitting alone at lunch?',
      emotion: 'loneliness and nervousness',
    ),
    _EmotionStory(
      narration:
          'Ananya worked really hard on her science project. She stayed up '
          'late and gave it her best. But when the results came, she did not '
          'win. Her friend who barely tried won instead.',
      question: 'How do you think Ananya felt about the results?',
      emotion: 'frustration and unfairness',
    ),
    _EmotionStory(
      narration:
          'Rohan found a lost puppy on the road. It was raining and the '
          'puppy was shivering. Rohan took it home, dried it off, and gave '
          'it warm milk. The puppy wagged its tail and licked his hand.',
      question: 'How do you think Rohan felt when the puppy licked his hand?',
      emotion: 'warmth and gratitude',
    ),
    _EmotionStory(
      narration:
          'Diya was about to go on stage for her first ever speech in front '
          'of the whole school. Her hands were shaking and her tummy felt funny.',
      question: 'How do you think Diya felt before going on stage?',
      emotion: 'nervousness and anxiety',
    ),
    _EmotionStory(
      narration:
          'Vikram built an amazing sandcastle at the beach. It took him two '
          'hours. Just as he finished, a big wave came and washed it all away.',
      question: 'How do you think Vikram felt when the wave destroyed his castle?',
      emotion: 'sadness and frustration',
    ),
  ];
}

/// A story scenario targeting a specific emotion.
class _EmotionStory {
  final String narration;
  final String question;
  final String emotion;

  const _EmotionStory({
    required this.narration,
    required this.question,
    required this.emotion,
  });
}
