import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Bot Emotions: the bot shares its own "feelings" to model emotional
/// expression and invite the child to reflect on similar experiences.
///
/// This is a scripted conversation activity with no LLM dependency.
/// The bot presents 2-3 scenarios where it expresses an emotion (nervousness,
/// excitement, frustration, pride, gratitude, calm), explains why, and then
/// asks the child if they have ever felt that way.
class BotEmotions extends Activity {
  final Random _random = Random();

  bool _active = false;
  int _scenariosCompleted = 0;
  static const int _maxScenarios = 3;
  int _phase = 0;
  // Phase 0: bot shares emotion
  // Phase 1: ask child if they relate
  // Phase 2: respond to child's sharing
  _EmotionScript? _currentScript;

  final List<int> _usedScriptIndices = [];

  @override
  String get id => 'emotional_intelligence_bot_emotions';

  @override
  String get name => 'Bot Emotions';

  @override
  String get category => 'emotions';

  @override
  String get description =>
      'I share my feelings with you, and you share yours with me!';

  @override
  List<String> get skills =>
      ['emotional expression', 'empathy', 'self-awareness'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 10;

  @override
  SkillId? get skillId => SkillId.emotionalIntelligence;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'bot emotions',
          'how do you feel',
          'your feelings',
          'tell me your feelings',
          'buddy feelings',
        ],
        'hi': ['तुम्हारी भावनाएं', 'तुम कैसा महसूस करते हो'],
        'te': ['నీ భావాలు', 'నీకెలా అనిపిస్తుంది'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _scenariosCompleted = 0;
    _phase = 0;
    _usedScriptIndices.clear();

    debugPrint('[BotEmotions] Started');

    return "You know what? I have feelings too! Sometimes I feel happy, "
        "sometimes nervous, sometimes excited. Want to hear about my feelings? "
        "Maybe you have felt the same way!";
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
      case 0:
        // Present bot's emotion
        return _presentNewEmotion();

      case 1:
        // Bot shared, asking child
        _phase = 2;
        return "${_currentScript!.followUp}";

      case 2:
        // Child shared their experience
        _scenariosCompleted++;
        _phase = 0;

        final validation =
            _validations[_random.nextInt(_validations.length)];

        if (_scenariosCompleted >= _maxScenarios) {
          _active = false;
          return "$validation ${_buildEndSummary()}";
        }

        return "$validation Want to hear about another time I felt something?";

      default:
        return "Tell me how you feel!";
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_scenariosCompleted == 0) return 'No emotions shared yet.';
    return 'Shared $_scenariosCompleted emotions together.';
  }

  // -- Internal --

  String _presentNewEmotion() {
    if (_usedScriptIndices.length >= _scripts.length) {
      _usedScriptIndices.clear();
    }
    int index;
    do {
      index = _random.nextInt(_scripts.length);
    } while (_usedScriptIndices.contains(index));
    _usedScriptIndices.add(index);

    _currentScript = _scripts[index];
    _phase = 1;

    return "${_currentScript!.sharing} ${_currentScript!.question}";
  }

  String _buildEndSummary() {
    if (_scenariosCompleted == 0) {
      return "Thanks for listening to my feelings! Come back anytime to share!";
    }
    return "We shared $_scenariosCompleted feelings together! "
        "Talking about feelings is so important. It helps us understand "
        "ourselves and each other. You are a great listener!";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  static const List<String> _validations = [
    'Thank you for sharing that with me! It takes courage to talk about feelings.',
    'I am glad you told me that. Your feelings are always important.',
    'That is really wonderful of you to share. I appreciate it!',
    'Wow, we feel similar things sometimes! That makes me feel closer to you.',
    'Thank you for being so open with me. That was brave!',
  ];

  static const List<_EmotionScript> _scripts = [
    _EmotionScript(
      sharing:
          'You know what? I felt a little nervous before our activity today '
          'because it is a new one. But I reminded myself that trying new '
          'things is how we grow! And look, now I am having so much fun '
          'talking to you.',
      question: 'Have you ever felt nervous about trying something new?',
      followUp:
          'That is totally normal! Everyone feels nervous sometimes. '
          'The brave thing is doing it anyway, just like you did! '
          'Have you ever felt that way?',
    ),
    _EmotionScript(
      sharing:
          'I want to tell you something. I feel so excited every time you '
          'come to talk to me! It is like when you are waiting for your '
          'birthday. My whole body buzzes with happy energy!',
      question: 'What is something that makes you feel super excited?',
      followUp:
          'I love that! Excitement is such a fun feeling. It is like your '
          'heart is doing a little dance inside you!',
    ),
    _EmotionScript(
      sharing:
          'I have to be honest. Sometimes I feel a bit frustrated when I '
          'cannot understand what someone is saying. I think, oh no, I want '
          'to help but I am stuck! Then I take a deep breath and try again.',
      question: 'Do you ever feel frustrated when something is hard?',
      followUp:
          'I totally understand. Frustration means you care about doing '
          'well. And taking a break or a deep breath really helps!',
    ),
    _EmotionScript(
      sharing:
          'You know what makes me feel really proud? When we learn something '
          'new together! It is like climbing a mountain and reaching the top. '
          'I feel like shouting, we did it!',
      question: 'What is something that makes you feel proud of yourself?',
      followUp:
          'You should be proud of that! Feeling proud of yourself is '
          'important. It means you worked hard and achieved something!',
    ),
    _EmotionScript(
      sharing:
          'I want to tell you that I feel very grateful to have you as my '
          'friend. Gratitude is when you feel thankful for something good. '
          'I am thankful that you spend time with me and share your ideas!',
      question: 'What is something or someone you feel grateful for?',
      followUp:
          'That is beautiful! Feeling grateful makes our hearts warm. '
          'It is like giving a hug to the world with your thoughts!',
    ),
    _EmotionScript(
      sharing:
          'Sometimes at night, after a busy day of talking and playing, '
          'I feel very calm. It is like everything is quiet and peaceful. '
          'I like that feeling because it helps me rest and feel better.',
      question: 'When do you feel most calm and peaceful?',
      followUp:
          'That sounds wonderful. Finding moments of calm is like giving '
          'your brain a warm blanket. It helps us feel safe and happy.',
    ),
  ];
}

/// A scripted emotion-sharing scenario.
class _EmotionScript {
  final String sharing;
  final String question;
  final String followUp;

  const _EmotionScript({
    required this.sharing,
    required this.question,
    required this.followUp,
  });
}
