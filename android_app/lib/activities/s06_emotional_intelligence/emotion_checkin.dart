import 'dart:async';

import 'package:flutter/foundation.dart';

import '../../core/llm/llm_router.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Data for handling a specific emotion.
class _EmotionData {
  final String name;
  final String validation;
  final String copingStrategy;

  const _EmotionData({
    required this.name,
    required this.validation,
    required this.copingStrategy,
  });
}

/// Emotional Check-in: help children identify and express emotions.
///
/// Teaches: emotional intelligence, self-awareness, coping skills, vocabulary.
///
/// Flow:
/// 1. Bot asks how the child is feeling.
/// 2. Child names an emotion.
/// 3. Bot validates the emotion and asks what happened.
/// 4. Child explains.
/// 5. Bot responds with empathy (LLM).
/// 6. Bot guides through a breathing exercise.
/// 7. Bot wraps up with encouragement.
class EmotionCheckin extends Activity {
  final LlmRouter _llmRouter;

  bool _active = false;
  int _score = 0;
  int _phase = 0;
  // Phase: 0=ask feeling, 1=detect emotion, 2=ask why, 3=empathy+breathing,
  //        4=breathing step 2, 5=breathing step 3, 6=wrap up

  String _detectedEmotion = '';

  static const Map<String, _EmotionData> _emotions = {
    'happy': _EmotionData(
      name: 'happy',
      validation: "I am so glad you are happy! Being happy is a wonderful feeling!",
      copingStrategy: "When we are happy, we can share our happiness with others. "
          "Can you think of something nice you can do for someone today?",
    ),
    'sad': _EmotionData(
      name: 'sad',
      validation: "Thank you for telling me you are sad. It is okay to feel sad sometimes. "
          "Everyone feels sad, even grown-ups.",
      copingStrategy: "When I feel sad, I like to take deep breaths. It helps me feel a little better.",
    ),
    'angry': _EmotionData(
      name: 'angry',
      validation: "I understand you feel angry. It is okay to feel angry. "
          "What matters is what we do with that feeling.",
      copingStrategy: "When we feel angry, taking deep breaths can help calm us down. "
          "Let's try that together.",
    ),
    'scared': _EmotionData(
      name: 'scared',
      validation: "It is okay to feel scared. Being scared is your body's way of keeping you safe. "
          "Even brave people feel scared sometimes.",
      copingStrategy: "When we feel scared, we can hug something soft, like a teddy bear, "
          "and take slow breaths.",
    ),
    'excited': _EmotionData(
      name: 'excited',
      validation: "That is wonderful! Being excited means something great is happening or about to happen!",
      copingStrategy: "When we are excited, we have so much energy! "
          "Let's use that energy for something fun!",
    ),
    'bored': _EmotionData(
      name: 'bored',
      validation: "Feeling bored is okay. It means your brain is looking for something interesting to do!",
      copingStrategy: "When we feel bored, we can use our imagination! "
          "Let's think of something fun to do together.",
    ),
    'frustrated': _EmotionData(
      name: 'frustrated',
      validation: "I know how it feels to be frustrated. It happens when things don't go the way we want.",
      copingStrategy: "When we feel frustrated, it helps to stop and take a break. "
          "Let's breathe together.",
    ),
    'lonely': _EmotionData(
      name: 'lonely',
      validation: "I am sorry you feel lonely. Everyone feels lonely sometimes. "
          "But remember, I am always here to talk to you!",
      copingStrategy: "When we feel lonely, we can talk to someone we love, or play with a toy, "
          "or talk to me!",
    ),
    'nervous': _EmotionData(
      name: 'nervous',
      validation: "It is okay to feel nervous. Your tummy might feel funny and that is normal.",
      copingStrategy: "When we feel nervous, we can squeeze our hands tight, then let go, "
          "and breathe slowly.",
    ),
  };

  EmotionCheckin({required LlmRouter llmRouter})
      : _llmRouter = llmRouter;

  // -- Activity metadata --

  @override
  String get id => 'emotions_checkin';

  @override
  String get name => 'Emotion Check-in';

  @override
  String get category => 'emotions';

  @override
  String get description =>
      'Talk about how you are feeling and learn to manage your emotions.';

  @override
  List<String> get skills => ['emotional intelligence', 'self-awareness', 'coping skills'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  SkillId? get skillId => SkillId.emotionalIntelligence;

  @override
  Map<String, List<String>> get voiceTriggers => const {
    'en': ['how am i feeling', 'emotion check', 'feeling check', 'talk about feelings', 'emotion game'],
    'hi': ['मैं कैसा हूं', 'भावना खेल', 'भावनाएं'],
    'te': ['నా భావాలు', 'ఎమోషన్ ఆట', 'ఎలా ఉన్నాను'],
  };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_detectedEmotion.isEmpty) return 'No check-in yet.';
    return 'Feeling: $_detectedEmotion. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _detectedEmotion = '';
    _phase = 0;
    _score = 0;
    _active = true;

    debugPrint('[EmotionCheckin] Started');

    return "Hey! How are you feeling right now? Are you happy, sad, angry, excited, "
        "scared, or something else? You can tell me anything!";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    switch (_phase) {
      case 0:
        // Detect emotion from response
        _detectedEmotion = _detectEmotion(lower);
        _score += 10;

        if (_detectedEmotion.isEmpty) {
          // Could not detect — ask again gently
          return "That is okay! Sometimes feelings are hard to name. "
              "Are you feeling happy, sad, angry, scared, or something else?";
        }

        final emotionData = _emotions[_detectedEmotion];
        _phase = 1;

        if (emotionData != null) {
          // For positive emotions, skip asking "why"
          if (_detectedEmotion == 'happy' || _detectedEmotion == 'excited') {
            _phase = 2;
            return "${emotionData.validation} "
                "Can you tell me what made you feel $_detectedEmotion?";
          }
          return "${emotionData.validation} "
              "Can you tell me what made you feel that way?";
        }

        return "Thank you for telling me how you feel. "
            "Can you tell me more about why you feel that way?";

      case 1:
        // Ask why they feel that way
        _phase = 2;
        return "Can you tell me what made you feel $_detectedEmotion?";

      case 2:
        // Child explained why — respond with empathy
        _score += 10;
        _phase = 3;

        final empathyResponse = await _generateEmpathy(childSaid);
        final emotionData = _emotions[_detectedEmotion];
        final coping = emotionData?.copingStrategy ??
            "Let's try some deep breaths together. Breathing always helps!";

        return "$empathyResponse $coping "
            "Want to try some deep breaths with me?";

      case 3:
        // Breathing exercise step 1
        _phase = 4;
        _score += 5;
        return "Great! Let's breathe together. Breathe in slowly through your nose. "
            "1, 2, 3. Now breathe out slowly through your mouth. 1, 2, 3. "
            "How does that feel?";

      case 4:
        // Breathing exercise step 2
        _phase = 5;
        _score += 5;
        return "Let's do it again! Breathe in. 1, 2, 3. "
            "And breathe out. 1, 2, 3. You are doing great!";

      case 5:
        // Breathing exercise step 3 + wrap up
        _phase = 6;
        _score += 5;
        return "One more time! Breathe in. 1, 2, 3. Breathe out. 1, 2, 3. "
            "Wonderful! How do you feel now?";

      case 6:
        // Final wrap up
        _score += 10;
        _active = false;
        return "I am proud of you for sharing your feelings with me! "
            "Remember, it is okay to feel all kinds of feelings. "
            "You can always talk to me about how you feel. You are amazing!";

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    debugPrint('[EmotionCheckin] Ended, emotion=$_detectedEmotion, score=$_score');

    return "Remember, all feelings are okay. Happy, sad, angry, scared. "
        "They are all part of being you! You can always talk to me. "
        "You are wonderful just the way you are!";
  }

  // -- Emotion detection --

  String _detectEmotion(String lower) {
    // English emotion words
    final emotionKeywords = <String, List<String>>{
      'happy': ['happy', 'glad', 'good', 'great', 'fine', 'wonderful', 'awesome', 'nice'],
      'sad': ['sad', 'unhappy', 'cry', 'crying', 'miss', 'down', 'blue'],
      'angry': ['angry', 'mad', 'upset', 'furious', 'annoyed'],
      'scared': ['scared', 'afraid', 'fear', 'frightened', 'worried', 'scary'],
      'excited': ['excited', 'thrilled', 'pumped', 'can not wait', 'yay'],
      'bored': ['bored', 'boring', 'nothing to do', 'dull'],
      'frustrated': ['frustrated', 'stuck', 'hard', 'difficult', 'can not do'],
      'lonely': ['lonely', 'alone', 'no friends', 'by myself', 'nobody'],
      'nervous': ['nervous', 'anxious', 'tummy hurts', 'butterflies', 'worried'],
    };

    // Hindi emotion words
    final hindiKeywords = <String, List<String>>{
      'happy': ['खुश', 'अच्छा', 'मज़ा'],
      'sad': ['उदास', 'दुखी', 'रो'],
      'angry': ['गुस्सा', 'नाराज़'],
      'scared': ['डर', 'भय', 'डरा'],
      'excited': ['उत्साहित', 'मज़ा'],
      'bored': ['बोर', 'ऊब'],
    };

    // Telugu emotion words
    final teluguKeywords = <String, List<String>>{
      'happy': ['సంతోషం', 'బాగా', 'ఆనందం'],
      'sad': ['బాధ', 'దుఃఖం', 'ఏడుపు'],
      'angry': ['కోపం', 'అసహనం'],
      'scared': ['భయం', 'భయపడుతున్నాను'],
    };

    // Check all keyword maps
    for (final entry in emotionKeywords.entries) {
      for (final keyword in entry.value) {
        if (lower.contains(keyword)) return entry.key;
      }
    }
    for (final entry in hindiKeywords.entries) {
      for (final keyword in entry.value) {
        if (lower.contains(keyword)) return entry.key;
      }
    }
    for (final entry in teluguKeywords.entries) {
      for (final keyword in entry.value) {
        if (lower.contains(keyword)) return entry.key;
      }
    }

    return '';
  }

  // -- LLM empathy generation --

  Future<String> _generateEmpathy(String childSaid) async {
    try {
      final provider = _llmRouter.getProvider();
      final instruction =
          'You are a caring, warm friend for a 3-6 year old child. '
          'The child is feeling $_detectedEmotion and said: "$childSaid". '
          'Respond with empathy in 1-2 short sentences. '
          'Validate their feelings. Show you understand. '
          'Never dismiss their feelings. Do not use emojis or markdown. '
          'Speak naturally and gently.';

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
      debugPrint('[EmotionCheckin] LLM error: $e');
    }

    // Fallback
    return "I understand. Thank you for telling me about that.";
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
