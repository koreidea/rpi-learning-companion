import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// Nutrition Navigator: kid describes what they ate and the bot gives
/// friendly, age-appropriate nutritional guidance.
///
/// Teaches: nutrition awareness, healthy eating habits, food groups,
/// body awareness.
///
/// Flow:
/// 1. Bot asks what the child ate today (breakfast, lunch, or snack).
/// 2. Child describes their meal.
/// 3. Bot analyzes with fun analogies (protein = building blocks, etc.).
/// 4. Bot suggests one small improvement.
/// 5. Shares a fun food fact.
///
/// Uses LLM for natural, context-aware nutritional guidance.
class NutritionNavigator extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _mealsDiscussed = 0;
  int _score = 0;
  int _maxMeals = 3;

  bool _waitingForMeal = false;
  bool _waitingForPlayAgain = false;

  final List<Map<String, String>> _conversationHistory = [];

  NutritionNavigator({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'health_nutrition_navigator';

  @override
  String get name => 'Nutrition Navigator';

  @override
  String get category => 'wellness';

  @override
  String get description =>
      'Tell me what you ate today and I will help you understand nutrition.';

  @override
  List<String> get skills => [
        'nutrition awareness',
        'healthy eating',
        'food groups',
        'body awareness',
      ];

  @override
  int get minAge => 4;

  @override
  int get maxAge => 10;

  @override
  SkillId? get skillId => SkillId.healthWellness;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'nutrition game',
          'what should i eat',
          'food game',
          'healthy food',
          'tell me about food',
          'nutrition navigator',
        ],
        'hi': [
          'खाने के बारे में बताओ',
          'स्वस्थ खाना',
          'पोषण',
        ],
        'te': [
          'ఆహారం గురించి చెప్పు',
          'ఆరోగ్యకరమైన ఆహారం',
          'పోషణ',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_mealsDiscussed == 0) return 'No meals discussed yet.';
    return 'Discussed $_mealsDiscussed meals. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _mealsDiscussed = 0;
    _score = 0;
    _conversationHistory.clear();
    _waitingForMeal = true;
    _waitingForPlayAgain = false;

    return 'Hello there, nutrition explorer! I would love to hear about what '
        'you ate today. Tell me about your breakfast, lunch, or any snack. '
        'What did you eat?';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return _buildEndSummary();
    }

    if (_waitingForPlayAgain) {
      _waitingForPlayAgain = false;
      if (_containsNo(lower)) {
        _active = false;
        return _buildEndSummary();
      }
      _waitingForMeal = true;
      return 'Great! Tell me about another meal or snack you had today.';
    }

    if (_waitingForMeal) {
      _waitingForMeal = false;
      return await _analyzeMeal(childSaid);
    }

    return 'Tell me what you ate and I will tell you all about it!';
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  Future<String> _analyzeMeal(String mealDescription) async {
    _conversationHistory.add({'role': 'user', 'content': mealDescription});

    final systemPrompt = 'You are a friendly nutrition guide for a child aged 3-10. '
        'The child just told you what they ate. Analyze their meal simply and positively. '
        'Rules: '
        '1. Start by saying something positive about what they ate. '
        '2. Use fun analogies: protein = building blocks for muscles, '
        'carbs = energy fuel for running and playing, vitamins = shields that '
        'protect you from getting sick, calcium = cement that makes bones strong. '
        '3. Suggest ONE small addition they could make (a fruit, a vegetable, water). '
        '4. Share one fun food fact. '
        '5. Never shame any food choice. Even candy and chips have something positive. '
        '6. Keep response to 4-5 sentences. No markdown, no bullets. '
        'Speak naturally as if talking to a child.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._conversationHistory,
    ];

    try {
      final provider = _llmRouter.getProvider();
      String response;

      if (onSpeakSentence != null) {
        response = await _streamWithTts(provider, messages);
      } else {
        final buffer = StringBuffer();
        await for (final token in provider.stream(messages)) {
          buffer.write(token);
        }
        response = buffer.toString().trim();
      }

      if (response.isEmpty) {
        response = _fallbackResponse();
      }

      _conversationHistory.add({'role': 'assistant', 'content': response});
      _mealsDiscussed++;
      _score += 15;

      if (_mealsDiscussed >= _maxMeals) {
        _active = false;
        return '$response ${_buildEndSummary()}';
      }

      _waitingForPlayAgain = true;
      return '$response Would you like to tell me about another meal?';
    } catch (e) {
      _mealsDiscussed++;
      _score += 10;
      _waitingForPlayAgain = true;
      return '${_fallbackResponse()} Would you like to tell me about another meal?';
    }
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
      final result = fullResponse.join(' ');
      return result.isNotEmpty ? result : _fallbackResponse();
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return _fallbackResponse();
    }
  }

  String _fallbackResponse() {
    const responses = [
      'That sounds like a good meal! Food gives our body energy to run, '
          'play, and think. Did you have any fruit today? Fruits are like '
          'vitamin shields that keep you healthy!',
      'Yummy! Every meal helps your body grow strong. Protein foods like '
          'dal, eggs, or milk are like building blocks for your muscles. '
          'Try adding a glass of milk if you can!',
      'That is great! Our body needs different types of food to work well. '
          'Vegetables are like tiny superheroes that fight germs inside your body. '
          'Can you try to eat one colorful vegetable today?',
    ];
    return responses[_rng.nextInt(responses.length)];
  }

  String _buildEndSummary() {
    if (_mealsDiscussed == 0) {
      return 'Come back anytime to talk about food and nutrition!';
    }
    return 'We talked about $_mealsDiscussed '
        'meal${_mealsDiscussed > 1 ? 's' : ''} today! '
        'Score: $_score points! Remember, eating colorful foods makes '
        'your body happy and strong!';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
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
}
