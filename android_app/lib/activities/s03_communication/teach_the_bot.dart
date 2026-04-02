import 'dart:math';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../audio/sentence_buffer.dart';

/// Callback type for streaming TTS during LLM responses.
typedef SpeakSentenceCallback = Future<void> Function(String sentence);

/// Activity where the child teaches the bot how to do something.
///
/// The bot plays dumb and asks clarifying questions, encouraging the child
/// to break down a process into clear steps. At the end the bot repeats
/// back what it learned, reinforcing the child's explanation skills.
class TeachTheBot extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  final Random _random = Random();

  bool _active = false;
  int _topicsCompleted = 0;
  int _turnsInTopic = 0;
  int _maxTurnsPerTopic = 4;
  int _maxTopics = 2;

  String? _currentTopic;
  bool _waitingForReady = false;
  bool _waitingForExplanation = false;
  bool _waitingForPlayAgain = false;

  final List<int> _usedTopicIndices = [];
  final List<Map<String, String>> _topicHistory = [];

  TeachTheBot({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'communication_teach_bot';

  @override
  String get name => 'Teach the Bot';

  @override
  String get category => 'communication';

  @override
  String get description =>
      'Teach me how to do something! Practice explaining things step by step.';

  @override
  List<String> get skills => ['verbal expression', 'sequencing', 'communication'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  SkillId? get skillId => SkillId.communication;

  @override
  Map<String, List<String>> get voiceTriggers => const {
    'en': ['teach you', 'let me teach', 'i will teach', 'teach the bot', 'teach buddy', 'i can teach'],
    'hi': ['सिखाता हूं', 'मैं सिखाऊंगा', 'सिखाती हूं'],
    'te': ['నేర్పిస్తా', 'నేను నేర్పిస్తాను'],
  };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _topicsCompleted = 0;
    _usedTopicIndices.clear();
    _waitingForReady = true;
    _waitingForExplanation = false;
    _waitingForPlayAgain = false;

    return "I want to learn something new from you! You are going to be "
        "my teacher today. Can you teach me how to do something? Ready?";
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
      return _presentTopic();
    }

    if (_waitingForPlayAgain) {
      _waitingForPlayAgain = false;
      if (_containsNo(lower)) {
        _active = false;
        return _buildEndSummary();
      }
      return _presentTopic();
    }

    if (_waitingForExplanation && _currentTopic != null) {
      return await _processExplanation(childSaid);
    }

    return "Go ahead, teach me! I am listening!";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_topicsCompleted == 0) return 'No topics taught yet.';
    return 'Taught me $_topicsCompleted ${_topicsCompleted == 1 ? 'thing' : 'things'} so far.';
  }

  // -- Internal --

  String _presentTopic() {
    // Pick an unused topic
    String? chosen;
    for (int i = 0; i < _topics.length; i++) {
      if (!_usedTopicIndices.contains(i)) {
        chosen = _topics[i];
        _usedTopicIndices.add(i);
        break;
      }
    }

    if (chosen == null) {
      _usedTopicIndices.clear();
      final idx = _random.nextInt(_topics.length);
      chosen = _topics[idx];
      _usedTopicIndices.add(idx);
    }

    _currentTopic = chosen;
    _turnsInTopic = 0;
    _topicHistory.clear();
    _waitingForExplanation = true;

    return "Okay teacher! Can you teach me how to $chosen? "
        "Tell me what to do first!";
  }

  Future<String> _processExplanation(String childSaid) async {
    _turnsInTopic++;
    _topicHistory.add({'role': 'user', 'content': childSaid});

    // Check if we should wrap up this topic
    if (_turnsInTopic >= _maxTurnsPerTopic) {
      _topicsCompleted++;
      _waitingForExplanation = false;

      final summary = await _getLlmSummary(childSaid);
      _topicHistory.add({'role': 'assistant', 'content': summary});

      if (_topicsCompleted >= _maxTopics) {
        _active = false;
        return "$summary ${_buildEndSummary()}";
      }

      _waitingForPlayAgain = true;
      return "$summary Can you teach me something else?";
    }

    // Ask a clarifying question via LLM
    final response = await _getLlmClarification(childSaid);
    _topicHistory.add({'role': 'assistant', 'content': response});
    return response;
  }

  Future<String> _getLlmClarification(String childSaid) async {
    final messages = <Map<String, String>>[
      {'role': 'system', 'content': _clarificationPrompt},
      {'role': 'assistant', 'content': 'Can you teach me how to $_currentTopic? Tell me what to do first!'},
    ];
    messages.addAll(_topicHistory);

    try {
      final provider = _llmRouter.getProvider();

      if (onSpeakSentence != null) {
        return await _streamWithTts(provider, messages);
      }

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }

      final response = buffer.toString().trim();
      if (response.isEmpty) return _fallbackClarification();
      return response;
    } catch (e) {
      return _fallbackClarification();
    }
  }

  Future<String> _getLlmSummary(String childSaid) async {
    final messages = <Map<String, String>>[
      {'role': 'system', 'content': _summaryPrompt},
      {'role': 'assistant', 'content': 'Can you teach me how to $_currentTopic? Tell me what to do first!'},
    ];
    messages.addAll(_topicHistory);

    // Add the final message if not already in history
    if (_topicHistory.isEmpty || _topicHistory.last['content'] != childSaid) {
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

      final response = buffer.toString().trim();
      if (response.isEmpty) return _fallbackSummary();
      return response;
    } catch (e) {
      return _fallbackSummary();
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
      return result.isNotEmpty ? result : _fallbackClarification();
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return _fallbackClarification();
    }
  }

  String _fallbackClarification() {
    const clarifications = [
      "Hmm, wait. Do I use my hands for that? Tell me more!",
      "Oh interesting! And then what do I do next?",
      "I think I understand! What is the next step?",
      "Ooh, okay! But how exactly do I do that part?",
    ];
    return clarifications[_random.nextInt(clarifications.length)];
  }

  String _fallbackSummary() {
    return "Thank you for teaching me how to $_currentTopic! "
        "You are a wonderful teacher! I learned so much from you!";
  }

  String _buildEndSummary() {
    if (_topicsCompleted == 0) {
      return "Thanks for being my teacher! Come back anytime to teach me more!";
    }
    return "You taught me $_topicsCompleted ${_topicsCompleted == 1 ? 'thing' : 'things'} today! "
        "You are the best teacher ever!";
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

  static const String _clarificationPrompt =
      'You are Buddy, a friendly learning companion for a 3-6 year old child. '
      'The child is teaching you how to do something. You are the student and '
      'they are the teacher. '
      'Ask one simple clarifying question about what they just said to help '
      'them practice explaining clearly. Act a little confused or curious so '
      'they have to explain more. For example: "Wait, do I use hot water or '
      'cold water?" or "Hmm, how do I hold it?" '
      'Rules: Keep it to 1-2 sentences. Be encouraging. Use simple words. '
      'Do not use markdown or bullet points. Speak naturally.';

  static const String _summaryPrompt =
      'You are Buddy, a friendly learning companion for a 3-6 year old child. '
      'The child has been teaching you how to do something. Now summarize what '
      'they taught you in a fun way. Start with "Thank you for teaching me!" '
      'and then repeat back the steps they described. Be very encouraging and '
      'tell them they are a great teacher. '
      'Rules: Keep it to 3-4 sentences. Use simple words. Do not use markdown '
      'or bullet points. Speak naturally as if talking to a young child.';

  static const List<String> _topics = [
    'brush your teeth',
    'make a sandwich',
    'draw a flower',
    'get dressed in the morning',
    'wash your hands',
    'feed a pet',
    'play your favorite game',
    'make your bed',
    'tie your shoes',
    'pack a school bag',
    'water a plant',
    'build with blocks',
  ];
}
