import 'dart:math';

import '../activity_base.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../audio/sentence_buffer.dart';

/// Callback type for streaming TTS during LLM responses.
typedef SpeakSentenceCallback = Future<void> Function(String sentence);

/// Role-play interview game where the child pretends to be a character.
///
/// The bot acts as an interviewer, asking fun questions about the child's
/// "adventures" as an astronaut, chef, explorer, etc. Uses the LLM with
/// character-specific system prompts to generate natural follow-up questions.
class InterviewGame extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  final Random _random = Random();

  bool _active = false;
  int _interviewsCompleted = 0;
  int _questionsAsked = 0;
  int _maxQuestions = 4;
  int _maxInterviews = 2;

  _InterviewScenario? _currentScenario;
  bool _waitingForReady = false;
  bool _waitingForAnswer = false;
  bool _waitingForPlayAgain = false;

  final List<int> _usedScenarioIndices = [];
  final List<Map<String, String>> _interviewHistory = [];

  InterviewGame({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'communication_interview';

  @override
  String get name => 'Interview Game';

  @override
  String get category => 'communication';

  @override
  String get description =>
      'Pretend to be a fun character and answer interview questions!';

  @override
  List<String> get skills =>
      ['verbal expression', 'imagination', 'communication', 'role-play'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  bool get isActive => _active;

  @override
  Future<String> start() async {
    _active = true;
    _interviewsCompleted = 0;
    _usedScenarioIndices.clear();
    _waitingForReady = true;
    _waitingForAnswer = false;
    _waitingForPlayAgain = false;

    return "Let's play the Interview Game! You get to pretend to be someone "
        "really cool and I will interview you! Ready?";
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
      return _startNewInterview();
    }

    if (_waitingForPlayAgain) {
      _waitingForPlayAgain = false;
      if (_containsNo(lower)) {
        _active = false;
        return _buildEndSummary();
      }
      return _startNewInterview();
    }

    if (_waitingForAnswer && _currentScenario != null) {
      return await _processAnswer(childSaid);
    }

    return "Go ahead, tell me about your adventure!";
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_interviewsCompleted == 0) return 'No interviews completed yet.';
    return 'Completed $_interviewsCompleted ${_interviewsCompleted == 1 ? 'interview' : 'interviews'}.';
  }

  // -- Internal --

  String _startNewInterview() {
    _InterviewScenario? chosen;
    for (int i = 0; i < _scenarios.length; i++) {
      if (!_usedScenarioIndices.contains(i)) {
        chosen = _scenarios[i];
        _usedScenarioIndices.add(i);
        break;
      }
    }

    if (chosen == null) {
      _usedScenarioIndices.clear();
      final idx = _random.nextInt(_scenarios.length);
      chosen = _scenarios[idx];
      _usedScenarioIndices.add(idx);
    }

    _currentScenario = chosen;
    _questionsAsked = 0;
    _interviewHistory.clear();
    _waitingForAnswer = true;

    return "Let's pretend you are ${chosen.characterIntro}! "
        "I am going to interview you. ${chosen.openingQuestion}";
  }

  Future<String> _processAnswer(String childSaid) async {
    _questionsAsked++;
    _interviewHistory.add({'role': 'user', 'content': childSaid});

    // Check if interview is done
    if (_questionsAsked >= _maxQuestions) {
      _interviewsCompleted++;
      _waitingForAnswer = false;

      // Generate a closing remark
      final closing = await _getLlmResponse(childSaid, isClosing: true);
      _interviewHistory.add({'role': 'assistant', 'content': closing});

      if (_interviewsCompleted >= _maxInterviews) {
        _active = false;
        return "$closing ${_buildEndSummary()}";
      }

      _waitingForPlayAgain = true;
      return "$closing Want to be someone else for the next interview?";
    }

    // Generate a follow-up question
    final response = await _getLlmResponse(childSaid, isClosing: false);
    _interviewHistory.add({'role': 'assistant', 'content': response});
    return response;
  }

  Future<String> _getLlmResponse(String childSaid, {required bool isClosing}) async {
    final scenario = _currentScenario!;
    final systemPrompt = isClosing
        ? _buildClosingPrompt(scenario)
        : _buildInterviewPrompt(scenario);

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      {'role': 'assistant', 'content': scenario.openingQuestion},
    ];
    messages.addAll(_interviewHistory);

    if (_interviewHistory.isEmpty ||
        _interviewHistory.last['content'] != childSaid) {
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
      if (response.isEmpty) {
        return isClosing ? _fallbackClosing(scenario) : _fallbackQuestion();
      }
      return response;
    } catch (e) {
      return isClosing ? _fallbackClosing(scenario) : _fallbackQuestion();
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
      return result.isNotEmpty ? result : _fallbackQuestion();
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return _fallbackQuestion();
    }
  }

  String _buildInterviewPrompt(_InterviewScenario scenario) {
    return 'You are Buddy, a friendly interviewer for a 3-6 year old child. '
        'The child is pretending to be ${scenario.character}. '
        '${scenario.context} '
        'React enthusiastically to their answer with a short excited comment, '
        'then ask one follow-up question about their adventure. '
        'Rules: Keep it to 2 sentences total (one reaction + one question). '
        'Use simple words. Be very excited and impressed by everything they say. '
        'Do not use markdown or bullet points. Speak naturally as if on a talk show.';
  }

  String _buildClosingPrompt(_InterviewScenario scenario) {
    return 'You are Buddy, a friendly interviewer for a 3-6 year old child. '
        'The child was pretending to be ${scenario.character}. '
        'This is the end of the interview. Thank the child for the amazing '
        'interview. Mention one specific thing they said that was interesting '
        'or creative. Tell them they are an amazing ${scenario.characterTitle}. '
        'Rules: Keep it to 2-3 sentences. Be very enthusiastic and celebratory. '
        'Do not use markdown. Speak naturally.';
  }

  String _fallbackQuestion() {
    const questions = [
      "Wow, that sounds amazing! Can you tell me more about that?",
      "Oh how exciting! What was the best part?",
      "That is so cool! What happened next?",
      "I cannot believe it! What did you do then?",
    ];
    return questions[_random.nextInt(questions.length)];
  }

  String _fallbackClosing(_InterviewScenario scenario) {
    return "Thank you so much for this wonderful interview, ${scenario.characterTitle}! "
        "You told the most amazing stories! You are truly incredible!";
  }

  String _buildEndSummary() {
    if (_interviewsCompleted == 0) {
      return "Thanks for playing the Interview Game! Come back anytime!";
    }
    return "We did $_interviewsCompleted amazing "
        "${_interviewsCompleted == 1 ? 'interview' : 'interviews'}! "
        "You are a wonderful storyteller!";
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

  // -- Scenario data --

  static const List<_InterviewScenario> _scenarios = [
    _InterviewScenario(
      character: 'an astronaut who just came back from space',
      characterTitle: 'astronaut',
      characterIntro: 'an astronaut who just flew back from space',
      openingQuestion:
          'Welcome back to Earth, astronaut! Tell me, what did you see up in space?',
      context:
          'The child is pretending to be an astronaut returning from a space mission. '
          'Ask about what they saw in space, if they met aliens, what they ate, '
          'if they visited any planets, and if they would go back.',
    ),
    _InterviewScenario(
      character: 'a chef who made the world\'s biggest cake',
      characterTitle: 'chef',
      characterIntro: 'an amazing chef who just made the biggest cake in the whole world',
      openingQuestion:
          'Chef, congratulations on making the world\'s biggest cake! '
          'How big was it? Tell me about it!',
      context:
          'The child is pretending to be a chef who baked the biggest cake ever. '
          'Ask about what flavor it was, how they made it, who helped them, '
          'what decorations they put on it, and who got to eat it.',
    ),
    _InterviewScenario(
      character: 'an explorer who found a hidden island',
      characterTitle: 'explorer',
      characterIntro: 'a brave explorer who just discovered a hidden island',
      openingQuestion:
          'Explorer, I heard you found a secret island! '
          'What did it look like when you first saw it?',
      context:
          'The child is pretending to be an explorer who discovered a hidden island. '
          'Ask about what animals they found, if there was treasure, what the '
          'island looked like, what they ate there, and if anyone else was there.',
    ),
    _InterviewScenario(
      character: 'a superhero who saved the city',
      characterTitle: 'superhero',
      characterIntro: 'a superhero who just saved the whole city',
      openingQuestion:
          'Superhero, you just saved the city! Tell me, what happened? '
          'What was the danger?',
      context:
          'The child is pretending to be a superhero. Ask about their superpower, '
          'what villain they defeated, how they saved everyone, if they have a '
          'costume, and what their superhero name is.',
    ),
    _InterviewScenario(
      character: 'a scientist who invented a flying car',
      characterTitle: 'scientist',
      characterIntro: 'a brilliant scientist who just invented a flying car',
      openingQuestion:
          'Scientist, you invented a flying car! That is incredible! '
          'How does it work? Tell me all about it!',
      context:
          'The child is pretending to be a scientist with a flying car. '
          'Ask about how they built it, what color it is, how fast it goes, '
          'where they flew, and what they want to invent next.',
    ),
    _InterviewScenario(
      character: 'a zookeeper who takes care of baby pandas',
      characterTitle: 'zookeeper',
      characterIntro: 'a zookeeper who takes care of adorable baby pandas',
      openingQuestion:
          'Zookeeper, I heard you take care of baby pandas! '
          'What are the baby pandas like? Tell me about them!',
      context:
          'The child is pretending to be a zookeeper caring for baby pandas. '
          'Ask about what the pandas eat, what their names are, if they are '
          'playful, what a typical day looks like, and what other animals they care for.',
    ),
    _InterviewScenario(
      character: 'a pirate who found a treasure chest',
      characterTitle: 'pirate',
      characterIntro: 'a pirate who just found a huge treasure chest',
      openingQuestion:
          'Ahoy, pirate! I heard you found treasure! '
          'Where did you find it? Tell me the whole story!',
      context:
          'The child is pretending to be a pirate who found treasure. '
          'Ask about their ship, what was in the treasure chest, if they had '
          'a map, who was on their crew, and what they will do with the treasure.',
    ),
  ];
}

class _InterviewScenario {
  final String character;
  final String characterTitle;
  final String characterIntro;
  final String openingQuestion;
  final String context;

  const _InterviewScenario({
    required this.character,
    required this.characterTitle,
    required this.characterIntro,
    required this.openingQuestion,
    required this.context,
  });
}
