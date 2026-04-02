import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// Pitch to Kore: kid pitches a business idea and the bot plays investor,
/// asking questions and scoring the pitch.
///
/// Teaches: presentation skills, business thinking, clarity of expression,
/// handling questions, entrepreneurial thinking.
///
/// Flow:
/// 1. Bot asks the child to pitch a business idea.
/// 2. Child presents their idea.
/// 3. Bot asks investor-style questions (who would buy, cost, competition, marketing).
/// 4. Bot rates the pitch: Problem (5), Solution (5), Practicality (5), Presentation (5).
/// 5. Total score out of 20.
class PitchToKore extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  bool _active = false;

  /// 0=intro, 1=who buys, 2=cost, 3=competition, 4=marketing, 5=scoring
  int _phase = 0;
  final List<Map<String, String>> _history = [];

  // Scores per category (0-5 each)
  int _problemScore = 0;
  int _solutionScore = 0;
  int _practicalityScore = 0;
  int _presentationScore = 0;

  PitchToKore({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'entrepreneurial_pitch_to_kore';

  @override
  String get name => 'Pitch to Kore';

  @override
  String get category => 'entrepreneurial';

  @override
  String get description =>
      'Pitch your business idea to me and I will ask investor questions.';

  @override
  List<String> get skills => [
        'presentation',
        'business thinking',
        'communication',
        'entrepreneurial thinking',
      ];

  @override
  int get minAge => 6;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.entrepreneurial;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'pitch to kore',
          'pitch my idea',
          'business pitch',
          'shark tank',
          'investor game',
        ],
        'hi': [
          'बिज़नेस पिच',
          'आइडिया बताओ',
          'निवेशक खेल',
        ],
        'te': [
          'వ్యాపార ఆలోచన చెప్పు',
          'పిచ్ చేయి',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    final total = _problemScore + _solutionScore + _practicalityScore + _presentationScore;
    return 'Pitch score: $total out of 20.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _phase = 0;
    _history.clear();
    _problemScore = 0;
    _solutionScore = 0;
    _practicalityScore = 0;
    _presentationScore = 0;

    return 'Welcome to Pitch to Kore! I am your friendly investor. '
        'You get to pitch me a business idea, any idea at all! '
        'It could be an app, a product, a service, anything you can imagine. '
        'I will ask some questions and then score your pitch out of 20. '
        'So, what is your big idea?';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      return await end();
    }

    _history.add({'role': 'user', 'content': childSaid});

    switch (_phase) {
      case 0:
        // Heard the pitch, ask who would buy
        _phase = 1;
        _problemScore = _assessResponse(childSaid);
        return await _getLlmResponse(
            'The child just pitched their business idea. '
            'Acknowledge it enthusiastically. Then ask: Who would buy this or use this? '
            'Who is your customer? Keep it to 2-3 sentences.');

      case 1:
        // Heard who buys, ask about cost
        _phase = 2;
        _solutionScore = _assessResponse(childSaid);
        return await _getLlmResponse(
            'The child answered who would buy their product. Respond positively. '
            'Then ask: How much would it cost to make? And how much would you '
            'sell it for? Keep it to 2-3 sentences.');

      case 2:
        // Heard about cost, ask about competition
        _phase = 3;
        return await _getLlmResponse(
            'The child talked about cost and pricing. Acknowledge their answer. '
            'Then ask: What if someone else is already making something like this? '
            'What makes yours special or different? Keep it to 2-3 sentences.');

      case 3:
        // Heard about competition, ask about marketing
        _phase = 4;
        _practicalityScore = _assessResponse(childSaid);
        return await _getLlmResponse(
            'The child talked about what makes their idea different. Good! '
            'Now ask: How would people find out about your product? '
            'How would you tell the world about it? Keep it to 2-3 sentences.');

      case 4:
        // Heard marketing plan, score the pitch
        _phase = 5;
        _presentationScore = _assessResponse(childSaid);
        _active = false;
        return _buildScoreCard();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    final total = _problemScore + _solutionScore + _practicalityScore + _presentationScore;
    if (total == 0) {
      return 'Come back anytime with your business ideas! Every great company '
          'started with someone just like you.';
    }
    return 'Thanks for pitching to me! Your score was $total out of 20. '
        'Keep thinking up ideas, you have a great entrepreneurial mind!';
  }

  /// Simple heuristic to score a response based on length and detail.
  int _assessResponse(String response) {
    final words = response.split(' ').length;
    if (words >= 20) return 5;
    if (words >= 12) return 4;
    if (words >= 6) return 3;
    if (words >= 3) return 2;
    return 1;
  }

  String _buildScoreCard() {
    final total = _problemScore + _solutionScore + _practicalityScore + _presentationScore;

    String rating;
    if (total >= 18) {
      rating = 'Incredible pitch! You are a natural entrepreneur!';
    } else if (total >= 14) {
      rating = 'Great pitch! You clearly thought this through!';
    } else if (total >= 10) {
      rating = 'Good effort! With some more detail, this could be even better!';
    } else {
      rating = 'Nice try! Keep practicing and your pitches will get stronger!';
    }

    return 'Okay, let me score your pitch! '
        'Problem clarity: $_problemScore out of 5. '
        'Solution creativity: $_solutionScore out of 5. '
        'Practicality: $_practicalityScore out of 5. '
        'Presentation: $_presentationScore out of 5. '
        'Total score: $total out of 20! '
        '$rating Remember, even the biggest companies started with a simple idea!';
  }

  Future<String> _getLlmResponse(String guidance) async {
    final systemPrompt = 'You are Kore, a friendly investor talking to a child aged 6-14. '
        'You are evaluating their business pitch. $guidance '
        'Rules: Be warm and encouraging. No markdown, no bullets. '
        'Speak naturally. Use simple language.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._history,
    ];

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
      if (response.isNotEmpty) {
        _history.add({'role': 'assistant', 'content': response});
        return response;
      }
      return _fallback();
    } catch (e) {
      return _fallback();
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
      if (result.isNotEmpty) _history.add({'role': 'assistant', 'content': result});
      return result.isNotEmpty ? result : _fallback();
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return _fallback();
    }
  }

  String _fallback() {
    return 'That is a really interesting idea! Tell me more about how it would work.';
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough'];
    for (final w in quitWords) {
      if (text.contains(w)) return true;
    }
    return false;
  }
}
