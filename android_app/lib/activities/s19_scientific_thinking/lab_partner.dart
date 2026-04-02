import 'dart:math';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_router.dart';
import '../../core/llm/llm_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../s02_creativity/what_if_machine.dart' show SpeakSentenceCallback;

/// A home experiment with step-by-step instructions.
class _Experiment {
  final String title;
  final String materials;
  final List<String> steps;
  final String scienceBehindIt;

  const _Experiment({
    required this.title,
    required this.materials,
    required this.steps,
    required this.scienceBehindIt,
  });
}

/// LabPartner: guide home experiments step by step.
///
/// Teaches: experimental method, following procedures, observation,
/// scientific explanation.
///
/// Flow:
/// 1. Bot presents an experiment and its materials.
/// 2. Walks through each step one at a time.
/// 3. Asks what the child observes at key steps.
/// 4. Explains the science behind the result using LLM.
class LabPartner extends Activity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;
  SpeakSentenceCallback? onSpeakSentence;

  final Random _rng = Random();

  bool _active = false;
  int _experimentsCompleted = 0;
  int _score = 0;
  int _currentStep = 0;

  /// 0=pick experiment, 1=list materials, 2=walk through steps,
  /// 3=observe, 4=explain science, 5=next or end
  int _phase = 0;
  _Experiment? _currentExperiment;
  final List<int> _usedIndices = [];
  final List<Map<String, String>> _history = [];

  static const List<_Experiment> _experiments = [
    _Experiment(
      title: 'Volcano in a Cup',
      materials: 'baking soda, vinegar, a cup, food coloring (optional), '
          'and a tray to catch the mess',
      steps: [
        'Put 2 spoonfuls of baking soda in the cup.',
        'Add a few drops of food coloring if you have it.',
        'Now slowly pour some vinegar into the cup. Watch what happens!',
      ],
      scienceBehindIt: 'The baking soda and vinegar react to make carbon '
          'dioxide gas. The gas makes all those bubbles and fizz!',
    ),
    _Experiment(
      title: 'Dancing Raisins',
      materials: 'a clear glass, sparkling water or soda, and a few raisins',
      steps: [
        'Fill the glass with sparkling water or clear soda.',
        'Drop 4 or 5 raisins into the glass.',
        'Watch the raisins for a minute. What do they do?',
      ],
      scienceBehindIt: 'The bubbles of carbon dioxide gas stick to the bumpy '
          'raisins and lift them up. At the top the bubbles pop and the '
          'raisins sink back down. Up and down they dance!',
    ),
    _Experiment(
      title: 'Rainbow Milk',
      materials: 'a plate, whole milk, food coloring (2-3 colors), '
          'dish soap, and a cotton swab',
      steps: [
        'Pour enough milk to cover the bottom of the plate.',
        'Add a few drops of different food colors around the milk.',
        'Dip a cotton swab in dish soap, then touch it to the center '
            'of the milk. Watch the colors!',
      ],
      scienceBehindIt: 'The dish soap breaks apart the fat in the milk. '
          'As the fat molecules run away from the soap, they push the '
          'food coloring around, creating swirling rainbow patterns!',
    ),
    _Experiment(
      title: 'Pepper Run Away',
      materials: 'a bowl of water, ground black pepper, and dish soap',
      steps: [
        'Fill a bowl with water.',
        'Sprinkle pepper all over the surface of the water.',
        'Dip your finger in dish soap, then touch the center of the water. '
            'Watch the pepper!',
      ],
      scienceBehindIt: 'Water has surface tension, like a thin skin on top. '
          'The pepper floats on this skin. Soap breaks the surface tension, '
          'and the water pulls away from your finger, carrying the pepper '
          'to the edges!',
    ),
    _Experiment(
      title: 'Walking Water',
      materials: 'three glasses, water, food coloring (red and blue), '
          'and two paper towels',
      steps: [
        'Fill the first and third glasses with water. Leave the middle '
            'glass empty.',
        'Add red food coloring to the first glass and blue to the third.',
        'Fold two paper towels into strips. Put one end of each strip in '
            'a colored glass and the other end in the empty middle glass.',
        'Wait about 30 minutes and check what happens to the middle glass!',
      ],
      scienceBehindIt: 'The water climbs up the tiny holes in the paper '
          'towel. This is called capillary action! The red and blue water '
          'meet in the middle glass and mix to make purple.',
    ),
    _Experiment(
      title: 'Invisible Ink',
      materials: 'lemon juice, a cotton swab or small brush, white paper, '
          'and a lamp or hair dryer for heat',
      steps: [
        'Dip the cotton swab in lemon juice.',
        'Write a secret message on the white paper using the lemon juice.',
        'Let it dry completely. The message will be invisible!',
        'Hold the paper near a warm lamp or use a hair dryer on it. '
            'Watch your message appear!',
      ],
      scienceBehindIt: 'Lemon juice is made of carbon compounds. When you '
          'heat the paper, the carbon in the lemon juice oxidizes and turns '
          'brown before the paper does, revealing your secret message!',
    ),
  ];

  LabPartner({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'lab_partner';

  @override
  String get name => 'Lab Partner';

  @override
  String get category => 'science';

  @override
  String get description =>
      'Do fun home science experiments with step-by-step guidance.';

  @override
  List<String> get skills => [
        'experimental method',
        'following procedures',
        'observation',
        'scientific explanation',
      ];

  @override
  int get minAge => 4;

  @override
  int get maxAge => 12;

  @override
  SkillId? get skillId => SkillId.scientificThinking;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'lab partner',
          'do an experiment',
          'science experiment',
          'home experiment',
          'let us experiment',
        ],
        'hi': [
          'प्रयोग करो',
          'विज्ञान प्रयोग',
          'घर पर प्रयोग',
        ],
        'te': [
          'ప్రయోగం చేయి',
          'సైన్స్ ప్రయోగం',
          'ఇంట్లో ప్రయోగం',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_experimentsCompleted == 0 && _currentExperiment == null) {
      return 'No experiments done yet.';
    }
    return '$_experimentsCompleted experiment${_experimentsCompleted != 1 ? 's' : ''} '
        'completed. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _active = true;
    _experimentsCompleted = 0;
    _score = 0;
    _currentStep = 0;
    _phase = 0;
    _usedIndices.clear();
    _history.clear();

    return 'Welcome to the Science Lab! I am your lab partner today. '
        'We are going to do real experiments together! '
        '${_pickExperiment()}';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      return await end();
    }

    switch (_phase) {
      case 1:
        // Child confirmed they have materials, start steps
        if (_isReady(lower)) {
          _phase = 2;
          _currentStep = 0;
          return _presentStep();
        }
        return 'No worries! Take your time gathering the materials. '
            'Tell me when you are ready!';

      case 2:
        // Walking through steps
        _currentStep++;
        _score += 10;

        if (_currentStep >= _currentExperiment!.steps.length) {
          // All steps done, ask what they observed
          _phase = 3;
          return 'You finished all the steps! Now here is the important '
              'scientist question: what did you observe? What happened?';
        }
        return 'Great job! ${_presentStep()}';

      case 3:
        // Child described observation, explain the science
        _phase = 4;
        _score += 15;
        _history.add({'role': 'user', 'content': childSaid});

        final explanation = await _getLlmResponse(
          childSaid,
          'The child just completed the experiment '
          '"${_currentExperiment!.title}" and observed: "$childSaid". '
          'Validate their observation. Then explain the science behind it '
          'in simple terms: ${_currentExperiment!.scienceBehindIt} '
          'Make it fun and easy to understand. 3-4 sentences.',
        );
        return '$explanation Would you like to try another experiment?';

      case 4:
        // Ask if they want another
        _experimentsCompleted++;
        if (_isAffirmative(lower) &&
            _experimentsCompleted < _experiments.length) {
          _history.clear();
          return _pickExperiment();
        }
        return await end();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_experimentsCompleted == 0 && _currentStep == 0) {
      return 'Come back to the Science Lab anytime! There are so many '
          'cool experiments to try.';
    }
    if (_currentStep > 0) _experimentsCompleted++;
    return 'What a great science session! You completed '
        '$_experimentsCompleted '
        'experiment${_experimentsCompleted != 1 ? 's' : ''}. '
        'Score: $_score points! Keep experimenting and asking questions!';
  }

  String _pickExperiment() {
    final available = <int>[];
    for (int i = 0; i < _experiments.length; i++) {
      if (!_usedIndices.contains(i)) available.add(i);
    }
    if (available.isEmpty) {
      _usedIndices.clear();
      for (int i = 0; i < _experiments.length; i++) {
        available.add(i);
      }
    }

    final idx = available[_rng.nextInt(available.length)];
    _usedIndices.add(idx);
    _currentExperiment = _experiments[idx];
    _currentStep = 0;
    _phase = 1;

    return 'Today we are doing: ${_currentExperiment!.title}! '
        'Here is what you need: ${_currentExperiment!.materials}. '
        'Do you have everything ready? Say ready when you are!';
  }

  String _presentStep() {
    final step = _currentExperiment!.steps[_currentStep];
    final stepNum = _currentStep + 1;
    final totalSteps = _currentExperiment!.steps.length;
    return 'Step $stepNum of $totalSteps: $step '
        'Tell me when you are done!';
  }

  Future<String> _getLlmResponse(String childSaid, String guidance) async {
    final systemPrompt =
        'You are Kore, a friendly lab partner for kids aged 4-12. '
        'You guide home science experiments and explain the science '
        'in simple, exciting terms. '
        '$guidance '
        'Rules: Keep explanations simple. Celebrate the child\'s observations. '
        'Make science feel magical and fun. '
        'No markdown, no bullets, no emojis. Speak naturally.';

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
      return _currentExperiment!.scienceBehindIt;
    } catch (e) {
      return _currentExperiment!.scienceBehindIt;
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
      if (result.isNotEmpty) {
        _history.add({'role': 'assistant', 'content': result});
      }
      return result.isNotEmpty ? result : _currentExperiment!.scienceBehindIt;
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return _currentExperiment!.scienceBehindIt;
    }
  }

  bool _isReady(String text) {
    const words = ['ready', 'yes', 'got it', 'have it', 'okay', 'ok',
      'let us go', 'start', 'go', 'हाँ', 'तैयार', 'అవును', 'సిద్ధం'];
    return words.any((w) => text.contains(w));
  }

  bool _isAffirmative(String text) {
    const words = ['yes', 'yeah', 'sure', 'okay', 'another', 'more',
      'next', 'one more', 'हाँ', 'और', 'అవును', 'ఇంకొకటి'];
    return words.any((w) => text.contains(w));
  }

  bool _containsQuit(String text) {
    const quitWords = ['quit', 'exit', 'stop', 'done', 'finish', 'enough',
      'no', 'no thanks'];
    return quitWords.any((w) => text.contains(w));
  }
}
