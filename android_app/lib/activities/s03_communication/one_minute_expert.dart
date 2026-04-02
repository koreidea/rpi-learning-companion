import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../core/llm/llm_router.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../activity_types.dart';

/// One Minute Expert: timed speaking challenge where the child explains a
/// topic in 60 seconds.
///
/// Teaches: communication, public speaking, confidence, and content knowledge.
/// The bot picks a topic, starts a 60-second countdown, the child explains,
/// and the bot gives constructive LLM-powered feedback.
class OneMinuteExpert extends TimerActivity {
  final LlmRouter _llmRouter;
  final Random _random = Random();

  bool _active = false;
  Duration _elapsed = Duration.zero;
  bool _isPaused = false;
  Timer? _timer;
  int _round = 0;
  static const int _maxRounds = 3;
  String? _currentTopic;
  _Phase _phase = _Phase.idle;
  String _childExplanation = '';

  final List<int> _usedTopicIndices = [];

  /// Optional callback invoked each second with remaining time.
  void Function(Duration remaining)? onTick;

  /// Optional callback invoked when the timer completes.
  Future<String> Function()? onComplete;

  OneMinuteExpert({
    required LlmRouter llmRouter,
    this.onTick,
    this.onComplete,
  }) : _llmRouter = llmRouter;

  @override
  String get id => 'communication_one_minute_expert';

  @override
  String get name => 'One Minute Expert';

  @override
  String get category => 'communication';

  @override
  String get description =>
      'Explain a topic in 60 seconds and get helpful feedback!';

  @override
  List<String> get skills =>
      ['public speaking', 'communication', 'confidence', 'knowledge'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.communication;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'one minute expert',
          'explain game',
          'timed talk',
          'quick explain',
          'expert game',
        ],
        'hi': ['एक मिनट एक्सपर्ट', 'समझाओ खेल', 'जल्दी बोलो'],
        'te': ['ఒక నిమిషం నిపుణుడు', 'వివరించు ఆట'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  Duration get totalDuration => const Duration(seconds: 60);

  @override
  Duration get elapsed => _elapsed;

  @override
  bool get isPaused => _isPaused;

  @override
  Future<void> pause() async {
    _isPaused = true;
    _timer?.cancel();
  }

  @override
  Future<void> resume() async {
    _isPaused = false;
    _startTimer();
  }

  @override
  void onTimerTick(Duration remaining) {
    onTick?.call(remaining);
  }

  @override
  Future<String> onTimerComplete() async {
    _timer?.cancel();
    _phase = _Phase.givingFeedback;
    return await _generateFeedback();
  }

  @override
  Future<String> start() async {
    _active = true;
    _round = 0;
    _usedTopicIndices.clear();
    _phase = _Phase.presentingTopic;

    return _presentNewTopic();
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _timer?.cancel();
      _active = false;
      return _buildEndSummary();
    }

    switch (_phase) {
      case _Phase.idle:
        return "Say ready when you want to start explaining!";

      case _Phase.presentingTopic:
        // Child acknowledged, start the timer
        _phase = _Phase.timerRunning;
        _elapsed = Duration.zero;
        _childExplanation = '';
        _startTimer();
        return "Go! You have 60 seconds. Tell me everything you know about "
            "$_currentTopic!";

      case _Phase.timerRunning:
        // Accumulate what the child says during the timer
        _childExplanation += ' $childSaid';
        return null; // Do not interrupt during the timer

      case _Phase.givingFeedback:
        // Child responded to feedback, offer next round
        _round++;
        if (_round >= _maxRounds) {
          _active = false;
          return "You were amazing! ${_buildEndSummary()}";
        }
        _phase = _Phase.waitingForNext;
        return "Want to try another topic?";

      case _Phase.waitingForNext:
        if (_containsNo(lower)) {
          _active = false;
          return _buildEndSummary();
        }
        return _presentNewTopic();
    }
  }

  @override
  Future<String> end() async {
    _timer?.cancel();
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_round == 0) return 'No topics explained yet.';
    return 'Explained $_round ${_round == 1 ? 'topic' : 'topics'}.';
  }

  // -- Internal --

  void _startTimer() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_isPaused) return;
      _elapsed += const Duration(seconds: 1);
      final remaining = totalDuration - _elapsed;
      onTimerTick(remaining);

      if (_elapsed >= totalDuration) {
        timer.cancel();
        onTimerComplete().then((_) {});
      }
    });
  }

  String _presentNewTopic() {
    if (_usedTopicIndices.length >= _topics.length) {
      _usedTopicIndices.clear();
    }
    int index;
    do {
      index = _random.nextInt(_topics.length);
    } while (_usedTopicIndices.contains(index));
    _usedTopicIndices.add(index);

    _currentTopic = _topics[index];
    _phase = _Phase.presentingTopic;

    return "Your topic is: $_currentTopic! You will have 60 seconds to "
        "explain everything you know about it. Say ready when you want to start!";
  }

  Future<String> _generateFeedback() async {
    if (_childExplanation.trim().isEmpty) {
      _phase = _Phase.givingFeedback;
      return "It looks like the time ran out! That is okay. Sometimes topics "
          "are tricky. You will do great next time!";
    }

    try {
      final provider = _llmRouter.getProvider();
      final prompt =
          'You are a friendly communication coach for children aged 5-14. '
          'A child was asked to explain "$_currentTopic" in 60 seconds. '
          'Here is what they said: "$_childExplanation" '
          'Give brief, encouraging feedback in 2-3 sentences. '
          'Mention ONE thing they did well and ONE specific tip to improve. '
          'Focus on: clarity, confidence, and completeness. '
          'Do NOT focus on accent or grammar. '
          'Do not use emojis or markdown. Speak naturally and warmly.';

      final messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': _childExplanation.trim()},
      ];

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }

      final result = buffer.toString().trim();
      if (result.isNotEmpty) {
        _phase = _Phase.givingFeedback;
        return "Time is up! $result";
      }
    } catch (e) {
      debugPrint('[OneMinuteExpert] LLM error: $e');
    }

    _phase = _Phase.givingFeedback;
    return "Time is up! Great job explaining $_currentTopic! You spoke "
        "clearly and shared some interesting ideas. Next time, try adding "
        "an example to make your explanation even better!";
  }

  String _buildEndSummary() {
    if (_round == 0) {
      return "Thanks for trying One Minute Expert! Come back anytime to practice!";
    }
    return "You explained $_round ${_round == 1 ? 'topic' : 'topics'} like a "
        "pro! Every time you practice explaining things, you become a better "
        "communicator. Keep it up!";
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

  static const List<String> _topics = [
    'what a rainbow is',
    'how volcanoes work',
    'the ocean and the creatures in it',
    'outer space and the planets',
    'dinosaurs',
    'why we have seasons',
    'how rain happens',
    'how plants grow',
    'the moon and its phases',
    'how cars work',
    'why birds can fly',
    'how fish breathe underwater',
    'what earthquakes are',
    'how bees make honey',
    'why the sky is blue',
  ];
}

enum _Phase {
  idle,
  presentingTopic,
  timerRunning,
  givingFeedback,
  waitingForNext,
}
