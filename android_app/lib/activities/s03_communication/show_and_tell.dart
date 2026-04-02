import 'dart:math';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../../vision/camera_manager.dart';
import '../../vision/vision_describer.dart';

/// Show-and-tell activity that uses the camera to see what the child shows.
///
/// The child holds up an object, the bot uses GPT-4o-mini vision to identify
/// it, then asks 3-4 follow-up questions to practice verbal expression and
/// storytelling. Each show-and-tell round takes about 1-2 minutes.
class ShowAndTell extends Activity {
  final CameraManager _cameraManager;
  final VisionDescriber _visionDescriber;
  final String _apiKey;

  final Random _random = Random();

  bool _active = false;
  int _itemsShown = 0;
  int _maxItems = 3;

  // Current item state
  _ShowAndTellPhase _phase = _ShowAndTellPhase.idle;
  int _followUpIndex = 0;
  String? _identifiedObject;

  @override
  String get id => 'communication_show_and_tell';

  @override
  String get name => 'Show and Tell';

  @override
  String get category => 'communication';

  @override
  String get description =>
      'Show something to the camera and tell me all about it!';

  @override
  List<String> get skills => ['verbal expression', 'storytelling', 'communication'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  SkillId? get skillId => SkillId.communication;

  @override
  Map<String, List<String>> get voiceTriggers => const {
    'en': ['show and tell', 'show you something', 'look at this', 'see what i have', 'let me show'],
    'hi': ['दिखाओ', 'देखो ये', 'ये देखो'],
    'te': ['చూపిస్తా', 'చూడు ఇది'],
  };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  /// Create a ShowAndTell activity.
  ///
  /// Requires a [CameraManager] for capturing frames, a [VisionDescriber]
  /// for identifying objects via the vision API, and an [apiKey] for the
  /// OpenAI API.
  ShowAndTell({
    required CameraManager cameraManager,
    required VisionDescriber visionDescriber,
    required String apiKey,
  })  : _cameraManager = cameraManager,
        _visionDescriber = visionDescriber,
        _apiKey = apiKey;

  @override
  Future<String> start() async {
    _active = true;
    _itemsShown = 0;
    _phase = _ShowAndTellPhase.waitingForObject;
    _followUpIndex = 0;
    _identifiedObject = null;

    return "Time for show and tell! Hold up something you want to show me "
        "and say ready when I should look!";
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
      case _ShowAndTellPhase.idle:
        return "Hold up something you want to show me and say ready!";

      case _ShowAndTellPhase.waitingForObject:
        return await _captureAndIdentify();

      case _ShowAndTellPhase.askingFollowUps:
        return _handleFollowUpResponse(childSaid);

      case _ShowAndTellPhase.waitingForNextItem:
        if (_containsNo(lower)) {
          _active = false;
          return _buildEndSummary();
        }
        _phase = _ShowAndTellPhase.waitingForObject;
        return "Great! Hold up your next item and say ready when I should look!";
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    return _buildEndSummary();
  }

  @override
  String get progressSummary {
    if (_itemsShown == 0) return 'No items shown yet.';
    return 'Showed $_itemsShown ${_itemsShown == 1 ? 'item' : 'items'} so far.';
  }

  // -- Internal --

  Future<String> _captureAndIdentify() async {
    // Initialize camera if needed
    final cameraReady = await _cameraManager.init();
    if (!cameraReady) {
      return "Hmm, I cannot access the camera right now. Can you try again?";
    }

    // Capture a frame
    final frame = await _cameraManager.captureFrame();
    if (frame == null || frame.isEmpty) {
      return "I could not take a picture. Hold the object up and say ready again!";
    }

    // Send to vision API
    final description = await _visionDescriber.describe(
      imageBytes: frame,
      apiKey: _apiKey,
      userPrompt: 'A child is doing show and tell. Identify the main object '
          'they are holding up. Describe it briefly in a fun, excited way '
          'for a young child. Keep it to 1-2 sentences.',
    );

    _identifiedObject = description;
    _phase = _ShowAndTellPhase.askingFollowUps;
    _followUpIndex = 0;

    // Ask the first follow-up question
    return "$description ${_followUpQuestions[0]}";
  }

  String _handleFollowUpResponse(String childSaid) {
    _followUpIndex++;

    if (_followUpIndex >= _followUpQuestions.length) {
      // Done with this item
      _itemsShown++;
      _identifiedObject = null;

      if (_itemsShown >= _maxItems) {
        _active = false;
        return "Thank you for sharing that with me! You did a wonderful "
            "show and tell! ${_buildEndSummary()}";
      }

      _phase = _ShowAndTellPhase.waitingForNextItem;
      return "${_encouragements[_random.nextInt(_encouragements.length)]} "
          "Thank you for telling me all about it! "
          "Do you want to show me something else?";
    }

    // Acknowledge what they said and ask the next follow-up
    final ack = _acknowledgements[_random.nextInt(_acknowledgements.length)];
    return "$ack ${_followUpQuestions[_followUpIndex]}";
  }

  String _buildEndSummary() {
    if (_itemsShown == 0) {
      return "Thanks for playing Show and Tell! Come back anytime to show me your stuff!";
    }
    return "You showed me $_itemsShown ${_itemsShown == 1 ? 'thing' : 'things'}! "
        "You are a great presenter! I loved hearing about your stuff!";
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

  static const List<String> _followUpQuestions = [
    'Where did you get it?',
    'What is your favorite thing about it?',
    'What can you do with it?',
    'Would you like to tell me anything else about it?',
  ];

  static const List<String> _acknowledgements = [
    'Oh, that is so cool!',
    'Wow, that is really interesting!',
    'I love hearing about that!',
    'That is wonderful!',
    'How fun!',
    'That is a great story!',
  ];

  static const List<String> _encouragements = [
    'You did a wonderful show and tell!',
    'You are such a great storyteller!',
    'I love how you described that!',
    'What a great presenter you are!',
  ];
}

enum _ShowAndTellPhase {
  idle,
  waitingForObject,
  askingFollowUps,
  waitingForNextItem,
}
