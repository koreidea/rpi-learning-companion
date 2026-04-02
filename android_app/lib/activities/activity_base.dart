import '../models/age_band.dart';
import '../models/skill.dart';

/// Base class for structured learning activities.
///
/// Each activity is a multi-turn voice interaction with a specific learning goal.
/// Activities are designed for 3-6 year old children and last 3-5 minutes.
/// All text output is designed for TTS (natural spoken language, no markdown).
abstract class Activity {
  /// Unique identifier, e.g. 'coding_sequence'.
  String get id;

  /// Human-readable display name.
  String get name;

  /// Category: 'coding', 'thinking', 'communication', 'creativity',
  /// 'world', 'emotions', 'science', 'math'.
  String get category;

  /// Short description of what this activity teaches.
  String get description;

  /// 21st century skills covered, e.g. ['sequencing', 'logical thinking'].
  List<String> get skills;

  /// Minimum recommended age.
  int get minAge;

  /// Maximum recommended age.
  int get maxAge;

  /// The primary 21st century skill this activity develops.
  /// Returns null for activities that have not been mapped to a skill yet.
  SkillId? get skillId => null;

  /// Voice trigger phrases that launch this activity, keyed by language code.
  /// Keys: 'en', 'hi', 'te'. Values: list of trigger phrases (lowercase).
  /// Returns an empty map for activities that have not defined triggers yet.
  Map<String, List<String>> get voiceTriggers => const {};

  /// The target age band for this activity.
  AgeBand get targetAgeBand => AgeBand.nursery;

  /// Whether the activity is currently in progress.
  bool get isActive;

  /// Start the activity. Returns the intro text to speak aloud.
  Future<String> start();

  /// Process the child's spoken response and return the bot's next message.
  ///
  /// Returns `null` when the activity is complete.
  /// Implementations must handle unexpected/off-topic responses gracefully.
  Future<String?> processResponse(String childSaid);

  /// End the activity early. Returns a friendly goodbye/summary message.
  Future<String> end();

  /// A short spoken summary of current score or progress.
  String get progressSummary;
}
