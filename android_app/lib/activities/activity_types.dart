import 'package:shared_preferences/shared_preferences.dart';

import '../models/age_band.dart';
import '../models/skill.dart';
import 'activity_base.dart';

/// Activity with daily episode/continuation support.
///
/// Subclasses persist their episode number so the child can resume
/// a multi-day storyline or challenge sequence each session.
abstract class EpisodicActivity extends Activity {
  /// The current episode number (1-based).
  int get currentEpisode;

  /// The SharedPreferences key used to persist episode progress.
  String get episodeKey;

  /// Save current episode progress to persistent storage.
  Future<void> saveProgress();

  /// Load previously saved episode progress from persistent storage.
  Future<void> loadProgress();

  /// Whether a new episode is available today (not yet completed today).
  bool get hasNewEpisode;
}

/// Activity with a countdown timer for timed challenges or Pomodoro sessions.
///
/// The timer ticks each second, calling [onTimerTick] with the remaining
/// duration, and fires [onTimerComplete] when it reaches zero.
abstract class TimerActivity extends Activity {
  /// Total duration of the timer.
  Duration get totalDuration;

  /// How much time has elapsed since the timer started.
  Duration get elapsed;

  /// Whether the timer is currently paused.
  bool get isPaused;

  /// Pause the timer.
  Future<void> pause();

  /// Resume the timer after a pause.
  Future<void> resume();

  /// Called each second while the timer is running.
  void onTimerTick(Duration remaining);

  /// Called when the timer reaches zero. Returns a completion message.
  Future<String> onTimerComplete();
}

/// Activity with virtual state such as money, inventory, or scores.
///
/// The simulation state is a flexible map that persists across sessions,
/// allowing complex multi-turn activities like shops or adventures.
abstract class SimulationActivity extends Activity {
  /// The current simulation state as a key-value map.
  Map<String, dynamic> get simulationState;

  /// Save the simulation state to persistent storage.
  Future<void> saveSimulationState();

  /// Load previously saved simulation state from persistent storage.
  Future<void> loadSimulationState();
}

/// Multi-participant activity with turn management.
///
/// Tracks a list of participants, manages whose turn it is, and provides
/// personalized feedback for each participant at the end.
abstract class GroupActivity extends Activity {
  /// List of participant names.
  List<String> get participants;

  /// Name of the participant whose turn it is, or null if not started.
  String? get currentParticipant;

  /// Set the list of participant names for this session.
  Future<void> setParticipants(List<String> names);

  /// Advance to the next participant's turn.
  Future<void> nextTurn();

  /// Get personalized feedback for each participant at session end.
  Map<String, String> getPerParticipantFeedback();
}

/// Activity that is relevant on specific calendar dates (festivals, holidays).
///
/// The activity checks whether today matches a known event and adapts
/// its content accordingly.
abstract class CalendarActivity extends Activity {
  /// Check if this activity has relevant content for the given [date].
  bool isRelevantToday(DateTime date);

  /// Get the event or festival name for the given [date].
  ///
  /// Returns null if no event is relevant on that date.
  String? getTodayEvent(DateTime date);
}
