import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// A goal the child is working toward.
class Goal {
  /// Unique identifier for this goal.
  final String id;

  /// Display title of the goal.
  final String title;

  /// Optional skill ID this goal is associated with.
  final String? skillId;

  /// When the goal was created.
  final DateTime createdAt;

  /// Optional target date for achieving the goal.
  final DateTime? targetDate;

  /// Whether the goal has been completed.
  final bool completed;

  /// Milestone descriptions for incremental progress.
  final List<String> milestones;

  /// Creates a [Goal] with all metadata.
  Goal({
    required this.id,
    required this.title,
    this.skillId,
    required this.createdAt,
    this.targetDate,
    this.completed = false,
    this.milestones = const [],
  });

  /// Serialize this goal to a JSON-compatible map.
  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'skillId': skillId,
        'createdAt': createdAt.toIso8601String(),
        'targetDate': targetDate?.toIso8601String(),
        'completed': completed,
        'milestones': milestones,
      };

  /// Deserialize a [Goal] from a JSON map.
  factory Goal.fromJson(Map<String, dynamic> json) => Goal(
        id: json['id'] as String? ?? '',
        title: json['title'] as String? ?? '',
        skillId: json['skillId'] as String?,
        createdAt: DateTime.tryParse(json['createdAt'] as String? ?? '') ??
            DateTime.now(),
        targetDate: json['targetDate'] != null
            ? DateTime.tryParse(json['targetDate'] as String)
            : null,
        completed: json['completed'] as bool? ?? false,
        milestones: (json['milestones'] as List<dynamic>?)
                ?.map((m) => m as String)
                .toList() ??
            const [],
      );

  /// Return a copy with the [completed] flag set to true.
  Goal copyWithCompleted() => Goal(
        id: id,
        title: title,
        skillId: skillId,
        createdAt: createdAt,
        targetDate: targetDate,
        completed: true,
        milestones: milestones,
      );
}

/// A daily habit the child is building.
class Habit {
  /// Unique identifier for this habit.
  final String id;

  /// Display title of the habit.
  final String title;

  /// Optional skill ID this habit is associated with.
  final String? skillId;

  /// When the habit tracking started.
  final DateTime startDate;

  /// Dates on which the habit was completed.
  final List<DateTime> completedDates;

  /// Creates a [Habit] with all metadata.
  Habit({
    required this.id,
    required this.title,
    this.skillId,
    required this.startDate,
    List<DateTime>? completedDates,
  }) : completedDates = completedDates ?? [];

  /// Current consecutive-day streak ending today or yesterday.
  int get streakDays {
    if (completedDates.isEmpty) return 0;

    final sorted = List<DateTime>.from(completedDates)
      ..sort((a, b) => b.compareTo(a));
    final today = _dateOnly(DateTime.now());
    final yesterday = today.subtract(const Duration(days: 1));

    // Streak must include today or yesterday to be current.
    final latest = _dateOnly(sorted.first);
    if (latest != today && latest != yesterday) return 0;

    int streak = 1;
    for (int i = 1; i < sorted.length; i++) {
      final current = _dateOnly(sorted[i - 1]);
      final previous = _dateOnly(sorted[i]);
      if (current.difference(previous).inDays == 1) {
        streak++;
      } else {
        break;
      }
    }
    return streak;
  }

  /// Total number of days the habit was completed.
  int get totalDays => completedDates.length;

  /// Whether the habit has been completed today.
  bool get isCompletedToday {
    final today = _dateOnly(DateTime.now());
    return completedDates.any((d) => _dateOnly(d) == today);
  }

  /// Serialize this habit to a JSON-compatible map.
  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'skillId': skillId,
        'startDate': startDate.toIso8601String(),
        'completedDates':
            completedDates.map((d) => d.toIso8601String()).toList(),
      };

  /// Deserialize a [Habit] from a JSON map.
  factory Habit.fromJson(Map<String, dynamic> json) => Habit(
        id: json['id'] as String? ?? '',
        title: json['title'] as String? ?? '',
        skillId: json['skillId'] as String?,
        startDate: DateTime.tryParse(json['startDate'] as String? ?? '') ??
            DateTime.now(),
        completedDates: (json['completedDates'] as List<dynamic>?)
                ?.map((d) => DateTime.parse(d as String))
                .toList() ??
            [],
      );

  static DateTime _dateOnly(DateTime dt) => DateTime(dt.year, dt.month, dt.day);
}

/// A daily journal reflection entry.
class DailyReflection {
  /// The date this reflection was written.
  final DateTime date;

  /// The reflection text.
  final String text;

  /// Optional mood tag (e.g. 'happy', 'curious', 'tired').
  final String? mood;

  /// Creates a [DailyReflection].
  DailyReflection({
    required this.date,
    required this.text,
    this.mood,
  });

  /// Serialize this reflection to a JSON-compatible map.
  Map<String, dynamic> toJson() => {
        'date': date.toIso8601String(),
        'text': text,
        'mood': mood,
      };

  /// Deserialize a [DailyReflection] from a JSON map.
  factory DailyReflection.fromJson(Map<String, dynamic> json) =>
      DailyReflection(
        date: DateTime.tryParse(json['date'] as String? ?? '') ?? DateTime.now(),
        text: json['text'] as String? ?? '',
        mood: json['mood'] as String?,
      );
}

/// Service for tracking goals, habits, and daily reflections.
///
/// All data is persisted to SharedPreferences as JSON and loaded on [init].
class TrackingService {
  static const _goalsKey = 'tracking_goals';
  static const _habitsKey = 'tracking_habits';
  static const _reflectionsKey = 'tracking_reflections';

  List<Goal> _goals = [];
  List<Habit> _habits = [];
  List<DailyReflection> _reflections = [];

  SharedPreferences? _prefs;

  /// Initialize the service by loading persisted data from SharedPreferences.
  ///
  /// Must be called once at app startup before accessing any data.
  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
    _loadGoals();
    _loadHabits();
    _loadReflections();
    debugPrint('[TrackingService] Loaded ${_goals.length} goals, '
        '${_habits.length} habits, ${_reflections.length} reflections');
  }

  // -- Goals --

  /// All goals that have not been completed.
  List<Goal> get activeGoals =>
      _goals.where((g) => !g.completed).toList();

  /// All goals including completed ones.
  List<Goal> get allGoals => List.unmodifiable(_goals);

  /// Add a new goal and persist.
  Future<void> addGoal(Goal goal) async {
    _goals.add(goal);
    await _save();
  }

  /// Mark a goal as completed by its [id] and persist.
  Future<void> completeGoal(String id) async {
    final index = _goals.indexWhere((g) => g.id == id);
    if (index == -1) return;
    _goals[index] = _goals[index].copyWithCompleted();
    await _save();
  }

  /// Remove a goal by its [id] and persist.
  Future<void> removeGoal(String id) async {
    _goals.removeWhere((g) => g.id == id);
    await _save();
  }

  // -- Habits --

  /// All habits currently being tracked.
  List<Habit> get activeHabits => List.unmodifiable(_habits);

  /// Add a new habit and persist.
  Future<void> addHabit(Habit habit) async {
    _habits.add(habit);
    await _save();
  }

  /// Mark a habit as completed for the given [date] and persist.
  ///
  /// Ignores duplicate completions for the same date.
  Future<void> markHabitComplete(String id, DateTime date) async {
    final habit = _habits.firstWhere(
      (h) => h.id == id,
      orElse: () => throw StateError('Habit $id not found'),
    );
    final dateOnly = DateTime(date.year, date.month, date.day);
    final alreadyDone = habit.completedDates.any(
      (d) => DateTime(d.year, d.month, d.day) == dateOnly,
    );
    if (!alreadyDone) {
      habit.completedDates.add(dateOnly);
      await _save();
    }
  }

  /// Remove a habit by its [id] and persist.
  Future<void> removeHabit(String id) async {
    _habits.removeWhere((h) => h.id == id);
    await _save();
  }

  /// Get the longest streak ever achieved for a habit by [habitId].
  int getLongestStreak(String habitId) {
    final habit = _habits.firstWhere(
      (h) => h.id == habitId,
      orElse: () => throw StateError('Habit $habitId not found'),
    );
    if (habit.completedDates.isEmpty) return 0;

    final sorted = List<DateTime>.from(habit.completedDates)
      ..sort();
    int longest = 1;
    int current = 1;
    for (int i = 1; i < sorted.length; i++) {
      final prev = DateTime(sorted[i - 1].year, sorted[i - 1].month, sorted[i - 1].day);
      final curr = DateTime(sorted[i].year, sorted[i].month, sorted[i].day);
      if (curr.difference(prev).inDays == 1) {
        current++;
        if (current > longest) longest = current;
      } else if (curr != prev) {
        current = 1;
      }
    }
    return longest;
  }

  // -- Reflections --

  /// Add a daily reflection and persist.
  Future<void> addReflection(DailyReflection reflection) async {
    _reflections.add(reflection);
    await _save();
  }

  /// Get the reflection written on a specific [date], if any.
  DailyReflection? getReflectionForDate(DateTime date) {
    final target = DateTime(date.year, date.month, date.day);
    try {
      return _reflections.firstWhere(
        (r) => DateTime(r.date.year, r.date.month, r.date.day) == target,
      );
    } catch (_) {
      return null;
    }
  }

  /// The most recent 7 reflections, newest first.
  List<DailyReflection> get recentReflections {
    final sorted = List<DailyReflection>.from(_reflections)
      ..sort((a, b) => b.date.compareTo(a.date));
    return sorted.take(7).toList();
  }

  // -- Persistence --

  Future<void> _save() async {
    final prefs = _prefs;
    if (prefs == null) return;

    try {
      await prefs.setString(
        _goalsKey,
        jsonEncode(_goals.map((g) => g.toJson()).toList()),
      );
      await prefs.setString(
        _habitsKey,
        jsonEncode(_habits.map((h) => h.toJson()).toList()),
      );
      await prefs.setString(
        _reflectionsKey,
        jsonEncode(_reflections.map((r) => r.toJson()).toList()),
      );
    } catch (e) {
      debugPrint('[TrackingService] Failed to save: $e');
    }
  }

  void _loadGoals() {
    final raw = _prefs?.getString(_goalsKey);
    if (raw == null || raw.isEmpty) return;
    try {
      final list = jsonDecode(raw) as List<dynamic>;
      _goals = list
          .map((e) => Goal.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (e) {
      debugPrint('[TrackingService] Failed to load goals: $e');
    }
  }

  void _loadHabits() {
    final raw = _prefs?.getString(_habitsKey);
    if (raw == null || raw.isEmpty) return;
    try {
      final list = jsonDecode(raw) as List<dynamic>;
      _habits = list
          .map((e) => Habit.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (e) {
      debugPrint('[TrackingService] Failed to load habits: $e');
    }
  }

  void _loadReflections() {
    final raw = _prefs?.getString(_reflectionsKey);
    if (raw == null || raw.isEmpty) return;
    try {
      final list = jsonDecode(raw) as List<dynamic>;
      _reflections = list
          .map((e) => DailyReflection.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (e) {
      debugPrint('[TrackingService] Failed to load reflections: $e');
    }
  }
}
