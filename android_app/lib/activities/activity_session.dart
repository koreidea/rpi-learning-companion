import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Per-activity statistics.
class ActivityStats {
  final String activityId;
  int timesPlayed;
  int bestScore;
  int totalScore;
  DateTime? lastPlayed;

  ActivityStats({
    required this.activityId,
    this.timesPlayed = 0,
    this.bestScore = 0,
    this.totalScore = 0,
    this.lastPlayed,
  });

  double get averageScore =>
      timesPlayed > 0 ? totalScore / timesPlayed : 0.0;

  Map<String, dynamic> toJson() => {
        'activityId': activityId,
        'timesPlayed': timesPlayed,
        'bestScore': bestScore,
        'totalScore': totalScore,
        'lastPlayed': lastPlayed?.toIso8601String(),
      };

  factory ActivityStats.fromJson(Map<String, dynamic> json) => ActivityStats(
        activityId: json['activityId'] as String,
        timesPlayed: json['timesPlayed'] as int? ?? 0,
        bestScore: json['bestScore'] as int? ?? 0,
        totalScore: json['totalScore'] as int? ?? 0,
        lastPlayed: json['lastPlayed'] != null
            ? DateTime.tryParse(json['lastPlayed'] as String)
            : null,
      );
}

/// Tracks activity session stats and play streaks.
///
/// Persists data to SharedPreferences as JSON so it survives app restarts.
class ActivitySession {
  static const String _statsKey = 'activity_stats';
  static const String _streakKey = 'activity_streak';
  static const String _lastPlayDateKey = 'activity_last_play_date';

  SharedPreferences? _prefs;
  final Map<String, ActivityStats> _stats = {};
  int _currentStreak = 0;
  DateTime? _lastPlayDate;

  /// Initialize from SharedPreferences. Call once at app startup.
  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
    _loadStats();
    _loadStreak();
    debugPrint('[ActivitySession] Loaded ${_stats.length} activity stats, '
        'streak=$_currentStreak');
  }

  void _loadStats() {
    final raw = _prefs?.getString(_statsKey);
    if (raw == null || raw.isEmpty) return;

    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      for (final entry in map.entries) {
        _stats[entry.key] =
            ActivityStats.fromJson(entry.value as Map<String, dynamic>);
      }
    } catch (e) {
      debugPrint('[ActivitySession] Failed to load stats: $e');
    }
  }

  void _loadStreak() {
    _currentStreak = _prefs?.getInt(_streakKey) ?? 0;
    final dateStr = _prefs?.getString(_lastPlayDateKey);
    if (dateStr != null) {
      _lastPlayDate = DateTime.tryParse(dateStr);
    }
  }

  Future<void> _saveStats() async {
    final map = <String, dynamic>{};
    for (final entry in _stats.entries) {
      map[entry.key] = entry.value.toJson();
    }
    await _prefs?.setString(_statsKey, jsonEncode(map));
  }

  Future<void> _saveStreak() async {
    await _prefs?.setInt(_streakKey, _currentStreak);
    if (_lastPlayDate != null) {
      await _prefs?.setString(
          _lastPlayDateKey, _lastPlayDate!.toIso8601String());
    }
  }

  /// Record that an activity was completed with the given score.
  Future<void> recordCompletion(String activityId, int score) async {
    final stats = _stats.putIfAbsent(
      activityId,
      () => ActivityStats(activityId: activityId),
    );

    stats.timesPlayed++;
    stats.totalScore += score;
    if (score > stats.bestScore) {
      stats.bestScore = score;
    }
    stats.lastPlayed = DateTime.now();

    // Update streak
    _updateStreak();

    await _saveStats();
    await _saveStreak();

    debugPrint('[ActivitySession] Recorded: $activityId score=$score '
        'total=${stats.timesPlayed} best=${stats.bestScore}');
  }

  void _updateStreak() {
    final today = _dateOnly(DateTime.now());

    if (_lastPlayDate == null) {
      // First ever play
      _currentStreak = 1;
      _lastPlayDate = today;
      return;
    }

    final lastDate = _dateOnly(_lastPlayDate!);
    final diff = today.difference(lastDate).inDays;

    if (diff == 0) {
      // Already played today, streak unchanged
      return;
    } else if (diff == 1) {
      // Consecutive day
      _currentStreak++;
      _lastPlayDate = today;
    } else {
      // Streak broken
      _currentStreak = 1;
      _lastPlayDate = today;
    }
  }

  DateTime _dateOnly(DateTime dt) => DateTime(dt.year, dt.month, dt.day);

  /// Get stats for a specific activity. Returns zeroed stats if never played.
  ActivityStats getStats(String activityId) {
    return _stats[activityId] ??
        ActivityStats(activityId: activityId);
  }

  /// Current consecutive-day play streak.
  int getCurrentStreak() => _currentStreak;

  /// Total number of activities completed across all types.
  int get totalCompletions {
    int total = 0;
    for (final s in _stats.values) {
      total += s.timesPlayed;
    }
    return total;
  }
}
