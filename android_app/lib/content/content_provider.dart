import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../models/age_band.dart';
import '../models/skill.dart';

/// A single piece of learning content associated with a skill.
class ContentItem {
  /// Unique identifier for this content item.
  final String id;

  /// The skill this content belongs to.
  final SkillId skillId;

  /// Content type: 'story', 'framework', 'case_study', 'fun_fact',
  /// or 'activity_idea'.
  final String type;

  /// Target age band: 'nursery', 'junior', or 'senior'.
  final String ageBand;

  /// Display title of this content item.
  final String title;

  /// The main content body text.
  final String body;

  /// Key takeaway or lesson, if any.
  final String? lesson;

  /// Tags for search and filtering.
  final List<String> tags;

  /// Creates a [ContentItem] with all metadata.
  ContentItem({
    required this.id,
    required this.skillId,
    required this.type,
    required this.ageBand,
    required this.title,
    required this.body,
    this.lesson,
    this.tags = const [],
  });

  /// Deserialize a [ContentItem] from a JSON map.
  ///
  /// The [skillId] is passed separately since it is typically the parent
  /// key in the bundled JSON structure.
  factory ContentItem.fromJson(Map<String, dynamic> json, SkillId skillId) {
    return ContentItem(
      id: json['id'] as String? ?? '',
      skillId: skillId,
      type: json['type'] as String? ?? 'story',
      ageBand: json['ageBand'] as String? ?? 'nursery',
      title: json['title'] as String? ?? '',
      body: json['body'] as String? ?? '',
      lesson: json['lesson'] as String?,
      tags: (json['tags'] as List<dynamic>?)
              ?.map((t) => t as String)
              .toList() ??
          const [],
    );
  }
}

/// Loads and caches learning content from bundled JSON assets.
///
/// Content is organized by skill and can be filtered by age band.
/// JSON assets are expected at `assets/content/<skillId>.json`.
class ContentProvider {
  final Map<SkillId, List<ContentItem>> _cache = {};
  final Random _random = Random();

  /// Load content items for a [skill], optionally filtered by [band].
  ///
  /// Results are cached after the first load. Returns an empty list if
  /// the asset file is missing or cannot be parsed.
  Future<List<ContentItem>> getForSkill(
    SkillId skill, {
    AgeBand? band,
  }) async {
    if (!_cache.containsKey(skill)) {
      await _loadSkillContent(skill);
    }

    final items = _cache[skill] ?? [];
    if (band == null) return items;

    return items.where((item) => item.ageBand == band.name).toList();
  }

  /// Get the encyclopedia section title for a [skill].
  String getEncyclopediaTitle(SkillId skill) {
    return SkillRegistry.get(skill).encyclopediaTitle;
  }

  /// Get a random content item for a [skill] and [band] suitable for
  /// daily presentation.
  ///
  /// Uses the current date as a seed component so the same item is returned
  /// throughout the day but changes the next day. Returns null if no
  /// matching content is found.
  Future<ContentItem?> getDailyContent(SkillId skill, AgeBand band) async {
    final items = await getForSkill(skill, band: band);
    if (items.isEmpty) return null;

    // Use date-based index for consistent daily selection.
    final today = DateTime.now();
    final daySeed = today.year * 10000 + today.month * 100 + today.day;
    final index = (daySeed + skill.index) % items.length;
    return items[index];
  }

  /// Clear all cached content. Useful when switching profiles or languages.
  void clearCache() {
    _cache.clear();
  }

  Future<void> _loadSkillContent(SkillId skill) async {
    final assetPath = 'assets/content/${skill.name}.json';
    try {
      final raw = await rootBundle.loadString(assetPath);
      final jsonList = jsonDecode(raw) as List<dynamic>;
      _cache[skill] = jsonList
          .map((e) => ContentItem.fromJson(e as Map<String, dynamic>, skill))
          .toList();
    } catch (e) {
      debugPrint('[ContentProvider] Could not load $assetPath: $e');
      _cache[skill] = [];
    }
  }
}
