import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../content/content_provider.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../../services/illustration_service.dart';
import '../widgets/content_illustration.dart';

/// Scrollable reader for encyclopedia content (stories, frameworks, fun facts).
///
/// Loads real JSON content from [ContentProvider] and displays it grouped
/// by type with expandable cards. When expanded, each card can generate
/// a DALL-E educational illustration or fall back to the procedural one.
class EncyclopediaReader extends ConsumerStatefulWidget {
  final Skill skill;

  const EncyclopediaReader({super.key, required this.skill});

  @override
  ConsumerState<EncyclopediaReader> createState() =>
      _EncyclopediaReaderState();
}

class _EncyclopediaReaderState extends ConsumerState<EncyclopediaReader> {
  final ContentProvider _contentProvider = ContentProvider();
  AgeBand _selectedBand = AgeBand.nursery;
  List<ContentItem>? _items;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadContent();
  }

  Future<void> _loadContent() async {
    setState(() => _loading = true);
    try {
      final items = await _contentProvider.getForSkill(
        widget.skill.id,
        band: _selectedBand,
      );
      if (mounted) {
        setState(() {
          _items = items;
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _items = [];
          _loading = false;
        });
      }
    }
  }

  void _onBandChanged(AgeBand band) {
    setState(() => _selectedBand = band);
    _loadContent();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = Color(widget.skill.colorValue);

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Age band selector
        Row(
          children: AgeBand.values.map((band) {
            final selected = band == _selectedBand;
            return Padding(
              padding: const EdgeInsets.only(right: 8),
              child: ChoiceChip(
                label: Text(band.label),
                selected: selected,
                selectedColor: color.withValues(alpha: 0.15),
                labelStyle: TextStyle(
                  color:
                      selected ? color : theme.colorScheme.onSurfaceVariant,
                  fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
                ),
                side: BorderSide(
                  color: selected ? color : Colors.transparent,
                ),
                showCheckmark: false,
                onSelected: (_) => _onBandChanged(band),
              ),
            );
          }).toList(),
        ),
        const SizedBox(height: 20),

        // Header
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.06),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(Icons.auto_stories, color: color, size: 24),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      widget.skill.encyclopediaTitle,
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: color,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                'Explore stories, frameworks, and fun facts about '
                '${widget.skill.shortName.toLowerCase()}.',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 20),

        // Content
        if (_loading)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 48),
            child: Center(child: CircularProgressIndicator()),
          )
        else if (_items == null || _items!.isEmpty)
          _buildEmptyState(theme)
        else
          ..._buildContentSections(theme, color),
      ],
    );
  }

  Widget _buildEmptyState(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        border: Border.all(
          color: theme.colorScheme.outlineVariant,
          width: 1,
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Icon(
            Icons.library_books_outlined,
            size: 40,
            color:
                theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.4),
          ),
          const SizedBox(height: 10),
          Text(
            'No content yet for this level',
            style: theme.textTheme.titleSmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Try selecting a different age band above.',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant
                  .withValues(alpha: 0.7),
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  List<Widget> _buildContentSections(ThemeData theme, Color color) {
    final items = _items!;

    // Group by type in display order
    const typeOrder = [
      'story',
      'fun_fact',
      'framework',
      'case_study',
      'activity_idea',
    ];
    final grouped = <String, List<ContentItem>>{};
    for (final item in items) {
      grouped.putIfAbsent(item.type, () => []).add(item);
    }

    final widgets = <Widget>[];
    for (final type in typeOrder) {
      final group = grouped[type];
      if (group == null || group.isEmpty) continue;

      final label = _typeLabel(type);
      final icon = _typeIcon(type);

      // Section header
      widgets.add(
        Padding(
          padding: const EdgeInsets.only(top: 8, bottom: 12),
          child: Row(
            children: [
              Icon(icon, size: 18, color: color),
              const SizedBox(width: 8),
              Text(
                '$label (${group.length})',
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: color,
                ),
              ),
            ],
          ),
        ),
      );

      // Content cards
      for (final item in group) {
        widgets.add(
          _ExpandableContentCard(
            key: ValueKey(item.id),
            item: item,
            color: color,
          ),
        );
        widgets.add(const SizedBox(height: 10));
      }
    }

    return widgets;
  }

  String _typeLabel(String type) {
    switch (type) {
      case 'story':
        return 'Stories';
      case 'fun_fact':
        return 'Fun Facts';
      case 'framework':
        return 'Frameworks';
      case 'case_study':
        return 'Case Studies';
      case 'activity_idea':
        return 'Activity Ideas';
      default:
        return type;
    }
  }

  IconData _typeIcon(String type) {
    switch (type) {
      case 'story':
        return Icons.menu_book;
      case 'fun_fact':
        return Icons.lightbulb_outline;
      case 'framework':
        return Icons.architecture;
      case 'case_study':
        return Icons.school;
      case 'activity_idea':
        return Icons.play_circle_outline;
      default:
        return Icons.article_outlined;
    }
  }
}

/// An expandable card that shows a content item's title, body, lesson, tags,
/// and an optional DALL-E generated educational illustration.
class _ExpandableContentCard extends ConsumerStatefulWidget {
  final ContentItem item;
  final Color color;

  const _ExpandableContentCard({
    super.key,
    required this.item,
    required this.color,
  });

  @override
  ConsumerState<_ExpandableContentCard> createState() =>
      _ExpandableContentCardState();
}

class _ExpandableContentCardState
    extends ConsumerState<_ExpandableContentCard> {
  bool _expanded = false;

  /// Path to the cached/generated illustration file, or null.
  String? _imagePath;

  /// Whether we are currently generating an illustration.
  bool _generating = false;

  /// Error message if generation failed.
  String? _error;

  /// Whether we have checked the cache for this item.
  bool _cacheChecked = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final item = widget.item;
    final color = widget.color;
    final serviceAsync = ref.watch(illustrationServiceProvider);
    final service = serviceAsync.valueOrNull;

    // When card is expanded, check cache and auto-generate if not cached
    if (_expanded && !_cacheChecked && service != null) {
      _cacheChecked = true;
      _checkCacheAndAutoGenerate(service);
    }

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: InkWell(
        onTap: () => setState(() => _expanded = !_expanded),
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Title row with illustration thumbnail and expand indicator
              Row(
                children: [
                  // Show cached DALL-E thumbnail if available, else procedural
                  RepaintBoundary(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: SizedBox(
                        width: 48,
                        height: 48,
                        child: _imagePath != null
                            ? Image.file(
                                File(_imagePath!),
                                fit: BoxFit.cover,
                                width: 48,
                                height: 48,
                              )
                            : ContentIllustration(
                                item: item,
                                skillColor: color,
                                height: 48,
                              ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      item.title,
                      style: theme.textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  Icon(
                    _expanded
                        ? Icons.keyboard_arrow_up
                        : Icons.keyboard_arrow_down,
                    size: 20,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ],
              ),

              // Expanded content
              if (_expanded) ...[
                const SizedBox(height: 12),

                // Illustration area
                _buildIllustrationArea(theme, item, color, service),

                const SizedBox(height: 12),
                Text(
                  item.body,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                    height: 1.6,
                  ),
                ),

                // Lesson box
                if (item.lesson != null && item.lesson!.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: color.withValues(alpha: 0.08),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: color.withValues(alpha: 0.2),
                      ),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.tips_and_updates,
                            size: 16, color: color),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            item.lesson!,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: color,
                              fontWeight: FontWeight.w500,
                              height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],

                // Tags
                if (item.tags.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 6,
                    runSpacing: 4,
                    children: item.tags.map((tag) {
                      return Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 3,
                        ),
                        decoration: BoxDecoration(
                          color:
                              theme.colorScheme.surfaceContainerHighest,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          tag,
                          style: theme.textTheme.bodySmall?.copyWith(
                            fontSize: 10,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ],
              ],
            ],
          ),
        ),
      ),
    );
  }

  /// Builds the illustration area: generated image, generate button,
  /// loading state, or procedural fallback.
  Widget _buildIllustrationArea(
    ThemeData theme,
    ContentItem item,
    Color color,
    IllustrationService? service,
  ) {
    // If we have a generated image, show it (tappable for full-screen)
    if (_imagePath != null) {
      return GestureDetector(
        onTap: () => _openFullScreenImage(context, _imagePath!, item.title),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Image.file(
            File(_imagePath!),
            width: double.infinity,
            height: 220,
            fit: BoxFit.cover,
          ),
        ),
      );
    }

    // If currently generating, show loading state
    if (_generating) {
      return Container(
        width: double.infinity,
        height: 220,
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.06),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withValues(alpha: 0.15)),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: 32,
              height: 32,
              child: CircularProgressIndicator(
                strokeWidth: 3,
                color: color,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              'Creating illustration...',
              style: theme.textTheme.bodySmall?.copyWith(
                color: color,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'This may take a few seconds',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant
                    .withValues(alpha: 0.5),
                fontSize: 11,
              ),
            ),
          ],
        ),
      );
    }

    // Show error with retry, or procedural fallback
    if (_error != null && service != null) {
      return GestureDetector(
        onTap: () => _generateIllustration(service),
        child: Container(
          width: double.infinity,
          height: 140,
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.06),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: color.withValues(alpha: 0.15)),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.refresh, color: color, size: 28),
              const SizedBox(height: 8),
              Text(
                'Tap to retry generating diagram',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: color,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      );
    }

    // Procedural fallback (no API key or not yet triggered)
    return ContentIllustration(
      item: item,
      skillColor: color,
      height: 140,
    );
  }

  /// Check the cache first; if not cached, auto-generate the illustration.
  Future<void> _checkCacheAndAutoGenerate(IllustrationService service) async {
    final path = await service.getCachedPath(widget.item);
    if (path != null && mounted) {
      setState(() => _imagePath = path);
      return;
    }
    // Not cached — auto-generate
    _generateIllustration(service);
  }

  /// Generate a new illustration via DALL-E 3.
  Future<void> _generateIllustration(IllustrationService service) async {
    if (!mounted) return;
    setState(() {
      _generating = true;
      _error = null;
    });

    final path = await service.getIllustration(widget.item);

    if (!mounted) return;

    if (path != null) {
      setState(() {
        _imagePath = path;
        _generating = false;
      });
    } else {
      setState(() {
        _generating = false;
        _error = 'Could not generate illustration. Tap to retry.';
      });
    }
  }

  /// Open the generated image in a full-screen viewer with pinch-to-zoom.
  void _openFullScreenImage(
      BuildContext context, String imagePath, String title) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => _FullScreenImageViewer(
          imagePath: imagePath,
          title: title,
        ),
      ),
    );
  }
}

/// Full-screen image viewer with pinch-to-zoom and pan.
class _FullScreenImageViewer extends StatelessWidget {
  final String imagePath;
  final String title;

  const _FullScreenImageViewer({
    required this.imagePath,
    required this.title,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(
          title,
          style: const TextStyle(fontSize: 14),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ),
      body: Center(
        child: InteractiveViewer(
          minScale: 0.5,
          maxScale: 4.0,
          child: Image.file(
            File(imagePath),
            fit: BoxFit.contain,
          ),
        ),
      ),
    );
  }
}
