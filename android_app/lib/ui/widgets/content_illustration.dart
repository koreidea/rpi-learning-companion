import 'dart:math';
import 'package:flutter/material.dart';

import '../../content/content_provider.dart';
import 'illustration_shapes.dart';
import 'illustration_themes.dart';

/// Configuration for a procedurally generated illustration.
///
/// All fields are deterministic given a [ContentItem] so the same item always
/// produces the exact same illustration.
class IllustrationConfig {
  /// Dominant colour derived from the parent skill.
  final Color primaryColor;

  /// Secondary colour (hue-shifted 120 degrees from primary).
  final Color secondaryColor;

  /// Seed for the pseudo-random generator (from content id hash).
  final int seed;

  /// Background scatter shapes chosen by skill theme.
  final List<ShapeType> sceneShapes;

  /// Optional accent shape derived from the first matching content tag.
  final ShapeType? accentShape;

  /// Content type string (story, fun_fact, framework, case_study, etc.).
  final String contentType;

  const IllustrationConfig({
    required this.primaryColor,
    required this.secondaryColor,
    required this.seed,
    required this.sceneShapes,
    this.accentShape,
    required this.contentType,
  });

  /// Build a deterministic config from a content item and its skill colour.
  factory IllustrationConfig.fromContentItem(ContentItem item, Color skillColor) {
    // Use a combined hash of id + title + ageBand for maximum uniqueness.
    final combinedSeed = '${item.id}_${item.title}_${item.ageBand}'.hashCode;

    // Shift hue based on the item's unique seed for colour variety.
    final hsl = HSLColor.fromColor(skillColor);
    final hueShift = (combinedSeed.abs() % 60) - 30; // -30 to +30 degrees
    final primaryHsl = hsl.withHue((hsl.hue + hueShift) % 360);
    final secondaryHsl = hsl.withHue((hsl.hue + 120 + hueShift) % 360);
    final primaryColor = primaryHsl.toColor();
    final secondaryColor = secondaryHsl.toColor();

    // Mix scene shapes with extra shapes based on item seed for variety.
    final scene = skillScenes[item.skillId];
    final baseShapes = scene?.shapes ?? const [ShapeType.star, ShapeType.hexagon, ShapeType.cloud];
    final allShapeTypes = ShapeType.values;
    final rng = Random(combinedSeed);
    final extraShape = allShapeTypes[rng.nextInt(allShapeTypes.length)];
    final shapes = [...baseShapes, extraShape];

    // Find accent shape from first matching tag, fall back to title-based.
    ShapeType? accent;
    for (final tag in item.tags) {
      final lower = tag.toLowerCase();
      for (final entry in tagShapes.entries) {
        if (lower.contains(entry.key.toLowerCase())) {
          accent = entry.value;
          break;
        }
      }
      if (accent != null) break;
    }
    // If no tag match, pick based on title hash for uniqueness.
    accent ??= allShapeTypes[item.title.hashCode.abs() % allShapeTypes.length];

    return IllustrationConfig(
      primaryColor: primaryColor,
      secondaryColor: secondaryColor,
      seed: combinedSeed,
      sceneShapes: shapes,
      accentShape: accent,
      contentType: item.type,
    );
  }
}

/// CustomPainter that renders a layered, seed-deterministic illustration.
class IllustrationPainter extends CustomPainter {
  final IllustrationConfig config;

  IllustrationPainter({required this.config});

  @override
  void paint(Canvas canvas, Size size) {
    final rng = Random(config.seed);

    _paintBackground(canvas, size);
    _paintBackgroundShapes(canvas, size, rng);
    _paintAccentShape(canvas, size, rng);
    _paintTypeBadge(canvas, size);
  }

  void _paintBackground(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final gradient = RadialGradient(
      center: Alignment.center,
      radius: 1.2,
      colors: [
        config.primaryColor.withValues(alpha: 0.10),
        config.primaryColor.withValues(alpha: 0.04),
      ],
    );
    canvas.drawRect(rect, Paint()..shader = gradient.createShader(rect));
  }

  void _paintBackgroundShapes(Canvas canvas, Size size, Random rng) {
    final shapes = config.sceneShapes;
    // Draw 3-5 shapes scattered across the canvas.
    final count = shapes.length + rng.nextInt(3);

    for (int i = 0; i < count; i++) {
      final shapeType = shapes[i % shapes.length];
      final shapeSize = 20.0 + rng.nextDouble() * 40.0;
      final x = rng.nextDouble() * size.width;
      final y = rng.nextDouble() * size.height;
      final useSecondary = rng.nextBool();
      final opacity = 0.10 + rng.nextDouble() * 0.15;
      final rotation = rng.nextDouble() * pi * 2;

      final color = useSecondary ? config.secondaryColor : config.primaryColor;
      final paint = Paint()
        ..color = color.withValues(alpha: opacity)
        ..isAntiAlias = true
        ..style = PaintingStyle.fill;

      canvas.save();
      canvas.translate(x, y);
      canvas.rotate(rotation);

      _drawShape(canvas, shapeType, shapeSize, paint);

      canvas.restore();
    }
  }

  void _paintAccentShape(Canvas canvas, Size size, Random rng) {
    final accent = config.accentShape;
    if (accent == null) return;

    // Position at roughly golden-ratio horizontal placement.
    final x = size.width * 0.62;
    final y = size.height * 0.45 + (rng.nextDouble() - 0.5) * size.height * 0.15;
    final shapeSize = min(size.width, size.height) * 0.28;
    final opacity = 0.30 + rng.nextDouble() * 0.15;

    final paint = Paint()
      ..color = config.primaryColor.withValues(alpha: opacity)
      ..isAntiAlias = true
      ..style = PaintingStyle.fill;

    canvas.save();
    canvas.translate(x, y);
    _drawShape(canvas, accent, shapeSize, paint);
    canvas.restore();
  }

  void _paintTypeBadge(Canvas canvas, Size size) {
    // Tiny badge in bottom-right corner indicating content type.
    final badgeSize = min(size.width, size.height) * 0.12;
    final bx = size.width - badgeSize * 1.2;
    final by = size.height - badgeSize * 1.2;
    final paint = Paint()
      ..color = config.secondaryColor.withValues(alpha: 0.35)
      ..isAntiAlias = true
      ..style = PaintingStyle.fill;

    final ShapeType badge;
    switch (config.contentType) {
      case 'story':
        badge = ShapeType.book;
      case 'fun_fact':
        badge = ShapeType.lightbulb;
      case 'framework':
        badge = ShapeType.gear;
      case 'case_study':
        badge = ShapeType.puzzle;
      case 'activity_idea':
        badge = ShapeType.star;
      default:
        badge = ShapeType.star;
    }

    canvas.save();
    canvas.translate(bx, by);
    _drawShape(canvas, badge, badgeSize, paint);
    canvas.restore();
  }

  /// Dispatches to the appropriate shape drawing method.
  void _drawShape(Canvas canvas, ShapeType type, double size, Paint paint) {
    final half = size / 2;
    final bounds = Rect.fromCenter(center: Offset.zero, width: size, height: size);

    switch (type) {
      case ShapeType.star:
        IllustrationShapes.drawStar(canvas, Offset.zero, half, paint);
      case ShapeType.book:
        IllustrationShapes.drawBook(canvas, bounds, paint);
      case ShapeType.flask:
        IllustrationShapes.drawFlask(canvas, bounds, paint);
      case ShapeType.leaf:
        IllustrationShapes.drawLeaf(canvas, bounds, paint);
      case ShapeType.gear:
        IllustrationShapes.drawGear(canvas, bounds, paint);
      case ShapeType.lightbulb:
        IllustrationShapes.drawLightbulb(canvas, bounds, paint);
      case ShapeType.rocket:
        IllustrationShapes.drawRocket(canvas, bounds, paint);
      case ShapeType.wave:
        IllustrationShapes.drawWave(canvas, bounds, paint);
      case ShapeType.hexagon:
        IllustrationShapes.drawHexagon(canvas, Offset.zero, half, paint);
      case ShapeType.dome:
        IllustrationShapes.drawDome(canvas, bounds, paint);
      case ShapeType.atom:
        IllustrationShapes.drawAtom(canvas, Offset.zero, half, paint);
      case ShapeType.heart:
        IllustrationShapes.drawHeart(canvas, Offset.zero, half, paint);
      case ShapeType.cloud:
        IllustrationShapes.drawCloud(canvas, bounds, paint);
      case ShapeType.mountain:
        IllustrationShapes.drawMountain(canvas, bounds, paint);
      case ShapeType.tree:
        IllustrationShapes.drawTree(canvas, bounds, paint);
      case ShapeType.coin:
        IllustrationShapes.drawCoin(canvas, Offset.zero, half, paint);
      case ShapeType.music:
        IllustrationShapes.drawMusic(canvas, bounds, paint);
      case ShapeType.globe:
        IllustrationShapes.drawGlobe(canvas, Offset.zero, half, paint);
      case ShapeType.pencil:
        IllustrationShapes.drawPencil(canvas, bounds, paint);
      case ShapeType.puzzle:
        IllustrationShapes.drawPuzzle(canvas, bounds, paint);
      case ShapeType.shield:
        IllustrationShapes.drawShield(canvas, bounds, paint);
      case ShapeType.clock:
        IllustrationShapes.drawClock(canvas, Offset.zero, half, paint);
    }
  }

  @override
  bool shouldRepaint(IllustrationPainter oldDelegate) => false;
}

/// A widget that displays a procedurally generated illustration for a
/// [ContentItem], themed to its parent skill colour.
///
/// The illustration is deterministic (seeded by content id) and wrapped in a
/// [RepaintBoundary] for performance.
class ContentIllustration extends StatelessWidget {
  final ContentItem item;
  final Color skillColor;
  final double height;

  const ContentIllustration({
    super.key,
    required this.item,
    required this.skillColor,
    this.height = 120,
  });

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: CustomPaint(
          size: Size(double.infinity, height),
          painter: IllustrationPainter(
            config: IllustrationConfig.fromContentItem(item, skillColor),
          ),
        ),
      ),
    );
  }
}
