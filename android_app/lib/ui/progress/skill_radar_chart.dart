import 'dart:math';
import 'package:flutter/material.dart';

import '../../models/skill.dart';

/// A 20-point radar (spider) chart for visualizing skill completion.
class SkillRadarChart extends StatelessWidget {
  final List<Skill> skills;
  final List<double> values; // 0..1 per skill

  const SkillRadarChart({
    super.key,
    required this.skills,
    required this.values,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _RadarChartPainter(
        skills: skills,
        values: values,
        theme: Theme.of(context),
      ),
      size: Size.infinite,
    );
  }
}

class _RadarChartPainter extends CustomPainter {
  final List<Skill> skills;
  final List<double> values;
  final ThemeData theme;

  _RadarChartPainter({
    required this.skills,
    required this.values,
    required this.theme,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final n = skills.length;
    if (n == 0) return;

    final center = Offset(size.width / 2, size.height / 2);
    final radius = min(size.width, size.height) / 2 - 36;
    final angleStep = 2 * pi / n;

    // Grid lines at 25%, 50%, 75%, 100%
    final gridPaint = Paint()
      ..color = theme.colorScheme.outlineVariant.withValues(alpha: 0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.5;

    for (final pct in [0.25, 0.5, 0.75, 1.0]) {
      final r = radius * pct;
      final path = Path();
      for (int i = 0; i < n; i++) {
        final angle = -pi / 2 + i * angleStep;
        final x = center.dx + r * cos(angle);
        final y = center.dy + r * sin(angle);
        if (i == 0) {
          path.moveTo(x, y);
        } else {
          path.lineTo(x, y);
        }
      }
      path.close();
      canvas.drawPath(path, gridPaint);
    }

    // Axis lines
    final axisPaint = Paint()
      ..color = theme.colorScheme.outlineVariant.withValues(alpha: 0.2)
      ..strokeWidth = 0.5;
    for (int i = 0; i < n; i++) {
      final angle = -pi / 2 + i * angleStep;
      canvas.drawLine(
        center,
        Offset(
          center.dx + radius * cos(angle),
          center.dy + radius * sin(angle),
        ),
        axisPaint,
      );
    }

    // Data polygon
    final dataPath = Path();
    final fillPaint = Paint()
      ..color = theme.colorScheme.primary.withValues(alpha: 0.15)
      ..style = PaintingStyle.fill;
    final strokePaint = Paint()
      ..color = theme.colorScheme.primary.withValues(alpha: 0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    for (int i = 0; i < n; i++) {
      final v = values.length > i ? values[i].clamp(0.0, 1.0) : 0.0;
      // Ensure at least a tiny value so the polygon is visible
      final r = radius * max(v, 0.03);
      final angle = -pi / 2 + i * angleStep;
      final x = center.dx + r * cos(angle);
      final y = center.dy + r * sin(angle);
      if (i == 0) {
        dataPath.moveTo(x, y);
      } else {
        dataPath.lineTo(x, y);
      }
    }
    dataPath.close();
    canvas.drawPath(dataPath, fillPaint);
    canvas.drawPath(dataPath, strokePaint);

    // Data points
    final dotPaint = Paint()
      ..color = theme.colorScheme.primary
      ..style = PaintingStyle.fill;
    for (int i = 0; i < n; i++) {
      final v = values.length > i ? values[i].clamp(0.0, 1.0) : 0.0;
      final r = radius * max(v, 0.03);
      final angle = -pi / 2 + i * angleStep;
      canvas.drawCircle(
        Offset(center.dx + r * cos(angle), center.dy + r * sin(angle)),
        2.5,
        dotPaint,
      );
    }

    // Labels
    for (int i = 0; i < n; i++) {
      final angle = -pi / 2 + i * angleStep;
      final labelR = radius + 20;
      final x = center.dx + labelR * cos(angle);
      final y = center.dy + labelR * sin(angle);

      final color = Color(skills[i].colorValue);
      final tp = TextPainter(
        text: TextSpan(
          text: _abbreviate(skills[i].shortName),
          style: TextStyle(
            fontSize: 8,
            fontWeight: FontWeight.w500,
            color: color,
          ),
        ),
        textDirection: TextDirection.ltr,
        textAlign: TextAlign.center,
      );
      tp.layout(maxWidth: 60);
      tp.paint(canvas, Offset(x - tp.width / 2, y - tp.height / 2));
    }
  }

  String _abbreviate(String name) {
    // Shorten long names for chart labels
    if (name.length <= 8) return name;
    final words = name.split(' ');
    if (words.length > 1) {
      return words.map((w) => w.length > 4 ? '${w.substring(0, 4)}.' : w).join(' ');
    }
    return '${name.substring(0, 7)}.';
  }

  @override
  bool shouldRepaint(_RadarChartPainter old) => true;
}
