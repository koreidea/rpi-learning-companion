import 'dart:math';
import 'package:flutter/material.dart';

/// Library of simple, child-friendly shape-drawing methods using Canvas/Path.
///
/// Each method draws a single decorative shape suitable for procedurally
/// generated encyclopedia illustrations. All shapes use soft, rounded lines
/// and anti-aliased rendering.
class IllustrationShapes {
  IllustrationShapes._();

  /// Draws a 5-pointed star centered at [center] with the given [radius].
  static void drawStar(Canvas canvas, Offset center, double radius, Paint paint) {
    final path = Path();
    final innerRadius = radius * 0.4;
    for (int i = 0; i < 10; i++) {
      final angle = (i * pi / 5) - pi / 2;
      final r = i.isEven ? radius : innerRadius;
      final point = Offset(center.dx + r * cos(angle), center.dy + r * sin(angle));
      if (i == 0) {
        path.moveTo(point.dx, point.dy);
      } else {
        path.lineTo(point.dx, point.dy);
      }
    }
    path.close();
    canvas.drawPath(path, paint);
  }

  /// Draws an open book shape within [bounds].
  static void drawBook(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final cy = bounds.center.dy;
    final hw = bounds.width / 2;
    final hh = bounds.height / 2;

    // Left page
    final left = Path()
      ..moveTo(cx, cy - hh * 0.8)
      ..quadraticBezierTo(cx - hw * 0.6, cy - hh * 0.6, cx - hw, cy - hh * 0.3)
      ..lineTo(cx - hw, cy + hh * 0.8)
      ..quadraticBezierTo(cx - hw * 0.5, cy + hh * 0.6, cx, cy + hh * 0.7)
      ..close();
    canvas.drawPath(left, paint);

    // Right page
    final right = Path()
      ..moveTo(cx, cy - hh * 0.8)
      ..quadraticBezierTo(cx + hw * 0.6, cy - hh * 0.6, cx + hw, cy - hh * 0.3)
      ..lineTo(cx + hw, cy + hh * 0.8)
      ..quadraticBezierTo(cx + hw * 0.5, cy + hh * 0.6, cx, cy + hh * 0.7)
      ..close();
    canvas.drawPath(right, paint);
  }

  /// Draws a science flask/beaker within [bounds].
  static void drawFlask(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final w = bounds.width;
    final h = bounds.height;
    final top = bounds.top;
    final bottom = bounds.bottom;

    final path = Path()
      ..moveTo(cx - w * 0.15, top)
      ..lineTo(cx + w * 0.15, top)
      ..lineTo(cx + w * 0.15, top + h * 0.35)
      ..lineTo(cx + w * 0.45, bottom - h * 0.08)
      ..quadraticBezierTo(cx + w * 0.48, bottom, cx + w * 0.35, bottom)
      ..lineTo(cx - w * 0.35, bottom)
      ..quadraticBezierTo(cx - w * 0.48, bottom, cx - w * 0.45, bottom - h * 0.08)
      ..lineTo(cx - w * 0.15, top + h * 0.35)
      ..close();
    canvas.drawPath(path, paint);

    // Neck rim
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(cx, top + 2), width: w * 0.38, height: 4),
        const Radius.circular(2),
      ),
      paint,
    );
  }

  /// Draws a nature leaf within [bounds].
  static void drawLeaf(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final cy = bounds.center.dy;
    final hw = bounds.width / 2;
    final hh = bounds.height / 2;

    final path = Path()
      ..moveTo(cx, cy - hh)
      ..quadraticBezierTo(cx + hw * 1.2, cy - hh * 0.3, cx, cy + hh)
      ..quadraticBezierTo(cx - hw * 1.2, cy - hh * 0.3, cx, cy - hh)
      ..close();
    canvas.drawPath(path, paint);

    // Stem line
    final stem = Paint()
      ..color = paint.color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(Offset(cx, cy - hh * 0.5), Offset(cx, cy + hh), stem);
  }

  /// Draws a mechanical gear with teeth within [bounds].
  static void drawGear(Canvas canvas, Rect bounds, Paint paint) {
    final center = bounds.center;
    final radius = min(bounds.width, bounds.height) / 2;
    final innerR = radius * 0.65;
    const teeth = 8;

    final path = Path();
    for (int i = 0; i < teeth; i++) {
      final angle = (i * 2 * pi / teeth);
      final a1 = angle - 0.2;
      final a2 = angle + 0.2;
      final nextA1 = angle + (2 * pi / teeth) - 0.2;

      if (i == 0) {
        path.moveTo(center.dx + radius * cos(a1), center.dy + radius * sin(a1));
      }
      path.lineTo(center.dx + radius * cos(a2), center.dy + radius * sin(a2));
      path.lineTo(center.dx + innerR * cos(a2 + 0.15), center.dy + innerR * sin(a2 + 0.15));
      path.lineTo(center.dx + innerR * cos(nextA1 - 0.15), center.dy + innerR * sin(nextA1 - 0.15));
      path.lineTo(center.dx + radius * cos(nextA1), center.dy + radius * sin(nextA1));
    }
    path.close();
    canvas.drawPath(path, paint);

    // Center hole
    final holePaint = Paint()
      ..color = paint.color.withValues(alpha: (paint.color.a * 0.3))
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawCircle(center, radius * 0.25, holePaint);
  }

  /// Draws an idea lightbulb within [bounds].
  static void drawLightbulb(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final w = bounds.width;
    final h = bounds.height;
    final top = bounds.top;

    // Bulb
    final bulbR = w * 0.35;
    canvas.drawCircle(Offset(cx, top + bulbR + h * 0.05), bulbR, paint);

    // Neck
    final path = Path()
      ..moveTo(cx - w * 0.18, top + bulbR + h * 0.15)
      ..lineTo(cx - w * 0.15, bounds.bottom - h * 0.12)
      ..quadraticBezierTo(cx - w * 0.15, bounds.bottom, cx, bounds.bottom)
      ..quadraticBezierTo(cx + w * 0.15, bounds.bottom, cx + w * 0.15, bounds.bottom - h * 0.12)
      ..lineTo(cx + w * 0.18, top + bulbR + h * 0.15)
      ..close();
    canvas.drawPath(path, paint);

    // Filament lines
    final linePaint = Paint()
      ..color = paint.color.withValues(alpha: (paint.color.a * 0.5))
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    canvas.drawLine(
      Offset(cx - w * 0.08, bounds.bottom - h * 0.18),
      Offset(cx + w * 0.08, bounds.bottom - h * 0.18),
      linePaint,
    );
    canvas.drawLine(
      Offset(cx - w * 0.06, bounds.bottom - h * 0.12),
      Offset(cx + w * 0.06, bounds.bottom - h * 0.12),
      linePaint,
    );
  }

  /// Draws a space rocket within [bounds].
  static void drawRocket(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final w = bounds.width;
    final h = bounds.height;
    final top = bounds.top;
    final bottom = bounds.bottom;

    // Body
    final body = Path()
      ..moveTo(cx, top)
      ..quadraticBezierTo(cx + w * 0.25, top + h * 0.25, cx + w * 0.2, bottom - h * 0.15)
      ..lineTo(cx - w * 0.2, bottom - h * 0.15)
      ..quadraticBezierTo(cx - w * 0.25, top + h * 0.25, cx, top)
      ..close();
    canvas.drawPath(body, paint);

    // Left fin
    final leftFin = Path()
      ..moveTo(cx - w * 0.2, bottom - h * 0.3)
      ..lineTo(cx - w * 0.42, bottom)
      ..lineTo(cx - w * 0.15, bottom - h * 0.1)
      ..close();
    canvas.drawPath(leftFin, paint);

    // Right fin
    final rightFin = Path()
      ..moveTo(cx + w * 0.2, bottom - h * 0.3)
      ..lineTo(cx + w * 0.42, bottom)
      ..lineTo(cx + w * 0.15, bottom - h * 0.1)
      ..close();
    canvas.drawPath(rightFin, paint);

    // Window
    final windowPaint = Paint()
      ..color = paint.color.withValues(alpha: (paint.color.a * 0.4))
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    canvas.drawCircle(Offset(cx, top + h * 0.38), w * 0.09, windowPaint);
  }

  /// Draws an ocean wave curve within [bounds].
  static void drawWave(Canvas canvas, Rect bounds, Paint paint) {
    final w = bounds.width;
    final h = bounds.height;
    final cy = bounds.center.dy;

    final path = Path()..moveTo(bounds.left, cy);
    path.cubicTo(
      bounds.left + w * 0.25, cy - h * 0.45,
      bounds.left + w * 0.35, cy - h * 0.45,
      bounds.left + w * 0.5, cy,
    );
    path.cubicTo(
      bounds.left + w * 0.65, cy + h * 0.45,
      bounds.left + w * 0.75, cy + h * 0.45,
      bounds.right, cy,
    );
    path.lineTo(bounds.right, bounds.bottom);
    path.lineTo(bounds.left, bounds.bottom);
    path.close();
    canvas.drawPath(path, paint);
  }

  /// Draws a regular hexagon centered at [center] with the given [radius].
  static void drawHexagon(Canvas canvas, Offset center, double radius, Paint paint) {
    final path = Path();
    for (int i = 0; i < 6; i++) {
      final angle = (i * pi / 3) - pi / 2;
      final point = Offset(center.dx + radius * cos(angle), center.dy + radius * sin(angle));
      if (i == 0) {
        path.moveTo(point.dx, point.dy);
      } else {
        path.lineTo(point.dx, point.dy);
      }
    }
    path.close();
    canvas.drawPath(path, paint);
  }

  /// Draws an Indian dome/architecture shape within [bounds].
  static void drawDome(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final w = bounds.width;
    final bottom = bounds.bottom;
    final top = bounds.top;

    final path = Path()
      ..moveTo(bounds.left + w * 0.1, bottom)
      ..lineTo(bounds.left + w * 0.1, bottom - bounds.height * 0.35)
      ..quadraticBezierTo(cx, top - bounds.height * 0.1, bounds.right - w * 0.1, bottom - bounds.height * 0.35)
      ..lineTo(bounds.right - w * 0.1, bottom)
      ..close();
    canvas.drawPath(path, paint);

    // Spire
    canvas.drawCircle(Offset(cx, top + bounds.height * 0.08), w * 0.06, paint);
  }

  /// Draws atomic orbitals centered at [center] with the given [radius].
  static void drawAtom(Canvas canvas, Offset center, double radius, Paint paint) {
    final strokePaint = Paint()
      ..color = paint.color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    // Three orbital ellipses
    for (int i = 0; i < 3; i++) {
      canvas.save();
      canvas.translate(center.dx, center.dy);
      canvas.rotate(i * pi / 3);
      canvas.drawOval(
        Rect.fromCenter(center: Offset.zero, width: radius * 2, height: radius * 0.7),
        strokePaint,
      );
      canvas.restore();
    }

    // Nucleus
    canvas.drawCircle(center, radius * 0.18, paint);
  }

  /// Draws a heart shape centered at [center] with the given [radius].
  static void drawHeart(Canvas canvas, Offset center, double radius, Paint paint) {
    final path = Path();
    final x = center.dx;
    final y = center.dy;
    final r = radius;

    path.moveTo(x, y + r * 0.6);
    path.cubicTo(x - r * 1.2, y - r * 0.3, x - r * 0.6, y - r * 1.0, x, y - r * 0.4);
    path.cubicTo(x + r * 0.6, y - r * 1.0, x + r * 1.2, y - r * 0.3, x, y + r * 0.6);
    path.close();
    canvas.drawPath(path, paint);
  }

  /// Draws a cloud shape within [bounds].
  static void drawCloud(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final cy = bounds.center.dy;
    final w = bounds.width;
    final h = bounds.height;

    canvas.drawOval(Rect.fromCenter(center: Offset(cx - w * 0.2, cy), width: w * 0.5, height: h * 0.6), paint);
    canvas.drawOval(Rect.fromCenter(center: Offset(cx + w * 0.15, cy), width: w * 0.5, height: h * 0.55), paint);
    canvas.drawOval(Rect.fromCenter(center: Offset(cx, cy - h * 0.15), width: w * 0.45, height: h * 0.55), paint);
    canvas.drawOval(Rect.fromCenter(center: Offset(cx, cy + h * 0.1), width: w * 0.7, height: h * 0.4), paint);
  }

  /// Draws mountain peaks within [bounds].
  static void drawMountain(Canvas canvas, Rect bounds, Paint paint) {
    final w = bounds.width;
    final bottom = bounds.bottom;

    // Back mountain
    final back = Path()
      ..moveTo(bounds.left + w * 0.15, bottom)
      ..lineTo(bounds.left + w * 0.45, bounds.top + bounds.height * 0.15)
      ..lineTo(bounds.right - w * 0.1, bottom)
      ..close();
    canvas.drawPath(back, Paint()..color = paint.color.withValues(alpha: (paint.color.a * 0.6)));

    // Front mountain
    final front = Path()
      ..moveTo(bounds.left, bottom)
      ..lineTo(bounds.left + w * 0.35, bounds.top)
      ..lineTo(bounds.left + w * 0.7, bottom)
      ..close();
    canvas.drawPath(front, paint);
  }

  /// Draws a simple tree within [bounds].
  static void drawTree(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final w = bounds.width;
    final h = bounds.height;
    final bottom = bounds.bottom;

    // Trunk
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(cx, bottom - h * 0.15), width: w * 0.15, height: h * 0.35),
        const Radius.circular(3),
      ),
      Paint()..color = paint.color.withValues(alpha: (paint.color.a * 0.7)),
    );

    // Canopy (three overlapping circles)
    canvas.drawCircle(Offset(cx, bounds.top + h * 0.3), w * 0.3, paint);
    canvas.drawCircle(Offset(cx - w * 0.18, bounds.top + h * 0.4), w * 0.25, paint);
    canvas.drawCircle(Offset(cx + w * 0.18, bounds.top + h * 0.4), w * 0.25, paint);
  }

  /// Draws a money coin centered at [center] with the given [radius].
  static void drawCoin(Canvas canvas, Offset center, double radius, Paint paint) {
    // Outer ring
    canvas.drawCircle(center, radius, paint);
    // Inner ring
    final innerPaint = Paint()
      ..color = paint.color.withValues(alpha: (paint.color.a * 0.5))
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    canvas.drawCircle(center, radius * 0.7, innerPaint);
    // Center mark
    canvas.drawCircle(center, radius * 0.15, Paint()..color = paint.color.withValues(alpha: (paint.color.a * 0.4)));
  }

  /// Draws a music note within [bounds].
  static void drawMusic(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final w = bounds.width;
    final h = bounds.height;
    final top = bounds.top;

    // Note head
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(cx - w * 0.08, bounds.bottom - h * 0.18),
        width: w * 0.35,
        height: h * 0.22,
      ),
      paint,
    );

    // Stem
    final stemPaint = Paint()
      ..color = paint.color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(
      Offset(cx + w * 0.08, bounds.bottom - h * 0.18),
      Offset(cx + w * 0.08, top + h * 0.1),
      stemPaint,
    );

    // Flag
    final flag = Path()
      ..moveTo(cx + w * 0.08, top + h * 0.1)
      ..quadraticBezierTo(cx + w * 0.35, top + h * 0.25, cx + w * 0.15, top + h * 0.4);
    canvas.drawPath(flag, stemPaint);
  }

  /// Draws an earth globe centered at [center] with the given [radius].
  static void drawGlobe(Canvas canvas, Offset center, double radius, Paint paint) {
    // Main circle
    canvas.drawCircle(center, radius, paint);

    // Longitude lines
    final linePaint = Paint()
      ..color = paint.color.withValues(alpha: (paint.color.a * 0.4))
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    canvas.drawOval(
      Rect.fromCenter(center: center, width: radius * 1.0, height: radius * 2),
      linePaint,
    );
    canvas.drawOval(
      Rect.fromCenter(center: center, width: radius * 1.6, height: radius * 2),
      linePaint,
    );

    // Latitude line
    canvas.drawLine(
      Offset(center.dx - radius, center.dy),
      Offset(center.dx + radius, center.dy),
      linePaint,
    );
  }

  /// Draws a pencil/writing tool within [bounds].
  static void drawPencil(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final w = bounds.width;
    final h = bounds.height;
    final top = bounds.top;
    final bottom = bounds.bottom;

    // Body (tilted rectangle)
    final body = Path()
      ..moveTo(cx - w * 0.12, top + h * 0.1)
      ..lineTo(cx + w * 0.12, top + h * 0.1)
      ..lineTo(cx + w * 0.12, bottom - h * 0.2)
      ..lineTo(cx - w * 0.12, bottom - h * 0.2)
      ..close();
    canvas.drawPath(body, paint);

    // Tip
    final tip = Path()
      ..moveTo(cx - w * 0.12, bottom - h * 0.2)
      ..lineTo(cx, bottom)
      ..lineTo(cx + w * 0.12, bottom - h * 0.2)
      ..close();
    canvas.drawPath(tip, Paint()..color = paint.color.withValues(alpha: (paint.color.a * 0.7)));

    // Eraser
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(cx - w * 0.12, top, w * 0.24, h * 0.12),
        const Radius.circular(3),
      ),
      Paint()..color = paint.color.withValues(alpha: (paint.color.a * 0.5)),
    );
  }

  /// Draws a puzzle piece within [bounds].
  static void drawPuzzle(Canvas canvas, Rect bounds, Paint paint) {
    final l = bounds.left;
    final t = bounds.top;
    final w = bounds.width;
    final h = bounds.height;
    final knob = w * 0.12;

    final path = Path()
      ..moveTo(l, t)
      ..lineTo(l + w * 0.35, t)
      // Top knob
      ..arcToPoint(
        Offset(l + w * 0.65, t),
        radius: Radius.circular(knob),
        clockwise: false,
      )
      ..lineTo(l + w, t)
      ..lineTo(l + w, t + h * 0.35)
      // Right knob
      ..arcToPoint(
        Offset(l + w, t + h * 0.65),
        radius: Radius.circular(knob),
        clockwise: false,
      )
      ..lineTo(l + w, t + h)
      ..lineTo(l + w * 0.65, t + h)
      // Bottom knob
      ..arcToPoint(
        Offset(l + w * 0.35, t + h),
        radius: Radius.circular(knob),
        clockwise: true,
      )
      ..lineTo(l, t + h)
      ..lineTo(l, t + h * 0.65)
      // Left knob
      ..arcToPoint(
        Offset(l, t + h * 0.35),
        radius: Radius.circular(knob),
        clockwise: true,
      )
      ..close();
    canvas.drawPath(path, paint);
  }

  /// Draws a shield/protection shape within [bounds].
  static void drawShield(Canvas canvas, Rect bounds, Paint paint) {
    final cx = bounds.center.dx;
    final w = bounds.width;
    final h = bounds.height;
    final top = bounds.top;

    final path = Path()
      ..moveTo(cx, top)
      ..lineTo(bounds.right - w * 0.05, top + h * 0.08)
      ..quadraticBezierTo(bounds.right - w * 0.05, top + h * 0.55, cx, bounds.bottom)
      ..quadraticBezierTo(bounds.left + w * 0.05, top + h * 0.55, bounds.left + w * 0.05, top + h * 0.08)
      ..close();
    canvas.drawPath(path, paint);
  }

  /// Draws a clock face centered at [center] with the given [radius].
  static void drawClock(Canvas canvas, Offset center, double radius, Paint paint) {
    // Face
    canvas.drawCircle(center, radius, paint);

    // Inner ring
    final ringPaint = Paint()
      ..color = paint.color.withValues(alpha: (paint.color.a * 0.4))
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    canvas.drawCircle(center, radius * 0.85, ringPaint);

    // Hands
    final handPaint = Paint()
      ..color = paint.color.withValues(alpha: (paint.color.a * 0.6))
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;

    // Hour hand (pointing ~10 o'clock)
    canvas.drawLine(
      center,
      Offset(center.dx - radius * 0.3, center.dy - radius * 0.4),
      handPaint,
    );

    // Minute hand (pointing ~2 o'clock)
    canvas.drawLine(
      center,
      Offset(center.dx + radius * 0.35, center.dy - radius * 0.5),
      handPaint,
    );

    // Center dot
    canvas.drawCircle(center, radius * 0.08, paint);
  }
}
