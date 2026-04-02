import 'dart:math';
import 'package:flutter/material.dart';
import 'face_state.dart';

/// CustomPainter that renders the Cozmo/EMO-style animated face.
/// Direct port of render_eye() and render_face() from tft_display.py.
class FacePainter extends CustomPainter {
  final FaceParams face;
  final double blinkMult;
  final bool isError;
  final List<Sparkle> sparkles;
  final double listenPhase;
  final int dotPhase;
  final int spinnerIdx;
  final double sleepZPhase;
  final String botState;

  FacePainter({
    required this.face,
    this.blinkMult = 1.0,
    this.isError = false,
    this.sparkles = const [],
    this.listenPhase = 0.0,
    this.dotPhase = 0,
    this.spinnerIdx = 0,
    this.sleepZPhase = 0.0,
    this.botState = 'ready',
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Scale from reference 480×320 to actual screen size
    final scaleX = size.width / 480;
    final scaleY = size.height / 320;
    final scale = min(scaleX, scaleY);
    final offsetX = (size.width - 480 * scale) / 2;
    final offsetY = (size.height - 320 * scale) / 2;

    canvas.save();
    canvas.translate(offsetX, offsetY);
    canvas.scale(scale);

    // Background
    canvas.drawRect(
      const Rect.fromLTWH(0, 0, 480, 320),
      Paint()..color = Colors.black,
    );

    // Eye centers (same as Pi layout)
    const leftCx = 132.0, rightCx = 348.0, eyeCy = 150.0;

    _drawEye(canvas, face.leftEye, leftCx, eyeCy);
    _drawEye(canvas, face.rightEye, rightCx, eyeCy);

    // ── State-specific overlays ──
    if (botState == 'listening') {
      _drawListenRings(canvas, 240, eyeCy);
    }
    if (botState == 'processing') {
      _drawProcessingDots(canvas, 240, 260);
    }
    if (botState == 'speaking' && face.mouthStyle == 'open') {
      _drawMouth(canvas, 240, 230, face.mouthOpen, face.mouthWidth);
    }
    if (botState == 'error' && face.mouthStyle == 'flat') {
      _drawFlatMouth(canvas, 240, 230, face.mouthWidth);
    }
    if (botState == 'loading') {
      _drawSpinner(canvas, 240, 260);
    }
    if (botState == 'sleeping') {
      _drawSleepZ(canvas, 300, 100);
    }

    // Sparkles
    _drawSparkles(canvas);

    canvas.restore();
  }

  void _drawEye(Canvas canvas, EyeParams eye, double cx, double cy) {
    final cr = eye.colorR.round().clamp(0, 255);
    final cg = eye.colorG.round().clamp(0, 255);
    final cb = eye.colorB.round().clamp(0, 255);
    final color = Color.fromARGB(255, cr, cg, cb);

    final hw = eye.width / 2;
    final hh = (eye.height / 2) * blinkMult; // Apply blink
    final double r = eye.cornerRadius.clamp(0.0, min(hw, hh).toDouble());

    if (isError) {
      final arm = max(24.0, min(hw, hh) * 0.7);
      final paint = Paint()
        ..color = color
        ..strokeWidth = 8
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round;
      canvas.drawLine(Offset(cx - arm, cy - arm), Offset(cx + arm, cy + arm), paint);
      canvas.drawLine(Offset(cx - arm, cy + arm), Offset(cx + arm, cy - arm), paint);
      return;
    }

    if (hh < 3) {
      // Thin line (sleeping/blink)
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(center: Offset(cx, cy), width: hw * 2, height: 4),
          const Radius.circular(2),
        ),
        Paint()..color = color,
      );
      return;
    }

    // Glow
    final ga = eye.glowAlpha.clamp(0.0, 1.0);
    final glowColor = Color.fromARGB(
        (255 * ga).round().clamp(0, 255), cr, cg, cb);
    const gx = 8.0;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
            center: Offset(cx, cy),
            width: (hw + gx) * 2,
            height: (hh + gx) * 2),
        Radius.circular(r + 6),
      ),
      Paint()..color = glowColor,
    );

    // Main eye body
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(cx, cy), width: hw * 2, height: hh * 2),
        Radius.circular(r),
      ),
      Paint()..color = color,
    );

    // Eyelid slope masks (top triangles for happy/angry expressions)
    if (eye.slopeLeft.abs() > 0.01 || eye.slopeRight.abs() > 0.01) {
      final clipPaint = Paint()..color = Colors.black;
      final slopeH = hh * 0.4;

      // Left slope
      if (eye.slopeLeft.abs() > 0.01) {
        final path = Path()
          ..moveTo(cx - hw - 2, cy - hh - 2)
          ..lineTo(cx, cy - hh - 2)
          ..lineTo(cx, cy - hh + slopeH * eye.slopeLeft.abs())
          ..close();
        canvas.drawPath(path, clipPaint);
      }
      // Right slope
      if (eye.slopeRight.abs() > 0.01) {
        final path = Path()
          ..moveTo(cx, cy - hh - 2)
          ..lineTo(cx + hw + 2, cy - hh - 2)
          ..lineTo(cx, cy - hh + slopeH * eye.slopeRight.abs())
          ..close();
        canvas.drawPath(path, clipPaint);
      }
    }

    // Pupil
    final ps = eye.pupilScale;
    final pupilR = min(hw, hh) * ps * 0.4;
    if (pupilR > 1) {
      final maxOx = hw - pupilR - 4;
      final maxOy = hh - pupilR - 4;
      final px = cx + eye.pupilX * maxOx;
      final py = cy + eye.pupilY * maxOy;
      canvas.drawCircle(
        Offset(px, py),
        pupilR,
        Paint()..color = Colors.black,
      );
      // Highlight
      final hlR = pupilR * 0.3;
      canvas.drawCircle(
        Offset(px - pupilR * 0.25, py - pupilR * 0.25),
        hlR,
        Paint()..color = Colors.white.withValues(alpha: 0.7),
      );
    }
  }

  void _drawListenRings(Canvas canvas, double cx, double cy) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    for (int i = 0; i < 3; i++) {
      final phase = (listenPhase + i * 0.33) % 1.0;
      final radius = 60 + phase * 80;
      final alpha = (1.0 - phase) * 0.4;
      paint.color = const Color.fromARGB(255, 30, 144, 255)
          .withValues(alpha: alpha);
      canvas.drawCircle(Offset(cx, cy), radius, paint);
    }
  }

  void _drawProcessingDots(Canvas canvas, double cx, double cy) {
    for (int i = 0; i < 3; i++) {
      final active = i < dotPhase;
      final color = active
          ? const Color.fromARGB(255, 255, 200, 50)
          : const Color.fromARGB(100, 255, 200, 50);
      canvas.drawCircle(
        Offset(cx - 20 + i * 20, cy),
        5,
        Paint()..color = color,
      );
    }
  }

  void _drawMouth(
      Canvas canvas, double cx, double cy, double openness, double width) {
    final mouthH = 8 + openness * 12;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
            center: Offset(cx, cy), width: width, height: mouthH),
        Radius.circular(mouthH / 2),
      ),
      Paint()..color = const Color.fromARGB(255, 180, 100, 255),
    );
  }

  void _drawFlatMouth(Canvas canvas, double cx, double cy, double width) {
    canvas.drawLine(
      Offset(cx - width / 2, cy),
      Offset(cx + width / 2, cy),
      Paint()
        ..color = const Color.fromARGB(255, 255, 60, 60)
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round,
    );
  }

  void _drawSpinner(Canvas canvas, double cx, double cy) {
    final paint = Paint()
      ..style = PaintingStyle.fill;
    for (int i = 0; i < 8; i++) {
      final angle = i * pi / 4;
      final active = i == spinnerIdx;
      paint.color = active
          ? const Color.fromARGB(255, 255, 200, 50)
          : const Color.fromARGB(80, 255, 200, 50);
      canvas.drawCircle(
        Offset(cx + 20 * cos(angle), cy + 20 * sin(angle)),
        4,
        paint,
      );
    }
  }

  void _drawSleepZ(Canvas canvas, double cx, double cy) {
    final phase = sleepZPhase;
    final textPainter = TextPainter(textDirection: TextDirection.ltr);
    for (int i = 0; i < 3; i++) {
      final p = (phase + i * 0.8) % (2 * pi);
      final yOff = -30.0 * (p / (2 * pi));
      final alpha = (1.0 - p / (2 * pi)).clamp(0.0, 1.0);
      final size = 12.0 + i * 4;
      textPainter.text = TextSpan(
        text: 'z',
        style: TextStyle(
          color: Colors.grey.withValues(alpha: alpha),
          fontSize: size,
          fontWeight: FontWeight.bold,
          fontStyle: FontStyle.italic,
        ),
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(cx + i * 15, cy + yOff - i * 15));
    }
  }

  void _drawSparkles(Canvas canvas) {
    for (final s in sparkles) {
      final alpha = (s.life / s.maxLife).clamp(0.0, 1.0);
      final paint = Paint()
        ..color = Colors.white.withValues(alpha: alpha * 0.8);
      canvas.drawCircle(Offset(s.x, s.y), s.size * alpha, paint);
    }
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) => true;
}
