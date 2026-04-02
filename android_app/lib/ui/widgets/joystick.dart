import 'dart:math';
import 'package:flutter/material.dart';

/// Direction from joystick
enum JoystickDirection {
  none,
  forward,
  backward,
  left,
  right,
  forwardLeft,
  forwardRight,
  backwardLeft,
  backwardRight,
}

/// Callback with direction and distance (0-1)
typedef JoystickCallback = void Function(JoystickDirection direction, double distance);

/// On-screen joystick for manual car control.
class Joystick extends StatefulWidget {
  final JoystickCallback onMove;
  final VoidCallback onStop;
  final double size;

  const Joystick({
    super.key,
    required this.onMove,
    required this.onStop,
    this.size = 150,
  });

  @override
  State<Joystick> createState() => _JoystickState();
}

class _JoystickState extends State<Joystick> {
  double _dx = 0, _dy = 0;
  bool _active = false;

  @override
  Widget build(BuildContext context) {
    final radius = widget.size / 2;
    final knobRadius = radius * 0.4;

    return GestureDetector(
      onPanStart: (d) => _onMove(d.localPosition, radius),
      onPanUpdate: (d) => _onMove(d.localPosition, radius),
      onPanEnd: (_) => _onRelease(),
      onPanCancel: _onRelease,
      child: SizedBox(
        width: widget.size,
        height: widget.size,
        child: CustomPaint(
          painter: _JoystickPainter(
            dx: _dx,
            dy: _dy,
            radius: radius,
            knobRadius: knobRadius,
            active: _active,
          ),
        ),
      ),
    );
  }

  void _onMove(Offset pos, double radius) {
    final cx = radius;
    final cy = radius;
    var dx = (pos.dx - cx) / radius;
    var dy = (pos.dy - cy) / radius;

    // Clamp to circle
    final dist = sqrt(dx * dx + dy * dy);
    if (dist > 1.0) {
      dx /= dist;
      dy /= dist;
    }

    setState(() {
      _dx = dx;
      _dy = dy;
      _active = true;
    });

    final direction = _getDirection(dx, dy);
    final distance = min(1.0, dist);
    widget.onMove(direction, distance);
  }

  void _onRelease() {
    setState(() {
      _dx = 0;
      _dy = 0;
      _active = false;
    });
    widget.onStop();
  }

  JoystickDirection _getDirection(double dx, double dy) {
    final dist = sqrt(dx * dx + dy * dy);
    if (dist < 0.2) return JoystickDirection.none;

    final angle = atan2(-dy, dx) * 180 / pi; // -dy because screen Y is inverted

    if (angle >= 67.5 && angle < 112.5) return JoystickDirection.forward;
    if (angle >= 112.5 && angle < 157.5) return JoystickDirection.forwardLeft;
    if (angle >= 157.5 || angle < -157.5) return JoystickDirection.left;
    if (angle >= -157.5 && angle < -112.5) return JoystickDirection.backwardLeft;
    if (angle >= -112.5 && angle < -67.5) return JoystickDirection.backward;
    if (angle >= -67.5 && angle < -22.5) return JoystickDirection.backwardRight;
    if (angle >= -22.5 && angle < 22.5) return JoystickDirection.right;
    if (angle >= 22.5 && angle < 67.5) return JoystickDirection.forwardRight;

    return JoystickDirection.none;
  }
}

class _JoystickPainter extends CustomPainter {
  final double dx, dy, radius, knobRadius;
  final bool active;

  _JoystickPainter({
    required this.dx,
    required this.dy,
    required this.radius,
    required this.knobRadius,
    required this.active,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(radius, radius);

    // Outer circle
    canvas.drawCircle(
      center,
      radius,
      Paint()
        ..color = Colors.white.withValues(alpha: 0.15)
        ..style = PaintingStyle.fill,
    );
    canvas.drawCircle(
      center,
      radius,
      Paint()
        ..color = Colors.white.withValues(alpha: 0.3)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // Knob
    final knobPos = Offset(
      radius + dx * (radius - knobRadius),
      radius + dy * (radius - knobRadius),
    );
    canvas.drawCircle(
      knobPos,
      knobRadius,
      Paint()
        ..color = active
            ? Colors.orange.withValues(alpha: 0.8)
            : Colors.white.withValues(alpha: 0.4),
    );
  }

  @override
  bool shouldRepaint(_JoystickPainter old) => true;
}
