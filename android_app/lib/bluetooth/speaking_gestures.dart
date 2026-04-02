import 'dart:async';
import 'dart:math';
import 'car_chassis.dart';

/// Subtle car movements while the bot is speaking — port of car.py gestures.
class SpeakingGestures {
  final CarChassis _car;
  bool _active = false;
  Timer? _timer;
  final _rng = Random();

  SpeakingGestures(this._car);

  bool get active => _active;

  void start() {
    if (_active || !_car.connected) return;
    _active = true;
    _loop();
  }

  void stop() {
    _active = false;
    _timer?.cancel();
    _timer = null;
    if (_car.connected) _car.stop();
  }

  void _loop() {
    if (!_active) return;

    // 60% chance of pause, 20% wiggle, 20% rock
    final roll = _rng.nextDouble();
    if (roll < 0.6) {
      _gesturePause();
    } else if (roll < 0.8) {
      _gestureWiggle();
    } else {
      _gestureRock();
    }
  }

  void _gestureWiggle() {
    const speed = 196; // ~Arduino speed 7
    _car.spinLeft(speed: speed);
    _timer = Timer(const Duration(milliseconds: 100), () {
      _car.spinRight(speed: speed);
      _timer = Timer(const Duration(milliseconds: 100), () {
        _car.stop();
        _timer = Timer(
          Duration(milliseconds: 600 + _rng.nextInt(400)),
          _loop,
        );
      });
    });
  }

  void _gestureRock() {
    const speed = 196;
    _car.forward(speed: speed);
    _timer = Timer(const Duration(milliseconds: 100), () {
      _car.stop();
      _timer = Timer(const Duration(milliseconds: 100), () {
        _car.backward(speed: speed);
        _timer = Timer(const Duration(milliseconds: 100), () {
          _car.stop();
          _timer = Timer(
            Duration(milliseconds: 600 + _rng.nextInt(400)),
            _loop,
          );
        });
      });
    });
  }

  void _gesturePause() {
    _timer = Timer(
      Duration(milliseconds: 800 + _rng.nextInt(700)),
      _loop,
    );
  }
}
