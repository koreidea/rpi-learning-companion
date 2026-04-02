import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart'
    as mlkit;

import '../bluetooth/car_chassis.dart';
import 'camera_manager.dart';
import 'person_detector.dart';

/// Follow mode state — tracks what the follower is doing.
enum FollowState {
  /// Not running.
  stopped,

  /// Actively following a detected person.
  following,

  /// Person lost — spinning to search.
  searching,

  /// Waiting (person lost briefly, holding position).
  waiting,
}

/// Person-following mode: uses camera + ML Kit person detection + car steering.
///
/// Port of rpi/modules/follow.py for Flutter/Android.
///
/// Steering logic:
/// - Frame divided into left/center/right thirds
/// - Person in left third  -> turn left
/// - Person in right third -> turn right
/// - Person centered       -> go forward
/// - Person too close (bbox area > 40% of frame) -> stop/back up
/// - Person too far  (bbox area < 5% of frame)   -> go forward faster
/// - No person detected -> stop and wait, then search
class FollowMode {
  final CarChassis _car;
  final CameraManager _camera;
  final PersonDetector _detector;

  FollowState _state = FollowState.stopped;
  bool _processingFrame = false;
  DateTime? _lastPersonTime;

  /// Callback for state changes (UI can listen).
  void Function(FollowState state)? onStateChanged;

  // ── Tuning constants (matching rpi/modules/follow.py) ──

  /// Base movement speed (0-255).
  static const int baseSpeed = 150;

  /// Fast approach speed when person is very far.
  static const int fastSpeed = 220;

  /// Turn speed for pure spin corrections.
  static const int turnSpeed = 180;

  /// Person bbox area > this fraction of frame -> too close, back up.
  static const double tooCloseThreshold = 0.40;

  /// Person bbox area < this fraction of frame -> too far, go faster.
  static const double tooFarThreshold = 0.05;

  /// Person bbox area in this range -> comfortable distance, hold.
  static const double comfortableMinArea = 0.10;

  /// Horizontal dead zone: person center within middle third is "centered".
  /// Expressed as fraction from center (0.167 = 1/6 of frame on each side).
  static const double deadZoneFraction = 0.167;

  /// Seconds without detection before switching to search mode.
  static const double searchTimeoutSeconds = 3.0;

  /// Target frame processing rate.
  static const int targetFps = 10;
  static const Duration _frameInterval =
      Duration(milliseconds: 1000 ~/ targetFps);

  FollowMode({
    required CarChassis car,
    required CameraManager camera,
    PersonDetector? detector,
  })  : _car = car,
        _camera = camera,
        _detector = detector ?? PersonDetector();

  FollowState get state => _state;
  bool get isActive => _state != FollowState.stopped;

  /// Start follow mode.
  ///
  /// Initializes the camera for streaming and the person detector,
  /// then begins processing frames at ~10 FPS.
  Future<bool> start() async {
    if (isActive) {
      debugPrint('[FollowMode] Already active');
      return true;
    }

    if (!_car.connected) {
      debugPrint('[FollowMode] Car not connected — cannot start');
      return false;
    }

    // Initialize detector
    _detector.init();

    // Ensure camera is initialized
    if (!_camera.isInitialized) {
      final ok = await _camera.init();
      if (!ok) {
        debugPrint('[FollowMode] Camera init failed');
        return false;
      }
    }

    // Reinitialize camera for NV21 streaming (ML Kit needs raw frames)
    final streamOk = await _camera.reinitForStreaming();
    if (!streamOk) {
      debugPrint('[FollowMode] Camera reinit for streaming failed');
      return false;
    }

    _lastPersonTime = DateTime.now();
    _setState(FollowState.waiting);

    // Start camera image stream
    final started = await _camera.startImageStream(_onCameraFrame);
    if (!started) {
      debugPrint('[FollowMode] Failed to start image stream');
      _setState(FollowState.stopped);
      return false;
    }

    debugPrint('[FollowMode] Started (ML Kit person detection, ~$targetFps FPS)');
    return true;
  }

  /// Stop follow mode.
  Future<void> stop() async {
    if (!isActive) return;

    await _camera.stopImageStream();

    if (_car.connected) {
      await _car.stop();
    }

    _setState(FollowState.stopped);
    _processingFrame = false;
    debugPrint('[FollowMode] Stopped');
  }

  // ── Frame processing ──

  /// Throttled frame callback from camera stream.
  /// We skip frames to maintain target FPS and avoid overwhelming the detector.
  DateTime _lastFrameTime = DateTime.fromMillisecondsSinceEpoch(0);

  void _onCameraFrame(CameraImage image) {
    if (!isActive) return;

    final now = DateTime.now();
    if (now.difference(_lastFrameTime) < _frameInterval) return;
    _lastFrameTime = now;

    // Don't queue up frames — only process one at a time
    if (_processingFrame) return;

    _processFrameAsync(image);
  }

  Future<void> _processFrameAsync(CameraImage image) async {
    if (_processingFrame || !isActive) return;
    _processingFrame = true;

    try {
      final rotation = _mapSensorRotation(_camera.sensorOrientation);
      final detection = await _detector.detectFromCameraImage(
        image,
        rotation: rotation,
      );

      if (!isActive) return; // Stopped while processing

      if (detection != null) {
        _lastPersonTime = DateTime.now();
        _setState(FollowState.following);
        await _steerTowardPerson(detection);
      } else {
        _handleNoPerson();
      }
    } catch (e) {
      debugPrint('[FollowMode] Frame processing error: $e');
    } finally {
      _processingFrame = false;
    }
  }

  /// Convert camera sensor orientation to ML Kit InputImageRotation.
  mlkit.InputImageRotation _mapSensorRotation(int sensorDegrees) {
    switch (sensorDegrees) {
      case 0:
        return mlkit.InputImageRotation.rotation0deg;
      case 90:
        return mlkit.InputImageRotation.rotation90deg;
      case 180:
        return mlkit.InputImageRotation.rotation180deg;
      case 270:
        return mlkit.InputImageRotation.rotation270deg;
      default:
        return mlkit.InputImageRotation.rotation0deg;
    }
  }

  // ── Steering logic (port of rpi/modules/follow.py _steer_to_center) ──

  Future<void> _steerTowardPerson(PersonDetection detection) async {
    if (!_car.connected) return;

    final areaRatio = detection.areaRatio;
    final normalizedOffset = detection.normalizedOffsetX;
    final centered = normalizedOffset.abs() < deadZoneFraction;

    // 1. Too close -> stop or back up
    if (areaRatio > tooCloseThreshold) {
      debugPrint('[FollowMode] Too close (area=${areaRatio.toStringAsFixed(2)}), backing up');
      await _car.backward(speed: baseSpeed ~/ 2);
      return;
    }

    // 2. Comfortable distance + centered -> hold position
    if (areaRatio >= comfortableMinArea && centered) {
      debugPrint('[FollowMode] Centered + comfortable distance, holding');
      await _car.stop();
      return;
    }

    // 3. Off-center + close enough -> just spin to re-center
    if (!centered && areaRatio >= comfortableMinArea) {
      final correctionSpeed = _proportionalSpeed(normalizedOffset, turnSpeed);
      if (normalizedOffset > 0) {
        debugPrint('[FollowMode] Close + right, spin right (speed=$correctionSpeed)');
        await _car.spinRight(speed: correctionSpeed);
      } else {
        debugPrint('[FollowMode] Close + left, spin left (speed=$correctionSpeed)');
        await _car.spinLeft(speed: correctionSpeed);
      }
      return;
    }

    // 4. Far away -> approach
    final approachSpeed = areaRatio < tooFarThreshold ? fastSpeed : baseSpeed;

    if (centered) {
      // Centered + far -> drive straight forward
      debugPrint('[FollowMode] Centered + far (area=${areaRatio.toStringAsFixed(2)}), forward');
      await _car.forward(speed: approachSpeed);
    } else {
      // Off-center + far -> diagonal approach
      final correctionSpeed = _proportionalSpeed(normalizedOffset, approachSpeed);
      if (normalizedOffset > 0) {
        debugPrint('[FollowMode] Far + right, forward-right');
        await _car.forwardRight(speed: correctionSpeed);
      } else {
        debugPrint('[FollowMode] Far + left, forward-left');
        await _car.forwardLeft(speed: correctionSpeed);
      }
    }
  }

  /// Calculate speed proportional to horizontal offset.
  /// Larger offset = faster correction. Returns speed in 0-255 range.
  int _proportionalSpeed(double normalizedOffset, int maxSpeed) {
    // Scale: offset 0.167 (edge of dead zone) -> 50% speed
    //        offset 1.0   (far edge of frame) -> 100% speed
    final factor = (normalizedOffset.abs() / 1.0).clamp(0.3, 1.0);
    return (maxSpeed * factor).round().clamp(80, 255);
  }

  // ── No-person handling ──

  void _handleNoPerson() {
    if (_lastPersonTime == null) {
      _setState(FollowState.waiting);
      return;
    }

    final elapsed = DateTime.now().difference(_lastPersonTime!).inMilliseconds / 1000.0;

    if (elapsed > searchTimeoutSeconds) {
      // Lost person for a while -> search by spinning
      _setState(FollowState.searching);
      _searchSpin(elapsed);
    } else {
      // Brief loss -> hold position
      _setState(FollowState.waiting);
      _car.stop();
    }
  }

  /// Search pattern: alternate spinning left/right every 4 seconds.
  void _searchSpin(double elapsedSeconds) {
    if (!_car.connected) return;

    // Alternate direction every 4 seconds
    if (elapsedSeconds.toInt() % 8 < 4) {
      _car.spinLeft(speed: turnSpeed);
    } else {
      _car.spinRight(speed: turnSpeed);
    }
  }

  // ── State management ──

  void _setState(FollowState newState) {
    if (_state == newState) return;
    _state = newState;
    debugPrint('[FollowMode] State -> $newState');
    onStateChanged?.call(newState);
  }

  /// Release all resources.
  Future<void> dispose() async {
    await stop();
    _detector.dispose();
  }
}
