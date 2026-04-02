import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';

/// Manages camera lifecycle: initialization, preview, frame capture, and disposal.
///
/// Wraps the `camera` package with proper permission handling, camera switching,
/// and single-frame JPEG capture for vision features.
class CameraManager {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  int _currentCameraIndex = 0;
  bool _initialized = false;
  bool _initializing = false;

  /// Whether the camera is initialized and ready for capture.
  bool get isInitialized => _initialized && _controller != null && _controller!.value.isInitialized;

  /// The active camera controller, if initialized.
  CameraController? get controller => _controller;

  /// The current camera description (front/back).
  CameraDescription? get currentCamera =>
      _cameras != null && _cameras!.isNotEmpty ? _cameras![_currentCameraIndex] : null;

  /// Initialize the camera subsystem.
  ///
  /// Requests camera permission, enumerates available cameras, and starts
  /// the back camera by default. Falls back to front camera if back is unavailable.
  Future<bool> init({CameraLensDirection preferred = CameraLensDirection.back}) async {
    if (_initialized) return true;
    if (_initializing) return false;
    _initializing = true;

    try {
      // Request permission
      final status = await Permission.camera.request();
      if (!status.isGranted) {
        debugPrint('[CameraManager] Camera permission denied');
        return false;
      }

      // Enumerate cameras
      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        debugPrint('[CameraManager] No cameras available');
        return false;
      }

      // Find preferred camera
      _currentCameraIndex = _cameras!.indexWhere((c) => c.lensDirection == preferred);
      if (_currentCameraIndex < 0) {
        _currentCameraIndex = 0;
      }

      await _startCamera(_cameras![_currentCameraIndex]);
      _initialized = true;
      debugPrint('[CameraManager] Initialized with ${_cameras![_currentCameraIndex].lensDirection}');
      return true;
    } catch (e) {
      debugPrint('[CameraManager] Init error: $e');
      return false;
    } finally {
      _initializing = false;
    }
  }

  /// Start or restart the camera controller for the given camera.
  Future<void> _startCamera(CameraDescription camera) async {
    // Dispose previous controller if any
    await _controller?.dispose();

    _controller = CameraController(
      camera,
      // Medium resolution is a good balance between quality and speed
      // for vision API calls. High res wastes bandwidth; low res loses detail.
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    await _controller!.initialize();
  }

  /// Capture a single frame as JPEG bytes.
  ///
  /// Returns null if the camera is not initialized or capture fails.
  Future<Uint8List?> captureFrame() async {
    if (!isInitialized) {
      debugPrint('[CameraManager] Cannot capture: not initialized');
      return null;
    }

    try {
      final xFile = await _controller!.takePicture();
      final bytes = await xFile.readAsBytes();
      debugPrint('[CameraManager] Captured frame: ${bytes.length} bytes');
      return bytes;
    } catch (e) {
      debugPrint('[CameraManager] Capture error: $e');
      return null;
    }
  }

  /// Switch between front and back cameras.
  ///
  /// Returns true if the switch succeeded, false otherwise.
  Future<bool> switchCamera() async {
    if (_cameras == null || _cameras!.length < 2) {
      debugPrint('[CameraManager] Cannot switch: less than 2 cameras');
      return false;
    }

    try {
      _currentCameraIndex = (_currentCameraIndex + 1) % _cameras!.length;
      await _startCamera(_cameras![_currentCameraIndex]);
      debugPrint('[CameraManager] Switched to ${_cameras![_currentCameraIndex].lensDirection}');
      return true;
    } catch (e) {
      debugPrint('[CameraManager] Switch error: $e');
      return false;
    }
  }

  /// Start the camera preview (initialize if needed).
  Future<bool> startPreview() async {
    if (!_initialized) {
      return await init();
    }
    if (_controller != null && !_controller!.value.isInitialized) {
      await _startCamera(_cameras![_currentCameraIndex]);
    }
    return isInitialized;
  }

  /// Stop the camera preview and release resources temporarily.
  ///
  /// Call [startPreview] or [init] to restart.
  Future<void> stopPreview() async {
    if (_controller != null) {
      await _controller!.dispose();
      _controller = null;
    }
  }

  // ── Image streaming for real-time vision (follow mode) ──

  bool _streaming = false;

  /// Whether the camera is currently streaming frames.
  bool get isStreaming => _streaming;

  /// Start streaming camera frames for real-time processing.
  ///
  /// The camera must be re-initialized in NV21/YUV format for streaming
  /// (not JPEG). Call [stopImageStream] when done.
  /// Each frame is delivered to the [onFrame] callback.
  Future<bool> startImageStream(void Function(CameraImage image) onFrame) async {
    if (!isInitialized) {
      debugPrint('[CameraManager] Cannot stream: not initialized');
      return false;
    }
    if (_streaming) {
      debugPrint('[CameraManager] Already streaming');
      return true;
    }

    try {
      // If the controller was initialized in JPEG mode, we need to
      // reinitialize in YUV/NV21 for streaming. Check current format.
      // startImageStream works with the controller's current format.
      await _controller!.startImageStream(onFrame);
      _streaming = true;
      debugPrint('[CameraManager] Image stream started');
      return true;
    } catch (e) {
      debugPrint('[CameraManager] startImageStream error: $e');
      return false;
    }
  }

  /// Stop the image stream.
  Future<void> stopImageStream() async {
    if (!_streaming || _controller == null) return;

    try {
      await _controller!.stopImageStream();
    } catch (e) {
      debugPrint('[CameraManager] stopImageStream error: $e');
    }
    _streaming = false;
    debugPrint('[CameraManager] Image stream stopped');
  }

  /// Re-initialize the camera in NV21 format for ML Kit streaming.
  ///
  /// ML Kit requires NV21 (Android) or BGRA (iOS) format for real-time
  /// processing. The default JPEG mode does not support startImageStream
  /// with raw frames. Call this before [startImageStream].
  Future<bool> reinitForStreaming() async {
    if (_cameras == null || _cameras!.isEmpty) {
      debugPrint('[CameraManager] Cannot reinit: no cameras');
      return false;
    }

    try {
      await _controller?.dispose();

      _controller = CameraController(
        _cameras![_currentCameraIndex],
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.nv21,
      );

      await _controller!.initialize();
      debugPrint('[CameraManager] Reinited for streaming (NV21)');
      return true;
    } catch (e) {
      debugPrint('[CameraManager] reinitForStreaming error: $e');
      return false;
    }
  }

  /// Get the sensor orientation of the current camera in degrees.
  int get sensorOrientation {
    if (_cameras == null || _cameras!.isEmpty) return 0;
    return _cameras![_currentCameraIndex].sensorOrientation;
  }

  /// Fully dispose all camera resources.
  ///
  /// After calling dispose, you must call [init] again to use the camera.
  Future<void> dispose() async {
    await stopImageStream();
    await _controller?.dispose();
    _controller = null;
    _cameras = null;
    _initialized = false;
    _initializing = false;
    debugPrint('[CameraManager] Disposed');
  }
}
