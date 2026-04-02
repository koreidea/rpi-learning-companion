import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';

/// Bounding box of a detected person in frame coordinates.
class PersonDetection {
  /// Center X in frame pixels.
  final double cx;

  /// Center Y in frame pixels.
  final double cy;

  /// Bounding box width in pixels.
  final double width;

  /// Bounding box height in pixels.
  final double height;

  /// Detection confidence (0.0 to 1.0).
  final double confidence;

  /// Frame dimensions used for this detection.
  final int frameWidth;
  final int frameHeight;

  const PersonDetection({
    required this.cx,
    required this.cy,
    required this.width,
    required this.height,
    required this.confidence,
    required this.frameWidth,
    required this.frameHeight,
  });

  /// Person width as fraction of frame width (proxy for distance).
  double get sizeRatio => width / frameWidth;

  /// Person area as fraction of total frame area.
  double get areaRatio => (width * height) / (frameWidth * frameHeight);

  /// Horizontal offset from center, normalized to [-1, 1].
  /// Negative = left of center, positive = right of center.
  double get normalizedOffsetX => (cx - frameWidth / 2) / (frameWidth / 2);

  @override
  String toString() =>
      'PersonDetection(cx=${cx.toStringAsFixed(0)}, cy=${cy.toStringAsFixed(0)}, '
      '${width.toStringAsFixed(0)}x${height.toStringAsFixed(0)}, '
      'conf=${confidence.toStringAsFixed(2)}, sizeRatio=${sizeRatio.toStringAsFixed(2)})';
}

/// Detects persons in camera frames using Google ML Kit object detection.
///
/// Uses the base (built-in) object detector which recognizes broad categories
/// including "Person". No custom model download required.
class PersonDetector {
  ObjectDetector? _detector;
  bool _initialized = false;

  /// Minimum confidence to accept a detection.
  static const double confidenceThreshold = 0.5;

  /// The base detector uses category labels; we match by text (case-insensitive).
  static const String _personLabelAlt = 'person';

  bool get initialized => _initialized;

  /// Initialize the object detector.
  /// Call once before calling [detect].
  void init() {
    if (_initialized) return;

    final options = ObjectDetectorOptions(
      mode: DetectionMode.stream,
      classifyObjects: true,
      multipleObjects: true,
    );

    _detector = ObjectDetector(options: options);
    _initialized = true;
    debugPrint('[PersonDetector] Initialized (ML Kit base detector, stream mode)');
  }

  /// Detect persons in a [CameraImage] from the camera package.
  ///
  /// Returns the largest detected person, or null if none found.
  /// The [rotation] parameter should match the camera sensor orientation.
  Future<PersonDetection?> detectFromCameraImage(
    CameraImage image, {
    required InputImageRotation rotation,
  }) async {
    if (!_initialized || _detector == null) {
      debugPrint('[PersonDetector] Not initialized');
      return null;
    }

    final inputImage = _convertCameraImage(image, rotation);
    if (inputImage == null) return null;

    return _runDetection(inputImage, image.width, image.height);
  }

  /// Run detection on an [InputImage] and return the largest person.
  Future<PersonDetection?> _runDetection(
    InputImage inputImage,
    int frameWidth,
    int frameHeight,
  ) async {
    try {
      final objects = await _detector!.processImage(inputImage);

      PersonDetection? best;
      double bestArea = 0;

      for (final obj in objects) {
        // Check if any label matches "person"
        bool isPerson = false;
        double personConfidence = 0.0;

        for (final label in obj.labels) {
          final labelText = label.text.toLowerCase();
          if (labelText == _personLabelAlt ||
              labelText.contains('person') ||
              labelText.contains('human') ||
              labelText.contains('people')) {
            isPerson = true;
            personConfidence = label.confidence;
            break;
          }
        }

        if (!isPerson || personConfidence < confidenceThreshold) continue;

        final rect = obj.boundingBox;
        final area = rect.width * rect.height;

        if (area > bestArea) {
          bestArea = area;
          best = PersonDetection(
            cx: rect.center.dx,
            cy: rect.center.dy,
            width: rect.width,
            height: rect.height,
            confidence: personConfidence,
            frameWidth: frameWidth,
            frameHeight: frameHeight,
          );
        }
      }

      if (best != null) {
        debugPrint('[PersonDetector] $best');
      }

      return best;
    } catch (e) {
      debugPrint('[PersonDetector] Detection error: $e');
      return null;
    }
  }

  /// Convert a [CameraImage] to ML Kit's [InputImage].
  ///
  /// Android camera streams NV21 (YUV_420_888), iOS streams BGRA8888.
  /// ML Kit requires the raw bytes plus metadata describing the format.
  InputImage? _convertCameraImage(CameraImage image, InputImageRotation rotation) {
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null) {
      debugPrint('[PersonDetector] Unsupported image format: ${image.format.raw}');
      return null;
    }

    if (image.planes.isEmpty) return null;

    // Concatenate all plane bytes into a single buffer
    final allBytes = WriteBuffer();
    for (final plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }

    final metadata = InputImageMetadata(
      size: ui.Size(image.width.toDouble(), image.height.toDouble()),
      rotation: rotation,
      format: format,
      bytesPerRow: image.planes.first.bytesPerRow,
    );

    return InputImage.fromBytes(
      bytes: allBytes.done().buffer.asUint8List(),
      metadata: metadata,
    );
  }

  /// Release detector resources.
  void dispose() {
    _detector?.close();
    _detector = null;
    _initialized = false;
    debugPrint('[PersonDetector] Disposed');
  }
}
