import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

/// Voice Activity Detection with ML-based Silero VAD and energy-based fallback.
///
/// Attempts to use the Silero VAD ONNX model for accurate speech detection.
/// If the model is not available, falls back to energy-based RMS thresholding
/// (matching the approach in AudioCaptureService).
class VadDetector {
  static const String modelFileName = 'silero_vad.onnx';
  static const String _tag = '[VAD]';

  /// RMS threshold for energy-based fallback. Calibrated for PCM16 16kHz.
  static const double _defaultEnergyThreshold = 700.0;

  String? _modelPath;
  bool _mlAvailable = false;
  bool _initialized = false;

  /// Custom energy threshold, can be updated via calibration.
  double energyThreshold = _defaultEnergyThreshold;

  // When onnxruntime_flutter is enabled, store the session here.
  // ignore: unused_field
  dynamic _onnxSession;

  /// Initialize the VAD detector.
  ///
  /// Checks for the Silero VAD ONNX model file. Falls back to energy-based
  /// detection if the model is not present.
  Future<void> init() async {
    final docsDir = await getApplicationDocumentsDirectory();
    _modelPath = '${docsDir.path}/models/$modelFileName';

    final modelFile = File(_modelPath!);
    if (await modelFile.exists()) {
      final size = await modelFile.length();
      debugPrint(
        '$_tag Silero VAD model found: $_modelPath '
        '(${(size / 1024).toStringAsFixed(0)} KB)',
      );
      _mlAvailable = await _loadOnnxModel();
    } else {
      debugPrint('$_tag Silero VAD model not found, using energy-based fallback');
      _mlAvailable = false;
    }

    _initialized = true;
    debugPrint('$_tag Initialized (ML: $_mlAvailable)');
  }

  /// Whether the ML-based Silero VAD is available.
  bool get isMlAvailable => _mlAvailable;

  /// Whether the detector has been initialized.
  bool get isInitialized => _initialized;

  /// Path where the Silero VAD model is expected.
  String? get modelPath => _modelPath;

  /// Load the ONNX model. Returns true on success.
  Future<bool> _loadOnnxModel() async {
    try {
      // TODO: Uncomment when onnxruntime_flutter is added as a dependency:
      //
      // import 'package:onnxruntime_flutter/onnxruntime_flutter.dart';
      //
      // _onnxSession = await OrtSession.fromFile(_modelPath!);
      // debugPrint('$_tag ONNX session created');
      // return true;

      debugPrint('$_tag onnxruntime_flutter not yet enabled');
      return false;
    } catch (e) {
      debugPrint('$_tag Failed to load ONNX model: $e');
      return false;
    }
  }

  /// Detect whether an audio chunk contains speech.
  ///
  /// [audioChunk] should be raw PCM16 mono 16kHz bytes.
  /// Returns true if speech is detected.
  ///
  /// Uses Silero VAD if the ML model is available, otherwise falls back
  /// to energy-based RMS thresholding.
  bool isSpeech(Uint8List audioChunk) {
    if (!_initialized) {
      debugPrint('$_tag Not initialized, call init() first');
      return false;
    }

    if (_mlAvailable) {
      return _isSpeechML(audioChunk);
    } else {
      return _isSpeechEnergy(audioChunk);
    }
  }

  /// ML-based speech detection using Silero VAD ONNX model.
  bool _isSpeechML(Uint8List audioChunk) {
    // TODO: Uncomment when onnxruntime_flutter is enabled:
    //
    // // Convert PCM16 bytes to float32 array normalized to [-1, 1]
    // final floats = _pcm16ToFloat32(audioChunk);
    //
    // // Run inference
    // final input = OrtValueTensor.fromList(floats, [1, floats.length]);
    // final outputs = _onnxSession.run({'input': input});
    // final probability = (outputs['output'] as List<double>)[0];
    //
    // return probability > 0.5;

    // Fallback to energy-based until ONNX is enabled
    return _isSpeechEnergy(audioChunk);
  }

  /// Energy-based speech detection using RMS thresholding.
  ///
  /// Matches the approach used in AudioCaptureService for consistency.
  bool _isSpeechEnergy(Uint8List audioChunk) {
    final rms = _computeRMS(audioChunk);
    return rms > energyThreshold;
  }

  /// Compute RMS (Root Mean Square) energy of PCM16 audio data.
  double _computeRMS(Uint8List data) {
    if (data.length < 2) return 0;
    final bd = ByteData.sublistView(data);
    double sum = 0;
    final n = data.length ~/ 2;
    for (int i = 0; i < n; i++) {
      final s = bd.getInt16(i * 2, Endian.little).toDouble();
      sum += s * s;
    }
    return sqrt(sum / n);
  }

  /// Calibrate the energy threshold from ambient noise samples.
  ///
  /// [noiseSamples] should be a list of RMS values collected during silence.
  /// Sets the threshold to 2x the 75th percentile of noise, with a minimum
  /// of [_defaultEnergyThreshold].
  void calibrate(List<double> noiseSamples) {
    if (noiseSamples.isEmpty) return;
    final sorted = List<double>.from(noiseSamples)..sort();
    final p75 = sorted[(sorted.length * 0.75).floor()];
    energyThreshold = max(_defaultEnergyThreshold, p75 * 2.0);
    debugPrint(
      '$_tag Calibrated energy threshold: '
      '${energyThreshold.toStringAsFixed(0)} '
      '(noise p75: ${p75.toStringAsFixed(0)})',
    );
  }

  /// Convert PCM16 bytes to normalized float32 array [-1.0, 1.0].
  /// Used for ML model input.
  // ignore: unused_element
  Float32List _pcm16ToFloat32(Uint8List data) {
    final bd = ByteData.sublistView(data);
    final n = data.length ~/ 2;
    final floats = Float32List(n);
    for (int i = 0; i < n; i++) {
      floats[i] = bd.getInt16(i * 2, Endian.little) / 32768.0;
    }
    return floats;
  }

  void dispose() {
    // TODO: Uncomment when onnxruntime_flutter is enabled:
    // (_onnxSession as OrtSession?)?.release();
    _onnxSession = null;
    _mlAvailable = false;
    _initialized = false;
    debugPrint('$_tag Disposed');
  }
}
