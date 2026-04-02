import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

/// Offline Speech-to-Text using whisper.cpp via whisper_flutter_new.
///
/// Checks for the model file at the app documents directory. If the model
/// is not downloaded, throws [ModelNotAvailableException]. When the model
/// is present and the whisper_flutter_new package is enabled, performs
/// local transcription matching the CloudSTT interface.
class ModelNotAvailableException implements Exception {
  final String modelName;
  final String expectedPath;
  final String message;

  ModelNotAvailableException({
    required this.modelName,
    required this.expectedPath,
    String? message,
  }) : message = message ??
            'Model "$modelName" not found at "$expectedPath". '
                'Download it from the Model Manager in parent settings.';

  @override
  String toString() => 'ModelNotAvailableException: $message';
}

class OfflineSTT {
  static const String modelFileName = 'ggml-tiny.bin';
  static const String _tag = '[OfflineSTT]';

  String? _modelPath;
  bool _initialized = false;

  // When whisper_flutter_new is enabled, store the Whisper instance here.
  // ignore: unused_field
  dynamic _whisper;

  /// Initialize and verify the model file exists.
  Future<void> init() async {
    final docsDir = await getApplicationDocumentsDirectory();
    _modelPath = '${docsDir.path}/models/$modelFileName';

    final modelFile = File(_modelPath!);
    if (await modelFile.exists()) {
      final size = await modelFile.length();
      debugPrint('$_tag Model found: $_modelPath (${(size / 1024 / 1024).toStringAsFixed(1)} MB)');
      _initialized = true;
      _initWhisper();
    } else {
      debugPrint('$_tag Model not found at $_modelPath');
      _initialized = false;
    }
  }

  /// Whether the model file is available for offline transcription.
  bool get isAvailable => _initialized;

  /// Path where the model file is expected.
  String? get modelPath => _modelPath;

  /// Initialize the whisper engine. Requires whisper_flutter_new package
  /// to be uncommented in pubspec.yaml and imported here.
  void _initWhisper() {
    // TODO: Uncomment when whisper_flutter_new is added as a dependency:
    //
    // import 'package:whisper_flutter_new/whisper_flutter_new.dart';
    //
    // _whisper = Whisper(
    //   model: WhisperModel.fromPath(_modelPath!),
    //   language: 'auto',
    // );
    debugPrint('$_tag Whisper engine placeholder initialized');
  }

  /// Transcribe PCM16 mono 16kHz audio to text.
  ///
  /// Matches the CloudSTT interface signature.
  /// [audioData] should be raw PCM16 mono 16kHz bytes.
  /// [language] is optional; pass null for auto-detect.
  Future<String> transcribe(Uint8List audioData, {String? language}) async {
    if (!_initialized || _modelPath == null) {
      throw ModelNotAvailableException(
        modelName: modelFileName,
        expectedPath: _modelPath ?? '<unknown>',
      );
    }

    final modelFile = File(_modelPath!);
    if (!await modelFile.exists()) {
      _initialized = false;
      throw ModelNotAvailableException(
        modelName: modelFileName,
        expectedPath: _modelPath!,
        message: 'Model file was deleted after initialization. '
            'Please re-download from Model Manager.',
      );
    }

    debugPrint('$_tag Transcribing ${audioData.length} bytes...');

    // TODO: Uncomment when whisper_flutter_new is enabled:
    //
    // // Write PCM data to a temporary WAV file for whisper
    // final tempDir = await getTemporaryDirectory();
    // final tempWav = File('${tempDir.path}/offline_stt_input.wav');
    // await tempWav.writeAsBytes(_pcmToWav(audioData));
    //
    // final result = await (_whisper as Whisper).transcribe(
    //   audio: tempWav.path,
    //   language: language,
    // );
    //
    // await tempWav.delete();
    // final text = result.text.trim();
    // debugPrint('$_tag Transcript: "$text"');
    // return text;

    // Placeholder until whisper_flutter_new is enabled
    debugPrint('$_tag whisper_flutter_new not yet enabled — returning empty');
    return '';
  }

  /// Convert raw PCM16 mono 16kHz to WAV format for whisper input.
  Uint8List _pcmToWav(Uint8List pcmData, {int sampleRate = 16000, int channels = 1}) {
    final byteRate = sampleRate * channels * 2;
    final blockAlign = channels * 2;
    final dataSize = pcmData.length;
    final fileSize = 36 + dataSize;

    final buffer = ByteData(44 + dataSize);

    // RIFF header
    buffer.setUint8(0, 0x52); // R
    buffer.setUint8(1, 0x49); // I
    buffer.setUint8(2, 0x46); // F
    buffer.setUint8(3, 0x46); // F
    buffer.setUint32(4, fileSize, Endian.little);
    buffer.setUint8(8, 0x57);  // W
    buffer.setUint8(9, 0x41);  // A
    buffer.setUint8(10, 0x56); // V
    buffer.setUint8(11, 0x45); // E

    // fmt chunk
    buffer.setUint8(12, 0x66); // f
    buffer.setUint8(13, 0x6D); // m
    buffer.setUint8(14, 0x74); // t
    buffer.setUint8(15, 0x20); // space
    buffer.setUint32(16, 16, Endian.little);
    buffer.setUint16(20, 1, Endian.little);
    buffer.setUint16(22, channels, Endian.little);
    buffer.setUint32(24, sampleRate, Endian.little);
    buffer.setUint32(28, byteRate, Endian.little);
    buffer.setUint16(32, blockAlign, Endian.little);
    buffer.setUint16(34, 16, Endian.little);

    // data chunk
    buffer.setUint8(36, 0x64); // d
    buffer.setUint8(37, 0x61); // a
    buffer.setUint8(38, 0x74); // t
    buffer.setUint8(39, 0x61); // a
    buffer.setUint32(40, dataSize, Endian.little);

    final bytes = buffer.buffer.asUint8List();
    bytes.setRange(44, 44 + dataSize, pcmData);
    return bytes;
  }

  void dispose() {
    // TODO: Uncomment when whisper_flutter_new is enabled:
    // (_whisper as Whisper?)?.dispose();
    _whisper = null;
    _initialized = false;
    debugPrint('$_tag Disposed');
  }
}
