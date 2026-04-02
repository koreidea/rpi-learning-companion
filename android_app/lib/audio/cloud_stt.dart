import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

/// Cloud Speech-to-Text via OpenAI Whisper API.
/// Sends recorded audio to OpenAI's API and returns transcript.
class CloudSTT {
  final Dio _dio = Dio();
  String _apiKey;

  CloudSTT({required String apiKey}) : _apiKey = apiKey;

  void updateApiKey(String key) => _apiKey = key;

  /// Transcribe PCM16 audio bytes to text using OpenAI Whisper API.
  /// [audioData] should be raw PCM16 mono 16kHz.
  Future<String> transcribe(Uint8List audioData, {String? language}) async {
    if (_apiKey.isEmpty) {
      debugPrint('[CloudSTT] No API key set');
      return '';
    }

    try {
      debugPrint('[CloudSTT] Transcribing ${audioData.length} bytes...');

      // Convert PCM16 to WAV format (OpenAI requires a file format)
      final wavData = _pcmToWav(audioData, sampleRate: 16000, channels: 1);

      final map = <String, dynamic>{
        'file': MultipartFile.fromBytes(wavData, filename: 'audio.wav'),
        'model': 'whisper-1',
        'response_format': 'json',
      };
      // Only force language if specified; otherwise let Whisper auto-detect
      if (language != null) map['language'] = language;
      final formData = FormData.fromMap(map);

      final response = await _dio.post(
        'https://api.openai.com/v1/audio/transcriptions',
        data: formData,
        options: Options(
          headers: {
            'Authorization': 'Bearer $_apiKey',
          },
          sendTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(seconds: 30),
        ),
      );

      final text = response.data['text']?.toString().trim() ?? '';
      debugPrint('[CloudSTT] Transcript: "$text"');
      return text;
    } catch (e) {
      debugPrint('[CloudSTT] Error: $e');
      return '';
    }
  }

  /// Convert raw PCM16 to WAV format.
  Uint8List _pcmToWav(Uint8List pcmData, {required int sampleRate, required int channels}) {
    final byteRate = sampleRate * channels * 2; // 16-bit = 2 bytes
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
    buffer.setUint8(15, 0x20); // (space)
    buffer.setUint32(16, 16, Endian.little); // chunk size
    buffer.setUint16(20, 1, Endian.little);  // PCM format
    buffer.setUint16(22, channels, Endian.little);
    buffer.setUint32(24, sampleRate, Endian.little);
    buffer.setUint32(28, byteRate, Endian.little);
    buffer.setUint16(32, blockAlign, Endian.little);
    buffer.setUint16(34, 16, Endian.little); // bits per sample

    // data chunk
    buffer.setUint8(36, 0x64); // d
    buffer.setUint8(37, 0x61); // a
    buffer.setUint8(38, 0x74); // t
    buffer.setUint8(39, 0x61); // a
    buffer.setUint32(40, dataSize, Endian.little);

    // Copy PCM data
    final bytes = buffer.buffer.asUint8List();
    bytes.setRange(44, 44 + dataSize, pcmData);

    return bytes;
  }
}
