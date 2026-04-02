import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';

/// Voice Activity Detection states.
enum _VadState { waitingForSpeech, speechActive, silenceAfterSpeech }

/// Captures audio from the microphone as PCM16 at 16kHz.
/// Uses energy-based VAD to detect speech start and end.
class AudioCaptureService {
  AudioRecorder? _recorder;
  bool _isRecording = false;

  bool get isRecording => _isRecording;

  /// Record with VAD: waits for speech, records until silence after speech.
  Future<Uint8List?> recordWithVAD({
    Duration maxDuration = const Duration(seconds: 15),
    Duration silenceAfterSpeech = const Duration(milliseconds: 1500),
    Duration maxWaitForSpeech = const Duration(seconds: 5),
  }) async {
    await _cleanup();
    _recorder = AudioRecorder();

    final chunks = <Uint8List>[];
    StreamSubscription<List<int>>? sub;
    Timer? maxTimer;
    final completer = Completer<Uint8List?>();

    try {
      debugPrint('[AudioCapture] Starting VAD recording...');

      final stream = await _recorder!.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        autoGain: true,
        echoCancel: true,
        noiseSuppress: true,
      ));
      _isRecording = true;

      var state = _VadState.waitingForSpeech;
      final startTime = DateTime.now();
      DateTime? lastSpeechTime;

      // Collect RMS samples for calibration
      final calibrationRms = <double>[];
      double speechThreshold = 700; // Will be calibrated
      bool calibrated = false;

      void finish(String reason) {
        if (completer.isCompleted) return;
        debugPrint('[AudioCapture] Done: $reason (${chunks.length} chunks)');
        maxTimer?.cancel();
        sub?.cancel();
        _cleanup();

        if (chunks.isEmpty || state == _VadState.waitingForSpeech) {
          completer.complete(null);
        } else {
          completer.complete(_mergeChunks(chunks));
        }
      }

      maxTimer = Timer(maxDuration, () => finish('max duration'));

      sub = stream.listen(
        (data) {
          if (completer.isCompleted) return;

          final bytes = Uint8List.fromList(data);
          chunks.add(bytes);

          final rms = _computeRMS(bytes);
          final elapsed = DateTime.now().difference(startTime);

          // Calibrate from first ~500ms (collect 8 samples)
          if (!calibrated) {
            calibrationRms.add(rms);
            if (calibrationRms.length >= 8) {
              calibrated = true;
              calibrationRms.sort();
              // Use 75th percentile as noise estimate
              final noiseEst = calibrationRms[(calibrationRms.length * 0.75).floor()];
              // Speech threshold: must be well above noise
              // Minimum 700 to avoid false positives from ambient noise
              speechThreshold = max(700, noiseEst * 2.0);
              debugPrint('[AudioCapture] Calibrated — noise ~${noiseEst.toStringAsFixed(0)}, threshold: ${speechThreshold.toStringAsFixed(0)}');
            }
            return;
          }

          // Log periodically
          if (chunks.length % 15 == 0) {
            debugPrint('[AudioCapture] RMS: ${rms.toStringAsFixed(0)} | ${state.name} | ${elapsed.inMilliseconds}ms');
          }

          switch (state) {
            case _VadState.waitingForSpeech:
              if (rms > speechThreshold) {
                state = _VadState.speechActive;
                lastSpeechTime = DateTime.now();
                debugPrint('[AudioCapture] >> Speech START (RMS: ${rms.toStringAsFixed(0)} > ${speechThreshold.toStringAsFixed(0)})');
              } else if (elapsed > maxWaitForSpeech) {
                finish('no speech detected');
              }

            case _VadState.speechActive:
              if (rms > speechThreshold) {
                lastSpeechTime = DateTime.now();
              } else {
                // Dropped below threshold → start silence timer
                state = _VadState.silenceAfterSpeech;
                debugPrint('[AudioCapture] >> Silence started (RMS: ${rms.toStringAsFixed(0)} < ${speechThreshold.toStringAsFixed(0)})');
              }

            case _VadState.silenceAfterSpeech:
              if (rms > speechThreshold) {
                // Speech resumed
                state = _VadState.speechActive;
                lastSpeechTime = DateTime.now();
                debugPrint('[AudioCapture] >> Speech resumed (RMS: ${rms.toStringAsFixed(0)})');
              } else if (lastSpeechTime != null) {
                final silenceDur = DateTime.now().difference(lastSpeechTime!);
                if (silenceDur > silenceAfterSpeech) {
                  finish('speech ended (${silenceDur.inMilliseconds}ms silence)');
                }
              }
          }
        },
        onError: (e) {
          debugPrint('[AudioCapture] Stream error: $e');
          finish('error');
        },
        onDone: () {
          finish('stream ended');
        },
      );
    } catch (e) {
      debugPrint('[AudioCapture] Error: $e');
      await _cleanup();
      if (!completer.isCompleted) {
        completer.complete(chunks.isEmpty ? null : _mergeChunks(chunks));
      }
    }

    return completer.future;
  }

  Future<void> _cleanup() async {
    _isRecording = false;
    try {
      await _recorder?.stop();
    } catch (_) {}
    try {
      _recorder?.dispose();
    } catch (_) {}
    _recorder = null;
  }

  Future<void> stop() async => await _cleanup();

  Future<bool> hasPermission() async {
    final r = AudioRecorder();
    try {
      final ok = await r.hasPermission();
      r.dispose();
      return ok;
    } catch (_) {
      r.dispose();
      return false;
    }
  }

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

  Uint8List _mergeChunks(List<Uint8List> chunks) {
    final total = chunks.fold<int>(0, (s, c) => s + c.length);
    final result = Uint8List(total);
    int off = 0;
    for (final c in chunks) {
      result.setRange(off, off + c.length, c);
      off += c.length;
    }
    return result;
  }

  void dispose() => _cleanup();
}
