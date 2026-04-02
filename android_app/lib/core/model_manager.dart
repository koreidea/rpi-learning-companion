import 'dart:async';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

/// Describes a downloadable ML model.
class ModelInfo {
  final String id;
  final String displayName;
  final String fileName;
  final String downloadUrl;
  final int estimatedSizeBytes;
  final String description;

  const ModelInfo({
    required this.id,
    required this.displayName,
    required this.fileName,
    required this.downloadUrl,
    required this.estimatedSizeBytes,
    required this.description,
  });
}

/// Status of a model on the device.
enum ModelStatus {
  notDownloaded,
  downloading,
  ready,
  error,
}

/// Tracks download progress and status for a single model.
class ModelState {
  final ModelInfo info;
  ModelStatus status;
  double progress; // 0.0 to 1.0
  int downloadedBytes;
  String? errorMessage;

  ModelState({
    required this.info,
    this.status = ModelStatus.notDownloaded,
    this.progress = 0.0,
    this.downloadedBytes = 0,
    this.errorMessage,
  });

  /// Human-readable size string.
  String get estimatedSizeDisplay {
    final mb = info.estimatedSizeBytes / (1024 * 1024);
    if (mb >= 1024) {
      return '${(mb / 1024).toStringAsFixed(1)} GB';
    }
    return '${mb.toStringAsFixed(0)} MB';
  }

  /// Human-readable downloaded size.
  String get downloadedSizeDisplay {
    final mb = downloadedBytes / (1024 * 1024);
    if (mb >= 1024) {
      return '${(mb / 1024).toStringAsFixed(1)} GB';
    }
    return '${mb.toStringAsFixed(0)} MB';
  }
}

/// Callback for download progress updates.
typedef ModelProgressCallback = void Function(String modelId, double progress);

/// Manages ML model downloads, storage, and lifecycle.
///
/// Models are stored in `<app_documents>/models/`.
/// Provides download with progress tracking, deletion, and status checks.
class ModelManager {
  static const String _tag = '[ModelManager]';
  static const String _modelsSubdir = 'models';

  /// Registry of all available models.
  static const List<ModelInfo> availableModels = [
    ModelInfo(
      id: 'whisper-tiny',
      displayName: 'Whisper Tiny (STT)',
      fileName: 'ggml-tiny.bin',
      downloadUrl:
          'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
      estimatedSizeBytes: 77 * 1024 * 1024, // ~77 MB
      description: 'Offline speech-to-text. Supports multiple languages.',
    ),
    ModelInfo(
      id: 'qwen-3b',
      displayName: 'Qwen 2.5 3B (LLM)',
      fileName: 'qwen2.5-3b-instruct-q4_k_m.gguf',
      downloadUrl:
          'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf',
      estimatedSizeBytes: 2048 * 1024 * 1024, // ~2 GB
      description: 'Offline language model for conversations. Large download.',
    ),
    ModelInfo(
      id: 'silero-vad',
      displayName: 'Silero VAD',
      fileName: 'silero_vad.onnx',
      downloadUrl:
          'https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx',
      estimatedSizeBytes: 2 * 1024 * 1024, // ~2 MB
      description:
          'ML-based voice activity detection. Small, improves accuracy.',
    ),
  ];

  final Dio _dio = Dio();
  String? _modelsDir;
  final Map<String, ModelState> _states = {};
  final Map<String, CancelToken> _activeDownloads = {};

  /// Initialize the model manager.
  /// Scans the models directory and determines status of each model.
  Future<void> init() async {
    final docsDir = await getApplicationDocumentsDirectory();
    _modelsDir = '${docsDir.path}/$_modelsSubdir';

    // Ensure models directory exists
    final dir = Directory(_modelsDir!);
    if (!await dir.exists()) {
      await dir.create(recursive: true);
      debugPrint('$_tag Created models directory: $_modelsDir');
    }

    // Initialize state for each model
    for (final model in availableModels) {
      final file = File('$_modelsDir/${model.fileName}');
      final exists = await file.exists();
      int fileSize = 0;
      if (exists) {
        fileSize = await file.length();
      }

      _states[model.id] = ModelState(
        info: model,
        status: exists ? ModelStatus.ready : ModelStatus.notDownloaded,
        progress: exists ? 1.0 : 0.0,
        downloadedBytes: fileSize,
      );

      if (exists) {
        debugPrint(
          '$_tag ${model.displayName}: ready '
          '(${(fileSize / 1024 / 1024).toStringAsFixed(1)} MB)',
        );
      }
    }

    debugPrint('$_tag Initialized with ${_states.length} models');
  }

  /// Get the models directory path.
  String? get modelsDir => _modelsDir;

  /// Get the current state of all models.
  List<ModelState> get allModelStates => _states.values.toList();

  /// Get the state of a specific model.
  ModelState? getModelState(String modelId) => _states[modelId];

  /// Check if a specific model is downloaded and ready.
  bool isModelReady(String modelId) {
    return _states[modelId]?.status == ModelStatus.ready;
  }

  /// Get the full file path for a model.
  String? getModelPath(String modelId) {
    final state = _states[modelId];
    if (state == null || _modelsDir == null) return null;
    return '$_modelsDir/${state.info.fileName}';
  }

  /// Download a model with progress tracking.
  ///
  /// [onProgress] is called with (modelId, progress) where progress is 0.0-1.0.
  /// Returns true if the download completed successfully.
  Future<bool> downloadModel(
    String modelId, {
    ModelProgressCallback? onProgress,
  }) async {
    final state = _states[modelId];
    if (state == null) {
      debugPrint('$_tag Unknown model: $modelId');
      return false;
    }

    if (state.status == ModelStatus.downloading) {
      debugPrint('$_tag ${state.info.displayName} is already downloading');
      return false;
    }

    if (state.status == ModelStatus.ready) {
      debugPrint('$_tag ${state.info.displayName} is already downloaded');
      return true;
    }

    final filePath = '$_modelsDir/${state.info.fileName}';
    final tempPath = '$filePath.downloading';
    final cancelToken = CancelToken();

    _activeDownloads[modelId] = cancelToken;
    state.status = ModelStatus.downloading;
    state.progress = 0.0;
    state.downloadedBytes = 0;
    state.errorMessage = null;

    debugPrint(
      '$_tag Downloading ${state.info.displayName} '
      'from ${state.info.downloadUrl}',
    );

    try {
      await _dio.download(
        state.info.downloadUrl,
        tempPath,
        cancelToken: cancelToken,
        onReceiveProgress: (received, total) {
          if (total > 0) {
            state.progress = received / total;
          } else {
            // Estimate progress from known size
            state.progress = received / state.info.estimatedSizeBytes;
          }
          state.downloadedBytes = received;
          onProgress?.call(modelId, state.progress);
        },
        options: Options(
          receiveTimeout: const Duration(minutes: 30),
          followRedirects: true,
          maxRedirects: 5,
        ),
      );

      // Move temp file to final location
      final tempFile = File(tempPath);
      if (await tempFile.exists()) {
        await tempFile.rename(filePath);
        final finalSize = await File(filePath).length();

        state.status = ModelStatus.ready;
        state.progress = 1.0;
        state.downloadedBytes = finalSize;
        onProgress?.call(modelId, 1.0);

        debugPrint(
          '$_tag ${state.info.displayName} downloaded successfully '
          '(${(finalSize / 1024 / 1024).toStringAsFixed(1)} MB)',
        );
        return true;
      } else {
        throw Exception('Downloaded file not found after download');
      }
    } on DioException catch (e) {
      if (e.type == DioExceptionType.cancel) {
        debugPrint('$_tag Download cancelled: ${state.info.displayName}');
        state.status = ModelStatus.notDownloaded;
        state.errorMessage = 'Download cancelled';
      } else {
        debugPrint('$_tag Download failed: ${state.info.displayName}: $e');
        state.status = ModelStatus.error;
        state.errorMessage = e.message ?? 'Download failed';
      }
      state.progress = 0.0;
      state.downloadedBytes = 0;

      // Clean up temp file
      try {
        final tempFile = File(tempPath);
        if (await tempFile.exists()) await tempFile.delete();
      } catch (_) {}

      return false;
    } catch (e) {
      debugPrint('$_tag Download error: ${state.info.displayName}: $e');
      state.status = ModelStatus.error;
      state.errorMessage = e.toString();
      state.progress = 0.0;
      state.downloadedBytes = 0;

      try {
        final tempFile = File(tempPath);
        if (await tempFile.exists()) await tempFile.delete();
      } catch (_) {}

      return false;
    } finally {
      _activeDownloads.remove(modelId);
    }
  }

  /// Cancel an ongoing download.
  void cancelDownload(String modelId) {
    final token = _activeDownloads[modelId];
    if (token != null && !token.isCancelled) {
      token.cancel('User cancelled');
      debugPrint('$_tag Cancelling download: $modelId');
    }
  }

  /// Delete a downloaded model file.
  ///
  /// Returns true if the file was deleted or did not exist.
  Future<bool> deleteModel(String modelId) async {
    final state = _states[modelId];
    if (state == null || _modelsDir == null) return false;

    // Cancel any ongoing download first
    cancelDownload(modelId);

    final filePath = '$_modelsDir/${state.info.fileName}';
    final file = File(filePath);

    try {
      if (await file.exists()) {
        await file.delete();
        debugPrint('$_tag Deleted: ${state.info.displayName}');
      }

      state.status = ModelStatus.notDownloaded;
      state.progress = 0.0;
      state.downloadedBytes = 0;
      state.errorMessage = null;
      return true;
    } catch (e) {
      debugPrint('$_tag Failed to delete ${state.info.displayName}: $e');
      return false;
    }
  }

  /// Get total disk space used by all downloaded models in bytes.
  Future<int> getTotalDiskUsage() async {
    if (_modelsDir == null) return 0;

    int total = 0;
    final dir = Directory(_modelsDir!);
    if (!await dir.exists()) return 0;

    await for (final entity in dir.list()) {
      if (entity is File) {
        total += await entity.length();
      }
    }
    return total;
  }

  /// Human-readable total disk usage.
  Future<String> getTotalDiskUsageDisplay() async {
    final bytes = await getTotalDiskUsage();
    final mb = bytes / (1024 * 1024);
    if (mb >= 1024) {
      return '${(mb / 1024).toStringAsFixed(1)} GB';
    }
    return '${mb.toStringAsFixed(0)} MB';
  }

  void dispose() {
    // Cancel all active downloads
    for (final entry in _activeDownloads.entries) {
      if (!entry.value.isCancelled) {
        entry.value.cancel('Manager disposed');
      }
    }
    _activeDownloads.clear();
    _dio.close();
    debugPrint('$_tag Disposed');
  }
}
