import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import 'llm_provider.dart';

/// Offline LLM provider using llama.cpp via fllama package.
///
/// Checks for the GGUF model file in the app documents directory.
/// If the model is not downloaded, the stream yields an error.
/// When the fllama package is enabled and the model is present,
/// performs local streaming token generation.
class OfflineLlm implements LlmProvider {
  static const String modelFileName = 'qwen2.5-3b-instruct-q4_k_m.gguf';
  static const String _tag = '[OfflineLlm]';

  String? _modelPath;
  bool _modelAvailable = false;

  // When fllama is enabled, store the model context here.
  // ignore: unused_field
  dynamic _llamaContext;

  @override
  String get name => 'offline-qwen2.5-3b';

  /// Initialize the provider and check for the model file.
  Future<void> init() async {
    final docsDir = await getApplicationDocumentsDirectory();
    _modelPath = '${docsDir.path}/models/$modelFileName';

    final modelFile = File(_modelPath!);
    if (await modelFile.exists()) {
      final size = await modelFile.length();
      debugPrint(
        '$_tag Model found: $_modelPath '
        '(${(size / 1024 / 1024).toStringAsFixed(0)} MB)',
      );
      _modelAvailable = true;
      await _loadModel();
    } else {
      debugPrint('$_tag Model not found at $_modelPath');
      _modelAvailable = false;
    }
  }

  /// Whether the model file is downloaded and available.
  bool get isAvailable => _modelAvailable;

  /// Path where the model file is expected.
  String? get modelPath => _modelPath;

  /// Load the model into memory. Requires fllama package.
  Future<void> _loadModel() async {
    // TODO: Uncomment when fllama is added as a dependency:
    //
    // import 'package:fllama/fllama.dart';
    //
    // _llamaContext = await FllamaModel.fromFile(
    //   _modelPath!,
    //   contextSize: 2048,
    //   gpuLayers: 0,  // CPU-only on most Android devices
    // );
    debugPrint('$_tag Model loader placeholder initialized');
  }

  @override
  Stream<String> stream(List<Map<String, String>> messages) async* {
    if (!_modelAvailable || _modelPath == null) {
      yield '[Offline model not available. '
          'Download "$modelFileName" from Model Manager in parent settings.]';
      return;
    }

    // Verify file still exists at runtime
    final modelFile = File(_modelPath!);
    if (!await modelFile.exists()) {
      _modelAvailable = false;
      yield '[Model file was deleted. Please re-download from Model Manager.]';
      return;
    }

    debugPrint('$_tag Generating response for ${messages.length} messages...');

    // Build the prompt in ChatML format for Qwen models
    final prompt = _buildChatMLPrompt(messages);

    // TODO: Uncomment when fllama is enabled:
    //
    // final controller = StreamController<String>();
    //
    // final request = FllamaInferenceRequest(
    //   contextId: _llamaContext!.id,
    //   input: prompt,
    //   maxTokens: 512,
    //   temperature: 0.7,
    //   topP: 0.9,
    //   onToken: (String token) {
    //     controller.add(token);
    //   },
    //   onComplete: () {
    //     controller.close();
    //   },
    //   onError: (String error) {
    //     controller.addError(Exception(error));
    //     controller.close();
    //   },
    // );
    //
    // fllamaInference(request);
    // yield* controller.stream;
    // return;

    // Placeholder until fllama is enabled
    debugPrint('$_tag fllama not yet enabled — yielding placeholder');
    yield '[Offline LLM not yet configured. '
        'Enable fllama package in pubspec.yaml to use local inference.]';
  }

  /// Build a ChatML-formatted prompt string from message list.
  /// Qwen models use the ChatML format:
  ///   <|im_start|>system\n...<|im_end|>
  ///   <|im_start|>user\n...<|im_end|>
  ///   <|im_start|>assistant\n
  String _buildChatMLPrompt(List<Map<String, String>> messages) {
    final buffer = StringBuffer();

    for (final msg in messages) {
      final role = msg['role'] ?? 'user';
      final content = msg['content'] ?? '';
      buffer.writeln('<|im_start|>$role');
      buffer.writeln(content);
      buffer.writeln('<|im_end|>');
    }

    // Start the assistant turn
    buffer.writeln('<|im_start|>assistant');
    return buffer.toString();
  }

  /// Release model resources.
  void dispose() {
    // TODO: Uncomment when fllama is enabled:
    // (_llamaContext as FllamaContext?)?.dispose();
    _llamaContext = null;
    _modelAvailable = false;
    debugPrint('$_tag Disposed');
  }
}
