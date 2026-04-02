import 'dart:async';
import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import 'llm_provider.dart';

/// OpenAI GPT provider with streaming support.
class OpenAIProvider implements LlmProvider {
  final Dio _dio = Dio();
  String apiKey;
  final String model;

  OpenAIProvider({
    required this.apiKey,
    this.model = 'gpt-4o-mini',
  });

  @override
  String get name => 'openai';

  @override
  Stream<String> stream(List<Map<String, String>> messages) async* {
    if (apiKey.isEmpty) {
      debugPrint('[OpenAI] No API key');
      return;
    }

    try {
      debugPrint('[OpenAI] Streaming from $model...');
      final response = await _dio.post(
        'https://api.openai.com/v1/chat/completions',
        data: {
          'model': model,
          'messages': messages,
          'stream': true,
          'max_tokens': 300,
          'temperature': 0.7,
        },
        options: Options(
          headers: {
            'Authorization': 'Bearer $apiKey',
            'Content-Type': 'application/json',
          },
          responseType: ResponseType.stream,
          sendTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(seconds: 60),
        ),
      );

      final stream = response.data.stream as Stream<List<int>>;
      String buffer = '';

      await for (final chunk in stream) {
        buffer += utf8.decode(chunk);

        // SSE format: each event is "data: {...}\n\n"
        while (buffer.contains('\n')) {
          final idx = buffer.indexOf('\n');
          final line = buffer.substring(0, idx).trim();
          buffer = buffer.substring(idx + 1);

          if (line.isEmpty) continue;
          if (line == 'data: [DONE]') return;
          if (!line.startsWith('data: ')) continue;

          try {
            final json = jsonDecode(line.substring(6));
            final delta = json['choices']?[0]?['delta']?['content'];
            if (delta != null && delta is String && delta.isNotEmpty) {
              yield delta;
            }
          } catch (_) {
            // Skip malformed JSON
          }
        }
      }
    } catch (e) {
      debugPrint('[OpenAI] Stream error: $e');
    }
  }
}
