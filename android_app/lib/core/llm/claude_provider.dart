import 'dart:async';
import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import 'llm_provider.dart';

/// Anthropic Claude provider with streaming support.
class ClaudeProvider implements LlmProvider {
  final Dio _dio = Dio();
  String apiKey;
  final String model;

  ClaudeProvider({
    required this.apiKey,
    this.model = 'claude-sonnet-4-20250514',
  });

  @override
  String get name => 'claude';

  @override
  Stream<String> stream(List<Map<String, String>> messages) async* {
    if (apiKey.isEmpty) {
      debugPrint('[Claude] No API key');
      return;
    }

    try {
      debugPrint('[Claude] Streaming from $model...');

      // Extract system message
      String? systemPrompt;
      final chatMessages = <Map<String, String>>[];
      for (final msg in messages) {
        if (msg['role'] == 'system') {
          systemPrompt = msg['content'];
        } else {
          chatMessages.add(msg);
        }
      }

      final body = <String, dynamic>{
        'model': model,
        'max_tokens': 300,
        'stream': true,
        'messages': chatMessages,
      };
      if (systemPrompt != null) {
        body['system'] = systemPrompt;
      }

      final response = await _dio.post(
        'https://api.anthropic.com/v1/messages',
        data: body,
        options: Options(
          headers: {
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
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

        while (buffer.contains('\n')) {
          final idx = buffer.indexOf('\n');
          final line = buffer.substring(0, idx).trim();
          buffer = buffer.substring(idx + 1);

          if (line.isEmpty) continue;
          if (!line.startsWith('data: ')) continue;

          try {
            final json = jsonDecode(line.substring(6));
            final type = json['type'];
            if (type == 'content_block_delta') {
              final text = json['delta']?['text'];
              if (text != null && text is String && text.isNotEmpty) {
                yield text;
              }
            }
          } catch (_) {
            // Skip malformed JSON
          }
        }
      }
    } catch (e) {
      debugPrint('[Claude] Stream error: $e');
    }
  }
}
