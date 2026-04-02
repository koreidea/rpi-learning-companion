import 'dart:async';
import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import 'llm_provider.dart';

/// Google Gemini provider with streaming support.
class GeminiProvider implements LlmProvider {
  final Dio _dio = Dio();
  String apiKey;
  final String model;

  GeminiProvider({
    required this.apiKey,
    this.model = 'gemini-2.0-flash',
  });

  @override
  String get name => 'gemini';

  @override
  Stream<String> stream(List<Map<String, String>> messages) async* {
    if (apiKey.isEmpty) {
      debugPrint('[Gemini] No API key');
      return;
    }

    try {
      debugPrint('[Gemini] Streaming from $model...');

      // Convert OpenAI-style messages to Gemini format
      final contents = <Map<String, dynamic>>[];
      String? systemInstruction;

      for (final msg in messages) {
        if (msg['role'] == 'system') {
          systemInstruction = msg['content'];
          continue;
        }
        contents.add({
          'role': msg['role'] == 'assistant' ? 'model' : 'user',
          'parts': [
            {'text': msg['content']}
          ],
        });
      }

      final body = <String, dynamic>{
        'contents': contents,
        'generationConfig': {
          'maxOutputTokens': 300,
          'temperature': 0.7,
        },
      };

      if (systemInstruction != null) {
        body['systemInstruction'] = {
          'parts': [
            {'text': systemInstruction}
          ],
        };
      }

      final response = await _dio.post(
        'https://generativelanguage.googleapis.com/v1beta/models/$model:streamGenerateContent?alt=sse&key=$apiKey',
        data: body,
        options: Options(
          headers: {'Content-Type': 'application/json'},
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
            final candidates = json['candidates'] as List?;
            if (candidates != null && candidates.isNotEmpty) {
              final parts = candidates[0]['content']?['parts'] as List?;
              if (parts != null && parts.isNotEmpty) {
                final text = parts[0]['text']?.toString();
                if (text != null && text.isNotEmpty) {
                  yield text;
                }
              }
            }
          } catch (_) {
            // Skip malformed JSON
          }
        }
      }
    } catch (e) {
      debugPrint('[Gemini] Stream error: $e');
    }
  }
}
