import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

/// Sends camera frames to GPT-4o-mini vision API and returns child-friendly descriptions.
///
/// Uses the OpenAI Chat Completions API with image content to describe
/// what the camera sees, in a fun and simple way for 3-6 year olds.
class VisionDescriber {
  final Dio _dio;

  static const String _systemPrompt =
      'You are Buddy, a friendly learning companion for a 3-6 year old child. '
      'Describe what you see in simple, fun words. Keep it to 2-3 sentences. '
      'Be enthusiastic! If you see a person, describe what they\'re doing. '
      'Always respond in the same language as the user\'s last message.';

  static const String _apiUrl = 'https://api.openai.com/v1/chat/completions';
  static const String _model = 'gpt-4o-mini';
  static const int _maxTokens = 200;
  static const Duration _timeout = Duration(seconds: 30);

  VisionDescriber({Dio? dio}) : _dio = dio ?? Dio();

  /// Describe what the camera sees.
  ///
  /// [imageBytes] — JPEG image data from the camera.
  /// [apiKey] — OpenAI API key.
  /// [userPrompt] — Optional user prompt to give context (e.g., the transcript
  ///   that triggered the vision command). Defaults to a generic prompt.
  /// [language] — Language hint for the response ('en', 'hi', 'te').
  ///
  /// Returns a text description, or an error message string on failure.
  Future<String> describe({
    required Uint8List imageBytes,
    required String apiKey,
    String? userPrompt,
    String language = 'en',
  }) async {
    if (apiKey.isEmpty) {
      return 'I need an OpenAI API key to use my eyes. Please set one up in settings.';
    }

    try {
      // Encode image to base64
      final base64Image = base64Encode(imageBytes);
      final dataUrl = 'data:image/jpeg;base64,$base64Image';

      // Build language-aware user prompt
      final prompt = _buildUserPrompt(userPrompt, language);

      final response = await _dio.post(
        _apiUrl,
        options: Options(
          headers: {
            'Authorization': 'Bearer $apiKey',
            'Content-Type': 'application/json',
          },
          sendTimeout: _timeout,
          receiveTimeout: _timeout,
        ),
        data: {
          'model': _model,
          'max_tokens': _maxTokens,
          'messages': [
            {
              'role': 'system',
              'content': _systemPrompt,
            },
            {
              'role': 'user',
              'content': [
                {
                  'type': 'text',
                  'text': prompt,
                },
                {
                  'type': 'image_url',
                  'image_url': {
                    'url': dataUrl,
                    'detail': 'low',
                  },
                },
              ],
            },
          ],
        },
      );

      if (response.statusCode == 200) {
        final data = response.data;
        final content = data['choices']?[0]?['message']?['content'] as String?;
        if (content != null && content.isNotEmpty) {
          debugPrint('[VisionDescriber] Got description: ${content.substring(0, content.length.clamp(0, 80))}');
          return content.trim();
        }
        debugPrint('[VisionDescriber] Empty content in response');
        return 'Hmm, I looked but could not figure out what I am seeing. Let me try again!';
      }

      debugPrint('[VisionDescriber] API error: ${response.statusCode} ${response.statusMessage}');
      return 'Oops, my eyes are not working right now. Let us try again in a moment!';
    } on DioException catch (e) {
      debugPrint('[VisionDescriber] DioException: ${e.type} ${e.message}');
      if (e.type == DioExceptionType.connectionTimeout ||
          e.type == DioExceptionType.receiveTimeout) {
        return 'It is taking too long to see. Let us try again!';
      }
      if (e.response?.statusCode == 401) {
        return 'My API key does not seem to be working. Can you check it in settings?';
      }
      if (e.response?.statusCode == 429) {
        return 'I have been looking too much! Let us wait a little bit and try again.';
      }
      return 'Oops, something went wrong with my eyes. Let us try again!';
    } catch (e) {
      debugPrint('[VisionDescriber] Unexpected error: $e');
      return 'Oops, something went wrong. Let us try again!';
    }
  }

  /// Build a language-appropriate user prompt.
  String _buildUserPrompt(String? userPrompt, String language) {
    if (userPrompt != null && userPrompt.isNotEmpty) {
      return 'The child said: "$userPrompt". Now describe what you see in the image.';
    }

    switch (language) {
      case 'hi':
        return 'Describe what you see in the image. Respond in Hindi.';
      case 'te':
        return 'Describe what you see in the image. Respond in Telugu.';
      default:
        return 'Describe what you see in the image.';
    }
  }
}
