import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path_provider/path_provider.dart';

import 'package:shared_preferences/shared_preferences.dart';

import '../content/content_provider.dart';

/// Service that generates and caches educational illustrations for content
/// items using the DALL-E 3 API.
///
/// Images are cached on disk so that each content item only needs to be
/// generated once. The prompt is automatically derived from the item's
/// title and body to produce kid-friendly, labeled educational diagrams.
class IllustrationService {
  final String apiKey;
  final Dio _dio = Dio();
  String? _cacheDir;

  IllustrationService({required this.apiKey});

  /// Get or generate an illustration for a content item.
  /// Returns the local file path, or null if generation fails.
  Future<String?> getIllustration(ContentItem item) async {
    final cacheDir = await _getCacheDir();
    final fileName = '${item.id}.png';
    final filePath = '$cacheDir/$fileName';

    // Return cached if exists
    if (await File(filePath).exists()) {
      return filePath;
    }

    // Generate via DALL-E 3
    try {
      final prompt = _buildPrompt(item);
      final imageUrl = await _generateImage(prompt);
      if (imageUrl == null) return null;

      // Download and cache
      final response = await _dio.get<List<int>>(
        imageUrl,
        options: Options(responseType: ResponseType.bytes),
      );
      final file = File(filePath);
      await file.writeAsBytes(response.data!);
      return filePath;
    } catch (e) {
      debugPrint('[IllustrationService] Error generating image: $e');
      return null;
    }
  }

  /// Check if an illustration is already cached.
  Future<bool> isCached(ContentItem item) async {
    final cacheDir = await _getCacheDir();
    return File('$cacheDir/${item.id}.png').exists();
  }

  /// Get the cached file path for an item, or null if not cached.
  Future<String?> getCachedPath(ContentItem item) async {
    final cacheDir = await _getCacheDir();
    final filePath = '$cacheDir/${item.id}.png';
    if (await File(filePath).exists()) {
      return filePath;
    }
    return null;
  }

  String _buildPrompt(ContentItem item) {
    final bodySnippet =
        item.body.length > 400 ? item.body.substring(0, 400) : item.body;
    return 'Create a visually self-explanatory educational infographic diagram for children aged 5-12. '
        'Topic: "${item.title}". '
        'Key concepts to show visually: $bodySnippet '
        '\n\nSTYLE REQUIREMENTS (follow strictly): '
        '- Like the USGS Water Cycle diagram or DK Encyclopedia illustrations '
        '- Show the PROCESS or CONCEPT visually with a clear landscape/scene '
        '- Add LABELED ARROWS showing each step or part (e.g., "Evaporation", "Condensation", "Precipitation") '
        '- Use NUMBERED STEPS if it is a process (Step 1, Step 2, Step 3...) '
        '- Include SHORT TEXT ANNOTATIONS next to each part explaining what happens '
        '- Use BRIGHT, SATURATED COLORS with a light/white background '
        '- Add cute illustrated characters or mascots where appropriate (like water droplets with faces) '
        '- Include a TITLE at the top of the diagram '
        '- Add a small FAST FACTS or KEY TAKEAWAY box in one corner '
        '- Cross-section or cutaway views for topics about internals (like Earth layers, body parts) '
        '- Use ICONS and SYMBOLS to reinforce concepts '
        '- Make it so a child can UNDERSTAND THE TOPIC JUST BY LOOKING AT THE IMAGE without reading any separate text '
        '- NO photorealistic style - use colorful illustrated infographic style '
        '- Landscape orientation preferred';
  }

  Future<String?> _generateImage(String prompt) async {
    try {
      final response = await _dio.post(
        'https://api.openai.com/v1/images/generations',
        options: Options(
          headers: {
            'Authorization': 'Bearer $apiKey',
            'Content-Type': 'application/json',
          },
        ),
        data: {
          'model': 'dall-e-3',
          'prompt': prompt,
          'n': 1,
          'size': '1024x1024',
          'quality': 'standard',
          'style': 'natural',
        },
      );

      final data = response.data as Map<String, dynamic>;
      final images = data['data'] as List<dynamic>;
      if (images.isNotEmpty) {
        return images[0]['url'] as String;
      }
      return null;
    } on DioException catch (e) {
      debugPrint(
        '[IllustrationService] DALL-E API error: '
        '${e.response?.statusCode} ${e.response?.data}',
      );
      return null;
    }
  }

  Future<String> _getCacheDir() async {
    if (_cacheDir != null) return _cacheDir!;
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.path}/illustrations');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    _cacheDir = dir.path;
    return _cacheDir!;
  }

  /// Get total cache size in bytes.
  Future<int> getCacheSize() async {
    final cacheDir = await _getCacheDir();
    final dir = Directory(cacheDir);
    if (!await dir.exists()) return 0;
    int total = 0;
    await for (final entity in dir.list()) {
      if (entity is File) {
        total += await entity.length();
      }
    }
    return total;
  }

  /// Get the number of cached illustrations.
  Future<int> getCachedCount() async {
    final cacheDir = await _getCacheDir();
    final dir = Directory(cacheDir);
    if (!await dir.exists()) return 0;
    int count = 0;
    await for (final entity in dir.list()) {
      if (entity is File && entity.path.endsWith('.png')) {
        count++;
      }
    }
    return count;
  }

  /// Clear all cached illustrations.
  Future<void> clearCache() async {
    final cacheDir = await _getCacheDir();
    final dir = Directory(cacheDir);
    if (await dir.exists()) {
      await dir.delete(recursive: true);
      await dir.create();
    }
    _cacheDir = null;
  }
}

/// Riverpod provider for the illustration service.
///
/// Returns null if no OpenAI API key is configured, since DALL-E requires it.
/// Reads directly from SharedPreferences to avoid depending on orchestrator
/// initialization timing.
final illustrationServiceProvider =
    FutureProvider<IllustrationService?>((ref) async {
  final prefs = await SharedPreferences.getInstance();
  final apiKey = prefs.getString('api_key_openai') ?? '';
  if (apiKey.isEmpty) return null;
  return IllustrationService(apiKey: apiKey);
});
