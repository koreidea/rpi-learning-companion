import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';

/// Plays nursery rhyme songs from assets.
/// Songs are bundled as assets/songs/*.wav (or .mp3).
class SongPlayer {
  final AudioPlayer _player = AudioPlayer();
  bool _isPlaying = false;
  String? _currentSong;

  bool get isPlaying => _isPlaying;
  String? get currentSong => _currentSong;

  /// Play a song by name (e.g. 'twinkle_twinkle').
  /// Looks for the file in assets/songs/ with .wav or .mp3 extension.
  Future<void> play(String songName) async {
    try {
      debugPrint('[SongPlayer] Playing: $songName');
      _isPlaying = true;
      _currentSong = songName;

      // Try .wav first, then .mp3
      try {
        await _player.play(AssetSource('songs/$songName.wav'));
      } catch (_) {
        try {
          await _player.play(AssetSource('songs/$songName.mp3'));
        } catch (e) {
          debugPrint('[SongPlayer] Song not found: $songName ($e)');
          _isPlaying = false;
          _currentSong = null;
          return;
        }
      }

      // Listen for completion
      _player.onPlayerComplete.listen((_) {
        _isPlaying = false;
        _currentSong = null;
      });
    } catch (e) {
      debugPrint('[SongPlayer] Error: $e');
      _isPlaying = false;
      _currentSong = null;
    }
  }

  /// Stop the currently playing song.
  Future<void> stop() async {
    await _player.stop();
    _isPlaying = false;
    _currentSong = null;
  }

  /// Get human-readable song name.
  static String displayName(String songName) {
    return songName
        .replaceAll('_', ' ')
        .replaceAll('-', ' ')
        .split(' ')
        .map((w) => w.isNotEmpty ? '${w[0].toUpperCase()}${w.substring(1)}' : '')
        .join(' ');
  }

  void dispose() {
    _player.dispose();
  }
}
