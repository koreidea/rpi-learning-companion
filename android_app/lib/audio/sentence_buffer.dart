/// Port of rpi/audio/sentence_buffer.py.
/// Collects streaming LLM tokens and yields complete sentences for TTS.
class SentenceBuffer {
  String _buffer = '';
  final int _minChars;

  /// Sentence-ending pattern: . ! ? followed by optional whitespace.
  static final _sentenceEnd = RegExp(r'[.!?]+\s*$');

  /// Split on sentence boundary: punctuation followed by whitespace.
  static final _sentenceSplit = RegExp(r'(?<=[.!?])\s+');

  SentenceBuffer({int minChars = 5}) : _minChars = minChars;

  /// Clear the buffer for a new interaction.
  void reset() {
    _buffer = '';
  }

  /// Feed a token from the LLM stream.
  /// Returns a complete sentence if a boundary was found, else null.
  String? feed(String token) {
    _buffer += token;

    // Don't check until we have enough text
    if (_buffer.length < _minChars) return null;

    // Look for sentence boundaries
    final parts = _buffer.split(_sentenceSplit);

    if (parts.length > 1) {
      // We have a complete sentence + remaining text
      final sentence = parts[0].trim();
      _buffer = parts.sublist(1).join(' ');
      if (sentence.isNotEmpty) return sentence;
    }

    // Also check if buffer ends with sentence punctuation
    if (_sentenceEnd.hasMatch(_buffer) && _buffer.length >= _minChars) {
      final sentence = _buffer.trim();
      _buffer = '';
      if (sentence.isNotEmpty) return sentence;
    }

    return null;
  }

  /// Flush remaining text when the LLM stream ends.
  String? flush() {
    final remaining = _buffer.trim();
    _buffer = '';
    return remaining.isNotEmpty ? remaining : null;
  }
}
