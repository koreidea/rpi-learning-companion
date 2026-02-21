import re
from typing import Optional

from loguru import logger

# Sentence-ending patterns
SENTENCE_END = re.compile(r'[.!?]+\s*$')
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


class SentenceBuffer:
    """Collects streaming LLM tokens and yields complete sentences.

    This enables the streaming TTS pipeline:
    LLM streams tokens → SentenceBuffer groups into sentences → TTS synthesizes per sentence

    For a 3-6 year old audience, sentences are short (5-15 words), so
    buffering to sentence boundaries adds minimal latency while enabling
    natural-sounding TTS output.
    """

    def __init__(self, min_chars: int = 5):
        self._buffer = ""
        self._min_chars = min_chars  # Minimum chars before we check for sentence end

    def reset(self):
        """Clear the buffer for a new interaction."""
        self._buffer = ""

    def feed(self, token: str) -> Optional[str]:
        """Feed a token from the LLM stream.

        Args:
            token: A single token or token chunk from the LLM.

        Returns:
            A complete sentence if a sentence boundary was found, else None.
        """
        self._buffer += token

        # Don't check for sentences until we have enough text
        if len(self._buffer) < self._min_chars:
            return None

        # Look for sentence boundaries
        parts = SENTENCE_SPLIT.split(self._buffer, maxsplit=1)

        if len(parts) > 1:
            # We have a complete sentence + remaining text
            sentence = parts[0].strip()
            self._buffer = parts[1] if len(parts) > 1 else ""

            if sentence:
                logger.debug("Sentence ready: '{}'", sentence[:60])
                return sentence

        # Also check if buffer ends with sentence punctuation
        if SENTENCE_END.search(self._buffer) and len(self._buffer) >= self._min_chars:
            sentence = self._buffer.strip()
            self._buffer = ""
            if sentence:
                logger.debug("Sentence ready: '{}'", sentence[:60])
                return sentence

        return None

    def flush(self) -> Optional[str]:
        """Flush any remaining text in the buffer.

        Called when the LLM stream ends to get the last partial sentence.

        Returns:
            Remaining text, or None if buffer is empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            logger.debug("Flushing remaining: '{}'", remaining[:60])
            return remaining
        return None
