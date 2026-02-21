"""Tests for audio pipeline components."""
import pytest
from audio.sentence_buffer import SentenceBuffer


class TestSentenceBuffer:
    def test_empty_feed(self):
        buf = SentenceBuffer()
        assert buf.feed("") is None

    def test_single_sentence(self):
        buf = SentenceBuffer(min_chars=5)
        assert buf.feed("Hello") is None
        assert buf.feed(" world.") is None  # No space after period
        result = buf.feed(" Next")
        assert result == "Hello world."

    def test_multiple_sentences(self):
        buf = SentenceBuffer(min_chars=5)
        buf.feed("First sentence. ")
        result = buf.feed("Second")
        # "First sentence." should be yielded
        assert result is not None

    def test_question_mark(self):
        buf = SentenceBuffer(min_chars=5)
        buf.feed("Is this cool? ")
        result = buf.feed("Yes")
        assert result is not None
        assert "?" in result

    def test_exclamation(self):
        buf = SentenceBuffer(min_chars=5)
        buf.feed("Wow that is great! ")
        result = buf.feed("Indeed")
        assert result is not None

    def test_flush(self):
        buf = SentenceBuffer(min_chars=5)
        buf.feed("Remaining text")
        result = buf.flush()
        assert result == "Remaining text"

    def test_flush_empty(self):
        buf = SentenceBuffer()
        assert buf.flush() is None

    def test_reset(self):
        buf = SentenceBuffer()
        buf.feed("Some text")
        buf.reset()
        assert buf.flush() is None
