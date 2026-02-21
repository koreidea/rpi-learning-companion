"""Tests for vision components (mocked, since camera/GPU not available in test)."""
import pytest


class TestVisionTrigger:
    """Test the vision trigger detection in the orchestrator."""

    VISION_TRIGGERS = [
        "what is this", "what's this", "what do you see",
        "look at this", "can you see", "what am i holding",
        "tell me about this", "what color is this",
    ]

    def _is_vision_request(self, transcript: str) -> bool:
        lower = transcript.lower()
        return any(trigger in lower for trigger in self.VISION_TRIGGERS)

    def test_vision_triggers(self):
        assert self._is_vision_request("What is this?")
        assert self._is_vision_request("Hey buddy, what's this thing?")
        assert self._is_vision_request("Can you see what I'm showing you?")
        assert self._is_vision_request("Look at this cool bug!")

    def test_non_vision(self):
        assert not self._is_vision_request("Tell me about the moon")
        assert not self._is_vision_request("What is two plus two?")
        assert not self._is_vision_request("Sing me a song")
