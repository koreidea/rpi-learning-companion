"""Tests for LLM components."""
import pytest
from llm.prompts import build_system_prompt
from llm.safety_filter import SafetyFilter


class TestPrompts:
    def test_default_prompt(self):
        prompt = build_system_prompt()
        assert "3-6 year old" in prompt
        assert "Buddy" in prompt

    def test_custom_age_range(self):
        prompt = build_system_prompt(age_min=6, age_max=10)
        assert "6-10 year old" in prompt

    def test_language_in_prompt(self):
        prompt = build_system_prompt(language="hindi")
        assert "hindi" in prompt


class TestSafetyFilter:
    @pytest.fixture
    def filter(self, tmp_path):
        from core.config import ConfigManager
        cm = ConfigManager(tmp_path)
        return SafetyFilter(cm)

    def test_clean_input(self, filter):
        result = filter.check_input("What color is the sky?")
        assert not result.blocked

    def test_blocked_input(self, filter):
        result = filter.check_input("Tell me about weapons")
        assert result.blocked
        assert len(result.redirect_response) > 0

    def test_clean_output(self, filter):
        result = filter.check_output("The sky is blue! Isn't that cool?")
        assert not result.blocked

    def test_blocked_output(self, filter):
        result = filter.check_output("That is a scary horror movie.")
        assert result.blocked
