"""Integration tests for the full pipeline (mocked components)."""
import pytest
from core.config import ConfigManager
from core.state import SharedState, BotState, LLMMode


class TestSharedState:
    def test_initial_state(self):
        state = SharedState()
        assert state.bot_state == BotState.SETUP
        assert state.is_running
        assert not state.is_ready

    def test_state_transitions(self):
        state = SharedState()
        state.set_state(BotState.READY)
        assert state.is_ready

        state.set_state(BotState.LISTENING)
        assert not state.is_ready
        assert state.bot_state == BotState.LISTENING

    def test_stop(self):
        state = SharedState()
        assert state.is_running
        state.request_stop()
        assert not state.is_running


class TestConfigIntegration:
    def test_online_offline_toggle(self, tmp_path):
        cm = ConfigManager(tmp_path)
        assert not cm.is_online

        cm.update(mode="online")
        assert cm.is_online

        cm.update(mode="offline")
        assert not cm.is_online

    def test_consent_flow(self, tmp_path):
        cm = ConfigManager(tmp_path)
        assert not cm.has_consent

        cm.update_nested("privacy", consent_given=True, consent_timestamp="2024-01-01T00:00:00Z")
        assert cm.has_consent

        cm.update_nested("privacy", consent_given=False)
        assert not cm.has_consent
