"""Tests for privacy and DPDP compliance components."""
import pytest
import json
from pathlib import Path

from privacy.encryption import DataEncryption
from privacy.audit_log import AuditLog
from core.config import ConfigManager


class TestEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        enc = DataEncryption(pin="1234", device_id="test-device-123")
        plaintext = "Hello, this is secret data!"
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext
        assert enc.decrypt(ciphertext) == plaintext

    def test_different_pins_different_ciphertext(self):
        enc1 = DataEncryption(pin="1234", device_id="device-1")
        enc2 = DataEncryption(pin="5678", device_id="device-1")
        ct1 = enc1.encrypt("test")
        ct2 = enc2.encrypt("test")
        assert ct1 != ct2

    def test_wrong_pin_fails(self):
        enc1 = DataEncryption(pin="1234", device_id="device-1")
        enc2 = DataEncryption(pin="5678", device_id="device-1")
        ciphertext = enc1.encrypt("secret")
        with pytest.raises(Exception):
            enc2.decrypt(ciphertext)

    def test_bytes_roundtrip(self):
        enc = DataEncryption(pin="1234", device_id="test")
        data = b"binary data here"
        encrypted = enc.encrypt_bytes(data)
        assert enc.decrypt_bytes(encrypted) == data


class TestAuditLog:
    @pytest.fixture
    def audit(self, tmp_path):
        return AuditLog(tmp_path / "sessions")

    @pytest.mark.asyncio
    async def test_log_session(self, audit):
        await audit.log_session(duration_seconds=5.0, topic_category="animals")
        sessions = await audit.get_recent_sessions()
        assert len(sessions) == 1
        assert sessions[0]["topic_category"] == "animals"
        assert sessions[0]["duration_seconds"] == 5.0

    @pytest.mark.asyncio
    async def test_no_conversation_content(self, audit):
        await audit.log_session(duration_seconds=3.0, topic_category="math")
        sessions = await audit.get_all_sessions()
        for session in sessions:
            # Ensure no conversation text is stored
            assert "transcript" not in session
            assert "text" not in session
            assert "audio" not in session
            assert "conversation" not in session

    @pytest.mark.asyncio
    async def test_stats(self, audit):
        await audit.log_session(duration_seconds=5.0, topic_category="animals")
        await audit.log_session(duration_seconds=3.0, topic_category="math")
        await audit.log_session(duration_seconds=4.0, topic_category="animals")
        stats = await audit.get_stats()
        assert stats["total_sessions"] == 3
        assert stats["topics"]["animals"] == 2
        assert stats["topics"]["math"] == 1

    @pytest.mark.asyncio
    async def test_erase_all(self, audit):
        await audit.log_session(duration_seconds=5.0)
        await audit.erase_all()
        sessions = await audit.get_all_sessions()
        assert len(sessions) == 0


class TestConfigManager:
    def test_default_config(self, tmp_path):
        cm = ConfigManager(tmp_path)
        assert cm.config.mode == "offline"
        assert not cm.has_consent
        assert not cm.is_setup_complete

    def test_save_load(self, tmp_path):
        cm = ConfigManager(tmp_path)
        cm.update(mode="online", provider="claude")
        cm.save()

        cm2 = ConfigManager(tmp_path)
        assert cm2.config.mode == "online"
        assert cm2.config.provider == "claude"

    def test_update_nested(self, tmp_path):
        cm = ConfigManager(tmp_path)
        cm.update_nested("child", age_min=5, age_max=8)
        assert cm.config.child.age_min == 5
        assert cm.config.child.age_max == 8

    def test_reset(self, tmp_path):
        cm = ConfigManager(tmp_path)
        cm.update(mode="online")
        cm.save()
        cm.reset()
        assert cm.config.mode == "offline"
