import base64
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger


class DataEncryption:
    """AES-256 encryption for data at rest.

    Key is derived from parent PIN + device-specific salt using PBKDF2.
    """

    def __init__(self, pin: str, device_id: str):
        self._fernet = self._derive_key(pin, device_id)

    @staticmethod
    def _derive_key(pin: str, salt: str) -> Fernet:
        """Derive an encryption key from PIN and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(pin.encode()))
        return Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string and return base64-encoded ciphertext."""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt base64-encoded ciphertext and return plaintext string."""
        return self._fernet.decrypt(ciphertext.encode()).decode()

    def encrypt_bytes(self, data: bytes) -> bytes:
        """Encrypt raw bytes."""
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        """Decrypt raw bytes."""
        return self._fernet.decrypt(data)
