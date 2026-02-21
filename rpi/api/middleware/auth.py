import hashlib
import hmac
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic(auto_error=False)


def hash_pin(pin: str, salt: str) -> str:
    """Hash a PIN with a salt using SHA-256."""
    return hashlib.pbkdf2_hmac(
        "sha256", pin.encode(), salt.encode(), iterations=100_000
    ).hex()


def verify_pin(pin: str, stored_hash: str, salt: str) -> bool:
    """Verify a PIN against a stored hash."""
    computed = hash_pin(pin, salt)
    return hmac.compare_digest(computed, stored_hash)


async def require_parent_auth(request: Request) -> None:
    """Dependency: require parent PIN authentication.

    The PIN is sent as a header: X-Parent-PIN
    """
    config_manager = request.app.state.config_manager
    config = config_manager.config

    # If no setup done yet, allow access (for initial consent flow)
    if not config.device.first_boot_complete:
        return

    pin = request.headers.get("X-Parent-PIN")
    if not pin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Parent PIN required.",
        )

    # PIN hash is stored in the config
    device_salt = config.device.device_id
    stored_hash = getattr(config, '_pin_hash', None)

    # For simplicity, we store the pin hash in a separate field
    # This gets set during the consent/setup flow
    pin_hash_path = config_manager.data_dir / "pin.hash"
    if not pin_hash_path.exists():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No PIN configured. Complete setup first.",
        )

    stored_hash = pin_hash_path.read_text().strip()
    if not verify_pin(pin, stored_hash, device_salt):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid PIN.",
        )
