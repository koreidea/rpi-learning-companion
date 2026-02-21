from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel

from api.middleware.auth import hash_pin

router = APIRouter()


class SetupRequest(BaseModel):
    pin: str  # 4-6 digit parent PIN
    consent_given: bool
    child_age_min: int = 3
    child_age_max: int = 6


class ConsentStatus(BaseModel):
    consent_given: bool
    consent_timestamp: str | None
    setup_complete: bool


@router.get("/status")
async def get_consent_status(request: Request):
    """Get current consent status. No auth needed (used during setup)."""
    config = request.app.state.config_manager.config
    return ConsentStatus(
        consent_given=config.privacy.consent_given,
        consent_timestamp=config.privacy.consent_timestamp,
        setup_complete=config.device.first_boot_complete,
    )


@router.post("/setup")
async def complete_setup(body: SetupRequest, request: Request):
    """First-time setup: set PIN and give consent.

    This is the DPDP consent flow:
    - Parent explicitly checks consent checkbox
    - Sets a PIN for future access
    - Configures child age range
    """
    cm = request.app.state.config_manager

    if cm.config.device.first_boot_complete:
        return {"error": "Setup already complete. Use settings to modify."}

    if not body.consent_given:
        return {"error": "Parental consent is required to proceed."}

    if len(body.pin) < 4 or len(body.pin) > 6:
        return {"error": "PIN must be 4-6 digits."}

    if not body.pin.isdigit():
        return {"error": "PIN must contain only digits."}

    # Hash and store the PIN
    device_salt = cm.config.device.device_id
    pin_hash = hash_pin(body.pin, device_salt)
    pin_path = cm.data_dir / "pin.hash"
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    pin_path.write_text(pin_hash)

    # Update config with consent
    timestamp = datetime.now(timezone.utc).isoformat()
    cm.update_nested("privacy", consent_given=True, consent_timestamp=timestamp)
    cm.update_nested("child", age_min=body.child_age_min, age_max=body.child_age_max)
    cm.update_nested("device", first_boot_complete=True)

    return {
        "status": "setup_complete",
        "consent_timestamp": timestamp,
        "message": "Consent recorded. The learning companion is now active.",
    }


@router.post("/revoke")
async def revoke_consent(request: Request):
    """Revoke consent — stops all interaction, enters setup mode."""
    cm = request.app.state.config_manager

    cm.update_nested("privacy", consent_given=False, consent_timestamp=None)
    cm.update_nested("device", first_boot_complete=False)

    # Remove PIN
    pin_path = cm.data_dir / "pin.hash"
    if pin_path.exists():
        pin_path.unlink()

    state = request.app.state.shared_state
    from core.state import BotState
    state.set_state(BotState.SETUP)

    return {"status": "consent_revoked", "message": "All interaction stopped. Setup required."}
