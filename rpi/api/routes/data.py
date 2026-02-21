import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from api.middleware.auth import require_parent_auth

router = APIRouter()


class EraseConfirmation(BaseModel):
    pin: str  # Must re-enter PIN to confirm erasure


@router.delete("/erase-all")
async def erase_all_data(
    body: EraseConfirmation, request: Request, _=Depends(require_parent_auth)
):
    """DPDP Right to Erasure: delete ALL user data and reset to factory state.

    This removes:
    - All session logs
    - All configuration
    - Consent records
    - PIN
    """
    cm = request.app.state.config_manager
    data_dir = cm.data_dir

    # Verify PIN one more time
    from api.middleware.auth import verify_pin
    pin_path = data_dir / "pin.hash"
    if pin_path.exists():
        stored_hash = pin_path.read_text().strip()
        if not verify_pin(body.pin, stored_hash, cm.config.device.device_id):
            return {"error": "Invalid PIN. Erasure cancelled."}

    # Delete all data files
    sessions_dir = data_dir / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)

    # Remove config and PIN
    for f in data_dir.iterdir():
        if f.is_file():
            f.unlink()

    # Reset in-memory config
    cm.reset()

    # Reset state to setup mode
    state = request.app.state.shared_state
    from core.state import BotState
    state.set_state(BotState.SETUP)
    state.is_model_loaded = False

    return {
        "status": "erased",
        "message": "All data has been permanently deleted. Device is reset to setup mode.",
    }


@router.get("/export")
async def export_data(request: Request, _=Depends(require_parent_auth)):
    """Export all stored data (for transparency / DPDP compliance).

    Returns only metadata — no audio or conversation text is ever stored.
    """
    cm = request.app.state.config_manager
    data_dir = cm.data_dir

    from privacy.audit_log import AuditLog
    audit = AuditLog(data_dir / "sessions")
    sessions = await audit.get_all_sessions()

    return {
        "device_id": cm.config.device.device_id,
        "consent_timestamp": cm.config.privacy.consent_timestamp,
        "child_settings": cm.config.child.model_dump(),
        "total_sessions": len(sessions),
        "sessions": sessions,
        "note": "No audio recordings or conversation text is ever stored.",
    }
