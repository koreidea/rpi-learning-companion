from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from api.middleware.auth import require_parent_auth

router = APIRouter()


class MicToggle(BaseModel):
    enabled: bool


class CameraToggle(BaseModel):
    enabled: bool


@router.put("/mic")
async def toggle_mic(
    body: MicToggle, request: Request, _=Depends(require_parent_auth)
):
    """Enable or disable the microphone."""
    state = request.app.state.shared_state
    state.mic_enabled = body.enabled

    cm = request.app.state.config_manager
    cm.update_nested("hardware", mic_enabled=body.enabled)

    return {"mic_enabled": body.enabled}


@router.put("/camera")
async def toggle_camera(
    body: CameraToggle, request: Request, _=Depends(require_parent_auth)
):
    """Enable or disable the camera."""
    state = request.app.state.shared_state
    state.camera_enabled = body.enabled

    cm = request.app.state.config_manager
    cm.update_nested("hardware", camera_enabled=body.enabled)

    return {"camera_enabled": body.enabled}


@router.post("/stop-response")
async def stop_response(request: Request, _=Depends(require_parent_auth)):
    """Stop the current response and go back to listening mode."""
    state = request.app.state.shared_state
    state.interrupt_event.set()
    return {"status": "stopping"}


@router.post("/restart")
async def restart_bot(request: Request, _=Depends(require_parent_auth)):
    """Restart the bot (reload models)."""
    state = request.app.state.shared_state
    from core.state import BotState

    state.set_state(BotState.LOADING)
    # The main loop will detect this and reload
    return {"status": "restarting"}
