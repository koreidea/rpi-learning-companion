from fastapi import APIRouter, Depends, Request

from api.middleware.auth import require_parent_auth

router = APIRouter()


@router.get("/")
async def get_dashboard(request: Request, _=Depends(require_parent_auth)):
    """Get dashboard data: usage stats, session history, device status."""
    state = request.app.state.shared_state
    config = request.app.state.config_manager.config

    # Read session logs from the audit log
    from privacy.audit_log import AuditLog
    from core.main import DATA_DIR

    audit = AuditLog(DATA_DIR / "sessions")
    sessions = await audit.get_recent_sessions(limit=50)
    stats = await audit.get_stats()

    return {
        "device": {
            "state": state.bot_state.value,
            "mode": config.mode,
            "provider": config.provider,
            "model_loaded": state.is_model_loaded,
            "mic_enabled": state.mic_enabled,
            "camera_enabled": state.camera_enabled,
        },
        "child": config.child.model_dump(),
        "stats": stats,
        "recent_sessions": sessions,
    }


@router.get("/status")
async def get_device_status(request: Request, _=Depends(require_parent_auth)):
    """Quick device status check."""
    state = request.app.state.shared_state
    return {
        "state": state.bot_state.value,
        "model_loaded": state.is_model_loaded,
        "last_error": state.last_error,
    }
