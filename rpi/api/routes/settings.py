from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from typing import Optional

from api.middleware.auth import require_parent_auth

router = APIRouter()


class ModeUpdate(BaseModel):
    mode: str  # "online" or "offline"


class ProviderUpdate(BaseModel):
    provider: str  # "openai", "gemini", or "claude"


class APIKeyUpdate(BaseModel):
    openai: Optional[str] = None
    gemini: Optional[str] = None
    claude: Optional[str] = None


class ChildSettingsUpdate(BaseModel):
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    language: Optional[str] = None


class HardwareUpdate(BaseModel):
    wake_word: Optional[str] = None


@router.get("/")
async def get_settings(request: Request, _=Depends(require_parent_auth)):
    """Get all current settings."""
    config = request.app.state.config_manager.config
    # Return settings without sensitive API keys (masked)
    data = config.model_dump()
    for key in data.get("api_keys", {}):
        val = data["api_keys"][key]
        if val:
            data["api_keys"][key] = val[:4] + "****" + val[-4:] if len(val) > 8 else "****"
    return data


@router.put("/mode")
async def update_mode(
    body: ModeUpdate, request: Request, _=Depends(require_parent_auth)
):
    """Switch between online and offline mode."""
    if body.mode not in ("online", "offline"):
        return {"error": "Mode must be 'online' or 'offline'"}

    cm = request.app.state.config_manager
    cm.update(mode=body.mode)

    state = request.app.state.shared_state
    from core.state import LLMMode
    state.llm_mode = LLMMode(body.mode)

    return {"mode": body.mode, "status": "updated"}


@router.put("/provider")
async def update_provider(
    body: ProviderUpdate, request: Request, _=Depends(require_parent_auth)
):
    """Select cloud LLM provider."""
    if body.provider not in ("openai", "gemini", "claude"):
        return {"error": "Provider must be 'openai', 'gemini', or 'claude'"}

    cm = request.app.state.config_manager
    cm.update(provider=body.provider)
    request.app.state.shared_state.active_provider = body.provider

    return {"provider": body.provider, "status": "updated"}


@router.put("/api-keys")
async def update_api_keys(
    body: APIKeyUpdate, request: Request, _=Depends(require_parent_auth)
):
    """Update API keys for cloud providers."""
    cm = request.app.state.config_manager
    updates = {}
    if body.openai is not None:
        updates["openai"] = body.openai
    if body.gemini is not None:
        updates["gemini"] = body.gemini
    if body.claude is not None:
        updates["claude"] = body.claude

    if updates:
        cm.update_nested("api_keys", **updates)

    return {"status": "updated"}


@router.put("/child")
async def update_child_settings(
    body: ChildSettingsUpdate, request: Request, _=Depends(require_parent_auth)
):
    """Update child age and language settings."""
    cm = request.app.state.config_manager
    updates = body.model_dump(exclude_none=True)
    if updates:
        cm.update_nested("child", **updates)
    return {"status": "updated", "child": cm.config.child.model_dump()}


@router.put("/hardware")
async def update_hardware_settings(
    body: HardwareUpdate, request: Request, _=Depends(require_parent_auth)
):
    """Update hardware settings like wake word."""
    cm = request.app.state.config_manager
    updates = body.model_dump(exclude_none=True)
    if updates:
        cm.update_nested("hardware", **updates)
    return {"status": "updated", "hardware": cm.config.hardware.model_dump()}
