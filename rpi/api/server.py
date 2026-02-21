from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import ConfigManager
from core.state import SharedState


def create_app(config_manager: ConfigManager, state: SharedState) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="RPi Learning Companion", version="1.0.0")

    # CORS for local network access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store references for route handlers
    app.state.config_manager = config_manager
    app.state.shared_state = state

    # Import and register routes
    from api.routes.settings import router as settings_router
    from api.routes.consent import router as consent_router
    from api.routes.dashboard import router as dashboard_router
    from api.routes.control import router as control_router
    from api.routes.data import router as data_router

    app.include_router(settings_router, prefix="/api/settings", tags=["settings"])
    app.include_router(consent_router, prefix="/api/consent", tags=["consent"])
    app.include_router(dashboard_router, prefix="/api/dashboard", tags=["dashboard"])
    app.include_router(control_router, prefix="/api/control", tags=["control"])
    app.include_router(data_router, prefix="/api/data", tags=["data"])

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "bot_state": state.bot_state.value,
            "mode": config_manager.config.mode,
            "consent": config_manager.has_consent,
            "model_loaded": state.is_model_loaded,
        }

    # Serve React static files (built app)
    # MUST be mounted AFTER all API routes — mount at "/" catches everything
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
