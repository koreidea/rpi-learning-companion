from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from api.middleware.auth import require_parent_auth

router = APIRouter()

SONGS_DIR = Path(__file__).resolve().parent.parent.parent / "audio" / "songs"


class MicToggle(BaseModel):
    enabled: bool


class CameraToggle(BaseModel):
    enabled: bool


class VolumeSet(BaseModel):
    level: int  # 0-100


@router.put("/mic")
async def toggle_mic(
    body: MicToggle, request: Request, _=Depends(require_parent_auth)
):
    """Enable or disable the microphone.

    When disabled, the interaction loop pauses (no wake word listening).
    When re-enabled, the bot resumes listening.
    """
    state = request.app.state.shared_state
    state.mic_enabled = body.enabled

    cm = request.app.state.config_manager
    cm.update_nested("hardware", mic_enabled=body.enabled)

    # If disabling mid-response, also interrupt
    if not body.enabled:
        state.interrupt_event.set()

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


@router.put("/volume")
async def set_volume(
    body: VolumeSet, request: Request, _=Depends(require_parent_auth)
):
    """Set the speaker volume (0-100)."""
    state = request.app.state.shared_state
    audio_player = getattr(request.app.state, "audio_player", None)

    level = max(0, min(100, body.level))
    state.volume = level

    if audio_player:
        audio_player.set_volume(level)

    return {"volume": level}


@router.get("/volume")
async def get_volume(request: Request):
    """Get the current speaker volume."""
    state = request.app.state.shared_state
    return {"volume": state.volume}


class RemoteText(BaseModel):
    text: str


@router.post("/send-text")
async def send_text(
    body: RemoteText, request: Request, _=Depends(require_parent_auth)
):
    """Send text to the bot as if the child spoke it.

    Used when the phone is far from the bot — speak into the phone,
    the Web Speech API transcribes it, and this endpoint injects
    the text into the bot's processing pipeline.
    """
    state = request.app.state.shared_state
    text = body.text.strip()
    if not text:
        return {"error": "Empty text"}

    # Set remote text and trigger wake event to break out of wake word wait
    state.remote_text = text
    state.wake_event.set()
    return {"status": "sent", "text": text}


@router.post("/wake")
async def trigger_wake(request: Request, _=Depends(require_parent_auth)):
    """Trigger wake word remotely (just wake, no text)."""
    state = request.app.state.shared_state
    state.wake_event.set()
    return {"status": "woken"}


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


# ── Song Library ──────────────────────────────────────────────────────────


@router.get("/songs")
async def list_songs(request: Request):
    """List all available songs in the library."""
    if not SONGS_DIR.exists():
        return {"songs": []}
    songs = []
    for f in sorted(SONGS_DIR.glob("*.wav")):
        name = f.stem.replace("_", " ").replace("-", " ").title()
        size_kb = f.stat().st_size // 1024
        songs.append({"filename": f.name, "name": name, "size_kb": size_kb})
    return {"songs": songs}


class SongPlay(BaseModel):
    filename: str


@router.post("/songs/play")
async def play_song(
    body: SongPlay, request: Request, _=Depends(require_parent_auth)
):
    """Play a song from the library."""
    audio_player = getattr(request.app.state, "audio_player", None)
    if not audio_player:
        return {"error": "Audio player not available"}

    song_path = SONGS_DIR / body.filename
    if not song_path.exists():
        return {"error": "Song not found"}

    await audio_player.play_file(song_path, timeout=300)
    return {"status": "played", "song": body.filename}


@router.post("/songs/stop")
async def stop_song(request: Request, _=Depends(require_parent_auth)):
    """Stop currently playing song."""
    audio_player = getattr(request.app.state, "audio_player", None)
    if audio_player:
        await audio_player.stop()
    return {"status": "stopped"}
