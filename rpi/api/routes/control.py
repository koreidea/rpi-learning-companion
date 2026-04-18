from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile
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


# ── Sketch Tracing from Image ───────────────────────────────────────────


@router.post("/sketch-from-image")
async def sketch_from_image(
    request: Request,
    file: UploadFile = File(...),
    _=Depends(require_parent_auth),
):
    """Upload an image and convert it to clean line art for tracing.

    Uses GrabCut foreground isolation + XDoG (Extended Difference of
    Gaussians) to produce a clean, artistic line drawing projected
    onto the table for the child to trace.
    """
    import asyncio

    from loguru import logger

    try:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            return {"error": "Image too large (max 10MB)"}

        # Process in thread to avoid blocking
        loop = asyncio.get_event_loop()
        pil_image = await loop.run_in_executor(None, _image_to_line_art, contents)

        if pil_image is None:
            return {"error": "Could not convert image to line art"}

        # Send directly to projector
        state = request.app.state.shared_state
        state.trace_image = {
            "name": file.filename or "uploaded image",
            "image": pil_image,
        }
        state.wake_event.set()

        logger.info("Line art generated from '{}'", file.filename)
        return {"status": "ok"}

    except Exception as e:
        logger.error("sketch-from-image error: {}", e)
        return {"error": str(e)}


def _image_to_line_art(image_bytes: bytes):
    """Convert image to clean edge-only line art for tracing.

    Pipeline:
      1. Decode → resize
      2. GrabCut to isolate foreground (removes background)
      3. Heavy smoothing to destroy shadows/textures/gradients
      4. Color quantization (posterize) — flattens remaining gradients
      5. Canny edge detection on the flat, posterized image
      6. Keep only significant edges, remove noise
      7. Return as PIL Image (white lines on black background)
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Decode
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Resize
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # ── 1. Isolate foreground ──
    fg_mask = _extract_foreground(img, h, w)

    # ── 2. Aggressively smooth to kill shadows and textures ──
    # Median blur destroys texture while keeping hard edges
    smooth = cv2.medianBlur(img, 7)
    # Multiple bilateral passes — smooths gradients, preserves only strong edges
    for _ in range(5):
        smooth = cv2.bilateralFilter(smooth, 9, 100, 100)

    # ── 3. Color quantization (posterize to ~6 colors) ──
    # This flattens any remaining shadows/gradients into flat regions
    # so edges are only at real object boundaries
    data = smooth.reshape((-1, 3)).astype(np.float32)
    K = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(img.shape)

    # ── 4. Edge detection on the clean, flat image ──
    gray_q = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)

    # Canny with higher thresholds — only strong structural edges
    edges = cv2.Canny(gray_q, 80, 200)

    # ── 5. Mask to foreground only ──
    edges = cv2.bitwise_and(edges, edges, mask=fg_mask)

    # ── 6. Clean up ──
    # Close small gaps in edge lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

    # Thicken lines for projection visibility
    kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.dilate(edges, kernel_thick, iterations=1)

    # Remove small noise blobs
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_peri = (h + w) * 0.03  # minimum 3% of image perimeter
    for c in contours:
        if cv2.arcLength(c, True) < min_peri:
            cv2.drawContours(edges, [c], -1, 0, thickness=cv2.FILLED)

    # Convert to PIL (white lines on black)
    pil_img = Image.fromarray(edges).convert("RGB")
    return pil_img


def _extract_foreground(img, h, w):
    """Extract foreground mask using GrabCut + fallback."""
    import cv2
    import numpy as np

    mask = np.zeros((h, w), np.uint8)

    try:
        # GrabCut with a border-based rectangle
        border = max(8, int(min(h, w) * 0.05))
        rect = (border, border, w - 2 * border, h - 2 * border)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

        # Check if enough foreground was found
        fg_ratio = np.sum(fg_mask > 0) / (h * w)
        if fg_ratio < 0.05:
            raise ValueError("GrabCut foreground too small")

        # Dilate to include edges at boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        return fg_mask

    except Exception:
        # Fallback: Otsu threshold to find main subject
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Use largest contour as main object
            largest = max(contours, key=cv2.contourArea)
            fg_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(fg_mask, [largest], -1, 255, thickness=cv2.FILLED)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
            return fg_mask

        # Last resort: entire image
        return np.ones((h, w), dtype=np.uint8) * 255
