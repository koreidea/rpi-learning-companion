"""ILI9488 TFT display driver — EMO/Cozmo-inspired animated face.

Drives a 3.5" 480x320 ILI9488 SPI display with:
- Rounded-rectangle eyes with glow effect (EMO/Cozmo style)
- Smooth parameter interpolation between states
- Sine-eased blink, gaze drift, micro-saccades, breathing pulse
- Per-state expressions: happy ^_^, wide listening, narrowed thinking
- Sparkle particles, processing dots, loading spinner
- Idle look-around: smooth left → center → right → center

Hardware wiring (SPI0):
  CS   → BCM8  (CE0)
  DC   → BCM23
  RST  → BCM25
  MOSI → BCM10
  SCK  → BCM11
  LED  → 3.3V  (always-on, no GPIO control)
"""

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import List

from loguru import logger

# ─── Display dimensions (landscape orientation) ──────────────────────────────
WIDTH = 480
HEIGHT = 320
CX = WIDTH // 2     # 240
CY = HEIGHT // 2    # 160

# ─── Layout: eyes fill the screen ────────────────────────────────────────────
# Eye width=180 (half=90). Gap=36px. Centers at 240 ± 108 = 132, 348.
# Left edge with glow: 132-90-8=34.  Right edge: 348+90+8=446.  Fits 480.
# Vertical: height=210 (half=105). Center=150. Top glow: 150-105-8=37. Bot: 150+105+8=263.
EYE_LCX = 132
EYE_RCX = 348
EYE_CY = 150

# ─── Colors ───────────────────────────────────────────────────────────────────
COLOR_BG = (0, 0, 0)

STATE_COLORS = {
    "ready": (255, 120, 0),       # Dark Orange
    "listening": (30, 144, 255),  # Blue
    "processing": (255, 200, 50), # Yellow
    "speaking": (180, 100, 255),  # Purple
    "dancing": (255, 50, 200),    # Hot Pink
    "error": (255, 60, 60),       # Red
    "setup": (80, 80, 80),        # Gray
    "loading": (255, 200, 50),    # Yellow
}


# ─── Frame rate tiers ─────────────────────────────────────────────────────────
# SPI@24MHz can do ~3MB/s; each frame is ~150KB → max ~20 FPS theoretical.
# Stay under 12 FPS to avoid SPI contention with LED backlight.
FPS_IDLE = 1.0          # Static idle (no animation active)
FPS_LOOK = 10.0         # During idle look-around — needs smooth motion
FPS_ACTIVE = 10.0       # Active states (listening, processing, speaking)
FPS_TRANSITION = 12.0   # State transitions
FPS_BLINK = 12.0        # Blink smoothness

# ─── Interpolation ────────────────────────────────────────────────────────────
LERP_ALPHA = 0.25       # Per-frame blend toward target (high FPS compensates)
COLOR_LERP_ALPHA = 0.15 # Slower color blend
PUPIL_LERP_ALPHA = 0.35 # Faster pupil tracking for snappy gaze


# ═════════════════════════════════════════════════════════════════════════════
#  Data Model
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EyeParams:
    """Animatable parameters for a single eye."""
    x: float = 0.0
    y: float = 0.0
    width: float = 90.0
    height: float = 75.0
    corner_radius: float = 22.0
    slope_left: float = 0.0
    slope_right: float = 0.0
    pupil_x: float = 0.0
    pupil_y: float = 0.0
    pupil_scale: float = 0.5
    color_r: float = 50.0
    color_g: float = 205.0
    color_b: float = 50.0
    glow_alpha: float = 0.3


@dataclass
class FaceParams:
    """Complete animatable face state."""
    left_eye: EyeParams = field(default_factory=EyeParams)
    right_eye: EyeParams = field(default_factory=EyeParams)
    mouth_open: float = 0.0
    mouth_width: float = 40.0
    mouth_style: str = "none"


@dataclass
class AnimState:
    """Scheduler bookkeeping for procedural animations."""
    blink_timer: float = 4.0
    blink_phase: float = -1.0
    blink_mult: float = 1.0         # Current blink multiplier (applied at render time)
    # Gaze drift (small random micro-movements)
    gaze_target_x: float = 0.0
    gaze_target_y: float = 0.0
    gaze_timer: float = 3.0
    # Micro-saccade
    saccade_ox: float = 0.0
    saccade_oy: float = 0.0
    saccade_timer: float = 2.0
    # Breathing
    breath_phase: float = 0.0
    # Processing dots
    dot_phase: int = 0
    dot_timer: float = 0.0
    # Speaking mouth
    mouth_phase: float = 0.0
    # Loading spinner
    spinner_idx: int = 0
    spinner_timer: float = 0.0
    # Sleep Z
    z_phase: float = 0.0
    # State transition
    transition_progress: float = 1.0
    # ── Listening pulse ──
    listen_phase: float = 0.0        # Continuous phase for pulse rings
    # ── Idle look-around ──
    look_phase: float = 0.0          # Time progress through the sequence
    look_active: bool = False        # Whether a look-around is happening
    look_cooldown: float = 5.0       # Seconds until next look-around
    _look_seq: list = field(default_factory=list)   # Current waypoint sequence
    _look_from: tuple = (0.0, 0.0)                  # Previous waypoint (for interpolation)
    # ── Tickle animation ──
    tickle_timer: float = 0.0          # Remaining seconds of tickle animation
    tickle_phase: float = 0.0          # Continuous phase for wiggle oscillation
    # ── Idle game carousel ──
    idle_timer: float = 0.0            # Seconds spent continuously in READY
    game_active: bool = False          # Whether an idle game is displayed
    game_timer: float = 0.0           # Time in current game (for cycling)
    game_index: int = 0               # Index into GAME_REGISTRY


@dataclass
class Heart:
    """Floating heart particle for tickle animation."""
    x: float = 0.0
    y: float = 0.0
    life: float = 0.0
    max_life: float = 1.5
    size: float = 6.0
    drift_x: float = 0.0   # horizontal drift speed


@dataclass
class Sparkle:
    x: float = 0.0
    y: float = 0.0
    life: float = 0.0
    max_life: float = 0.8
    size: float = 2.0


# ═════════════════════════════════════════════════════════════════════════════
#  Pure Utility Functions
# ═════════════════════════════════════════════════════════════════════════════

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_eye(cur: EyeParams, tgt: EyeParams, alpha: float, color_alpha: float,
             pupil_alpha: float = 0.35) -> EyeParams:
    return EyeParams(
        x=lerp(cur.x, tgt.x, alpha),
        y=lerp(cur.y, tgt.y, alpha),
        width=lerp(cur.width, tgt.width, alpha),
        height=lerp(cur.height, tgt.height, alpha),
        corner_radius=lerp(cur.corner_radius, tgt.corner_radius, alpha),
        slope_left=lerp(cur.slope_left, tgt.slope_left, alpha),
        slope_right=lerp(cur.slope_right, tgt.slope_right, alpha),
        pupil_x=lerp(cur.pupil_x, tgt.pupil_x, pupil_alpha),
        pupil_y=lerp(cur.pupil_y, tgt.pupil_y, pupil_alpha),
        pupil_scale=lerp(cur.pupil_scale, tgt.pupil_scale, alpha),
        color_r=lerp(cur.color_r, tgt.color_r, color_alpha),
        color_g=lerp(cur.color_g, tgt.color_g, color_alpha),
        color_b=lerp(cur.color_b, tgt.color_b, color_alpha),
        glow_alpha=lerp(cur.glow_alpha, tgt.glow_alpha, alpha),
    )


def lerp_face(cur: FaceParams, tgt: FaceParams, alpha: float, color_alpha: float,
              pupil_alpha: float = 0.35) -> FaceParams:
    return FaceParams(
        left_eye=lerp_eye(cur.left_eye, tgt.left_eye, alpha, color_alpha, pupil_alpha),
        right_eye=lerp_eye(cur.right_eye, tgt.right_eye, alpha, color_alpha, pupil_alpha),
        mouth_open=lerp(cur.mouth_open, tgt.mouth_open, alpha),
        mouth_width=lerp(cur.mouth_width, tgt.mouth_width, alpha),
        mouth_style=tgt.mouth_style,
    )


def params_converged(cur: FaceParams, tgt: FaceParams, thresh: float = 0.5) -> bool:
    for attr in ("width", "height", "pupil_x", "pupil_y", "pupil_scale",
                 "slope_left", "slope_right", "corner_radius"):
        if abs(getattr(cur.left_eye, attr) - getattr(tgt.left_eye, attr)) > thresh:
            return False
        if abs(getattr(cur.right_eye, attr) - getattr(tgt.right_eye, attr)) > thresh:
            return False
    for attr in ("color_r", "color_g", "color_b"):
        if abs(getattr(cur.left_eye, attr) - getattr(tgt.left_eye, attr)) > 2.0:
            return False
    return True


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ═════════════════════════════════════════════════════════════════════════════
#  Animation Updaters
# ═════════════════════════════════════════════════════════════════════════════

def update_blink(anim: AnimState, dt: float) -> float:
    """Smooth blink using sine-eased close/open.

    Returns a multiplier 0..1 for eye height.
    Phase progression: 0 → 1 over ~0.35s (dt * 2.8).
    The cosine curve gives natural acceleration: fast close, pause, fast open.
    Blink interval: 2-5 seconds (fairly frequent for liveliness).
    """
    if anim.blink_phase >= 0:
        anim.blink_phase += dt * 2.8  # slower → more visible blink
        if anim.blink_phase >= 1.0:
            anim.blink_phase = -1.0
            anim.blink_timer = random.uniform(2.0, 5.0)
            return 1.0
        return abs(math.cos(anim.blink_phase * math.pi))
    else:
        anim.blink_timer -= dt
        if anim.blink_timer <= 0:
            anim.blink_phase = 0.0
        return 1.0


def update_breathing(anim: AnimState, dt: float) -> float:
    anim.breath_phase += dt * 1.5
    if anim.breath_phase > 2 * math.pi:
        anim.breath_phase -= 2 * math.pi
    return 1.0 + 0.03 * math.sin(anim.breath_phase)


def update_gaze(anim: AnimState, dt: float):
    """Small random micro-drift (used only when NOT doing a look-around)."""
    anim.gaze_timer -= dt
    if anim.gaze_timer <= 0:
        anim.gaze_target_x = random.uniform(-0.15, 0.15)
        anim.gaze_target_y = random.uniform(-0.1, 0.1)
        anim.gaze_timer = random.uniform(2.0, 5.0)


def update_saccade(anim: AnimState, dt: float):
    anim.saccade_timer -= dt
    if anim.saccade_timer <= 0:
        anim.saccade_ox = random.uniform(-0.05, 0.05)
        anim.saccade_oy = random.uniform(-0.03, 0.03)
        anim.saccade_timer = random.uniform(1.0, 3.0)
    else:
        anim.saccade_ox *= 0.5
        anim.saccade_oy *= 0.5


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out using cosine: 0→0, 0.5→0.5, 1→1."""
    return (1.0 - math.cos(t * math.pi)) / 2.0


# Pre-defined look sequences — each is a list of (target_x, target_y) positions.
# The bot randomly picks one sequence per look-around cycle.
_LOOK_SEQUENCES = [
    # left → center → right → center
    [(-0.7, 0.0), (0.0, 0.0), (0.7, 0.0), (0.0, 0.0)],
    # top-left → top-right → center
    [(-0.6, -0.5), (0.6, -0.5), (0.0, 0.0)],
    # right → top → left → center
    [(0.7, 0.0), (0.0, -0.5), (-0.7, 0.0), (0.0, 0.0)],
    # top-left → center → top-right → center
    [(-0.5, -0.4), (0.0, 0.0), (0.5, -0.4), (0.0, 0.0)],
    # left → top → right → center
    [(-0.7, 0.0), (0.0, -0.5), (0.7, 0.0), (0.0, 0.0)],
    # top → left → top → right → center
    [(0.0, -0.5), (-0.6, -0.2), (0.0, -0.5), (0.6, -0.2), (0.0, 0.0)],
]


def update_idle_look(anim: AnimState, dt: float) -> tuple:
    """Smooth idle look-around using pre-defined direction sequences.

    Returns (pupil_x, pupil_y) — the actual target gaze position.
    Transitions between waypoints use ease-in-out for natural movement.
    Each waypoint holds for a brief pause, then smoothly moves to next.
    """
    if not anim.look_active:
        anim.look_cooldown -= dt
        if anim.look_cooldown <= 0:
            anim.look_active = True
            anim.look_phase = 0.0
            # Pick a random sequence
            anim._look_seq = random.choice(_LOOK_SEQUENCES)
            anim._look_from = (0.0, 0.0)  # start from center
        return (0.0, 0.0)

    seq = getattr(anim, '_look_seq', [(0.0, 0.0)])
    n = len(seq)

    # Time per waypoint: 0.6s move + 0.4s hold = 1.0s each
    move_time = 0.6
    hold_time = 0.4
    step_time = move_time + hold_time
    total_time = n * step_time

    anim.look_phase += dt

    if anim.look_phase >= total_time:
        anim.look_active = False
        anim.look_cooldown = random.uniform(3.0, 8.0)
        anim.look_phase = 0.0
        return (0.0, 0.0)

    # Which waypoint segment are we in?
    step_idx = min(int(anim.look_phase / step_time), n - 1)
    step_local = anim.look_phase - step_idx * step_time

    from_pos = getattr(anim, '_look_from', (0.0, 0.0))
    to_pos = seq[step_idx]

    if step_local < move_time:
        # Moving phase — ease in-out
        t = _ease_in_out(step_local / move_time)
        ox = from_pos[0] + (to_pos[0] - from_pos[0]) * t
        oy = from_pos[1] + (to_pos[1] - from_pos[1]) * t
    else:
        # Holding phase
        ox, oy = to_pos
        # Update from_pos for next segment
        anim._look_from = to_pos

    return (ox, oy)


def update_listen_pulse(anim: AnimState, dt: float) -> float:
    """Advance the listening pulse phase. Returns phase 0..inf (continuous)."""
    anim.listen_phase += dt * 1.8  # ~1.8 rings per second
    return anim.listen_phase


def update_dots(anim: AnimState, dt: float) -> int:
    anim.dot_timer += dt
    if anim.dot_timer >= 0.5:
        anim.dot_timer = 0.0
        anim.dot_phase = (anim.dot_phase + 1) % 4
    return anim.dot_phase


def update_mouth(anim: AnimState, dt: float) -> float:
    anim.mouth_phase += dt * 8.0
    return 0.3 + 0.7 * abs(math.sin(anim.mouth_phase))


def update_spinner(anim: AnimState, dt: float) -> int:
    anim.spinner_timer += dt
    if anim.spinner_timer >= 0.3:
        anim.spinner_timer = 0.0
        anim.spinner_idx = (anim.spinner_idx + 1) % 8
    return anim.spinner_idx


def update_sparkles(sparkles: List[Sparkle], dt: float, cx: float, cy: float,
                    spawn_chance: float = 0.12) -> List[Sparkle]:
    alive = []
    for s in sparkles:
        s.life -= dt
        if s.life > 0:
            alive.append(s)
    if random.random() < spawn_chance and len(alive) < 8:
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(75, 115)
        alive.append(Sparkle(
            x=cx + dist * math.cos(angle),
            y=cy + dist * math.sin(angle),
            life=0.8,
            max_life=0.8,
            size=random.uniform(2.0, 4.0),
        ))
    return alive


# ── Tickle Animation ────────────────────────────────────────────────────────

TICKLE_DURATION = 2.5       # seconds of tickle animation
TICKLE_COLOR = (255, 130, 180)  # Warm pink

def update_hearts(hearts: List[Heart], dt: float, cx: float, cy: float,
                  spawn: bool = False) -> List[Heart]:
    """Update floating heart particles — drift upward and fade."""
    alive = []
    for h in hearts:
        h.life -= dt
        h.y -= 45 * dt           # float upward
        h.x += h.drift_x * dt    # gentle side drift
        if h.life > 0:
            alive.append(h)
    if spawn and random.random() < 0.35 and len(alive) < 10:
        alive.append(Heart(
            x=cx + random.uniform(-90, 90),
            y=cy + random.uniform(-20, 40),
            life=1.5,
            max_life=1.5,
            size=random.uniform(4.0, 8.0),
            drift_x=random.uniform(-15, 15),
        ))
    return alive


def render_heart(draw, x: int, y: int, size: int, color: tuple):
    """Draw a tiny heart shape at (x, y)."""
    s = max(2, size)
    hs = s // 2
    # Two circles for the top bumps
    draw.ellipse([x - s, y - hs, x, y + hs], fill=color)
    draw.ellipse([x, y - hs, x + s, y + hs], fill=color)
    # Triangle for the bottom point
    draw.polygon([
        (x - s, y),
        (x + s, y),
        (x, y + s + hs),
    ], fill=color)


def render_hearts(draw, hearts: List[Heart]):
    """Render all floating hearts with fade."""
    for h in hearts:
        alpha = max(0, h.life / h.max_life)
        c = (int(TICKLE_COLOR[0] * alpha),
             int(TICKLE_COLOR[1] * alpha),
             int(TICKLE_COLOR[2] * alpha))
        render_heart(draw, int(h.x), int(h.y), int(h.size * alpha), c)


# ═════════════════════════════════════════════════════════════════════════════
#  Renderers
# ═════════════════════════════════════════════════════════════════════════════

def render_eye(draw, eye: EyeParams, is_error: bool = False):
    cr = int(eye.color_r)
    cg = int(eye.color_g)
    cb = int(eye.color_b)
    color = (cr, cg, cb)
    cx, cy = int(eye.x), int(eye.y)
    hw = int(eye.width / 2)
    hh = int(eye.height / 2)
    r = int(_clamp(eye.corner_radius, 0, min(hw, hh)))

    if is_error:
        arm = max(24, int(min(hw, hh) * 0.7))
        draw.line([cx - arm, cy - arm, cx + arm, cy + arm], fill=color, width=8)
        draw.line([cx - arm, cy + arm, cx + arm, cy - arm], fill=color, width=8)
        return

    if hh < 3:
        draw.rounded_rectangle(
            [cx - hw, cy - 2, cx + hw, cy + 2],
            radius=2, fill=color,
        )
        return

    # Glow
    ga = _clamp(eye.glow_alpha, 0, 1)
    glow = (int(cr * ga), int(cg * ga), int(cb * ga))
    gx = 8
    draw.rounded_rectangle(
        [cx - hw - gx, cy - hh - gx, cx + hw + gx, cy + hh + gx],
        radius=r + 6, fill=glow,
    )

    # Main eye
    draw.rounded_rectangle(
        [cx - hw, cy - hh, cx + hw, cy + hh],
        radius=r, fill=color,
    )

    # Eyelid slope mask
    sl = int(eye.slope_left)
    sr = int(eye.slope_right)
    if sl != 0 or sr != 0:
        top = cy - hh
        pts = [
            (cx - hw - 2, top - 2),
            (cx + hw + 2, top - 2),
            (cx + hw + 2, top + sr),
            (cx - hw - 2, top + sl),
        ]
        draw.polygon(pts, fill=COLOR_BG)

    # Pupil
    if eye.pupil_scale > 0.05:
        max_tx = hw * 0.45
        max_ty = hh * 0.35
        px = cx + int(eye.pupil_x * max_tx)
        py = cy + int(eye.pupil_y * max_ty)
        pw = int(hw * eye.pupil_scale)
        ph = int(hh * eye.pupil_scale)
        pr = max(4, int(r * 0.4))
        pcol = (
            min(255, 200 + int(cr * 0.2)),
            min(255, 200 + int(cg * 0.2)),
            min(255, 200 + int(cb * 0.2)),
        )
        draw.rounded_rectangle(
            [px - pw, py - ph, px + pw, py + ph],
            radius=pr, fill=pcol,
        )


def render_mouth(draw, face: FaceParams):
    cx, cy = CX, EYE_CY + 78
    cr = int(face.left_eye.color_r)
    cg = int(face.left_eye.color_g)
    cb = int(face.left_eye.color_b)
    color = (cr, cg, cb)

    if face.mouth_style == "open":
        mw = int(face.mouth_width / 2)
        mh = int(20 * face.mouth_open)
        if mh > 1:
            draw.rounded_rectangle(
                [cx - mw, cy - mh, cx + mw, cy + mh],
                radius=min(mw, mh), fill=color,
            )
    elif face.mouth_style == "smile":
        draw.arc(
            [cx - 30, cy - 12, cx + 30, cy + 18],
            start=0, end=180, fill=color, width=4,
        )
    elif face.mouth_style == "line":
        draw.line([cx - 25, cy, cx + 25, cy], fill=color, width=4)


def render_dots(draw, count: int, color: tuple):
    cx, cy = CX, EYE_CY + 78
    spacing, dr = 22, 7
    for i in range(count):
        x = cx + (i - 1) * spacing
        draw.ellipse([x - dr, cy - dr, x + dr, cy + dr], fill=color)


def render_spinner(draw, idx: int, color: tuple):
    cx, cy = CX, EYE_CY + 78
    n = 8
    for i in range(n):
        angle = (2 * math.pi * i / n) - math.pi / 2
        x = cx + int(24 * math.cos(angle))
        y = cy + int(24 * math.sin(angle))
        if i == idx:
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color)
        else:
            dim = (color[0] // 4, color[1] // 4, color[2] // 4)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=dim)


def render_sleep_z(draw, phase: float, font):
    chars = ["z", "z", "Z"]
    base_x = 250
    for i, ch in enumerate(chars):
        p = ((phase * 0.15 + i * 0.3) % 1.0)
        y = int(EYE_CY - p * 80)
        x = base_x + i * 22
        alpha = max(0.0, 1.0 - p)
        if alpha > 0.1:
            c = (int(100 * alpha), int(100 * alpha), int(100 * alpha))
            draw.text((x, y), ch, font=font, fill=c)


def render_listen_pulse(draw, phase: float, color: tuple):
    """Draw expanding concentric pulse rings between the eyes.

    3 rings at staggered phases, expanding from center outward.
    Each ring fades as it grows, creating a sonar/ripple effect.
    Rings are drawn as arcs (top half only) to stay between/above the eyes.
    """
    cx = CX  # Center between both eyes
    cy = EYE_CY
    n_rings = 3
    max_radius = 70  # Max expansion before ring fades out

    for i in range(n_rings):
        # Stagger each ring by 0.33 of a cycle
        ring_phase = (phase + i * 0.33) % 1.0
        radius = int(15 + ring_phase * max_radius)
        # Fade: bright at start, gone by end
        alpha = max(0.0, 1.0 - ring_phase)
        alpha = alpha * alpha  # Quadratic fade for nicer falloff
        if alpha < 0.05:
            continue
        rc = (int(color[0] * alpha * 0.7),
              int(color[1] * alpha * 0.7),
              int(color[2] * alpha * 0.7))
        # Draw full ellipse ring (thin outline)
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(bbox, outline=rc, width=2)


def render_sparkles(draw, sparkles: List[Sparkle], color: tuple):
    for s in sparkles:
        alpha = s.life / s.max_life
        c = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        r = int(s.size * alpha)
        if r >= 1:
            ix, iy = int(s.x), int(s.y)
            draw.ellipse([ix - r, iy - r, ix + r, iy + r], fill=c)


# ═════════════════════════════════════════════════════════════════════════════
#  State → Target Parameters
# ═════════════════════════════════════════════════════════════════════════════

def _make_eye(cx, cy, color, **kw) -> EyeParams:
    e = EyeParams(x=cx, y=cy, color_r=color[0], color_g=color[1], color_b=color[2])
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def compute_target(state: str) -> FaceParams:
    """Map bot state to target FaceParams.

    480x320 screen.  Eyes fill most of the area.
    Left eye center=132, right=348, vertical=150.
    Default eye: 180w × 210h.
    Left spans  42..222, right 258..438.  Gap=36px.
    With glow+12: 30..234 / 246..450.  Screen margins ~30px.
    Height 210 at cy=150: spans 45..255. With glow: 33..267. Bottom 53px for mouth.
    """
    color = STATE_COLORS.get(state, (80, 80, 80))
    lcx = float(EYE_LCX)
    rcx = float(EYE_RCX)
    cy = float(EYE_CY)

    if state == "ready":
        return FaceParams(
            left_eye=_make_eye(lcx, cy, color,
                               width=180, height=210, corner_radius=52, pupil_scale=0.40),
            right_eye=_make_eye(rcx, cy, color,
                                width=180, height=210, corner_radius=52, pupil_scale=0.40),
            mouth_style="none",
        )

    elif state == "listening":
        # Taller + dilated pupils + looking slightly up = attentive
        return FaceParams(
            left_eye=_make_eye(lcx, cy, color,
                               width=180, height=240, corner_radius=57,
                               pupil_scale=0.55, pupil_y=-0.15, glow_alpha=0.45),
            right_eye=_make_eye(rcx, cy, color,
                                width=180, height=240, corner_radius=57,
                                pupil_scale=0.55, pupil_y=-0.15, glow_alpha=0.45),
            mouth_style="none",
        )

    elif state == "processing":
        # Thinking — eyes look up-right (like a person recalling/thinking)
        return FaceParams(
            left_eye=_make_eye(lcx, cy, color,
                               width=180, height=150, corner_radius=42,
                               pupil_scale=0.40, pupil_x=0.5, pupil_y=-0.5,
                               slope_left=12, slope_right=-6),
            right_eye=_make_eye(rcx, cy, color,
                                width=180, height=150, corner_radius=42,
                                pupil_scale=0.40, pupil_x=0.5, pupil_y=-0.5,
                                slope_left=-6, slope_right=12),
            mouth_style="none",
        )

    elif state == "speaking":
        # Happy cheerful — soft upward-curved eyes with visible pupils
        return FaceParams(
            left_eye=_make_eye(lcx, cy, color,
                               width=180, height=150, corner_radius=68,
                               pupil_scale=0.35, pupil_y=-0.1,
                               slope_left=-38, slope_right=-12, glow_alpha=0.4),
            right_eye=_make_eye(rcx, cy, color,
                                width=180, height=150, corner_radius=68,
                                pupil_scale=0.35, pupil_y=-0.1,
                                slope_left=-12, slope_right=-38, glow_alpha=0.4),
            mouth_style="open", mouth_open=0.5, mouth_width=82,
        )

    elif state == "dancing":
        # Excited bouncy eyes — wide, dilated, happy
        return FaceParams(
            left_eye=_make_eye(lcx, cy, color,
                               width=200, height=200, corner_radius=65,
                               pupil_scale=0.50, glow_alpha=0.5,
                               slope_left=-30, slope_right=-10),
            right_eye=_make_eye(rcx, cy, color,
                                width=200, height=200, corner_radius=65,
                                pupil_scale=0.50, glow_alpha=0.5,
                                slope_left=-10, slope_right=-30),
            mouth_style="open", mouth_open=0.7, mouth_width=90,
        )

    elif state == "error":
        return FaceParams(
            left_eye=_make_eye(lcx, cy, color, width=150, height=150),
            right_eye=_make_eye(rcx, cy, color, width=150, height=150),
            mouth_style="line",
        )

    elif state == "loading":
        # Half-open sleepy
        return FaceParams(
            left_eye=_make_eye(lcx, cy, color,
                               width=180, height=98, corner_radius=42, pupil_scale=0.30),
            right_eye=_make_eye(rcx, cy, color,
                                width=180, height=98, corner_radius=42, pupil_scale=0.30),
            mouth_style="none",
        )

    else:  # setup, unknown
        return FaceParams(
            left_eye=_make_eye(lcx, cy, color,
                               width=180, height=210, corner_radius=52, pupil_scale=0.40),
            right_eye=_make_eye(rcx, cy, color,
                                width=180, height=210, corner_radius=52, pupil_scale=0.40),
            mouth_style="none",
        )


# ═════════════════════════════════════════════════════════════════════════════
#  TFTDisplay Class
# ═════════════════════════════════════════════════════════════════════════════

class TFTDisplay:

    # Shared SPI bus lock — touch controller acquires this during reads
    spi_lock = threading.Lock()

    def __init__(self):
        self._display = None
        self._draw = None
        self._image = None
        self._font_large = None
        self._font_small = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self._current_state = "loading"
        self._volume = 80
        self._bot_enabled = True
        self._frame = 0

        self._cur_params = FaceParams()
        self._tgt_params = FaceParams()
        self._anim = AnimState()
        self._sparkles: List[Sparkle] = []
        self._hearts: List[Heart] = []
        self._games: list = []                # Idle game instances (populated in _init_hardware)
        self._current_game = None             # Active idle game
        self._last_touch_seen: str | None = None  # track touch event changes

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self, state_ref):
        self._state_ref = state_ref
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._clear()

    def set_brightness(self, on: bool):
        pass

    # ── Hardware Init ──────────────────────────────────────────────────────

    def _init_hardware(self):
        try:
            import spidev
            import gpiod
            from gpiod.line import Direction, Value
            from PIL import Image, ImageDraw, ImageFont

            # GPIO setup via gpiod (Pi 5 uses gpiochip4 / RP1)
            self._dc_pin = 23
            self._rst_pin = 25
            self._gpio_request = gpiod.request_lines(
                "/dev/gpiochip4",
                consumer="ili9488",
                config={
                    self._dc_pin: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE),
                    self._rst_pin: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE),
                },
            )

            # SPI setup
            self._spi = spidev.SpiDev()
            self._spi.open(0, 0)  # SPI0, CE0
            self._spi.max_speed_hz = 48000000
            self._spi.mode = 0

            # Hardware reset
            self._gpio_set(self._rst_pin, True)
            time.sleep(0.01)
            self._gpio_set(self._rst_pin, False)
            time.sleep(0.01)
            self._gpio_set(self._rst_pin, True)
            time.sleep(0.12)

            # ILI9488 init sequence
            self._ili9488_init()

            self._display = True  # Flag indicating display is active
            self._image = Image.new("RGB", (WIDTH, HEIGHT), COLOR_BG)
            self._draw = ImageDraw.Draw(self._image)

            try:
                self._font_large = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                self._font_small = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except OSError:
                self._font_large = ImageFont.load_default()
                self._font_small = ImageFont.load_default()

            logger.info("ILI9488 display initialized ({}x{}, SPI@48MHz, EMO mode)", WIDTH, HEIGHT)

            # Instantiate idle screensaver games
            try:
                from display.games import GAME_REGISTRY
                self._games = [GameClass() for GameClass in GAME_REGISTRY]
                logger.info("Loaded {} idle games: {}", len(self._games),
                            ", ".join(g.name for g in self._games))
            except Exception as ge:
                logger.warning("Failed to load idle games: {}", ge)
                self._games = []

            return True

        except Exception as e:
            import traceback
            logger.warning("Display init failed: {}. Running without display.", e)
            logger.debug("Display traceback:\n{}", traceback.format_exc())
            return False

    def _gpio_set(self, pin, high):
        """Set a GPIO pin high or low via gpiod."""
        from gpiod.line import Value
        self._gpio_request.set_value(pin, Value.ACTIVE if high else Value.INACTIVE)

    def _send_command(self, cmd, data=None):
        """Send command byte, optionally followed by data bytes."""
        self._gpio_set(self._dc_pin, False)
        self._spi.writebytes([cmd])
        if data:
            self._gpio_set(self._dc_pin, True)
            self._spi.writebytes(list(data))

    def _send_data(self, data):
        """Send data bytes."""
        self._gpio_set(self._dc_pin, True)
        # SPI transfer in chunks (spidev limit is 4096 bytes)
        for i in range(0, len(data), 4096):
            self._spi.writebytes(data[i:i + 4096])

    def _ili9488_init(self):
        """ILI9488 initialization sequence."""
        # Software reset
        self._send_command(0x01)
        time.sleep(0.15)

        # Sleep out
        self._send_command(0x11)
        time.sleep(0.15)

        # Interface Pixel Format: 18-bit color (RGB666) for SPI
        self._send_command(0x3A, [0x66])

        # Memory Access Control: landscape rotation
        # MY=0, MX=1, MV=1, ML=0, BGR=1, MH=0 → 0x68
        self._send_command(0x36, [0x28])  # landscape 180° rotated (MY=0,MX=0,MV=1,BGR=1)

        # Power Control 1
        self._send_command(0xC0, [0x10, 0x10])
        # Power Control 2
        self._send_command(0xC1, [0x41])
        # VCOM Control
        self._send_command(0xC5, [0x00, 0x22, 0x80])

        # Positive Gamma Control
        self._send_command(0xE0, [0x00, 0x07, 0x10, 0x09, 0x17, 0x0B, 0x41,
                                   0x89, 0x4B, 0x0A, 0x0C, 0x0E, 0x18, 0x1B, 0x0F])
        # Negative Gamma Control
        self._send_command(0xE1, [0x00, 0x17, 0x1A, 0x04, 0x0E, 0x06, 0x2F,
                                   0x45, 0x43, 0x02, 0x0A, 0x09, 0x32, 0x36, 0x0F])

        # Display On
        self._send_command(0x29)
        time.sleep(0.05)

    def _set_window(self, x0, y0, x1, y1):
        """Set the drawing window."""
        self._send_command(0x2A, [x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF])
        self._send_command(0x2B, [y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF])
        self._send_command(0x2C)  # Memory Write

    # ── Main Loop ──────────────────────────────────────────────────────────

    def _run(self):
        if not self._init_hardware():
            return

        self._cur_params = compute_target("loading")
        self._cur_params.left_eye.x = float(EYE_LCX)
        self._cur_params.left_eye.y = float(EYE_CY)
        self._cur_params.right_eye.x = float(EYE_RCX)
        self._cur_params.right_eye.y = float(EYE_CY)
        self._tgt_params = compute_target("loading")
        last_time = time.monotonic()
        prev_state = "loading"

        while self._running:
            try:
                now = time.monotonic()
                dt = min(now - last_time, 1.0)
                last_time = now

                # 1. Read shared state
                new_state = self._state_ref.bot_state.value
                self._volume = self._state_ref.volume
                self._bot_enabled = self._state_ref.mic_enabled

                # 2. Detect state change
                if new_state != prev_state:
                    prev_state = new_state
                    self._current_state = new_state
                    self._anim.transition_progress = 0.0
                    # Reset look-around on state change
                    self._anim.look_active = False
                    self._anim.look_cooldown = random.uniform(3.0, 7.0)
                    # Reset idle timer & deactivate game on any state change
                    self._anim.idle_timer = 0.0
                    if self._anim.game_active:
                        self._anim.game_active = False
                        self._current_game = None

                # 2b. Detect tickle trigger (long_press touch event)
                touch_ev = getattr(self._state_ref, 'last_touch_event', None)
                if touch_ev == "long_press" and self._last_touch_seen != "long_press":
                    self._anim.tickle_timer = TICKLE_DURATION
                    self._anim.tickle_phase = 0.0
                self._last_touch_seen = touch_ev

                # Update tickle timer
                tickle_active = self._anim.tickle_timer > 0
                if tickle_active:
                    self._anim.tickle_timer -= dt
                    self._anim.tickle_phase += dt

                # 3. Disabled → sleeping mode
                if not self._bot_enabled:
                    self._render_sleeping()
                    self._push_frame()
                    time.sleep(3.0)
                    continue

                # 3b. Idle timer → game carousel
                if self._current_state == "ready" and not self._anim.game_active and self._games:
                    self._anim.idle_timer += dt
                    from display.games import IDLE_THRESHOLD
                    if self._anim.idle_timer >= IDLE_THRESHOLD:
                        self._anim.game_active = True
                        self._anim.game_timer = 0.0
                        self._anim.game_index = 0
                        self._current_game = self._games[0]
                        self._current_game.reset()
                        logger.info("Idle games started — {}", self._current_game.name)
                elif self._current_state != "ready":
                    self._anim.idle_timer = 0.0

                # 3c. Game carousel mode — skip face animations entirely
                if self._anim.game_active and self._current_game is not None:
                    from display.games import GAME_CYCLE_TIME, GameContext
                    # Cycle to next game every GAME_CYCLE_TIME seconds
                    self._anim.game_timer += dt
                    if self._anim.game_timer >= GAME_CYCLE_TIME:
                        self._anim.game_timer = 0.0
                        self._anim.game_index = (self._anim.game_index + 1) % len(self._games)
                        self._current_game = self._games[self._anim.game_index]
                        self._current_game.reset()
                        logger.info("Cycling to idle game: {}", self._current_game.name)

                    menu_open = getattr(self._state_ref, 'menu_open', False)
                    if menu_open:
                        self._render_menu()
                    else:
                        ctx = GameContext(self._draw, WIDTH, HEIGHT,
                                         self._font_large, self._font_small)
                        self._current_game.update(dt)
                        self._current_game.render(ctx)
                    self._push_frame()
                    self._frame += 1
                    time.sleep(1.0 / FPS_ACTIVE)
                    continue

                # 4. Compute target from state
                self._tgt_params = compute_target(self._current_state)

                # 5. Update procedural animations
                blink_mult = update_blink(self._anim, dt)
                self._anim.blink_mult = blink_mult
                breath_mult = update_breathing(self._anim, dt)

                # Idle look-around only in ready state
                if self._current_state == "ready":
                    look_ox, look_oy = update_idle_look(self._anim, dt)
                    update_gaze(self._anim, dt)
                    update_saccade(self._anim, dt)
                else:
                    look_ox, look_oy = 0.0, 0.0
                    update_gaze(self._anim, dt)
                    update_saccade(self._anim, dt)

                # Apply breathing to target (blink is applied at render time, not here)
                for eye in (self._tgt_params.left_eye, self._tgt_params.right_eye):
                    eye.height *= breath_mult
                    eye.width *= breath_mult

                # Apply gaze: look-around (big) + micro-drift (small) + saccade (tiny)
                if self._anim.look_active:
                    # During look-around, use the look offset as primary
                    gx = look_ox + self._anim.saccade_ox
                    gy = look_oy + self._anim.saccade_oy
                else:
                    gx = self._anim.gaze_target_x + self._anim.saccade_ox
                    gy = self._anim.gaze_target_y + self._anim.saccade_oy
                for eye in (self._tgt_params.left_eye, self._tgt_params.right_eye):
                    eye.pupil_x = _clamp(eye.pupil_x + gx, -1, 1)
                    eye.pupil_y = _clamp(eye.pupil_y + gy, -1, 1)

                # 6. Lerp current → target (blink NOT included — keeps lerp state clean)
                self._cur_params = lerp_face(
                    self._cur_params, self._tgt_params,
                    LERP_ALPHA, COLOR_LERP_ALPHA, PUPIL_LERP_ALPHA,
                )

                # 7. Advance transition
                self._anim.transition_progress = min(
                    1.0, self._anim.transition_progress + dt * 2.0)

                # 7b. Tickle override — applied AFTER lerp directly to _cur_params
                #     so wiggle/shape changes are instant, not smoothed away
                if tickle_active:
                    t = self._anim.tickle_phase
                    fade = min(1.0, self._anim.tickle_timer / 0.5)  # fade out last 0.5s

                    # Rapid wiggle — both eyes shake together
                    wiggle_x = math.sin(t * 20) * 8 * fade
                    # Vertical bounce
                    wiggle_y = abs(math.sin(t * 12)) * 4 * fade

                    for eye in (self._cur_params.left_eye, self._cur_params.right_eye):
                        # Squished into thin ^_^ happy crescents
                        eye.height = lerp(eye.height, 28.0, 0.5 * fade)
                        eye.width = lerp(eye.width, 140.0, 0.4 * fade)
                        eye.corner_radius = lerp(eye.corner_radius, 55.0, 0.5 * fade)
                        # Strong happy curve (anime ^_^)
                        eye.slope_left = -35.0 * fade + eye.slope_left * (1.0 - fade)
                        eye.slope_right = -35.0 * fade + eye.slope_right * (1.0 - fade)
                        # Hide pupils (eyes are squished shut happily)
                        eye.pupil_scale = lerp(eye.pupil_scale, 0.0, 0.6 * fade)
                        # Warm pink color — override directly
                        eye.color_r = lerp(eye.color_r, TICKLE_COLOR[0], 0.4 * fade)
                        eye.color_g = lerp(eye.color_g, TICKLE_COLOR[1], 0.4 * fade)
                        eye.color_b = lerp(eye.color_b, TICKLE_COLOR[2], 0.4 * fade)
                        eye.glow_alpha = lerp(eye.glow_alpha, 0.55, 0.3 * fade)
                        # Wiggle!
                        eye.x += wiggle_x
                        eye.y += wiggle_y

                # 8. State-specific animations
                state_color = STATE_COLORS.get(self._current_state, (80, 80, 80))

                if self._current_state == "listening":
                    lp = update_listen_pulse(self._anim, dt)
                    # Pulsing glow: eyes breathe brighter/dimmer while listening
                    pulse = 0.35 + 0.20 * math.sin(lp * math.pi * 2.0)
                    self._cur_params.left_eye.glow_alpha = pulse
                    self._cur_params.right_eye.glow_alpha = pulse
                    # Subtle pupil size pulse (dilate/contract)
                    pupil_pulse = 0.45 + 0.12 * math.sin(lp * math.pi * 2.0)
                    self._cur_params.left_eye.pupil_scale = pupil_pulse
                    self._cur_params.right_eye.pupil_scale = pupil_pulse

                if self._current_state == "speaking" and not tickle_active:
                    self._cur_params.mouth_open = update_mouth(self._anim, dt)

                if self._current_state == "processing":
                    update_dots(self._anim, dt)

                if self._current_state == "loading":
                    update_spinner(self._anim, dt)

                if self._current_state == "dancing":
                    # Excited wiggle like tickle — eyes shake side to side
                    wiggle_x = math.sin(self._frame * 1.2) * 12
                    bounce_y = math.sin(self._frame * 0.5) * 15
                    self._cur_params.left_eye.x += wiggle_x
                    self._cur_params.right_eye.x += wiggle_x
                    self._cur_params.left_eye.y += bounce_y
                    self._cur_params.right_eye.y += bounce_y
                    # Squash and stretch (excited breathing)
                    squash = math.sin(self._frame * 0.8) * 15
                    for eye in (self._cur_params.left_eye, self._cur_params.right_eye):
                        eye.width += squash
                        eye.height -= squash * 0.5
                    # Dilated excited pupils that look around
                    look = math.sin(self._frame * 0.3) * 0.6
                    self._cur_params.left_eye.pupil_x = look
                    self._cur_params.right_eye.pupil_x = look
                    self._cur_params.left_eye.pupil_scale = 0.55
                    self._cur_params.right_eye.pupil_scale = 0.55
                    # Big bouncy mouth
                    self._cur_params.mouth_open = 0.5 + 0.4 * abs(math.sin(self._frame * 0.6))
                    # Pulsing glow
                    glow = 0.45 + 0.25 * math.sin(self._frame * 0.6)
                    self._cur_params.left_eye.glow_alpha = glow
                    self._cur_params.right_eye.glow_alpha = glow

                # Sparkles
                if self._current_state == "dancing":
                    self._sparkles = update_sparkles(
                        self._sparkles, dt, CX, EYE_CY, spawn_chance=0.20)
                elif self._current_state in ("ready", "speaking"):
                    self._sparkles = update_sparkles(
                        self._sparkles, dt, CX, EYE_CY, spawn_chance=0.08)
                else:
                    self._sparkles = update_sparkles(
                        self._sparkles, dt, CX, EYE_CY, spawn_chance=0)

                # Hearts (tickle and dance animation)
                dance_active = self._current_state == "dancing"
                self._hearts = update_hearts(
                    self._hearts, dt, CX, EYE_CY,
                    spawn=tickle_active or dance_active,
                )

                # Tickle mouth bounce
                if tickle_active:
                    bounce = abs(math.sin(self._anim.tickle_phase * 8))
                    self._cur_params.mouth_open = bounce * 0.7 * min(1.0, self._anim.tickle_timer / 0.5)

                # 9. Determine if redraw needed
                needs_draw = (
                    self._anim.blink_phase >= 0
                    or self._anim.transition_progress < 0.95
                    or self._anim.look_active
                    or tickle_active
                    or len(self._hearts) > 0
                    or self._current_state in ("listening", "processing", "speaking", "loading", "dancing")
                    or len(self._sparkles) > 0
                    or not params_converged(self._cur_params, self._tgt_params)
                )

                # Check if menu or card UI is open
                menu_open = getattr(self._state_ref, 'menu_open', False)
                card_mode = getattr(self._state_ref, 'card_mode', 'off')
                if menu_open or card_mode != 'off':
                    needs_draw = True  # always redraw for UI overlays

                if needs_draw:
                    if card_mode != 'off':
                        self._render_cards(card_mode)
                    elif menu_open:
                        self._render_menu()
                    else:
                        self._render_frame(state_color)
                    self._push_frame()
                    self._frame += 1

                # 10. Adaptive sleep
                fps = self._get_target_fps()
                time.sleep(1.0 / fps)

            except Exception as e:
                logger.debug("Display update error: {}", e)
                time.sleep(1)

    # ── FPS Logic ──────────────────────────────────────────────────────────

    def _get_target_fps(self) -> float:
        # Card UI needs responsive redraw
        card_mode = getattr(self._state_ref, 'card_mode', 'off')
        if card_mode != 'off':
            return FPS_ACTIVE
        if self._anim.game_active:
            return FPS_ACTIVE
        if self._anim.blink_phase >= 0:
            return FPS_BLINK
        if self._anim.transition_progress < 0.95:
            return FPS_TRANSITION
        if self._anim.look_active:
            return FPS_LOOK
        if self._current_state in ("listening", "processing", "speaking", "loading", "dancing"):
            return FPS_ACTIVE
        return FPS_IDLE

    # ── Frame Push ─────────────────────────────────────────────────────────

    def _push_frame(self):
        with self._lock:
            import numpy as np
            # Set window to full screen
            with TFTDisplay.spi_lock:
                self._set_window(0, 0, WIDTH - 1, HEIGHT - 1)
                # Convert PIL RGB image to RGB666 (3 bytes per pixel, top 6 bits used)
                # Use numpy for fast masking instead of Python loop
                arr = np.frombuffer(self._image.tobytes(), dtype=np.uint8)
                buf = (arr & 0xFC).tobytes()
                self._gpio_set(self._dc_pin, True)
                # Send in chunks (spidev limit)
                for j in range(0, len(buf), 4096):
                    self._spi.writebytes(buf[j:j + 4096])

    # ── Render ─────────────────────────────────────────────────────────────

    def _render_frame(self, state_color: tuple):
        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=COLOR_BG)

        p = self._cur_params
        is_err = self._current_state == "error"

        # Listening pulse rings — drawn BEHIND eyes
        if self._current_state == "listening":
            render_listen_pulse(self._draw, self._anim.listen_phase, state_color)

        # Apply blink multiplier at render time (keeps lerp state clean)
        bm = self._anim.blink_mult
        if bm < 0.99:
            # Temporarily scale eye height for rendering
            orig_lh = p.left_eye.height
            orig_rh = p.right_eye.height
            p.left_eye.height *= bm
            p.right_eye.height *= bm
            render_eye(self._draw, p.left_eye, is_error=is_err)
            render_eye(self._draw, p.right_eye, is_error=is_err)
            # Restore so lerp state is untouched
            p.left_eye.height = orig_lh
            p.right_eye.height = orig_rh
        else:
            render_eye(self._draw, p.left_eye, is_error=is_err)
            render_eye(self._draw, p.right_eye, is_error=is_err)

        if self._sparkles:
            render_sparkles(self._draw, self._sparkles, state_color)

        # Floating hearts (tickle)
        if self._hearts:
            render_hearts(self._draw, self._hearts)

        if self._current_state == "processing":
            render_dots(self._draw, self._anim.dot_phase, state_color)
        elif self._anim.tickle_timer > 0:
            # Tickle mouth (bouncy open/close) — override normal mouth
            render_mouth(self._draw, p)
        elif self._current_state == "speaking":
            render_mouth(self._draw, p)
        elif self._current_state == "loading":
            render_spinner(self._draw, self._anim.spinner_idx, state_color)

    def _render_menu(self):
        """Render a settings menu overlay on the TFT display."""
        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(10, 10, 20))

        # Menu title bar
        self._draw.rounded_rectangle(
            [5, 5, WIDTH - 5, 35], radius=8, fill=(40, 40, 60)
        )
        self._draw.text(
            (WIDTH // 2 - 40, 10), "SETTINGS",
            fill=(180, 180, 220), font=self._font_large,
        )

        # Menu items — 7 items, fit within 320px height
        # title bar ends at 35, items start at 37
        # hint bar is 16px tall at bottom → available = 320 - 37 - 16 = 267px
        # 267 / 7 = 38px per item
        menu_items = self._get_menu_items()
        menu_idx = getattr(self._state_ref, 'menu_index', 0)
        item_height = 38
        start_y = 37

        for i, (label, value) in enumerate(menu_items):
            y = start_y + i * item_height
            is_selected = (i == menu_idx)

            if is_selected:
                # Highlighted item — colored background
                self._draw.rounded_rectangle(
                    [8, y, WIDTH - 8, y + item_height - 4],
                    radius=6, fill=(30, 100, 200),
                )
                text_color = (255, 255, 255)
                val_color = (200, 230, 255)
                # Selection indicator
                self._draw.text(
                    (14, y + 6), ">",
                    fill=(255, 255, 100), font=self._font_small,
                )
            else:
                text_color = (140, 140, 160)
                val_color = (100, 100, 120)

            # Label
            self._draw.text(
                (30, y + 6), label,
                fill=text_color, font=self._font_small,
            )
            # Value (right-aligned)
            if value:
                # Approximate right alignment
                val_x = WIDTH - 15 - len(value) * 8
                self._draw.text(
                    (val_x, y + 6), value,
                    fill=val_color, font=self._font_small,
                )

        # Bottom hint
        self._draw.text(
            (20, HEIGHT - 16), "Tap:Next  2x:Select  Hold:Close",
            fill=(80, 80, 100), font=self._font_small,
        )

    def _get_menu_items(self) -> list:
        """Build the menu items list with current values."""
        vol = getattr(self._state_ref, 'volume', 80)
        mode = getattr(self._state_ref, 'llm_mode', 'offline')
        mode_str = mode.value if hasattr(mode, 'value') else str(mode)
        mic = "On" if getattr(self._state_ref, 'mic_enabled', True) else "Off"
        proj = "On" if getattr(self._state_ref, 'projector_connected', False) else "Off"

        # Car connection status
        car_connecting = getattr(self._state_ref, 'car_connecting', False)
        car_connected = getattr(self._state_ref, 'car_connected', False)
        if car_connecting:
            car_str = "Connecting..."
        elif car_connected:
            car_str = "Connected"
        else:
            car_str = "Disconnected"

        return [
            ("Volume",     f"{vol}%"),
            ("Mode",       mode_str.capitalize()),
            ("Microphone", mic),
            ("Projector",  proj),
            ("Flashcards", ""),
            ("Car",        car_str),
            ("Sleep",      ""),
        ]

    # ── Card UI Rendering ──────────────────────────────────────────────

    def _render_cards(self, card_mode: str):
        """Render the card-based UI overlay."""
        if card_mode == "cards":
            self._render_card_selector()
        elif card_mode == "arts":
            self._render_arts_screen()
        elif card_mode == "encyclopedia":
            self._render_encyclopedia_screen()
        elif card_mode == "settings":
            self._render_settings_screen()
        elif card_mode == "settings_lang":
            self._render_language_screen()
        elif card_mode == "settings_wifi":
            self._render_wifi_screen()
        elif card_mode == "stories":
            self._render_stories_screen()
        elif card_mode == "story_reader":
            self._render_story_page()
        elif card_mode == "skill_detail":
            self._render_skill_detail()

    def _render_card_selector(self):
        """Render horizontally scrollable main cards (4 visible, swipe for more)."""
        from display.card_ui import MAIN_CARDS, draw_icon

        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(10, 10, 20))

        scroll = getattr(self._state_ref, 'card_scroll_offset', 0)
        card_w = 105
        card_h = 140
        gap = 8
        visible = 4
        total_w = visible * (card_w + gap) - gap
        start_x = (WIDTH - total_w) // 2
        card_y = (HEIGHT - card_h) // 2 - 10

        # Title
        self._draw.text(
            (WIDTH // 2 - 24, 12), "MENU",
            fill=(150, 150, 180), font=self._font_large,
        )

        # Left scroll arrow
        if scroll > 0:
            self._draw.text((4, HEIGHT // 2 - 10), "<", fill=(120, 120, 160), font=self._font_large)

        # Right scroll arrow
        if scroll + visible < len(MAIN_CARDS):
            self._draw.text((WIDTH - 18, HEIGHT // 2 - 10), ">", fill=(120, 120, 160), font=self._font_large)

        for vi in range(visible):
            ci = vi + scroll
            if ci >= len(MAIN_CARDS):
                break
            card = MAIN_CARDS[ci]
            x = start_x + vi * (card_w + gap)

            # Card background
            bg = card["bg"]
            border_color = card["color"]
            border_w = 2

            self._draw.rounded_rectangle(
                [x, card_y, x + card_w, card_y + card_h],
                radius=12, fill=bg, outline=border_color, width=border_w,
            )

            # Icon area
            icon_cx = x + card_w // 2
            icon_cy = card_y + 35
            draw_icon(self._draw, card["icon"], icon_cx, icon_cy, 28, card["color"])

            # Title text
            title_color = card["color"]
            tw = len(card["title"]) * 8
            self._draw.text(
                (icon_cx - tw // 2, card_y + 60), card["title"],
                fill=title_color, font=self._font_small,
            )
            if card["subtitle"]:
                sw = len(card["subtitle"]) * 8
                self._draw.text(
                    (icon_cx - sw // 2, card_y + 78), card["subtitle"],
                    fill=title_color, font=self._font_small,
                )

        # Page dots
        dot_y = card_y + card_h + 12
        total = len(MAIN_CARDS)
        dot_total_w = total * 12
        dot_start_x = (WIDTH - dot_total_w) // 2
        for i in range(total):
            dx = dot_start_x + i * 12
            in_view = scroll <= i < scroll + visible
            c = (150, 150, 180) if in_view else (50, 50, 70)
            self._draw.ellipse([dx, dot_y, dx + 6, dot_y + 6], fill=c)

        # Bottom hint
        self._draw.text(
            (WIDTH // 2 - 80, HEIGHT - 18), "Touch a card to select",
            fill=(80, 80, 100), font=self._font_small,
        )

    def _render_arts_screen(self):
        """Render Arts & Crafts sub-screen with drawing list."""
        from display.card_ui import ARTS_ITEMS

        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(15, 10, 20))

        sub_idx = getattr(self._state_ref, 'card_sub_index', 0)
        scroll = getattr(self._state_ref, 'card_scroll_offset', 0)

        # Header
        self._draw.rounded_rectangle([0, 0, WIDTH, 36], radius=0, fill=(50, 25, 10))
        self._draw.text(
            (12, 8), "< Arts & Crafts",
            fill=(255, 160, 80), font=self._font_large,
        )

        # Grid layout — 2 columns, scrollable rows
        col_w = 228
        row_h = 58
        gap = 6
        cols = 2
        start_y = 42
        visible_rows = 4  # fits in 320 - 42 - 20 = 258px → 4 rows of 58+6

        items = ARTS_ITEMS
        total_rows = (len(items) + cols - 1) // cols

        for i, item in enumerate(items):
            row = i // cols
            col = i % cols
            visual_row = row - scroll

            if visual_row < 0 or visual_row >= visible_rows:
                continue

            x = 6 + col * (col_w + gap)
            y = start_y + visual_row * (row_h + gap)
            is_sel = (i == sub_idx)

            # Item card
            bg = (40, 30, 45) if not is_sel else (60, 40, 70)
            border = item["color"] if is_sel else (50, 50, 60)
            self._draw.rounded_rectangle(
                [x, y, x + col_w, y + row_h],
                radius=8, fill=bg, outline=border, width=2 if is_sel else 1,
            )

            # Color dot
            dot_x = x + 14
            dot_cy = y + row_h // 2
            self._draw.ellipse(
                [dot_x - 6, dot_cy - 6, dot_x + 6, dot_cy + 6],
                fill=item["color"],
            )

            # Name and description
            name_color = (255, 255, 255) if is_sel else (180, 180, 200)
            desc_color = (180, 180, 200) if is_sel else (120, 120, 140)
            self._draw.text((dot_x + 14, y + 8), item["name"], fill=name_color, font=self._font_small)
            self._draw.text((dot_x + 14, y + 28), item["desc"], fill=desc_color, font=self._font_small)

        # Scroll indicators
        if scroll > 0:
            self._draw.text((WIDTH // 2 - 5, 38), "^", fill=(100, 100, 120), font=self._font_small)
        if scroll + visible_rows < total_rows:
            self._draw.text((WIDTH // 2 - 5, HEIGHT - 18), "v", fill=(100, 100, 120), font=self._font_small)

    def _render_encyclopedia_screen(self):
        """Render Encyclopedia sub-screen with 21st century skills."""
        from display.card_ui import SKILLS_DATA

        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(10, 12, 25))

        sub_idx = getattr(self._state_ref, 'card_sub_index', 0)
        scroll = getattr(self._state_ref, 'card_scroll_offset', 0)

        # Header
        self._draw.rounded_rectangle([0, 0, WIDTH, 36], radius=0, fill=(20, 30, 55))
        self._draw.text(
            (12, 8), "< Encyclopedia",
            fill=(120, 170, 255), font=self._font_large,
        )

        # List layout — vertical scrollable list
        row_h = 52
        gap = 4
        start_y = 42
        visible = 5  # fits in 320 - 42 - 10 = 268px → 5 rows of 52+4

        for i, skill in enumerate(SKILLS_DATA):
            visual_i = i - scroll
            if visual_i < 0 or visual_i >= visible:
                continue

            y = start_y + visual_i * (row_h + gap)
            is_sel = (i == sub_idx)

            # Row background
            bg = (25, 28, 45) if not is_sel else (35, 40, 65)
            border = skill["color"] if is_sel else (40, 45, 60)
            self._draw.rounded_rectangle(
                [6, y, WIDTH - 6, y + row_h],
                radius=8, fill=bg, outline=border, width=2 if is_sel else 1,
            )

            # Color bar on left
            self._draw.rounded_rectangle(
                [6, y, 14, y + row_h],
                radius=4, fill=skill["color"],
            )

            # Skill name and short description
            name_color = (255, 255, 255) if is_sel else (180, 180, 200)
            short_color = skill["color"] if is_sel else (120, 120, 140)
            self._draw.text((22, y + 6), skill["name"], fill=name_color, font=self._font_small)
            self._draw.text((22, y + 26), skill["short"], fill=short_color, font=self._font_small)

            # Activity count
            act_count = len(skill.get("activities", []))
            act_text = f"{act_count}"
            self._draw.text(
                (WIDTH - 35, y + 16), act_text,
                fill=(100, 100, 120), font=self._font_small,
            )

        # Scroll indicators
        if scroll > 0:
            self._draw.text((WIDTH // 2 - 5, 38), "^", fill=(100, 100, 120), font=self._font_small)
        if scroll + visible < len(SKILLS_DATA):
            self._draw.text((WIDTH // 2 - 5, HEIGHT - 16), "v", fill=(100, 100, 120), font=self._font_small)

    def _render_skill_detail(self):
        """Render a single skill's detail view with description and activities."""
        from display.card_ui import SKILLS_DATA

        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(10, 12, 25))

        sub_idx = getattr(self._state_ref, 'card_sub_index', 0)
        if sub_idx >= len(SKILLS_DATA):
            return
        skill = SKILLS_DATA[sub_idx]

        # Header with skill color
        self._draw.rounded_rectangle([0, 0, WIDTH, 40], radius=0, fill=skill["color"])
        self._draw.text((12, 10), f"< {skill['name']}", fill=(255, 255, 255), font=self._font_large)

        # Description
        desc = skill["desc"]
        # Word wrap at ~45 chars per line
        lines = []
        words = desc.split()
        line = ""
        for w in words:
            if len(line + " " + w) > 45:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)

        y = 50
        for ln in lines:
            self._draw.text((15, y), ln, fill=(200, 210, 230), font=self._font_small)
            y += 20

        # Activities section
        y += 10
        self._draw.text((15, y), "Activities:", fill=(180, 180, 200), font=self._font_small)
        y += 24

        act_idx = getattr(self._state_ref, 'card_scroll_offset', 0)
        activities = skill.get("activities", [])
        for i, act in enumerate(activities):
            if y + 36 > HEIGHT - 10:
                break
            is_sel = (i == act_idx)
            bg = (30, 35, 55) if not is_sel else (45, 50, 75)
            border = skill["color"] if is_sel else (40, 45, 60)
            self._draw.rounded_rectangle(
                [12, y, WIDTH - 12, y + 32],
                radius=6, fill=bg, outline=border, width=2 if is_sel else 1,
            )
            color = (255, 255, 255) if is_sel else (170, 170, 190)
            self._draw.text((24, y + 7), act, fill=color, font=self._font_small)
            y += 38

        # Bottom hint
        self._draw.text(
            (WIDTH // 2 - 60, HEIGHT - 16), "Tap to start activity",
            fill=(80, 80, 100), font=self._font_small,
        )

    def _render_settings_screen(self):
        """Render settings with large, touch-friendly tiles in a 2x4 grid."""
        from display.card_ui import SETTINGS_ITEMS, draw_icon

        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(10, 10, 20))

        # Header
        self._draw.rounded_rectangle([0, 0, WIDTH, 36], radius=0, fill=(35, 35, 50))
        self._draw.text((12, 6), "< Settings", fill=(180, 180, 220), font=self._font_large)

        # 2x4 grid of setting tiles
        cols, rows = 2, 4
        gap = 6
        pad_x, pad_top, pad_bot = 6, 40, 4
        tile_w = (WIDTH - pad_x * 2 - gap * (cols - 1)) // cols   # ~231
        tile_h = (HEIGHT - pad_top - pad_bot - gap * (rows - 1)) // rows  # ~67

        scroll = getattr(self._state_ref, 'card_scroll_offset', 0)

        # Get live values for display
        vol = getattr(self._state_ref, 'volume', 80)
        mode = getattr(self._state_ref, 'llm_mode', 'offline')
        mode_str = mode.value if hasattr(mode, 'value') else str(mode)
        mic_on = getattr(self._state_ref, 'mic_enabled', True)
        proj_on = getattr(self._state_ref, 'projector_connected', False)
        car_conn = getattr(self._state_ref, 'car_connected', False)
        car_ing = getattr(self._state_ref, 'car_connecting', False)
        wifi_ssid = getattr(self._state_ref, 'wifi_ssid', None)
        language = getattr(self._state_ref, 'language', 'en')

        lang_names = {"en": "English", "hi": "Hindi", "te": "Telugu"}
        values = {
            "volume": f"{vol}%",
            "language": lang_names.get(language, language),
            "mode": mode_str.capitalize(),
            "mic": "On" if mic_on else "Off",
            "projector": "On" if proj_on else "Off",
            "car": "Linking..." if car_ing else ("Linked" if car_conn else "Off"),
            "wifi": wifi_ssid or "Not connected",
            "sleep": "Zzz",
        }
        # Color for active/inactive values
        active_ids = set()
        if mic_on:
            active_ids.add("mic")
        if proj_on:
            active_ids.add("projector")
        if car_conn:
            active_ids.add("car")
        if wifi_ssid:
            active_ids.add("wifi")

        for idx, item in enumerate(SETTINGS_ITEMS):
            col = idx % cols
            row = idx // cols
            x = pad_x + col * (tile_w + gap)
            y = pad_top + row * (tile_h + gap)

            item_id = item["id"]
            color = item["color"]
            val = values.get(item_id, "")
            is_active = item_id in active_ids

            # Tile background
            bg = (25, 28, 40)
            self._draw.rounded_rectangle(
                [x, y, x + tile_w, y + tile_h],
                radius=10, fill=bg, outline=(50, 50, 65), width=1,
            )

            # Icon on the left
            icon_cx = x + 28
            icon_cy = y + tile_h // 2
            draw_icon(self._draw, item["icon"], icon_cx, icon_cy, 22, color)

            # Label — bigger text
            label_x = x + 52
            self._draw.text(
                (label_x, y + 10), item["label"],
                fill=(220, 220, 240), font=self._font_large,
            )

            # Value — right side, color-coded
            val_color = color if is_active else (120, 120, 140)
            # Truncate long wifi names
            display_val = val[:14] + ".." if len(val) > 16 else val
            val_w = len(display_val) * 8
            self._draw.text(
                (x + tile_w - val_w - 10, y + 14), display_val,
                fill=val_color, font=self._font_small,
            )

            # Active indicator dot
            if is_active:
                self._draw.ellipse(
                    [x + tile_w - 14, y + tile_h - 14, x + tile_w - 6, y + tile_h - 6],
                    fill=(80, 220, 100),
                )

    def _render_language_screen(self):
        """Render language selection screen with big buttons."""
        from display.card_ui import LANGUAGES

        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(10, 10, 20))

        # Header
        self._draw.rounded_rectangle([0, 0, WIDTH, 40], radius=0, fill=(60, 50, 15))
        self._draw.text((12, 8), "< Language", fill=(255, 200, 80), font=self._font_large)

        current_lang = getattr(self._state_ref, 'language', 'en')

        # 3 large buttons vertically
        btn_h = 72
        gap = 12
        start_y = 55
        pad_x = 20

        for i, lang in enumerate(LANGUAGES):
            y = start_y + i * (btn_h + gap)
            is_current = (lang["id"] == current_lang)

            # Button background
            bg = (35, 40, 55) if not is_current else (25, 50, 80)
            border = lang["color"] if is_current else (50, 55, 70)
            bw = 3 if is_current else 1
            self._draw.rounded_rectangle(
                [pad_x, y, WIDTH - pad_x, y + btn_h],
                radius=12, fill=bg, outline=border, width=bw,
            )

            # Flag/code badge
            badge_x = pad_x + 18
            badge_cy = y + btn_h // 2
            badge_color = lang["color"]
            self._draw.rounded_rectangle(
                [badge_x - 8, badge_cy - 16, badge_x + 28, badge_cy + 16],
                radius=6, fill=badge_color,
            )
            self._draw.text(
                (badge_x - 2, badge_cy - 10), lang["flag"],
                fill=(255, 255, 255), font=self._font_small,
            )

            # Language name — big
            self._draw.text(
                (badge_x + 44, y + 12), lang["name"],
                fill=(240, 240, 255) if is_current else (180, 180, 200),
                font=self._font_large,
            )

            # Check mark if selected
            if is_current:
                cx = WIDTH - pad_x - 30
                cy = y + btn_h // 2
                self._draw.text(
                    (cx, cy - 12), "OK",
                    fill=badge_color, font=self._font_large,
                )

    def _render_wifi_screen(self):
        """Render WiFi info screen with connection details."""
        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(10, 10, 20))

        # Header
        self._draw.rounded_rectangle([0, 0, WIDTH, 40], radius=0, fill=(15, 50, 45))
        self._draw.text((12, 8), "< WiFi Info", fill=(60, 230, 190), font=self._font_large)

        ssid = getattr(self._state_ref, 'wifi_ssid', None)
        ip = getattr(self._state_ref, 'wifi_ip', None)
        signal = getattr(self._state_ref, 'wifi_signal', 0)

        pad = 20
        y = 55

        if ssid:
            # Connected — show details in big cards
            # SSID card
            self._draw.rounded_rectangle(
                [pad, y, WIDTH - pad, y + 60],
                radius=10, fill=(25, 35, 50), outline=(50, 60, 80), width=1,
            )
            self._draw.text((pad + 15, y + 6), "Network", fill=(100, 120, 150), font=self._font_small)
            self._draw.text((pad + 15, y + 26), ssid, fill=(220, 240, 255), font=self._font_large)
            y += 72

            # IP card
            self._draw.rounded_rectangle(
                [pad, y, WIDTH - pad, y + 60],
                radius=10, fill=(25, 35, 50), outline=(50, 60, 80), width=1,
            )
            self._draw.text((pad + 15, y + 6), "IP Address", fill=(100, 120, 150), font=self._font_small)
            ip_display = ip or "Unknown"
            self._draw.text((pad + 15, y + 26), ip_display, fill=(220, 240, 255), font=self._font_large)
            y += 72

            # Signal strength card
            self._draw.rounded_rectangle(
                [pad, y, WIDTH - pad, y + 60],
                radius=10, fill=(25, 35, 50), outline=(50, 60, 80), width=1,
            )
            self._draw.text((pad + 15, y + 6), "Signal Strength", fill=(100, 120, 150), font=self._font_small)
            # Signal bar
            bar_x = pad + 15
            bar_y = y + 32
            bar_w = WIDTH - pad * 2 - 80
            bar_h = 14
            self._draw.rounded_rectangle(
                [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                radius=4, fill=(40, 45, 55),
            )
            fill_w = int(bar_w * signal / 100)
            if fill_w > 0:
                sig_color = (80, 220, 100) if signal > 60 else (255, 200, 50) if signal > 30 else (255, 80, 80)
                self._draw.rounded_rectangle(
                    [bar_x, bar_y, bar_x + fill_w, bar_y + bar_h],
                    radius=4, fill=sig_color,
                )
            self._draw.text(
                (bar_x + bar_w + 10, bar_y - 2), f"{signal}%",
                fill=(180, 180, 200), font=self._font_small,
            )
            y += 72

            # Dashboard URL
            self._draw.rounded_rectangle(
                [pad, y, WIDTH - pad, y + 50],
                radius=10, fill=(25, 30, 45), outline=(50, 60, 80), width=1,
            )
            self._draw.text((pad + 15, y + 6), "Parent Dashboard", fill=(100, 120, 150), font=self._font_small)
            url = f"http://{ip}:8080" if ip else "http://hairobo.local:8080"
            self._draw.text((pad + 15, y + 24), url, fill=(100, 200, 255), font=self._font_small)
        else:
            # Not connected
            cy = HEIGHT // 2 - 20
            self._draw.text(
                (WIDTH // 2 - 80, cy), "Not Connected",
                fill=(200, 100, 100), font=self._font_large,
            )
            self._draw.text(
                (WIDTH // 2 - 120, cy + 35),
                "Connect via hairobo.local",
                fill=(120, 120, 150), font=self._font_small,
            )

    def _render_stories_screen(self):
        """Render Bedtime Stories list — 2-column grid similar to arts."""
        from display.card_ui import BEDTIME_STORIES

        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(12, 8, 25))

        sub_idx = getattr(self._state_ref, 'card_sub_index', 0)
        scroll = getattr(self._state_ref, 'card_scroll_offset', 0)

        # Header
        self._draw.rounded_rectangle([0, 0, WIDTH, 36], radius=0, fill=(40, 20, 55))
        self._draw.text((12, 6), "< Bedtime Stories", fill=(200, 170, 255), font=self._font_large)

        # 2-column grid
        col_w = 228
        row_h = 58
        gap = 6
        cols = 2
        start_y = 42
        visible_rows = 4

        for i, story in enumerate(BEDTIME_STORIES):
            row = i // cols
            col = i % cols
            visual_row = row - scroll

            if visual_row < 0 or visual_row >= visible_rows:
                continue

            x = 6 + col * (col_w + gap)
            y = start_y + visual_row * (row_h + gap)
            is_sel = (i == sub_idx)

            bg = (30, 20, 40) if not is_sel else (50, 30, 65)
            border = story["color"] if is_sel else (40, 35, 55)
            self._draw.rounded_rectangle(
                [x, y, x + col_w, y + row_h],
                radius=8, fill=bg, outline=border, width=2 if is_sel else 1,
            )

            # Icon: sparkle for "imagine_story", crescent moon for others
            dot_x = x + 14
            dot_cy = y + row_h // 2
            if story.get("id") == "imagine_story":
                # Star/sparkle icon
                from display.card_ui import draw_icon
                draw_icon(self._draw, "star", dot_x, dot_cy, 14, story["color"])
            else:
                self._draw.ellipse(
                    [dot_x - 6, dot_cy - 6, dot_x + 6, dot_cy + 6],
                    fill=story["color"],
                )
                self._draw.ellipse(
                    [dot_x - 2, dot_cy - 6, dot_x + 8, dot_cy + 6],
                    fill=bg,  # crescent cutout
                )

            name_color = (240, 240, 255) if is_sel else (180, 170, 200)
            desc_color = (180, 170, 210) if is_sel else (110, 100, 130)
            self._draw.text((dot_x + 14, y + 8), story["title"], fill=name_color, font=self._font_small)
            self._draw.text((dot_x + 14, y + 28), story["desc"], fill=desc_color, font=self._font_small)

            # Page count
            pages = len(story.get("pages", []))
            self._draw.text(
                (x + col_w - 28, y + 8), f"{pages}p",
                fill=(80, 70, 100), font=self._font_small,
            )

        # Scroll indicators
        total_rows = (len(BEDTIME_STORIES) + cols - 1) // cols
        if scroll > 0:
            self._draw.text((WIDTH // 2 - 5, 38), "^", fill=(100, 80, 130), font=self._font_small)
        if scroll + visible_rows < total_rows:
            self._draw.text((WIDTH // 2 - 5, HEIGHT - 18), "v", fill=(100, 80, 130), font=self._font_small)

    def _render_story_page(self):
        """Render a story page — text only on TFT (visuals go to projector)."""
        from display.card_ui import BEDTIME_STORIES

        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(10, 8, 25))

        sub_idx = getattr(self._state_ref, 'card_sub_index', 0)
        page_idx = getattr(self._state_ref, 'card_scroll_offset', 0)

        # Check if this is an AI-generated story
        gen_story = getattr(self._state_ref, 'generated_story', None)
        if sub_idx == 0 and gen_story:
            story = gen_story
        elif sub_idx < len(BEDTIME_STORIES):
            story = BEDTIME_STORIES[sub_idx]
        else:
            return

        pages = story.get("pages", [])
        if not pages:
            # Still generating...
            self._draw.text((WIDTH // 2 - 60, HEIGHT // 2 - 20), "Creating story...",
                            fill=(200, 180, 255), font=self._font_large)
            self._draw.text((WIDTH // 2 - 80, HEIGHT // 2 + 15),
                            "Imagining your story & pictures",
                            fill=(120, 110, 150), font=self._font_small)
            return

        if page_idx >= len(pages):
            page_idx = len(pages) - 1
        page = pages[page_idx]

        accent = story.get("color", (255, 200, 80))

        # Story title at top
        self._draw.text((15, 8), story["title"], fill=accent, font=self._font_small)
        self._draw.text((WIDTH - 90, 8), f"Page {page_idx + 1}/{len(pages)}", fill=(120, 110, 150), font=self._font_small)

        # Divider
        self._draw.line([10, 26, WIDTH - 10, 26], fill=(40, 35, 60), width=1)

        # Text area: full height (no scene on TFT — scene is on projector)
        text_y = 34
        text = page["text"]

        # Word wrap at ~46 chars (slightly wider text area now)
        lines = []
        words = text.split()
        line = ""
        for w in words:
            if len(line + " " + w) > 46:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)

        # Center text vertically in available space
        avail_h = HEIGHT - 34 - 30  # Between title and bottom bar
        text_block_h = len(lines) * 22
        if text_block_h < avail_h:
            text_y += (avail_h - text_block_h) // 2

        for ln in lines:
            if text_y + 22 > HEIGHT - 30:
                break
            self._draw.text((20, text_y), ln, fill=(220, 220, 240), font=self._font_small)
            text_y += 22

        # Projector hint (subtle)
        self._draw.text((WIDTH // 2 - 55, HEIGHT - 52), "Scene on projector", fill=(60, 55, 80), font=self._font_small)

        # Bottom bar: page indicator + nav hints
        bar_y = HEIGHT - 26
        self._draw.rectangle([0, bar_y, WIDTH, HEIGHT], fill=(20, 15, 35))
        total = len(pages)

        # Page dots
        dot_w = total * 16
        dot_x = (WIDTH - dot_w) // 2
        for i in range(total):
            c = accent if i == page_idx else (50, 45, 65)
            self._draw.ellipse([dot_x + i * 16, bar_y + 8, dot_x + i * 16 + 8, bar_y + 16], fill=c)

        # Nav arrows
        if page_idx > 0:
            self._draw.text((8, bar_y + 4), "<", fill=(150, 140, 180), font=self._font_small)
        if page_idx < total - 1:
            self._draw.text((WIDTH - 16, bar_y + 4), ">", fill=(150, 140, 180), font=self._font_small)
        else:
            self._draw.text((WIDTH - 35, bar_y + 4), "End", fill=accent, font=self._font_small)

    def _render_sleeping(self):
        self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=COLOR_BG)
        color = (80, 80, 80)
        for cx in (EYE_LCX, EYE_RCX):
            self._draw.rounded_rectangle(
                [cx - 55, EYE_CY - 2, cx + 55, EYE_CY + 2],
                radius=2, fill=color)
        self._anim.z_phase += 1
        render_sleep_z(self._draw, self._anim.z_phase, self._font_large)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _clear(self):
        if self._display and self._image:
            try:
                self._draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(0, 0, 0))
                self._push_frame()
            except Exception:
                pass
