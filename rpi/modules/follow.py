"""Follow mode — bot follows a yellow wheel using HSV color tracking.

Tracks the bright yellow hub of a robot chassis wheel via OpenCV HSV
color filtering. No neural network needed — pure color detection,
very fast and lightweight.

Goal: Keep the yellow wheel centered horizontally in the camera frame.
- Wheel drifts LEFT  → spin left to re-center
- Wheel drifts RIGHT → spin right to re-center
- Wheel is SMALL (far away) → drive forward while centering
- Wheel is BIG (too close) → stop, wait for movement
- Wheel is centered + good size → stop, hold position

NOTE: Bot is mounted FACING BACKWARD on the car. The Arduino sketch
already handles direction reversal (motor pins swapped), so this
module sends logical directions directly — no swap needed here.

Usage:
    follower = FollowMode(car, camera, audio_capture)
    await follower.start()
    await follower.stop()
"""

import asyncio
import numpy as np
from typing import Optional

from loguru import logger

# ─── Camera frame constants ──────────────────────────────────
FRAME_W = 640
FRAME_H = 480
CENTER_X = FRAME_W // 2

# ─── Yellow wheel HSV range ──────────────────────────────────
# Yellow hub: H=20-40, S=100-255, V=100-255 (bright saturated yellow)
YELLOW_LOW = np.array([18, 80, 80], dtype=np.uint8)
YELLOW_HIGH = np.array([45, 255, 255], dtype=np.uint8)

# Minimum contour area (pixels²) to count as the wheel (filters noise)
MIN_CONTOUR_AREA = 500

# ─── Centering thresholds ────────────────────────────────────
DEAD_ZONE_X = 120           # pixels from center = "centered enough" (wide to avoid jitter)

# Wheel width as fraction of FRAME_W to judge distance
WHEEL_TOO_CLOSE = 0.50      # Wheel fills >50% → back up
WHEEL_CLOSE_ENOUGH = 0.15   # Wheel fills >15% → hold (close enough)
WHEEL_TOO_FAR = 0.06        # Wheel <6% → drive forward fast

# ─── Backward retreat ───────────────────────────────────────
SPEED_BACKWARD = 150        # Gentle reverse when too close

# ─── Speeds (gentle — bot is small, indoors) ────────────────
SPEED_FORWARD = 180         # Normal approach
SPEED_FAST = 200            # When wheel is very far
SPEED_TURN = 160            # Pure spin to re-center (gentle)

# ─── Timing ──────────────────────────────────────────────────
DETECT_INTERVAL = 0.05      # Small sleep between detections
NO_TARGET_TIMEOUT = 3.0     # Seconds without detection → start searching
SEARCH_SPIN_SPEED = 150     # Spin while searching (gentle)
SEARCH_SPIN_DURATION = 0.3  # Spin burst duration (seconds)
SEARCH_PAUSE_DURATION = 0.8 # Pause after spin to let camera detect


class FollowMode:
    """Yellow wheel following using HSV color tracking.

    Lightweight — pure OpenCV color filtering, no model needed.
    Tracks the bright yellow hub of a robot chassis wheel.
    """

    def __init__(self, car, camera, audio_capture):
        self._car = car
        self._camera = camera
        self._audio = audio_capture
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_detect_time = 0.0
        # Smoothing: rolling average of last N detections to reduce jitter
        self._cx_history: list[int] = []
        self._w_history: list[int] = []
        self._smooth_n = 5  # average over 5 frames

    @property
    def active(self) -> bool:
        return self._running

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._follow_loop())
        logger.info("Follow mode started (yellow wheel HSV tracking)")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._car and self._car.connected:
            await self._car.stop()
        logger.info("Follow mode stopped")

    # ── Main loop ────────────────────────────────────────────────

    async def _follow_loop(self):
        import time
        self._last_detect_time = time.time()
        search_direction_right = True

        while self._running:
            try:
                detection = await self._detect_yellow_wheel()

                if detection:
                    self._last_detect_time = time.time()
                    await self._steer_to_center(detection)
                else:
                    elapsed = time.time() - self._last_detect_time

                    if elapsed > NO_TARGET_TIMEOUT:
                        # Lost wheel — search with spin-stop-check pattern
                        logger.info("Follow: no yellow wheel, searching ({:.0f}s, dir={})",
                                    elapsed, "R" if search_direction_right else "L")

                        # Short spin burst to search for target
                        if search_direction_right:
                            await self._car.spin_right(speed=SEARCH_SPIN_SPEED)
                        else:
                            await self._car.spin_left(speed=SEARCH_SPIN_SPEED)
                        await asyncio.sleep(SEARCH_SPIN_DURATION)

                        # Stop and let camera check
                        await self._car.stop()
                        await asyncio.sleep(SEARCH_PAUSE_DURATION)

                        # Check again after stopping
                        detection = await self._detect_yellow_wheel()
                        if detection:
                            self._last_detect_time = time.time()
                            await self._steer_to_center(detection)
                            continue

                        # Switch direction every ~8 seconds
                        if int(elapsed) % 8 < 1:
                            search_direction_right = not search_direction_right
                    else:
                        # Brief loss — hold position
                        await self._car.stop()

                await asyncio.sleep(DETECT_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Follow loop error: {}", e)
                await asyncio.sleep(0.5)

        if self._car and self._car.connected:
            await self._car.stop()

    # ── Yellow wheel detection (HSV color tracking) ──────────────

    async def _detect_yellow_wheel(self) -> Optional[dict]:
        """Capture frame, detect yellow wheel using HSV color filtering."""
        if not self._camera:
            return None

        frame = await self._camera.capture()
        if frame is None:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._detect_sync, frame)

    def _detect_sync(self, frame: np.ndarray) -> Optional[dict]:
        """Find the largest yellow blob in the frame."""
        import cv2

        h, w = frame.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mask for yellow color
        mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)

        # Clean up noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest contour (should be the yellow wheel)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < MIN_CONTOUR_AREA:
            return None

        # Get bounding rect and center
        x, y, bw, bh = cv2.boundingRect(largest)
        cx = x + bw // 2
        cy = y + bh // 2

        logger.info("Follow: yellow wheel detected ({}×{}px, area={}, cx={})",
                    bw, bh, int(area), cx)

        return {
            "cx": cx,
            "cy": cy,
            "w": bw,
            "h": bh,
            "area": area,
        }

    # ── Steering: keep wheel centered ────────────────────────────

    async def _steer_to_center(self, det: dict):
        """Drive toward the yellow wheel and stop when close enough.

        Priority:
        1. Too close → STOP immediately
        2. Close enough + centered → STOP, hold
        3. Off-center + close → SPIN to re-center
        4. Centered + far → drive FORWARD
        5. Off-center + far → FORWARD + spin (diagonal)
        """
        # Smooth detection values over last N frames to reduce jitter
        self._cx_history.append(det["cx"])
        self._w_history.append(det["w"])
        if len(self._cx_history) > self._smooth_n:
            self._cx_history.pop(0)
            self._w_history.pop(0)

        wheel_cx = sum(self._cx_history) // len(self._cx_history)
        wheel_w = sum(self._w_history) // len(self._w_history)

        # Wheel size relative to frame width (distance indicator)
        size_ratio = wheel_w / FRAME_W

        # Horizontal offset from center
        offset_x = wheel_cx - CENTER_X
        centered = abs(offset_x) < DEAD_ZONE_X

        # ── 1. Too close: back up ──
        if size_ratio > WHEEL_TOO_CLOSE:
            logger.info("Follow: wheel too close ({:.0%}), backing up!", size_ratio)
            await self._car.backward(speed=SPEED_BACKWARD)
            return

        # ── 2. Close enough + centered: hold ──
        if size_ratio >= WHEEL_CLOSE_ENOUGH and centered:
            logger.info("Follow: wheel close & centered ({:.0%}), holding", size_ratio)
            await self._car.stop()
            return

        # ── 3. Off-center + close: spin to re-center ──
        # Arduino already handles backward-mount reversal, so send logical directions
        if not centered and size_ratio >= WHEEL_CLOSE_ENOUGH:
            if offset_x > 0:
                logger.info("Follow: wheel right (+{}px), spin right", int(offset_x))
                await self._car.spin_right(speed=SPEED_TURN)
            else:
                logger.info("Follow: wheel left ({}px), spin left", int(offset_x))
                await self._car.spin_left(speed=SPEED_TURN)
            return

        # ── 4. Far away: approach! ──
        speed = SPEED_FAST if size_ratio < WHEEL_TOO_FAR else SPEED_FORWARD

        if centered:
            logger.info("Follow: wheel centered, far ({:.0%}), forward!", size_ratio)
            await self._car.forward(speed=speed)
        else:
            # Diagonal approach — Arduino handles backward-mount reversal
            if offset_x > 0:
                logger.info("Follow: wheel far+right ({:.0%}), fwd-right!", size_ratio)
                await self._car.forward_right(speed=speed)
            else:
                logger.info("Follow: wheel far+left ({:.0%}), fwd-left!", size_ratio)
                await self._car.forward_left(speed=speed)
