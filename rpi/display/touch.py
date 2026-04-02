"""XPT2046 resistive touch screen driver for ILI9488 3.5" display.

Reads touch coordinates via SPI (shared bus with display) and detects
tap patterns: tap, double_tap, triple_tap, long_press, extra_long.

Hardware wiring:
  TCS  → BCM7  (Pin 26) — Touch chip select (SPI CE1)
  TCK  → BCM11 (Pin 23) — Shared with display SCK
  TDI  → BCM10 (Pin 19) — Shared with display MOSI
  TOO  → BCM9  (Pin 21) — Touch MISO
  PEN  → BCM5  (Pin 29) — Touch interrupt (active low)

Calibration (raw → screen 480x320, display rotated 180°):
  TOP-LEFT:     raw X=3623, Y=348
  TOP-RIGHT:    raw X=3680, Y=3694
  BOTTOM-LEFT:  raw X=547,  Y=325
  BOTTOM-RIGHT: raw X=450,  Y=3715
"""

import threading
import time
from enum import Enum
from typing import Callable, Optional, Tuple

from loguru import logger


# ─── Pin assignments ─────────────────────────────────────────
GPIO_CHIP = 4           # Pi 5 uses gpiochip4
PIN_PEN_IRQ = 5         # BCM5 (Pin 29) — touch interrupt, active low
SPI_BUS = 0
SPI_DEVICE = 1          # CE1 = BCM7

# ─── XPT2046 commands ────────────────────────────────────────
CMD_READ_X = 0xD0       # 12-bit X position
CMD_READ_Y = 0x90       # 12-bit Y position

# ─── Calibration constants ───────────────────────────────────
# Raw touch range from calibration
RAW_X_MIN = 500         # Bottom of screen
RAW_X_MAX = 3650        # Top of screen
RAW_Y_MIN = 335         # Left of screen
RAW_Y_MAX = 3705        # Right of screen

# Screen dimensions
SCREEN_W = 480
SCREEN_H = 320

# ─── Timing thresholds ──────────────────────────────────────
DEBOUNCE_MS = 50
LONG_PRESS_THRESHOLD = 1.0      # seconds
EXTRA_LONG_THRESHOLD = 2.0      # seconds
TAP_WINDOW = 0.45               # seconds between taps
POLL_HZ = 50                    # 50Hz polling (touch doesn't need 100Hz)
TOUCH_THRESHOLD = 100            # Minimum raw value to consider valid


class TouchEvent(str, Enum):
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    TRIPLE_TAP = "triple_tap"
    LONG_PRESS = "long_press"
    EXTRA_LONG = "extra_long"


class TouchController:
    """XPT2046 resistive touch controller with tap-pattern detection."""

    def __init__(self):
        self._spi = None
        self._gpio = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable] = []

        # Touch state machine
        self._is_pressed = False
        self._press_start: Optional[float] = None
        self._last_transition: float = 0.0
        self._tap_count: int = 0
        self._last_tap_time: float = 0.0
        self._long_press_fired = False
        self._extra_long_fired = False

        # Last screen coordinates
        self._last_x: int = 0
        self._last_y: int = 0

    def add_callback(self, callback: Callable[[TouchEvent], None]):
        """Register a callback for touch events."""
        self._callbacks.append(callback)

    @property
    def last_position(self) -> Tuple[int, int]:
        """Last touch position in screen coordinates (x, y)."""
        return self._last_x, self._last_y

    def start(self):
        """Initialize hardware and start polling thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _init_hardware(self) -> bool:
        """Initialize SPI for XPT2046 and GPIO for PEN interrupt."""
        try:
            import spidev
            import gpiod
            from gpiod.line import Direction, Bias

            # SPI for touch (CE1 = BCM7)
            self._spi = spidev.SpiDev()
            self._spi.open(SPI_BUS, SPI_DEVICE)
            self._spi.max_speed_hz = 500000  # 500kHz for reliable touch reads
            self._spi.mode = 0

            # PEN IRQ pin with internal pull-up
            self._gpio = gpiod.request_lines(
                f"/dev/gpiochip{GPIO_CHIP}",
                consumer="xpt2046_pen",
                config={
                    PIN_PEN_IRQ: gpiod.LineSettings(
                        direction=Direction.INPUT,
                        bias=Bias.PULL_UP,
                    ),
                },
            )

            logger.info(
                "XPT2046 touch initialized (SPI{}:{}, PEN=BCM{})",
                SPI_BUS, SPI_DEVICE, PIN_PEN_IRQ,
            )
            return True

        except Exception as e:
            logger.warning("Touch init failed: {}. Running without touch.", e)
            return False

    def _read_raw(self) -> Tuple[int, int]:
        """Read raw X/Y from XPT2046 via SPI (acquires shared bus lock)."""
        from display.tft_display import TFTDisplay
        with TFTDisplay.spi_lock:
            x_raw = self._spi.xfer2([CMD_READ_X, 0x00, 0x00])
            y_raw = self._spi.xfer2([CMD_READ_Y, 0x00, 0x00])
        x = ((x_raw[1] << 8) | x_raw[2]) >> 3
        y = ((y_raw[1] << 8) | y_raw[2]) >> 3
        return x, y

    def _raw_to_screen(self, raw_x: int, raw_y: int) -> Tuple[int, int]:
        """Convert raw touch coordinates to screen coordinates.

        Display is 480x320 landscape, rotated 180°.
        Raw X decreases top→bottom, Raw Y increases left→right.
        """
        # Clamp to calibration range
        rx = max(RAW_X_MIN, min(RAW_X_MAX, raw_x))
        ry = max(RAW_Y_MIN, min(RAW_Y_MAX, raw_y))

        # Map to screen coordinates
        screen_x = int((ry - RAW_Y_MIN) / (RAW_Y_MAX - RAW_Y_MIN) * (SCREEN_W - 1))
        screen_y = int((RAW_X_MAX - rx) / (RAW_X_MAX - RAW_X_MIN) * (SCREEN_H - 1))

        return max(0, min(SCREEN_W - 1, screen_x)), max(0, min(SCREEN_H - 1, screen_y))

    def _is_touching(self) -> bool:
        """Check PEN IRQ — active low when screen is touched."""
        from gpiod.line import Value
        return self._gpio.get_value(PIN_PEN_IRQ) == Value.INACTIVE

    def _dispatch(self, event: TouchEvent):
        """Send event to all registered callbacks."""
        logger.info("Touch event: {} at ({}, {})", event.value, self._last_x, self._last_y)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error("Touch callback error: {}", e)

    def _run(self):
        """Main polling loop — reads touch and detects tap patterns."""
        if not self._init_hardware():
            return

        poll_interval = 1.0 / POLL_HZ

        while self._running:
            try:
                now = time.monotonic()
                pressed = self._is_touching()

                # Read coordinates when touching
                if pressed:
                    raw_x, raw_y = self._read_raw()
                    if raw_x > TOUCH_THRESHOLD and raw_y > TOUCH_THRESHOLD:
                        self._last_x, self._last_y = self._raw_to_screen(raw_x, raw_y)

                # Debounce
                if pressed != self._is_pressed:
                    if (now - self._last_transition) < (DEBOUNCE_MS / 1000):
                        time.sleep(poll_interval)
                        continue
                    self._last_transition = now

                # ── Press started ──
                if pressed and not self._is_pressed:
                    self._is_pressed = True
                    self._press_start = now
                    self._long_press_fired = False
                    self._extra_long_fired = False

                # ── Currently held ──
                elif pressed and self._is_pressed:
                    hold_time = now - (self._press_start or now)

                    if hold_time >= EXTRA_LONG_THRESHOLD and not self._extra_long_fired:
                        self._extra_long_fired = True
                        self._long_press_fired = True
                        self._tap_count = 0
                        self._dispatch(TouchEvent.EXTRA_LONG)

                    elif hold_time >= LONG_PRESS_THRESHOLD and not self._long_press_fired:
                        self._long_press_fired = True
                        self._tap_count = 0
                        self._dispatch(TouchEvent.LONG_PRESS)

                # ── Released ──
                elif not pressed and self._is_pressed:
                    self._is_pressed = False
                    if not self._long_press_fired:
                        self._tap_count += 1
                        self._last_tap_time = now

                # ── Idle — check if tap window expired ──
                elif not pressed and not self._is_pressed:
                    if self._tap_count > 0:
                        if now - self._last_tap_time >= TAP_WINDOW:
                            if self._tap_count == 1:
                                self._dispatch(TouchEvent.TAP)
                            elif self._tap_count == 2:
                                self._dispatch(TouchEvent.DOUBLE_TAP)
                            elif self._tap_count >= 3:
                                self._dispatch(TouchEvent.TRIPLE_TAP)
                            self._tap_count = 0

                time.sleep(poll_interval)

            except Exception as e:
                logger.debug("Touch poll error: {}", e)
                time.sleep(0.1)

    def close(self):
        self.stop()
        if self._spi is not None:
            try:
                self._spi.close()
            except Exception:
                pass
        if self._gpio is not None:
            try:
                self._gpio.release()
            except Exception:
                pass
