"""SG90 servo controller for bot hands.

Hardware wiring:
  Left hand:  Signal → BCM12 (Pin 32), VCC → 5V, GND → GND
  Right hand: Signal → BCM13 (Pin 33), VCC → 5V, GND → GND

Both BCM12 and BCM13 support hardware PWM (PWM0/PWM1).
SG90 expects 50Hz PWM: 2.5% duty = 0°, 7.5% = 90°, 12.5% = 180°.
"""

import math
import random
import threading
import time

from loguru import logger

GPIO_CHIP = 4
PIN_LEFT = 12       # BCM12 — PWM0
PIN_RIGHT = 13      # BCM13 — PWM1

# SG90 PWM parameters
PWM_FREQ = 50       # 50Hz
# Duty cycle range (in %) for 0° to 90°
DUTY_MIN = 2.5      # 0°
DUTY_MAX = 7.5      # 90°
MAX_ANGLE = 90      # Maximum rotation


def _angle_to_duty(angle: float) -> float:
    """Convert angle (0-90) to duty cycle percentage."""
    angle = max(0, min(MAX_ANGLE, angle))
    return DUTY_MIN + (angle / MAX_ANGLE) * (DUTY_MAX - DUTY_MIN)


class ServoController:
    """Controls two SG90 servos for left and right bot hands."""

    def __init__(self):
        self._gpio = None
        self._available = False
        self._speaking = False
        self._speak_thread = None

    def start(self):
        """Initialize PWM on both servo pins."""
        try:
            import lgpio

            self._gpio = lgpio.gpiochip_open(GPIO_CHIP)
            lgpio.gpio_claim_output(self._gpio, PIN_LEFT)
            lgpio.gpio_claim_output(self._gpio, PIN_RIGHT)

            # Start PWM at rest position (left=90° mirrored, right=0°)
            lgpio.tx_pwm(self._gpio, PIN_LEFT, PWM_FREQ, _angle_to_duty(MAX_ANGLE))
            lgpio.tx_pwm(self._gpio, PIN_RIGHT, PWM_FREQ, _angle_to_duty(0))

            # Brief pause then stop PWM to avoid jitter at rest
            time.sleep(0.5)
            lgpio.tx_pwm(self._gpio, PIN_LEFT, 0, 0)
            lgpio.tx_pwm(self._gpio, PIN_RIGHT, 0, 0)

            self._available = True
            logger.info("Servo controller initialized (BCM{}, BCM{})", PIN_LEFT, PIN_RIGHT)

        except Exception as e:
            logger.warning("Servo init failed: {}. Running without servos.", e)
            self._available = False

    def stop(self):
        """Stop PWM and release GPIO."""
        if self._gpio is not None:
            try:
                import lgpio
                lgpio.tx_pwm(self._gpio, PIN_LEFT, 0, 0)
                lgpio.tx_pwm(self._gpio, PIN_RIGHT, 0, 0)
                lgpio.gpiochip_close(self._gpio)
            except Exception:
                pass
        self._available = False

    def set_angle(self, left: float, right: float):
        """Set both servos to specific angles (0-90)."""
        if not self._available:
            return
        try:
            import lgpio
            lgpio.tx_pwm(self._gpio, PIN_LEFT, PWM_FREQ, _angle_to_duty(left))
            lgpio.tx_pwm(self._gpio, PIN_RIGHT, PWM_FREQ, _angle_to_duty(right))
        except Exception as e:
            logger.debug("Servo set_angle error: {}", e)

    def rest(self):
        """Return to rest position (0°) then stop PWM."""
        if not self._available:
            return
        try:
            import lgpio
            lgpio.tx_pwm(self._gpio, PIN_LEFT, PWM_FREQ, _angle_to_duty(MAX_ANGLE))
            lgpio.tx_pwm(self._gpio, PIN_RIGHT, PWM_FREQ, _angle_to_duty(0))
            time.sleep(0.4)
            lgpio.tx_pwm(self._gpio, PIN_LEFT, 0, 0)
            lgpio.tx_pwm(self._gpio, PIN_RIGHT, 0, 0)
        except Exception as e:
            logger.debug("Servo rest error: {}", e)

    def tickle_wave(self):
        """Animated wave during tickle — runs in background thread."""
        if not self._available:
            return
        threading.Thread(target=self._tickle_animation, daemon=True).start()

    def _tickle_animation(self):
        """Sweep both hands 0° → 90° → 0° with a wiggle, matching tickle duration."""
        import math
        try:
            import lgpio
            logger.info("Tickle animation started")

            duration = 2.5  # Match TICKLE_DURATION
            start = time.time()

            while time.time() - start < duration:
                t = time.time() - start
                # Smooth sine wave: 0 → 90 → 0 over duration
                base = math.sin(math.pi * t / duration) * 90
                # Add quick wiggle oscillation
                wiggle = math.sin(t * 12) * 10 * math.sin(math.pi * t / duration)

                right_angle = max(0, min(MAX_ANGLE, base + wiggle))
                left_angle = MAX_ANGLE - max(0, min(MAX_ANGLE, base - wiggle))  # Mirrored

                lgpio.tx_pwm(self._gpio, PIN_LEFT, PWM_FREQ, _angle_to_duty(left_angle))
                lgpio.tx_pwm(self._gpio, PIN_RIGHT, PWM_FREQ, _angle_to_duty(right_angle))
                time.sleep(0.02)

            # Return to rest
            self.rest()

        except Exception as e:
            logger.warning("Tickle animation error: {}", e)

    # --- Speaking gestures (natural hand movement while talking) ---

    def start_speaking_gestures(self):
        """Start natural hand gestures while the bot speaks."""
        if not self._available or self._speaking:
            return
        self._speaking = True
        self._speak_thread = threading.Thread(target=self._speaking_loop, daemon=True)
        self._speak_thread.start()
        logger.info("Speaking gestures started")

    def stop_speaking_gestures(self):
        """Stop speaking gestures and return to rest."""
        if not self._speaking:
            return
        self._speaking = False
        if self._speak_thread:
            self._speak_thread.join(timeout=2.0)
            self._speak_thread = None
        self.rest()
        logger.info("Speaking gestures stopped")

    def _speaking_loop(self):
        """Natural hand gestures like a person explaining.

        Cycles through random gesture patterns:
          - Both hands rise together (emphasis)
          - Alternating hands (explaining left/right)
          - Small nods (thinking)
          - Wide open (excitement)
          - One hand gesture (pointing/directing)
        Each gesture transitions smoothly with easing.
        """
        try:
            import lgpio

            while self._speaking:
                gesture = random.choice([
                    self._gesture_emphasis,
                    self._gesture_alternate,
                    self._gesture_nod,
                    self._gesture_wide_open,
                    self._gesture_one_hand,
                    self._gesture_wave_explain,
                ])
                gesture()

                # Brief pause between gestures
                if self._speaking:
                    time.sleep(random.uniform(0.2, 0.5))

            self.rest()

        except Exception as e:
            logger.warning("Speaking gesture error: {}", e)

    def _smooth_move(self, left_start, left_end, right_start, right_end, duration=0.6):
        """Smoothly transition both servos using ease-in-out."""
        import lgpio
        steps = int(duration / 0.02)
        for i in range(steps + 1):
            if not self._speaking:
                return
            t = i / steps
            # Ease-in-out (smooth sine curve)
            ease = (1 - math.cos(t * math.pi)) / 2

            left = left_start + (left_end - left_start) * ease
            right = right_start + (right_end - right_start) * ease

            left_duty = _angle_to_duty(MAX_ANGLE - left)   # Mirrored
            right_duty = _angle_to_duty(right)

            lgpio.tx_pwm(self._gpio, PIN_LEFT, PWM_FREQ, left_duty)
            lgpio.tx_pwm(self._gpio, PIN_RIGHT, PWM_FREQ, right_duty)
            time.sleep(0.02)

    def _gesture_emphasis(self):
        """Both hands rise together — used for emphasis."""
        angle = random.uniform(30, 60)
        dur = random.uniform(0.4, 0.7)
        self._smooth_move(0, angle, 0, angle, dur)
        time.sleep(random.uniform(0.2, 0.5))
        self._smooth_move(angle, 0, angle, 0, dur)

    def _gesture_alternate(self):
        """Alternating hands — like explaining two sides."""
        angle = random.uniform(30, 55)
        dur = random.uniform(0.3, 0.5)
        # Left up, right stays
        self._smooth_move(0, angle, 0, 10, dur)
        time.sleep(0.2)
        # Switch — right up, left down
        self._smooth_move(angle, 10, 10, angle, dur)
        time.sleep(0.2)
        # Both back down
        self._smooth_move(10, 0, angle, 0, dur)

    def _gesture_nod(self):
        """Small subtle nods — thinking/agreeing."""
        for _ in range(random.randint(2, 4)):
            if not self._speaking:
                return
            angle = random.uniform(15, 30)
            self._smooth_move(0, angle, 0, angle, 0.25)
            self._smooth_move(angle, 0, angle, 0, 0.25)

    def _gesture_wide_open(self):
        """Both hands spread wide — excitement/big idea."""
        self._smooth_move(0, 70, 0, 70, 0.5)
        time.sleep(random.uniform(0.3, 0.6))
        self._smooth_move(70, 0, 70, 0, 0.5)

    def _gesture_one_hand(self):
        """One hand gestures while other stays low — directing."""
        hand = random.choice(["left", "right"])
        angle = random.uniform(35, 65)
        dur = random.uniform(0.4, 0.6)

        if hand == "left":
            self._smooth_move(0, angle, 0, 5, dur)
            time.sleep(random.uniform(0.3, 0.5))
            self._smooth_move(angle, 0, 5, 0, dur)
        else:
            self._smooth_move(0, 5, 0, angle, dur)
            time.sleep(random.uniform(0.3, 0.5))
            self._smooth_move(5, 0, angle, 0, dur)

    # --- Dance animation (synchronized with car dance) ---

    def dance(self):
        """Energetic hand dance — runs in background thread, ~5.5s to match car dance."""
        if not self._available:
            logger.warning("Servo dance skipped — servos not available")
            return
        logger.info("Starting servo dance thread")
        threading.Thread(target=self._dance_animation, daemon=True).start()

    def _dance_animation(self):
        """Synchronized hand movements matching the car dance sequence.

        Timeline matches car.dance():
          0.0-1.6s  Wiggles (4× left-right spin)  → Alternating hand pumps
          1.6-2.8s  Rock (4× fwd-back)             → Both hands wave up-down
          2.8-3.6s  Fast spin                       → Windmill (one up one down rotating)
          3.6-4.8s  More wiggles                    → Rapid clapping motion
          4.8-6.0s  Finale spin                     → Triumphant both arms high + wave
        """
        try:
            import lgpio
            logger.info("Dance hand animation started")

            # Phase 1: Alternating hand pumps (match wiggles) ~1.6s
            for _ in range(4):
                self._smooth_dance_move(0, 70, 70, 0, 0.2)
                self._smooth_dance_move(70, 0, 0, 70, 0.2)

            # Phase 2: Both hands wave up-down (match rock) ~1.2s
            for _ in range(4):
                self._smooth_dance_move(0, 60, 0, 60, 0.15)
                self._smooth_dance_move(60, 0, 60, 0, 0.15)

            # Phase 3: Windmill — one up while other down (match fast spin) ~0.8s
            for _ in range(4):
                self._smooth_dance_move(0, 90, 90, 0, 0.1)
                self._smooth_dance_move(90, 0, 0, 90, 0.1)

            # Phase 4: Rapid symmetric pumps (match more wiggles) ~1.2s
            for _ in range(6):
                self._smooth_dance_move(0, 50, 0, 50, 0.1)
                self._smooth_dance_move(50, 0, 50, 0, 0.1)

            # Phase 5: Triumphant — both arms high + wave (match finale spin) ~1.2s
            self._smooth_dance_move(0, 90, 0, 90, 0.3)
            for _ in range(3):
                self._smooth_dance_move(90, 60, 90, 60, 0.15)
                self._smooth_dance_move(60, 90, 60, 90, 0.15)
            self._smooth_dance_move(90, 0, 90, 0, 0.3)

            self.rest()
            logger.info("Dance hand animation finished")

        except Exception as e:
            logger.warning("Dance animation error: {}", e)

    def _smooth_dance_move(self, left_start, left_end, right_start, right_end, duration=0.3):
        """Fast smooth transition for dance moves (no speaking check)."""
        import lgpio
        steps = max(1, int(duration / 0.02))
        for i in range(steps + 1):
            t = i / steps
            ease = (1 - math.cos(t * math.pi)) / 2

            left = left_start + (left_end - left_start) * ease
            right = right_start + (right_end - right_start) * ease

            lgpio.tx_pwm(self._gpio, PIN_LEFT, PWM_FREQ, _angle_to_duty(MAX_ANGLE - left))
            lgpio.tx_pwm(self._gpio, PIN_RIGHT, PWM_FREQ, _angle_to_duty(right))
            time.sleep(0.02)

    def _gesture_wave_explain(self):
        """Flowing wave motion — like explaining a process."""
        dur = random.uniform(0.3, 0.4)
        for angle in [20, 45, 30, 55, 20, 0]:
            if not self._speaking:
                return
            other = max(0, 50 - angle)
            self._smooth_move(angle, angle, other, other, dur)
