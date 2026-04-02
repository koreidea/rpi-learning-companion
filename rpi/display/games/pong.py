"""Self-playing Pong game for the idle screensaver carousel."""

import math
import random
from dataclasses import dataclass, field

from .base import IdleGame, GameContext, clamp

# ─── Constants ───────────────────────────────────────────────────────────────
COURT_TOP = 28
COURT_BOTTOM = 310
PADDLE_LEFT_X = 20
PADDLE_RIGHT_X = 460
COLOR_BALL = (50, 205, 50)
COLOR_GLOW = (25, 100, 25)
COLOR_PADDLE_L = (30, 144, 255)
COLOR_PADDLE_R = (180, 100, 255)
COLOR_COURT = (30, 30, 35)
COLOR_SCORE = (55, 55, 60)
COLOR_BG = (0, 0, 0)


@dataclass
class PongState:
    """All state for the Pong game."""
    ball_x: float = 240.0
    ball_y: float = 160.0
    ball_vx: float = 150.0
    ball_vy: float = 90.0
    ball_radius: float = 6.0
    paddle_left_y: float = 160.0
    paddle_right_y: float = 160.0
    paddle_width: float = 8.0
    paddle_height: float = 50.0
    score_left: int = 0
    score_right: int = 0
    ai_left_target: float = 160.0
    ai_right_target: float = 160.0
    ai_left_timer: float = 0.0
    ai_right_timer: float = 0.0
    trail: list = field(default_factory=list)
    hit_flash_timer: float = 0.0
    hit_flash_side: str = ""
    score_flash_timer: float = 0.0
    serve_delay: float = 1.5


class PongGame(IdleGame):
    """Self-playing Pong — two AI paddles rally a ball."""

    def __init__(self):
        self._s = PongState()

    @property
    def name(self) -> str:
        return "Pong"

    def reset(self) -> None:
        self._s = PongState()
        self._reset_ball()

    def _reset_ball(self, serve_dir: int = 0):
        s = self._s
        s.ball_x = 240.0
        s.ball_y = float((COURT_TOP + COURT_BOTTOM) // 2)
        if serve_dir == 0:
            serve_dir = random.choice([-1, 1])
        speed = random.uniform(100, 130)
        angle = random.uniform(-0.4, 0.4)
        s.ball_vx = serve_dir * speed * math.cos(angle)
        s.ball_vy = speed * math.sin(angle)
        s.serve_delay = 1.2
        s.score_flash_timer = 0.8
        s.trail.clear()
        if s.score_left >= 5 or s.score_right >= 5:
            s.score_left = 0
            s.score_right = 0

    def _update_ai(self, dt: float):
        s = self._s
        paddle_speed = 140.0
        half_h = s.paddle_height / 2

        s.ai_left_timer -= dt
        if s.ai_left_timer <= 0:
            s.ai_left_timer = random.uniform(0.25, 0.65)
            if s.ball_vx < 0:
                s.ai_left_target = s.ball_y + random.uniform(-18, 18)
            else:
                s.ai_left_target = 160 + random.uniform(-25, 25)
        diff = s.ai_left_target - s.paddle_left_y
        s.paddle_left_y += clamp(diff, -paddle_speed * dt, paddle_speed * dt)
        s.paddle_left_y = clamp(s.paddle_left_y,
                                COURT_TOP + half_h, COURT_BOTTOM - half_h)

        s.ai_right_timer -= dt
        if s.ai_right_timer <= 0:
            s.ai_right_timer = random.uniform(0.25, 0.65)
            if s.ball_vx > 0:
                s.ai_right_target = s.ball_y + random.uniform(-18, 18)
            else:
                s.ai_right_target = 160 + random.uniform(-25, 25)
        diff = s.ai_right_target - s.paddle_right_y
        s.paddle_right_y += clamp(diff, -paddle_speed * dt, paddle_speed * dt)
        s.paddle_right_y = clamp(s.paddle_right_y,
                                 COURT_TOP + half_h, COURT_BOTTOM - half_h)

    def update(self, dt: float) -> None:
        s = self._s

        if s.serve_delay > 0:
            s.serve_delay -= dt
            self._update_ai(dt)
            return

        s.ball_x += s.ball_vx * dt
        s.ball_y += s.ball_vy * dt

        # Wall bounce
        if s.ball_y - s.ball_radius <= COURT_TOP:
            s.ball_y = COURT_TOP + s.ball_radius
            s.ball_vy = abs(s.ball_vy)
        elif s.ball_y + s.ball_radius >= COURT_BOTTOM:
            s.ball_y = COURT_BOTTOM - s.ball_radius
            s.ball_vy = -abs(s.ball_vy)

        half_h = s.paddle_height / 2
        half_w = s.paddle_width / 2

        # Left paddle collision
        if (s.ball_vx < 0
                and s.ball_x - s.ball_radius <= PADDLE_LEFT_X + half_w
                and s.ball_x - s.ball_radius >= PADDLE_LEFT_X - half_w - 6
                and abs(s.ball_y - s.paddle_left_y) < half_h + s.ball_radius):
            s.ball_x = PADDLE_LEFT_X + half_w + s.ball_radius
            s.ball_vx = abs(s.ball_vx)
            offset = (s.ball_y - s.paddle_left_y) / half_h
            s.ball_vy += offset * 55
            speed = math.sqrt(s.ball_vx**2 + s.ball_vy**2)
            if speed < 220:
                s.ball_vx *= 1.03
                s.ball_vy *= 1.03
            s.hit_flash_timer = 0.15
            s.hit_flash_side = "left"

        # Right paddle collision
        if (s.ball_vx > 0
                and s.ball_x + s.ball_radius >= PADDLE_RIGHT_X - half_w
                and s.ball_x + s.ball_radius <= PADDLE_RIGHT_X + half_w + 6
                and abs(s.ball_y - s.paddle_right_y) < half_h + s.ball_radius):
            s.ball_x = PADDLE_RIGHT_X - half_w - s.ball_radius
            s.ball_vx = -abs(s.ball_vx)
            offset = (s.ball_y - s.paddle_right_y) / half_h
            s.ball_vy += offset * 55
            speed = math.sqrt(s.ball_vx**2 + s.ball_vy**2)
            if speed < 220:
                s.ball_vx *= 1.03
                s.ball_vy *= 1.03
            s.hit_flash_timer = 0.15
            s.hit_flash_side = "right"

        # Scoring
        if s.ball_x < 0:
            s.score_right += 1
            self._reset_ball(serve_dir=1)
        elif s.ball_x > 480:
            s.score_left += 1
            self._reset_ball(serve_dir=-1)

        s.ball_vx = clamp(s.ball_vx, -250, 250)
        s.ball_vy = clamp(s.ball_vy, -250, 250)

        # Trail
        s.trail.append((s.ball_x, s.ball_y, 0.0))
        s.trail = [(x, y, a + dt) for x, y, a in s.trail if a + dt < 0.3]
        if len(s.trail) > 12:
            s.trail = s.trail[-12:]

        if s.hit_flash_timer > 0:
            s.hit_flash_timer -= dt
        if s.score_flash_timer > 0:
            s.score_flash_timer -= dt

        self._update_ai(dt)

    def render(self, ctx: GameContext) -> None:
        s = self._s
        d = ctx.draw
        W, H = ctx.width, ctx.height
        CX = W // 2

        d.rectangle([0, 0, W, H], fill=COLOR_BG)

        # Court lines
        d.line([0, COURT_TOP, W, COURT_TOP], fill=COLOR_COURT, width=1)
        d.line([0, COURT_BOTTOM, W, COURT_BOTTOM], fill=COLOR_COURT, width=1)
        for i in range(7):
            y = COURT_TOP + 8 + i * 31
            d.line([CX, y, CX, y + 18], fill=COLOR_COURT, width=2)

        # Score
        sc = COLOR_SCORE
        if s.score_flash_timer > 0:
            f = s.score_flash_timer / 0.8
            v = int(55 + 200 * f)
            sc = (v, v, v)
        d.text((CX - 38, 3), f"{s.score_left}", fill=sc, font=ctx.font_large)
        d.text((CX + 22, 3), f"{s.score_right}", fill=sc, font=ctx.font_large)

        # Ball trail
        for tx, ty, age in s.trail:
            alpha = max(0, 1.0 - age / 0.3)
            r = int(s.ball_radius * alpha * 0.7)
            if r >= 1:
                tc = (int(COLOR_GLOW[0] * alpha),
                      int(COLOR_GLOW[1] * alpha),
                      int(COLOR_GLOW[2] * alpha))
                d.ellipse([int(tx) - r, int(ty) - r,
                           int(tx) + r, int(ty) + r], fill=tc)

        # Ball
        bx, by = int(s.ball_x), int(s.ball_y)
        br = int(s.ball_radius)
        gr = br + 3
        d.ellipse([bx - gr, by - gr, bx + gr, by + gr], fill=COLOR_GLOW)
        d.ellipse([bx - br, by - br, bx + br, by + br], fill=COLOR_BALL)
        d.ellipse([bx - 2, by - 2, bx, by], fill=(150, 255, 150))

        # Paddles
        phh = int(s.paddle_height / 2)
        phw = int(s.paddle_width / 2)

        lx, ly = PADDLE_LEFT_X, int(s.paddle_left_y)
        lc = COLOR_PADDLE_L
        if s.hit_flash_timer > 0 and s.hit_flash_side == "left":
            f = s.hit_flash_timer / 0.15
            lc = (min(255, int(30 + 225 * f)),
                  min(255, int(144 + 111 * f)), 255)
        d.rounded_rectangle([lx - phw, ly - phh, lx + phw, ly + phh],
                            radius=3, fill=lc)
        gl = (lc[0] // 4, lc[1] // 4, lc[2] // 4)
        d.rounded_rectangle([lx - phw - 2, ly - phh - 2, lx + phw + 2, ly + phh + 2],
                            radius=4, outline=gl, width=1)

        rx, ry = PADDLE_RIGHT_X, int(s.paddle_right_y)
        rc = COLOR_PADDLE_R
        if s.hit_flash_timer > 0 and s.hit_flash_side == "right":
            f = s.hit_flash_timer / 0.15
            rc = (min(255, int(180 + 75 * f)),
                  min(255, int(100 + 155 * f)), 255)
        d.rounded_rectangle([rx - phw, ry - phh, rx + phw, ry + phh],
                            radius=3, fill=rc)
        gr_r = (rc[0] // 4, rc[1] // 4, rc[2] // 4)
        d.rounded_rectangle([rx - phw - 2, ry - phh - 2, rx + phw + 2, ry + phh + 2],
                            radius=4, outline=gr_r, width=1)

        # Serve delay pulse
        if s.serve_delay > 0:
            pulse = abs(math.sin(s.serve_delay * 5))
            pr = int(3 + 4 * pulse)
            pc = (int(70 * pulse), int(70 * pulse), int(70 * pulse))
            CY = H // 2
            d.ellipse([CX - pr, CY - pr, CX + pr, CY + pr], fill=pc)
