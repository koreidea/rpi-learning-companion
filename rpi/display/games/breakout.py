"""Self-playing Breakout game for the idle screensaver carousel."""

import math
import random
from dataclasses import dataclass, field

from .base import IdleGame, GameContext, clamp

# ─── Constants ───────────────────────────────────────────────────────────────
COLOR_BG = (0, 0, 0)
COLOR_BALL = (50, 205, 50)
COLOR_BALL_GLOW = (25, 100, 25)
COLOR_PADDLE = (30, 144, 255)
COLOR_PADDLE_FLASH = (100, 200, 255)
COLOR_BORDER = (30, 30, 35)
COLOR_SCORE = (55, 55, 60)
COLOR_LIFE = (255, 60, 60)

# Brick row colors (top to bottom)
BRICK_COLORS = [
    (255, 60, 60),     # Red
    (255, 200, 50),    # Yellow
    (50, 205, 50),     # Green
    (30, 144, 255),    # Blue
    (180, 100, 255),   # Purple
]

# Layout
BRICK_ROWS = 5
BRICK_COLS = 10
BRICK_W = 44
BRICK_H = 12
BRICK_GAP = 2
BRICK_AREA_TOP = 35
BRICK_OFFSET_X = (480 - (BRICK_COLS * (BRICK_W + BRICK_GAP) - BRICK_GAP)) // 2

PADDLE_Y = 300
PADDLE_W = 60
PADDLE_H = 8
BALL_R = 5


@dataclass
class Brick:
    x: int = 0
    y: int = 0
    color: tuple = (255, 255, 255)
    alive: bool = True


@dataclass
class Particle:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    life: float = 0.0
    color: tuple = (255, 255, 255)


@dataclass
class BreakoutState:
    ball_x: float = 240.0
    ball_y: float = 270.0
    ball_vx: float = 120.0
    ball_vy: float = -150.0
    paddle_x: float = 240.0
    bricks: list = field(default_factory=list)
    particles: list = field(default_factory=list)
    trail: list = field(default_factory=list)
    score: int = 0
    lives: int = 3
    serve_delay: float = 1.0
    hit_flash: float = 0.0
    ai_target: float = 240.0
    ai_timer: float = 0.0


class BreakoutGame(IdleGame):
    """Self-playing Breakout — AI paddle breaks bricks with a bouncing ball."""

    def __init__(self):
        self._s = BreakoutState()

    @property
    def name(self) -> str:
        return "Breakout"

    def reset(self) -> None:
        self._s = BreakoutState()
        self._init_bricks()
        self._serve()

    def _init_bricks(self):
        s = self._s
        s.bricks.clear()
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                bx = BRICK_OFFSET_X + col * (BRICK_W + BRICK_GAP)
                by = BRICK_AREA_TOP + row * (BRICK_H + BRICK_GAP)
                s.bricks.append(Brick(x=bx, y=by, color=BRICK_COLORS[row]))

    def _serve(self):
        s = self._s
        s.ball_x = s.paddle_x
        s.ball_y = PADDLE_Y - 12
        angle = random.uniform(-0.6, 0.6)
        speed = random.uniform(120, 150)
        s.ball_vx = speed * math.sin(angle)
        s.ball_vy = -speed * math.cos(angle)
        s.serve_delay = 1.0
        s.trail.clear()

    def _spawn_particles(self, x, y, color, count=4):
        s = self._s
        for _ in range(count):
            s.particles.append(Particle(
                x=x, y=y,
                vx=random.uniform(-80, 80),
                vy=random.uniform(-80, 80),
                life=random.uniform(0.2, 0.4),
                color=color,
            ))

    def update(self, dt: float) -> None:
        s = self._s

        # Update particles
        s.particles = [p for p in s.particles if p.life > 0]
        for p in s.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt

        # Serve delay
        if s.serve_delay > 0:
            s.serve_delay -= dt
            return

        # AI paddle
        s.ai_timer -= dt
        if s.ai_timer <= 0:
            s.ai_timer = random.uniform(0.15, 0.35)
            # Project where ball will reach paddle Y
            if s.ball_vy > 0:
                time_to_paddle = (PADDLE_Y - s.ball_y) / max(s.ball_vy, 1)
                projected_x = s.ball_x + s.ball_vx * time_to_paddle
                s.ai_target = projected_x + random.uniform(-12, 12)
            else:
                s.ai_target = 240 + random.uniform(-20, 20)
        diff = s.ai_target - s.paddle_x
        s.paddle_x += clamp(diff, -200 * dt, 200 * dt)
        s.paddle_x = clamp(s.paddle_x, PADDLE_W // 2 + 4, 476 - PADDLE_W // 2)

        # Ball movement
        s.ball_x += s.ball_vx * dt
        s.ball_y += s.ball_vy * dt

        # Wall bounce (left/right)
        if s.ball_x - BALL_R < 0:
            s.ball_x = BALL_R
            s.ball_vx = abs(s.ball_vx)
        elif s.ball_x + BALL_R > 480:
            s.ball_x = 480 - BALL_R
            s.ball_vx = -abs(s.ball_vx)

        # Top wall bounce
        if s.ball_y - BALL_R < 16:
            s.ball_y = 16 + BALL_R
            s.ball_vy = abs(s.ball_vy)

        # Paddle collision
        if (s.ball_vy > 0
                and s.ball_y + BALL_R >= PADDLE_Y
                and s.ball_y + BALL_R <= PADDLE_Y + PADDLE_H + 6
                and abs(s.ball_x - s.paddle_x) < PADDLE_W // 2 + BALL_R):
            s.ball_y = PADDLE_Y - BALL_R
            offset = (s.ball_x - s.paddle_x) / (PADDLE_W // 2)
            speed = math.sqrt(s.ball_vx**2 + s.ball_vy**2)
            speed = min(speed * 1.02, 250)
            angle = offset * 1.0  # ±1 radian range
            s.ball_vx = speed * math.sin(angle)
            s.ball_vy = -speed * math.cos(angle)
            s.hit_flash = 0.1

        # Brick collision
        for brick in s.bricks:
            if not brick.alive:
                continue
            # Simple AABB check
            if (s.ball_x + BALL_R > brick.x
                    and s.ball_x - BALL_R < brick.x + BRICK_W
                    and s.ball_y + BALL_R > brick.y
                    and s.ball_y - BALL_R < brick.y + BRICK_H):
                brick.alive = False
                s.score += 10
                self._spawn_particles(
                    brick.x + BRICK_W // 2, brick.y + BRICK_H // 2,
                    brick.color)
                # Determine bounce direction
                dx_left = abs((s.ball_x + BALL_R) - brick.x)
                dx_right = abs((s.ball_x - BALL_R) - (brick.x + BRICK_W))
                dy_top = abs((s.ball_y + BALL_R) - brick.y)
                dy_bottom = abs((s.ball_y - BALL_R) - (brick.y + BRICK_H))
                min_d = min(dx_left, dx_right, dy_top, dy_bottom)
                if min_d in (dx_left, dx_right):
                    s.ball_vx = -s.ball_vx
                else:
                    s.ball_vy = -s.ball_vy
                break  # One brick per frame

        # Ball lost
        if s.ball_y > 320:
            s.lives -= 1
            if s.lives <= 0:
                self.reset()
                return
            self._serve()

        # All bricks cleared
        if not any(b.alive for b in s.bricks):
            self._init_bricks()
            self._serve()

        # Trail
        s.trail.append((s.ball_x, s.ball_y, 0.0))
        s.trail = [(x, y, a + dt) for x, y, a in s.trail if a + dt < 0.25]
        if len(s.trail) > 10:
            s.trail = s.trail[-10:]

        if s.hit_flash > 0:
            s.hit_flash -= dt

    def render(self, ctx: GameContext) -> None:
        s = self._s
        d = ctx.draw
        W, H = ctx.width, ctx.height

        d.rectangle([0, 0, W, H], fill=COLOR_BG)

        # Score and lives
        d.text((6, 2), f"Score: {s.score}", fill=COLOR_SCORE, font=ctx.font_small)
        for i in range(s.lives):
            cx = W - 16 - i * 14
            d.ellipse([cx - 4, 4, cx + 4, 12], fill=COLOR_LIFE)

        # Border
        d.rectangle([0, 16, W - 1, H - 1], outline=COLOR_BORDER, width=1)

        # Bricks
        for brick in s.bricks:
            if not brick.alive:
                continue
            d.rounded_rectangle(
                [brick.x, brick.y, brick.x + BRICK_W, brick.y + BRICK_H],
                radius=2, fill=brick.color)
            # Highlight on top edge
            hl = (min(255, brick.color[0] + 60),
                  min(255, brick.color[1] + 60),
                  min(255, brick.color[2] + 60))
            d.line([brick.x + 2, brick.y + 1, brick.x + BRICK_W - 2, brick.y + 1],
                   fill=hl, width=1)

        # Particles
        for p in s.particles:
            alpha = max(0, p.life / 0.4)
            pc = (int(p.color[0] * alpha),
                  int(p.color[1] * alpha),
                  int(p.color[2] * alpha))
            r = int(2 * alpha)
            if r >= 1:
                d.ellipse([int(p.x) - r, int(p.y) - r,
                           int(p.x) + r, int(p.y) + r], fill=pc)

        # Ball trail
        for tx, ty, age in s.trail:
            alpha = max(0, 1.0 - age / 0.25)
            r = int(BALL_R * alpha * 0.6)
            if r >= 1:
                tc = (int(COLOR_BALL_GLOW[0] * alpha),
                      int(COLOR_BALL_GLOW[1] * alpha),
                      int(COLOR_BALL_GLOW[2] * alpha))
                d.ellipse([int(tx) - r, int(ty) - r,
                           int(tx) + r, int(ty) + r], fill=tc)

        # Ball
        bx, by = int(s.ball_x), int(s.ball_y)
        gr = BALL_R + 2
        d.ellipse([bx - gr, by - gr, bx + gr, by + gr], fill=COLOR_BALL_GLOW)
        d.ellipse([bx - BALL_R, by - BALL_R, bx + BALL_R, by + BALL_R],
                  fill=COLOR_BALL)
        d.ellipse([bx - 1, by - 1, bx + 1, by + 1], fill=(150, 255, 150))

        # Paddle
        pc = COLOR_PADDLE_FLASH if s.hit_flash > 0 else COLOR_PADDLE
        px = int(s.paddle_x)
        hw = PADDLE_W // 2
        d.rounded_rectangle(
            [px - hw, PADDLE_Y, px + hw, PADDLE_Y + PADDLE_H],
            radius=3, fill=pc)
        # Highlight
        d.line([px - hw + 2, PADDLE_Y + 1, px + hw - 2, PADDLE_Y + 1],
               fill=(min(255, pc[0] + 50), min(255, pc[1] + 50), 255),
               width=1)

        # Serve delay pulse
        if s.serve_delay > 0:
            pulse = abs(math.sin(s.serve_delay * 5))
            pr = int(3 + 3 * pulse)
            pc_d = (int(50 * pulse), int(50 * pulse), int(50 * pulse))
            d.ellipse([240 - pr, 160 - pr, 240 + pr, 160 + pr], fill=pc_d)
