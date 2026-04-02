"""Self-playing Space Invaders game for the idle screensaver carousel."""

import math
import random
from dataclasses import dataclass, field

from .base import IdleGame, GameContext, clamp

# ─── Constants ───────────────────────────────────────────────────────────────
ALIEN_COLS = 10
ALIEN_ROWS = 4
ALIEN_W = 20
ALIEN_H = 14
ALIEN_SPACE_X = 40
ALIEN_SPACE_Y = 28
ALIEN_START_Y = 38
ALIEN_START_X = 30

SHIP_Y = 300
SHIP_W = 22
SHIP_H = 12

BULLET_W = 2
BULLET_H = 6
BULLET_SPEED_PLAYER = 200
BULLET_SPEED_ALIEN = 100

COLOR_BG = (0, 0, 0)
COLOR_SCORE = (55, 55, 60)
COLOR_BORDER = (30, 30, 35)
COLOR_SHIP = (50, 205, 50)
COLOR_SHIP_BRIGHT = (100, 255, 100)
COLOR_BULLET_PLAYER = (100, 255, 100)
COLOR_BULLET_ALIEN = (255, 80, 80)
COLOR_LIFE = (50, 205, 50)

# Alien colors by row type
ALIEN_COLORS = [
    (180, 100, 255),   # Top row — purple
    (30, 144, 255),    # Mid rows — blue
    (30, 144, 255),
    (50, 205, 50),     # Bottom row — green
]

# Alien point values
ALIEN_POINTS = [40, 20, 20, 10]


@dataclass
class Explosion:
    x: float = 0.0
    y: float = 0.0
    life: float = 0.0
    color: tuple = (255, 255, 255)
    particles: list = field(default_factory=list)  # [(dx, dy, speed)]


@dataclass
class InvaderState:
    # Alien grid: True = alive
    aliens: list = field(default_factory=list)  # [row][col] = alive
    alien_x: float = 0.0          # Formation X offset
    alien_y: float = 0.0          # Formation Y offset
    alien_dir: int = 1            # +1 right, -1 left
    alien_move_timer: float = 0.0
    alien_move_interval: float = 0.55
    alien_frame: int = 0          # 0 or 1 for animation

    # Ship
    ship_x: float = 240.0

    # Bullets
    player_bullets: list = field(default_factory=list)  # [(x, y)]
    alien_bullets: list = field(default_factory=list)    # [(x, y)]

    # AI
    ai_target_x: float = 240.0
    ai_shoot_timer: float = 0.3
    ai_retarget_timer: float = 0.0

    # State
    score: int = 0
    lives: int = 3
    wave: int = 1
    explosions: list = field(default_factory=list)
    wave_clear_timer: float = 0.0
    death_timer: float = 0.0


class SpaceInvadersGame(IdleGame):
    """Self-playing Space Invaders — AI ship defends against descending aliens."""

    def __init__(self):
        self._s = InvaderState()

    @property
    def name(self) -> str:
        return "Invaders"

    def reset(self) -> None:
        self._s = InvaderState()
        self._s.wave = 1
        self._init_aliens()

    def _init_aliens(self):
        s = self._s
        s.aliens = [[True] * ALIEN_COLS for _ in range(ALIEN_ROWS)]
        s.alien_x = 0.0
        s.alien_y = 0.0
        s.alien_dir = 1
        s.alien_move_timer = 0.0
        s.alien_move_interval = max(0.2, 0.55 - (s.wave - 1) * 0.05)
        s.alien_frame = 0
        s.player_bullets.clear()
        s.alien_bullets.clear()

    def _alien_pos(self, row, col):
        """Get pixel position of alien center."""
        s = self._s
        x = ALIEN_START_X + col * ALIEN_SPACE_X + s.alien_x + ALIEN_W // 2
        y = ALIEN_START_Y + row * ALIEN_SPACE_Y + s.alien_y + ALIEN_H // 2
        return x, y

    def _alive_count(self):
        return sum(1 for row in self._s.aliens for a in row if a)

    def _spawn_explosion(self, x, y, color):
        parts = []
        for _ in range(6):
            angle = random.uniform(0, 6.28)
            speed = random.uniform(30, 80)
            parts.append((math.cos(angle), math.sin(angle), speed))
        self._s.explosions.append(Explosion(
            x=x, y=y, life=0.35, color=color, particles=parts))

    def _ai_find_target(self):
        """Find the most threatening column to target."""
        s = self._s
        # Find the lowest alive alien in each column
        lowest = {}
        for col in range(ALIEN_COLS):
            for row in range(ALIEN_ROWS - 1, -1, -1):
                if s.aliens[row][col]:
                    lowest[col] = row
                    break

        if not lowest:
            return s.ship_x

        # 85% target most threatening (lowest), 15% random
        if random.random() < 0.15 and len(lowest) > 1:
            col = random.choice(list(lowest.keys()))
        else:
            col = max(lowest, key=lowest.get)

        x, _ = self._alien_pos(lowest[col], col)
        return x + random.uniform(-8, 8)

    def update(self, dt: float) -> None:
        s = self._s

        # Update explosions
        s.explosions = [e for e in s.explosions if e.life > 0]
        for e in s.explosions:
            e.life -= dt

        # Wave clear pause
        if s.wave_clear_timer > 0:
            s.wave_clear_timer -= dt
            if s.wave_clear_timer <= 0:
                s.wave += 1
                self._init_aliens()
            return

        # Death pause
        if s.death_timer > 0:
            s.death_timer -= dt
            if s.death_timer <= 0:
                s.ship_x = 240.0
            return

        # Check wave cleared
        if self._alive_count() == 0:
            s.wave_clear_timer = 1.2
            return

        # ─── AI ship movement ──────────────────────────────────────────
        s.ai_retarget_timer -= dt
        if s.ai_retarget_timer <= 0:
            s.ai_retarget_timer = random.uniform(0.3, 0.7)
            s.ai_target_x = self._ai_find_target()

        diff = s.ai_target_x - s.ship_x
        s.ship_x += clamp(diff, -150 * dt, 150 * dt)
        s.ship_x = clamp(s.ship_x, SHIP_W // 2 + 4, 476 - SHIP_W // 2)

        # AI shooting
        s.ai_shoot_timer -= dt
        if s.ai_shoot_timer <= 0 and abs(diff) < 10:
            s.ai_shoot_timer = random.uniform(0.3, 0.5)
            if len(s.player_bullets) < 3:
                s.player_bullets.append([s.ship_x, SHIP_Y - 4])

        # ─── Alien movement ────────────────────────────────────────────
        s.alien_move_timer += dt
        if s.alien_move_timer >= s.alien_move_interval:
            s.alien_move_timer = 0
            s.alien_frame = 1 - s.alien_frame

            # Check if formation hits edge
            move_down = False
            for col in range(ALIEN_COLS):
                for row in range(ALIEN_ROWS):
                    if s.aliens[row][col]:
                        ax, _ = self._alien_pos(row, col)
                        if ax + ALIEN_W // 2 >= 470 and s.alien_dir > 0:
                            move_down = True
                        elif ax - ALIEN_W // 2 <= 10 and s.alien_dir < 0:
                            move_down = True

            if move_down:
                s.alien_dir *= -1
                s.alien_y += 8
                # Speed up when descending
                s.alien_move_interval = max(0.1, s.alien_move_interval * 0.95)
            else:
                s.alien_x += s.alien_dir * 8

        # Alien shooting
        if random.random() < 0.015:  # ~1.5% chance per frame
            # Pick a random bottom-row alien
            bottom_aliens = []
            for col in range(ALIEN_COLS):
                for row in range(ALIEN_ROWS - 1, -1, -1):
                    if s.aliens[row][col]:
                        bottom_aliens.append((row, col))
                        break
            if bottom_aliens:
                row, col = random.choice(bottom_aliens)
                ax, ay = self._alien_pos(row, col)
                if len(s.alien_bullets) < 4:
                    s.alien_bullets.append([ax, ay + ALIEN_H // 2])

        # ─── Bullet movement ──────────────────────────────────────────
        for b in s.player_bullets:
            b[1] -= BULLET_SPEED_PLAYER * dt
        s.player_bullets = [b for b in s.player_bullets if b[1] > 0]

        for b in s.alien_bullets:
            b[1] += BULLET_SPEED_ALIEN * dt
        s.alien_bullets = [b for b in s.alien_bullets if b[1] < 320]

        # ─── Collision: player bullets vs aliens ──────────────────────
        bullets_to_remove = []
        for bi, b in enumerate(s.player_bullets):
            hit = False
            for row in range(ALIEN_ROWS):
                for col in range(ALIEN_COLS):
                    if not s.aliens[row][col]:
                        continue
                    ax, ay = self._alien_pos(row, col)
                    if (abs(b[0] - ax) < ALIEN_W // 2 + 2
                            and abs(b[1] - ay) < ALIEN_H // 2 + 2):
                        s.aliens[row][col] = False
                        s.score += ALIEN_POINTS[row]
                        self._spawn_explosion(ax, ay, ALIEN_COLORS[row])
                        bullets_to_remove.append(bi)
                        hit = True
                        # Speed up as fewer aliens remain
                        alive = self._alive_count()
                        if alive > 0:
                            s.alien_move_interval = max(
                                0.08, 0.55 - (s.wave - 1) * 0.05
                                - (ALIEN_COLS * ALIEN_ROWS - alive) * 0.012)
                        break
                if hit:
                    break
        for bi in reversed(bullets_to_remove):
            s.player_bullets.pop(bi)

        # ─── Collision: alien bullets vs ship ─────────────────────────
        for b in s.alien_bullets[:]:
            if (abs(b[0] - s.ship_x) < SHIP_W // 2 + 2
                    and abs(b[1] - SHIP_Y) < SHIP_H // 2 + 4):
                s.alien_bullets.remove(b)
                self._spawn_explosion(s.ship_x, SHIP_Y, COLOR_SHIP)
                s.lives -= 1
                if s.lives <= 0:
                    self.reset()
                    return
                s.death_timer = 0.8
                break

        # ─── Aliens reach bottom ──────────────────────────────────────
        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                if s.aliens[row][col]:
                    _, ay = self._alien_pos(row, col)
                    if ay + ALIEN_H // 2 >= SHIP_Y - 5:
                        self.reset()
                        return

    def render(self, ctx: GameContext) -> None:
        s = self._s
        d = ctx.draw
        W, H = ctx.width, ctx.height

        d.rectangle([0, 0, W, H], fill=COLOR_BG)

        # Score and lives
        d.text((6, 2), f"Score: {s.score}", fill=COLOR_SCORE, font=ctx.font_small)
        d.text((220, 2), f"Wave {s.wave}", fill=COLOR_SCORE, font=ctx.font_small)
        for i in range(s.lives):
            # Small ship icons for lives
            lx = W - 20 - i * 18
            d.polygon([(lx, 12), (lx - 5, 4), (lx + 5, 4)], fill=COLOR_LIFE)

        # Wave clear text
        if s.wave_clear_timer > 0:
            alpha = min(1.0, s.wave_clear_timer / 0.6)
            tc = (int(50 * alpha), int(205 * alpha), int(50 * alpha))
            d.text((180, 140), f"WAVE {s.wave}!", fill=tc, font=ctx.font_large)
            # Still render aliens (frozen) and explosions below

        # Aliens
        for row in range(ALIEN_ROWS):
            for col in range(ALIEN_COLS):
                if not s.aliens[row][col]:
                    continue
                ax, ay = self._alien_pos(row, col)
                color = ALIEN_COLORS[row]
                hw = ALIEN_W // 2
                hh = ALIEN_H // 2

                # Draw alien body (simple pixel art with rectangles)
                d.rectangle([int(ax) - hw + 2, int(ay) - hh,
                             int(ax) + hw - 2, int(ay) + hh], fill=color)
                # "Arms" that animate
                arm_dy = -2 if s.alien_frame == 0 else 2
                d.rectangle([int(ax) - hw, int(ay) - hh + 3 + arm_dy,
                             int(ax) - hw + 3, int(ay) + hh - 2], fill=color)
                d.rectangle([int(ax) + hw - 3, int(ay) - hh + 3 + arm_dy,
                             int(ax) + hw, int(ay) + hh - 2], fill=color)
                # Eyes
                d.rectangle([int(ax) - 3, int(ay) - 2,
                             int(ax) - 1, int(ay) + 1], fill=(0, 0, 0))
                d.rectangle([int(ax) + 1, int(ay) - 2,
                             int(ax) + 3, int(ay) + 1], fill=(0, 0, 0))

        # Player bullets
        for b in s.player_bullets:
            bx, by = int(b[0]), int(b[1])
            d.rectangle([bx - 1, by - 3, bx + 1, by + 3],
                        fill=COLOR_BULLET_PLAYER)

        # Alien bullets
        for b in s.alien_bullets:
            bx, by = int(b[0]), int(b[1])
            d.rectangle([bx - 1, by - 3, bx + 1, by + 3],
                        fill=COLOR_BULLET_ALIEN)

        # Ship
        if s.death_timer <= 0:
            sx = int(s.ship_x)
            # Triangle ship
            d.polygon([(sx, SHIP_Y - SHIP_H),
                       (sx - SHIP_W // 2, SHIP_Y),
                       (sx + SHIP_W // 2, SHIP_Y)], fill=COLOR_SHIP)
            # Cockpit highlight
            d.polygon([(sx, SHIP_Y - SHIP_H + 3),
                       (sx - 3, SHIP_Y - 3),
                       (sx + 3, SHIP_Y - 3)], fill=COLOR_SHIP_BRIGHT)

        # Explosions
        for e in s.explosions:
            alpha = max(0, e.life / 0.35)
            for dx, dy, speed in e.particles:
                t = 0.35 - e.life  # time since explosion
                px = e.x + dx * speed * t
                py = e.y + dy * speed * t
                pc = (int(e.color[0] * alpha),
                      int(e.color[1] * alpha),
                      int(e.color[2] * alpha))
                r = int(2 * alpha)
                if r >= 1:
                    d.ellipse([int(px) - r, int(py) - r,
                               int(px) + r, int(py) + r], fill=pc)

        # Ground line
        d.line([0, 312, W, 312], fill=COLOR_BORDER, width=1)
