"""Self-playing Snake game for the idle screensaver carousel."""

import random
from dataclasses import dataclass, field

from .base import IdleGame, GameContext

# ─── Constants ───────────────────────────────────────────────────────────────
CELL = 10               # Cell size in pixels
COLS = 48               # 480 / 10
ROWS = 29               # 290 / 10 (top 30px reserved for score)
AREA_TOP = 30           # Y offset for play area
MOVE_INTERVAL = 0.14    # Seconds between moves

COLOR_BG = (0, 0, 0)
COLOR_HEAD = (100, 255, 100)
COLOR_BODY_BRIGHT = (50, 205, 50)
COLOR_BODY_DIM = (20, 100, 20)
COLOR_FOOD = (255, 200, 50)
COLOR_FOOD_GLOW = (120, 95, 20)
COLOR_GRID = (12, 12, 15)
COLOR_SCORE = (55, 55, 60)
COLOR_DEATH_FLASH = (255, 60, 60)
COLOR_BORDER = (30, 30, 35)

# Directions: (dx, dy)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


@dataclass
class SnakeState:
    body: list = field(default_factory=lambda: [(10, 11), (9, 11), (8, 11)])
    direction: tuple = (1, 0)
    food: tuple = (20, 11)
    move_timer: float = 0.0
    score: int = 0
    food_pulse: float = 0.0
    death_timer: float = 0.0
    dead: bool = False


class SnakeGame(IdleGame):
    """Self-playing Snake — AI chases food, avoids walls and itself."""

    def __init__(self):
        self._s = SnakeState()

    @property
    def name(self) -> str:
        return "Snake"

    def reset(self) -> None:
        self._s = SnakeState()
        self._spawn_food()

    def _spawn_food(self):
        s = self._s
        occupied = set(s.body)
        attempts = 0
        while attempts < 200:
            fx = random.randint(0, COLS - 1)
            fy = random.randint(0, ROWS - 1)
            if (fx, fy) not in occupied:
                s.food = (fx, fy)
                return
            attempts += 1
        # Fallback: just place it somewhere
        s.food = (COLS // 2, ROWS // 2)

    def _opposite(self, d):
        return (-d[0], -d[1])

    def _is_safe(self, pos):
        """Check if position is safe (in bounds and not on body)."""
        x, y = pos
        if x < 0 or x >= COLS or y < 0 or y >= ROWS:
            return False
        if pos in set(self._s.body[:-1]):  # tail will move away
            return False
        return True

    def _ai_decide(self):
        """Greedy AI: pick direction closest to food, avoiding death."""
        s = self._s
        hx, hy = s.body[0]
        fx, fy = s.food
        opposite = self._opposite(s.direction)

        # Score each candidate direction
        candidates = []
        for d in DIRECTIONS:
            if d == opposite:
                continue  # Can't reverse
            nx, ny = hx + d[0], hy + d[1]
            if self._is_safe((nx, ny)):
                dist = abs(nx - fx) + abs(ny - fy)
                candidates.append((dist, d))

        if not candidates:
            return  # All moves are death — keep going

        # 5% chance of random move for imperfection
        if random.random() < 0.05 and len(candidates) > 1:
            s.direction = random.choice(candidates)[1]
        else:
            candidates.sort(key=lambda c: c[0])
            s.direction = candidates[0][1]

    def update(self, dt: float) -> None:
        s = self._s

        # Death pause
        if s.dead:
            s.death_timer -= dt
            if s.death_timer <= 0:
                self.reset()
            return

        s.food_pulse += dt
        s.move_timer += dt

        if s.move_timer < MOVE_INTERVAL:
            return
        s.move_timer -= MOVE_INTERVAL

        # AI decides direction
        self._ai_decide()

        # Move head
        hx, hy = s.body[0]
        nx = hx + s.direction[0]
        ny = hy + s.direction[1]

        # Check death
        if nx < 0 or nx >= COLS or ny < 0 or ny >= ROWS or (nx, ny) in set(s.body):
            s.dead = True
            s.death_timer = 0.8
            return

        s.body.insert(0, (nx, ny))

        # Check food
        if (nx, ny) == s.food:
            s.score += 1
            self._spawn_food()
        else:
            s.body.pop()

    def render(self, ctx: GameContext) -> None:
        s = self._s
        d = ctx.draw
        W, H = ctx.width, ctx.height

        d.rectangle([0, 0, W, H], fill=COLOR_BG)

        # Score
        d.text((6, 2), f"Score: {s.score}", fill=COLOR_SCORE, font=ctx.font_small)

        # Border
        d.rectangle([0, AREA_TOP, W - 1, AREA_TOP + ROWS * CELL],
                     outline=COLOR_BORDER, width=1)

        # Grid dots (subtle, every 4 cells)
        for gx in range(0, COLS, 4):
            for gy in range(0, ROWS, 4):
                px = gx * CELL + CELL // 2
                py = AREA_TOP + gy * CELL + CELL // 2
                d.point((px, py), fill=COLOR_GRID)

        # Food with pulsing glow
        fx, fy = s.food
        fpx = fx * CELL + CELL // 2
        fpy = AREA_TOP + fy * CELL + CELL // 2
        import math
        pulse = 0.7 + 0.3 * math.sin(s.food_pulse * 4)
        gr = int(6 + 2 * pulse)
        glow_c = (int(COLOR_FOOD_GLOW[0] * pulse),
                  int(COLOR_FOOD_GLOW[1] * pulse),
                  int(COLOR_FOOD_GLOW[2] * pulse))
        d.ellipse([fpx - gr, fpy - gr, fpx + gr, fpy + gr], fill=glow_c)
        fr = int(3 + 1 * pulse)
        d.ellipse([fpx - fr, fpy - fr, fpx + fr, fpy + fr], fill=COLOR_FOOD)

        # Snake body
        body_len = len(s.body)
        for i, (bx, by) in enumerate(s.body):
            px = bx * CELL
            py = AREA_TOP + by * CELL

            if i == 0:
                # Head
                color = COLOR_DEATH_FLASH if s.dead else COLOR_HEAD
                d.rounded_rectangle([px + 1, py + 1, px + CELL - 1, py + CELL - 1],
                                     radius=3, fill=color)
                # Eyes on head
                if not s.dead:
                    dx, dy = s.direction
                    ecx = px + CELL // 2 + dx * 2
                    ecy = py + CELL // 2 + dy * 2
                    d.ellipse([ecx - 1, ecy - 1, ecx + 1, ecy + 1],
                              fill=(0, 0, 0))
            else:
                # Body gradient: bright near head, dim near tail
                t = i / max(body_len - 1, 1)
                r = int(COLOR_BODY_BRIGHT[0] * (1 - t) + COLOR_BODY_DIM[0] * t)
                g = int(COLOR_BODY_BRIGHT[1] * (1 - t) + COLOR_BODY_DIM[1] * t)
                b = int(COLOR_BODY_BRIGHT[2] * (1 - t) + COLOR_BODY_DIM[2] * t)
                d.rounded_rectangle([px + 1, py + 1, px + CELL - 1, py + CELL - 1],
                                     radius=2, fill=(r, g, b))

        # Death flash overlay
        if s.dead:
            # Red tint border flash
            d.rectangle([0, AREA_TOP, W - 1, AREA_TOP + ROWS * CELL],
                         outline=COLOR_DEATH_FLASH, width=2)
