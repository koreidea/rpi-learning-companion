"""Self-playing Tetris game for the idle screensaver carousel."""

import math
import random
from dataclasses import dataclass, field

from .base import IdleGame, GameContext, clamp

# ─── Constants ───────────────────────────────────────────────────────────────
BOARD_W = 10
BOARD_H = 20
CELL = 15
BOARD_X = 165  # (480 - 10*15) // 2 = 165
BOARD_Y = 10
BOARD_PX_W = BOARD_W * CELL  # 150
BOARD_PX_H = BOARD_H * CELL  # 300

COLOR_BG = (0, 0, 0)
COLOR_GRID = (15, 15, 20)
COLOR_BORDER = (35, 35, 40)
COLOR_SCORE = (55, 55, 60)
COLOR_FLASH = (255, 255, 255)
COLOR_GAMEOVER = (255, 60, 60)

# Piece colors (index 1-7)
PIECE_COLORS = [
    None,
    (0, 200, 200),     # I - cyan
    (255, 200, 50),    # O - yellow
    (180, 100, 255),   # T - purple
    (50, 205, 50),     # S - green
    (255, 60, 60),     # Z - red
    (30, 144, 255),    # J - blue
    (255, 140, 50),    # L - orange
]

# Piece shapes: each piece has 4 rotations, each rotation is list of (row, col) offsets
PIECES = {
    1: [  # I
        [(0,0),(0,1),(0,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0)],
        [(0,0),(0,1),(0,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0)],
    ],
    2: [  # O
        [(0,0),(0,1),(1,0),(1,1)],
        [(0,0),(0,1),(1,0),(1,1)],
        [(0,0),(0,1),(1,0),(1,1)],
        [(0,0),(0,1),(1,0),(1,1)],
    ],
    3: [  # T
        [(0,0),(0,1),(0,2),(1,1)],
        [(0,0),(1,0),(2,0),(1,1)],
        [(1,0),(1,1),(1,2),(0,1)],
        [(0,0),(1,0),(2,0),(1,-1)],
    ],
    4: [  # S
        [(0,1),(0,2),(1,0),(1,1)],
        [(0,0),(1,0),(1,1),(2,1)],
        [(0,1),(0,2),(1,0),(1,1)],
        [(0,0),(1,0),(1,1),(2,1)],
    ],
    5: [  # Z
        [(0,0),(0,1),(1,1),(1,2)],
        [(0,1),(1,0),(1,1),(2,0)],
        [(0,0),(0,1),(1,1),(1,2)],
        [(0,1),(1,0),(1,1),(2,0)],
    ],
    6: [  # J
        [(0,0),(1,0),(1,1),(1,2)],
        [(0,0),(0,1),(1,0),(2,0)],
        [(0,0),(0,1),(0,2),(1,2)],
        [(0,0),(1,0),(2,0),(2,-1)],
    ],
    7: [  # L
        [(0,2),(1,0),(1,1),(1,2)],
        [(0,0),(1,0),(2,0),(2,1)],
        [(0,0),(0,1),(0,2),(1,0)],
        [(0,0),(0,1),(1,1),(2,1)],
    ],
}


@dataclass
class TetrisState:
    board: list = field(default_factory=lambda: [[0] * BOARD_W for _ in range(BOARD_H)])
    piece_type: int = 0
    piece_x: int = 3
    piece_y: int = 0
    piece_rot: int = 0
    next_type: int = 0
    drop_timer: float = 0.0
    drop_interval: float = 0.8
    ai_timer: float = 0.0
    ai_target_x: int = 3
    ai_target_rot: int = 0
    ai_decided: bool = False
    score: int = 0
    lines: int = 0
    level: int = 1
    clear_flash: float = 0.0
    clearing_rows: list = field(default_factory=list)
    game_over: bool = False
    game_over_timer: float = 0.0
    game_over_row: int = BOARD_H  # For top-down red fill effect


class TetrisGame(IdleGame):
    """Self-playing Tetris — AI places pieces using heuristic scoring."""

    def __init__(self):
        self._s = TetrisState()

    @property
    def name(self) -> str:
        return "Tetris"

    def reset(self) -> None:
        self._s = TetrisState()
        self._s.piece_type = random.randint(1, 7)
        self._s.next_type = random.randint(1, 7)

    def _get_cells(self, ptype, rot, px, py):
        """Get board cells for a piece at position."""
        cells = []
        for dr, dc in PIECES[ptype][rot % 4]:
            cells.append((py + dr, px + dc))
        return cells

    def _fits(self, ptype, rot, px, py):
        """Check if piece fits at position."""
        for r, c in self._get_cells(ptype, rot, px, py):
            if r < 0 or r >= BOARD_H or c < 0 or c >= BOARD_W:
                return False
            if self._s.board[r][c] != 0:
                return False
        return True

    def _lock_piece(self):
        """Lock current piece onto board."""
        s = self._s
        cells = self._get_cells(s.piece_type, s.piece_rot, s.piece_x, s.piece_y)
        for r, c in cells:
            if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                s.board[r][c] = s.piece_type

        # Check for line clears
        full_rows = []
        for r in range(BOARD_H):
            if all(s.board[r][c] != 0 for c in range(BOARD_W)):
                full_rows.append(r)

        if full_rows:
            s.clearing_rows = full_rows
            s.clear_flash = 0.3
            s.lines += len(full_rows)
            s.score += [0, 100, 300, 500, 800][min(len(full_rows), 4)]
            s.level = 1 + s.lines // 5
            s.drop_interval = max(0.2, 0.8 - (s.level - 1) * 0.05)

    def _clear_rows(self):
        """Remove cleared rows and shift board down."""
        s = self._s
        for r in sorted(s.clearing_rows, reverse=True):
            del s.board[r]
            s.board.insert(0, [0] * BOARD_W)
        s.clearing_rows.clear()

    def _spawn_piece(self):
        """Spawn next piece at top."""
        s = self._s
        s.piece_type = s.next_type
        s.next_type = random.randint(1, 7)
        s.piece_x = 3
        s.piece_y = 0
        s.piece_rot = 0
        s.ai_decided = False
        s.ai_timer = 0.3  # Brief delay before AI decides

        if not self._fits(s.piece_type, s.piece_rot, s.piece_x, s.piece_y):
            s.game_over = True
            s.game_over_timer = 2.0
            s.game_over_row = BOARD_H

    def _ai_evaluate(self, ptype, rot, px):
        """Score a placement. Higher is better."""
        s = self._s
        # Simulate drop
        py = 0
        while self._fits(ptype, rot, px, py + 1):
            py += 1
        if not self._fits(ptype, rot, px, py):
            return -9999

        # Temporarily place piece
        temp_board = [row[:] for row in s.board]
        for r, c in self._get_cells(ptype, rot, px, py):
            if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                temp_board[r][c] = ptype

        # Compute metrics
        # 1. Aggregate height
        heights = []
        for c in range(BOARD_W):
            h = 0
            for r in range(BOARD_H):
                if temp_board[r][c] != 0:
                    h = BOARD_H - r
                    break
            heights.append(h)
        agg_height = sum(heights)

        # 2. Complete lines
        complete = sum(1 for r in range(BOARD_H)
                       if all(temp_board[r][c] != 0 for c in range(BOARD_W)))

        # 3. Holes
        holes = 0
        for c in range(BOARD_W):
            found_block = False
            for r in range(BOARD_H):
                if temp_board[r][c] != 0:
                    found_block = True
                elif found_block:
                    holes += 1

        # 4. Bumpiness
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(BOARD_W - 1))

        return (complete * 1.0
                - agg_height * 0.5
                - holes * 0.7
                - bumpiness * 0.2
                + random.uniform(-0.3, 0.3))  # Noise for imperfection

    def _ai_decide(self):
        """Choose best placement for current piece."""
        s = self._s
        best_score = -99999
        best_x = s.piece_x
        best_rot = s.piece_rot

        # 10% chance to skip evaluation (random drop)
        if random.random() < 0.10:
            s.ai_target_x = s.piece_x
            s.ai_target_rot = s.piece_rot
            s.ai_decided = True
            return

        for rot in range(4):
            for px in range(-2, BOARD_W + 1):
                if self._fits(s.piece_type, rot, px, 0):
                    score = self._ai_evaluate(s.piece_type, rot, px)
                    if score > best_score:
                        best_score = score
                        best_x = px
                        best_rot = rot

        s.ai_target_x = best_x
        s.ai_target_rot = best_rot
        s.ai_decided = True

    def update(self, dt: float) -> None:
        s = self._s

        # Game over animation
        if s.game_over:
            s.game_over_timer -= dt
            # Red fill from top
            s.game_over_row = max(0, int(BOARD_H * (s.game_over_timer / 2.0)))
            if s.game_over_timer <= 0:
                self.reset()
            return

        # Line clear flash
        if s.clear_flash > 0:
            s.clear_flash -= dt
            if s.clear_flash <= 0:
                self._clear_rows()
                self._spawn_piece()
            return

        # First piece
        if s.piece_type == 0:
            self._spawn_piece()
            return

        # AI decision
        s.ai_timer -= dt
        if s.ai_timer <= 0 and not s.ai_decided:
            self._ai_decide()

        # AI moves piece toward target (one step per update for animation)
        if s.ai_decided:
            if s.piece_rot != s.ai_target_rot:
                new_rot = (s.piece_rot + 1) % 4
                if self._fits(s.piece_type, new_rot, s.piece_x, s.piece_y):
                    s.piece_rot = new_rot
            elif s.piece_x < s.ai_target_x:
                if self._fits(s.piece_type, s.piece_rot, s.piece_x + 1, s.piece_y):
                    s.piece_x += 1
            elif s.piece_x > s.ai_target_x:
                if self._fits(s.piece_type, s.piece_rot, s.piece_x - 1, s.piece_y):
                    s.piece_x -= 1

        # Drop
        s.drop_timer += dt
        if s.drop_timer >= s.drop_interval:
            s.drop_timer = 0
            if self._fits(s.piece_type, s.piece_rot, s.piece_x, s.piece_y + 1):
                s.piece_y += 1
            else:
                self._lock_piece()
                if not s.clearing_rows:
                    self._spawn_piece()

    def render(self, ctx: GameContext) -> None:
        s = self._s
        d = ctx.draw
        W, H = ctx.width, ctx.height

        d.rectangle([0, 0, W, H], fill=COLOR_BG)

        # Board border
        d.rectangle([BOARD_X - 1, BOARD_Y - 1,
                      BOARD_X + BOARD_PX_W + 1, BOARD_Y + BOARD_PX_H + 1],
                     outline=COLOR_BORDER, width=1)

        # Grid lines (very subtle)
        for c in range(1, BOARD_W):
            x = BOARD_X + c * CELL
            d.line([x, BOARD_Y, x, BOARD_Y + BOARD_PX_H], fill=COLOR_GRID, width=1)
        for r in range(1, BOARD_H):
            y = BOARD_Y + r * CELL
            d.line([BOARD_X, y, BOARD_X + BOARD_PX_W, y], fill=COLOR_GRID, width=1)

        # Board cells
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                val = s.board[r][c]
                if val == 0:
                    continue

                # Game over red fill
                if s.game_over and r >= s.game_over_row:
                    color = COLOR_GAMEOVER
                else:
                    color = PIECE_COLORS[val]

                px = BOARD_X + c * CELL
                py = BOARD_Y + r * CELL
                d.rectangle([px + 1, py + 1, px + CELL - 1, py + CELL - 1],
                            fill=color)
                # Highlight
                hl = (min(255, color[0] + 40),
                      min(255, color[1] + 40),
                      min(255, color[2] + 40))
                d.line([px + 2, py + 1, px + CELL - 2, py + 1], fill=hl, width=1)

        # Line clear flash
        if s.clear_flash > 0:
            for r in s.clearing_rows:
                alpha = s.clear_flash / 0.3
                fc = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
                py = BOARD_Y + r * CELL
                d.rectangle([BOARD_X, py, BOARD_X + BOARD_PX_W, py + CELL], fill=fc)

        # Current piece
        if not s.game_over and s.piece_type > 0:
            color = PIECE_COLORS[s.piece_type]
            for r, c in self._get_cells(s.piece_type, s.piece_rot, s.piece_x, s.piece_y):
                if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                    px = BOARD_X + c * CELL
                    py = BOARD_Y + r * CELL
                    d.rectangle([px + 1, py + 1, px + CELL - 1, py + CELL - 1],
                                fill=color)
                    hl = (min(255, color[0] + 60),
                          min(255, color[1] + 60),
                          min(255, color[2] + 60))
                    d.line([px + 2, py + 1, px + CELL - 2, py + 1],
                           fill=hl, width=1)

            # Ghost piece (where it would land)
            ghost_y = s.piece_y
            while self._fits(s.piece_type, s.piece_rot, s.piece_x, ghost_y + 1):
                ghost_y += 1
            if ghost_y > s.piece_y:
                for r, c in self._get_cells(s.piece_type, s.piece_rot, s.piece_x, ghost_y):
                    if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                        px = BOARD_X + c * CELL
                        py = BOARD_Y + r * CELL
                        gc = (color[0] // 4, color[1] // 4, color[2] // 4)
                        d.rectangle([px + 1, py + 1, px + CELL - 1, py + CELL - 1],
                                    outline=gc, width=1)

        # Left panel: Score / Level / Lines
        d.text((8, 20), "SCORE", fill=COLOR_SCORE, font=ctx.font_small)
        d.text((8, 36), f"{s.score}", fill=(80, 80, 90), font=ctx.font_small)
        d.text((8, 60), "LEVEL", fill=COLOR_SCORE, font=ctx.font_small)
        d.text((8, 76), f"{s.level}", fill=(80, 80, 90), font=ctx.font_small)
        d.text((8, 100), "LINES", fill=COLOR_SCORE, font=ctx.font_small)
        d.text((8, 116), f"{s.lines}", fill=(80, 80, 90), font=ctx.font_small)

        # Right panel: Next piece preview
        d.text((340, 20), "NEXT", fill=COLOR_SCORE, font=ctx.font_small)
        if s.next_type > 0:
            nc = PIECE_COLORS[s.next_type]
            for dr, dc in PIECES[s.next_type][0]:
                px = 340 + dc * 14
                py = 40 + dr * 14
                d.rectangle([px, py, px + 13, py + 13], fill=nc)
