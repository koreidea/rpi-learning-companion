"""Idle screensaver games for the TFT display carousel."""

from .base import IdleGame, GameContext
from .pong import PongGame
from .snake import SnakeGame
from .breakout import BreakoutGame
from .tetris import TetrisGame
from .space_invaders import SpaceInvadersGame

# Games cycle in this order, switching every GAME_CYCLE_TIME seconds
GAME_REGISTRY = [PongGame, SnakeGame, BreakoutGame, TetrisGame, SpaceInvadersGame]

IDLE_THRESHOLD = 60.0      # Seconds in READY before games start
GAME_CYCLE_TIME = 60.0     # 1 minute per game
