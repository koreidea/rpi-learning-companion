"""Base class for idle screensaver games on the TFT display."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import ImageDraw, ImageFont


@dataclass
class GameContext:
    """Shared rendering context passed to all games."""
    draw: ImageDraw.ImageDraw
    width: int   # 480
    height: int  # 320
    font_large: ImageFont.ImageFont
    font_small: ImageFont.ImageFont


class IdleGame(ABC):
    """Abstract base class for self-playing idle screensaver games."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown on game transitions."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Initialize or re-initialize all game state for a fresh start."""
        ...

    @abstractmethod
    def update(self, dt: float) -> None:
        """Advance game simulation by dt seconds. Includes AI logic."""
        ...

    @abstractmethod
    def render(self, ctx: GameContext) -> None:
        """Draw the current frame onto ctx.draw. Must clear background first."""
        ...


def clamp(v, lo, hi):
    """Clamp value to [lo, hi] range."""
    return max(lo, min(hi, v))
