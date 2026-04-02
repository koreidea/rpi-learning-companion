"""Battery projector module — HDMI output for wall/ceiling projection.

Detects when a projector is connected via HDMI hotplug and provides
fullscreen display output for flashcards, alphabet/number walls,
and educational content.

Detection methods:
  1. HDMI hotplug: Pi detects second display via /sys/class/drm
  2. Manual: API toggle or voice command "projector on"

Output: Uses pygame to render fullscreen content on HDMI output.
The TFT display (SPI) continues showing the face independently.

Content modes:
  - flashcard: Large image + word on wall
  - alphabet: Letter grid (A-Z) with highlighted current letter
  - numbers: Number grid (1-20) with highlighted current number
  - story: Full-screen story illustration
  - blank: Black screen (projector standby)
"""

import asyncio
import os
import subprocess
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

# Content directories
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets" / "projector"

# Display settings
PROJ_WIDTH = 1280
PROJ_HEIGHT = 720


class ProjectorMode(str, Enum):
    OFF = "off"
    BLANK = "blank"
    FLASHCARD = "flashcard"
    ALPHABET = "alphabet"
    NUMBERS = "numbers"
    IMAGINE = "imagine"
    LOADING = "loading"


# ─── Flashcard data ──────────────────────────────────────────────
# Simple built-in flashcards (no images needed — rendered as text + emoji)
FLASHCARDS = {
    "animals": [
        ("Dog", "A friendly pet that barks!"),
        ("Cat", "A fluffy pet that meows!"),
        ("Elephant", "The biggest land animal!"),
        ("Lion", "The king of the jungle!"),
        ("Fish", "It lives in the water!"),
        ("Bird", "It can fly in the sky!"),
        ("Rabbit", "It loves to eat carrots!"),
        ("Monkey", "It loves bananas!"),
        ("Cow", "It gives us milk!"),
        ("Horse", "You can ride on it!"),
    ],
    "colors": [
        ("Red", "Like a fire truck!"),
        ("Blue", "Like the sky!"),
        ("Green", "Like the grass!"),
        ("Yellow", "Like the sun!"),
        ("Orange", "Like an orange fruit!"),
        ("Purple", "Like a grape!"),
        ("Pink", "Like a flamingo!"),
        ("White", "Like snow!"),
        ("Black", "Like the night sky!"),
        ("Brown", "Like chocolate!"),
    ],
    "shapes": [
        ("Circle", "Round like a ball!"),
        ("Square", "Has four equal sides!"),
        ("Triangle", "Has three sides!"),
        ("Star", "Shines in the sky!"),
        ("Heart", "Means love!"),
        ("Rectangle", "Like a door!"),
        ("Diamond", "Like a kite!"),
        ("Oval", "Like an egg!"),
    ],
    "fruits": [
        ("Apple", "Red and crunchy!"),
        ("Banana", "Yellow and curved!"),
        ("Orange", "Round and juicy!"),
        ("Grape", "Small and sweet!"),
        ("Mango", "The king of fruits!"),
        ("Watermelon", "Big and green outside!"),
        ("Strawberry", "Small and red!"),
        ("Pineapple", "Has a spiky crown!"),
    ],
}

# Color values for flashcard categories
CATEGORY_COLORS = {
    "animals": (76, 175, 80),
    "colors": (33, 150, 243),
    "shapes": (255, 152, 0),
    "fruits": (233, 30, 99),
}


class Projector:
    """HDMI projector output controller."""

    def __init__(self):
        self._connected = False
        self._mode = ProjectorMode.OFF
        self._pygame_initialized = False
        self._screen = None
        self._font_large = None
        self._font_medium = None
        self._font_small = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Current content state
        self._current_category: str = "animals"
        self._current_index: int = 0
        self._highlighted_letter: Optional[str] = None
        self._highlighted_number: Optional[int] = None
        self._needs_redraw = True

        # Image cache: {(category, name): pygame.Surface}
        self._image_cache: dict = {}

        # Imagination mode state
        self._generated_surface = None  # pygame.Surface of AI-generated image
        self._imagine_prompt: str = ""  # What the child asked to see
        self._loading_text: str = ""    # Loading message while generating

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> ProjectorMode:
        return self._mode

    # ── Detection ─────────────────────────────────────────────────

    def detect(self) -> bool:
        """Check if an HDMI display/projector is connected.

        Checks /sys/class/drm for connected HDMI outputs.
        Pi 5 has HDMI-0 and HDMI-1 ports.
        """
        try:
            drm_path = Path("/sys/class/drm")
            if not drm_path.exists():
                return False

            for card_dir in drm_path.iterdir():
                name = card_dir.name.lower()
                if "hdmi" not in name:
                    continue
                status_file = card_dir / "status"
                if status_file.exists():
                    status = status_file.read_text().strip()
                    if status == "connected":
                        logger.info("Projector detected: {} ({})", card_dir.name, status)
                        self._connected = True
                        return True

            self._connected = False
            return False
        except Exception as e:
            logger.debug("Projector detection error: {}", e)
            return False

    # ── Pygame initialization ─────────────────────────────────────

    def _init_pygame(self) -> bool:
        """Initialize pygame for HDMI output."""
        try:
            # Use the second display (HDMI) if available
            # SDL_VIDEO_WINDOW_POS and DISPLAY env vars help target HDMI
            os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

            import pygame
            # Only init display + font — NOT pygame.init() which grabs the
            # audio device and blocks aplay/TTS from working
            pygame.display.init()
            pygame.font.init()

            # Try to use the second display for projector
            # On Pi with TFT on SPI + HDMI projector, HDMI is typically :0
            info = pygame.display.Info()
            logger.info("Projector display: {}x{}", info.current_w, info.current_h)

            self._screen = pygame.display.set_mode(
                (PROJ_WIDTH, PROJ_HEIGHT),
                pygame.FULLSCREEN | pygame.NOFRAME,
            )
            pygame.display.set_caption("RPi Bot Projector")
            pygame.mouse.set_visible(False)

            # Load fonts
            self._font_large = pygame.font.Font(None, 180)
            self._font_medium = pygame.font.Font(None, 80)
            self._font_small = pygame.font.Font(None, 50)

            self._pygame_initialized = True
            logger.info("Projector pygame initialized ({}x{})", PROJ_WIDTH, PROJ_HEIGHT)
            return True

        except Exception as e:
            logger.warning("Projector pygame init failed: {}", e)
            return False

    # ── Start / Stop ──────────────────────────────────────────────

    def start(self):
        """Start the projector rendering thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the projector."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._cleanup_pygame()

    def _cleanup_pygame(self):
        if self._pygame_initialized:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._pygame_initialized = False
            self._screen = None

    # ── Mode switching ────────────────────────────────────────────

    def set_mode(self, mode: ProjectorMode):
        """Switch projector content mode."""
        with self._lock:
            self._mode = mode
            self._needs_redraw = True
        logger.info("Projector mode: {}", mode.value)

    def set_flashcard_category(self, category: str):
        """Set the flashcard category (animals, colors, shapes, fruits)."""
        if category in FLASHCARDS:
            with self._lock:
                self._current_category = category
                self._current_index = 0
                self._needs_redraw = True

    def next_flashcard(self):
        """Advance to next flashcard."""
        with self._lock:
            cards = FLASHCARDS.get(self._current_category, [])
            if cards:
                self._current_index = (self._current_index + 1) % len(cards)
                self._needs_redraw = True

    def prev_flashcard(self):
        """Go to previous flashcard."""
        with self._lock:
            cards = FLASHCARDS.get(self._current_category, [])
            if cards:
                self._current_index = (self._current_index - 1) % len(cards)
                self._needs_redraw = True

    def highlight_letter(self, letter: str):
        """Highlight a letter on the alphabet wall."""
        with self._lock:
            self._highlighted_letter = letter.upper() if letter else None
            self._needs_redraw = True

    def highlight_number(self, number: int):
        """Highlight a number on the number wall."""
        with self._lock:
            self._highlighted_number = number
            self._needs_redraw = True

    def get_current_flashcard(self) -> Optional[tuple]:
        """Get the current flashcard (name, description)."""
        cards = FLASHCARDS.get(self._current_category, [])
        if cards and 0 <= self._current_index < len(cards):
            return cards[self._current_index]
        return None

    # ── Imagination mode ────────────────────────────────────────────

    def show_loading_message(self, prompt: str):
        """Show a loading screen while DALL-E generates the image."""
        with self._lock:
            self._loading_text = prompt
            self._mode = ProjectorMode.LOADING
            self._needs_redraw = True

    def show_generated_image(self, pil_image, prompt: str):
        """Display an AI-generated PIL Image on the projector.

        Converts PIL Image → pygame Surface, scaled to fill the screen.
        """
        try:
            import pygame

            # PIL → pygame: convert to RGB bytes
            raw = pil_image.tobytes()
            surface = pygame.image.fromstring(
                raw, pil_image.size, pil_image.mode
            )

            # Scale to fill screen while maintaining aspect ratio
            img_w, img_h = surface.get_size()
            # Leave room for prompt text at bottom (50px)
            max_w = PROJ_WIDTH - 40
            max_h = PROJ_HEIGHT - 80
            scale = min(max_w / img_w, max_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            surface = pygame.transform.smoothscale(surface, (new_w, new_h))

            with self._lock:
                self._generated_surface = surface
                self._imagine_prompt = prompt
                self._mode = ProjectorMode.IMAGINE
                self._needs_redraw = True

            logger.info("Showing generated image for: '{}'", prompt)

        except Exception as e:
            logger.error("Failed to display generated image: {}", e)

    # ── Image loading ─────────────────────────────────────────────

    def _load_flashcard_image(self, pygame, category: str, name: str):
        """Load a flashcard image from assets, with caching.

        Returns a pygame.Surface scaled to fit the display, or None.
        """
        key = (category, name.lower())
        if key in self._image_cache:
            return self._image_cache[key]

        img_path = ASSETS_DIR / "flashcards" / category / f"{name.lower()}.png"
        if not img_path.exists():
            self._image_cache[key] = None
            return None

        try:
            surface = pygame.image.load(str(img_path)).convert_alpha()
            # Scale to fit left half of screen with padding
            max_h = int(PROJ_HEIGHT * 0.7)
            max_w = int(PROJ_WIDTH * 0.4)
            w, h = surface.get_size()
            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            surface = pygame.transform.smoothscale(surface, (new_w, new_h))
            self._image_cache[key] = surface
            return surface
        except Exception as e:
            logger.debug("Failed to load flashcard image {}: {}", img_path, e)
            self._image_cache[key] = None
            return None

    # ── Rendering ─────────────────────────────────────────────────

    def _run(self):
        """Main render loop."""
        if not self._init_pygame():
            return

        import pygame

        while self._running:
            try:
                # Process pygame events (required to keep window alive)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                        break

                with self._lock:
                    needs = self._needs_redraw
                    mode = self._mode

                if needs:
                    if mode == ProjectorMode.FLASHCARD:
                        self._render_flashcard(pygame)
                    elif mode == ProjectorMode.ALPHABET:
                        self._render_alphabet(pygame)
                    elif mode == ProjectorMode.NUMBERS:
                        self._render_numbers(pygame)
                    elif mode == ProjectorMode.IMAGINE:
                        self._render_imagine(pygame)
                    elif mode == ProjectorMode.LOADING:
                        self._render_loading(pygame)
                    elif mode == ProjectorMode.BLANK:
                        self._screen.fill((0, 0, 0))
                    else:
                        self._screen.fill((0, 0, 0))

                    pygame.display.flip()
                    with self._lock:
                        # Keep redrawing in LOADING mode for animation
                        if self._mode != ProjectorMode.LOADING:
                            self._needs_redraw = False

                time.sleep(0.05 if mode == ProjectorMode.LOADING else 0.05)

            except Exception as e:
                logger.debug("Projector render error: {}", e)
                time.sleep(0.5)

    def _render_flashcard(self, pygame):
        """Render current flashcard — image on left, word + description on right."""
        card = self.get_current_flashcard()
        if not card:
            return

        name, description = card
        category = self._current_category
        color = CATEGORY_COLORS.get(category, (255, 255, 255))

        self._screen.fill((20, 20, 30))

        # Try to load the image
        image = self._load_flashcard_image(pygame, category, name)

        if image:
            # ── Layout with image: image left, text right ──
            img_w, img_h = image.get_size()

            # Center image in left half
            img_x = (PROJ_WIDTH // 2 - img_w) // 2
            img_y = (PROJ_HEIGHT - img_h) // 2
            self._screen.blit(image, (img_x, img_y))

            # Text in right half
            text_cx = PROJ_WIDTH * 3 // 4

            # Category label (top of right half)
            cat_surf = self._font_small.render(
                category.upper(), True, (150, 150, 150)
            )
            cat_rect = cat_surf.get_rect(center=(text_cx, PROJ_HEIGHT // 2 - 120))
            self._screen.blit(cat_surf, cat_rect)

            # Large word
            word_surf = self._font_large.render(name, True, color)
            word_rect = word_surf.get_rect(center=(text_cx, PROJ_HEIGHT // 2))
            self._screen.blit(word_surf, word_rect)

            # Description
            desc_surf = self._font_medium.render(description, True, (200, 200, 200))
            desc_rect = desc_surf.get_rect(center=(text_cx, PROJ_HEIGHT // 2 + 120))
            self._screen.blit(desc_surf, desc_rect)

        else:
            # ── Fallback: text-only centered layout ──
            cat_surf = self._font_small.render(
                category.upper(), True, (150, 150, 150)
            )
            cat_rect = cat_surf.get_rect(center=(PROJ_WIDTH // 2, 50))
            self._screen.blit(cat_surf, cat_rect)

            word_surf = self._font_large.render(name, True, color)
            word_rect = word_surf.get_rect(center=(PROJ_WIDTH // 2, PROJ_HEIGHT // 2 - 40))
            self._screen.blit(word_surf, word_rect)

            desc_surf = self._font_medium.render(description, True, (200, 200, 200))
            desc_rect = desc_surf.get_rect(center=(PROJ_WIDTH // 2, PROJ_HEIGHT // 2 + 100))
            self._screen.blit(desc_surf, desc_rect)

        # Card counter (bottom center)
        cards = FLASHCARDS.get(category, [])
        counter = f"{self._current_index + 1} / {len(cards)}"
        count_surf = self._font_small.render(counter, True, (100, 100, 100))
        count_rect = count_surf.get_rect(center=(PROJ_WIDTH // 2, PROJ_HEIGHT - 40))
        self._screen.blit(count_surf, count_rect)

    def _render_alphabet(self, pygame):
        """Render alphabet wall — A-Z grid with highlighted letter."""
        self._screen.fill((15, 15, 25))

        # Title
        title_surf = self._font_medium.render("ALPHABET", True, (100, 100, 100))
        title_rect = title_surf.get_rect(center=(PROJ_WIDTH // 2, 40))
        self._screen.blit(title_surf, title_rect)

        # Grid: 7 columns × 4 rows (26 letters + 2 blanks)
        cols = 7
        rows = 4
        cell_w = PROJ_WIDTH // (cols + 1)
        cell_h = (PROJ_HEIGHT - 100) // (rows + 1)
        start_x = cell_w
        start_y = 90

        highlighted = self._highlighted_letter

        for i in range(26):
            letter = chr(65 + i)  # A-Z
            row = i // cols
            col = i % cols
            cx = start_x + col * cell_w + cell_w // 2
            cy = start_y + row * cell_h + cell_h // 2

            is_highlighted = (letter == highlighted)

            if is_highlighted:
                # Highlighted: big, bright, with colored background
                bg_rect = pygame.Rect(
                    cx - cell_w // 2 + 5, cy - cell_h // 2 + 5,
                    cell_w - 10, cell_h - 10
                )
                pygame.draw.rect(self._screen, (30, 144, 255), bg_rect, border_radius=12)
                letter_surf = self._font_large.render(letter, True, (255, 255, 255))
            else:
                # Normal: dimmer, smaller
                letter_surf = self._font_medium.render(letter, True, (120, 120, 140))

            letter_rect = letter_surf.get_rect(center=(cx, cy))
            self._screen.blit(letter_surf, letter_rect)

    def _render_numbers(self, pygame):
        """Render number wall — 1-20 grid with highlighted number."""
        self._screen.fill((15, 25, 15))

        # Title
        title_surf = self._font_medium.render("NUMBERS", True, (100, 100, 100))
        title_rect = title_surf.get_rect(center=(PROJ_WIDTH // 2, 40))
        self._screen.blit(title_surf, title_rect)

        # Grid: 5 columns × 4 rows
        cols = 5
        rows = 4
        cell_w = PROJ_WIDTH // (cols + 1)
        cell_h = (PROJ_HEIGHT - 100) // (rows + 1)
        start_x = cell_w
        start_y = 90

        highlighted = self._highlighted_number

        for i in range(20):
            num = i + 1
            row = i // cols
            col = i % cols
            cx = start_x + col * cell_w + cell_w // 2
            cy = start_y + row * cell_h + cell_h // 2

            is_highlighted = (num == highlighted)

            if is_highlighted:
                bg_rect = pygame.Rect(
                    cx - cell_w // 2 + 5, cy - cell_h // 2 + 5,
                    cell_w - 10, cell_h - 10
                )
                pygame.draw.rect(self._screen, (76, 175, 80), bg_rect, border_radius=12)
                num_surf = self._font_large.render(str(num), True, (255, 255, 255))
            else:
                num_surf = self._font_medium.render(str(num), True, (120, 140, 120))

            num_rect = num_surf.get_rect(center=(cx, cy))
            self._screen.blit(num_surf, num_rect)

    def _render_imagine(self, pygame):
        """Render the AI-generated image fullscreen with prompt text."""
        self._screen.fill((10, 10, 20))

        if self._generated_surface:
            img_w, img_h = self._generated_surface.get_size()
            # Center the image
            x = (PROJ_WIDTH - img_w) // 2
            y = (PROJ_HEIGHT - 50 - img_h) // 2  # leave room for text
            self._screen.blit(self._generated_surface, (x, y))

        # Show prompt at bottom
        if self._imagine_prompt:
            prompt_surf = self._font_small.render(
                self._imagine_prompt, True, (180, 180, 200)
            )
            prompt_rect = prompt_surf.get_rect(
                center=(PROJ_WIDTH // 2, PROJ_HEIGHT - 30)
            )
            self._screen.blit(prompt_surf, prompt_rect)

    def _render_loading(self, pygame):
        """Render a loading/generating screen."""
        self._screen.fill((15, 15, 25))

        # "Imagining..." text
        title_surf = self._font_medium.render(
            "Imagining...", True, (180, 130, 255)
        )
        title_rect = title_surf.get_rect(
            center=(PROJ_WIDTH // 2, PROJ_HEIGHT // 2 - 60)
        )
        self._screen.blit(title_surf, title_rect)

        # Show what's being generated
        if self._loading_text:
            desc_surf = self._font_small.render(
                self._loading_text, True, (150, 150, 170)
            )
            desc_rect = desc_surf.get_rect(
                center=(PROJ_WIDTH // 2, PROJ_HEIGHT // 2 + 20)
            )
            self._screen.blit(desc_surf, desc_rect)

        # Simple animated dots (3 dots cycling)
        dots = "." * (int(time.time() * 2) % 4)
        dots_surf = self._font_medium.render(dots, True, (180, 130, 255))
        dots_rect = dots_surf.get_rect(
            center=(PROJ_WIDTH // 2, PROJ_HEIGHT // 2 + 80)
        )
        self._screen.blit(dots_surf, dots_rect)

    # ── Cleanup ───────────────────────────────────────────────────

    def close(self):
        self.stop()

    def __del__(self):
        self.close()
