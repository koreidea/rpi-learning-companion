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

# Display settings (defaults, overridden by actual screen resolution)
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
    SKETCH = "sketch"
    TRACE = "trace"
    STORY = "story"


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

        # Rotation (0 or 180 degrees)
        self._rotated = False

        # Sketch mode state
        self._sketch_name: Optional[str] = None
        self._sketch_custom: Optional[dict] = None  # {"name":..., "steps":...}
        self._sketch_step: int = 0

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
        self._pending_image_data = None  # (raw_bytes, size) for thread-safe surface creation
        self._imagine_prompt: str = ""  # What the child asked to see
        self._loading_text: str = ""    # Loading message while generating

        # Trace mode state (line art from uploaded image)
        self._trace_surface = None      # pygame.Surface of line art
        self._trace_name: str = ""

        # Story mode state
        self._story_scene: str = ""         # Current scene ID from card_ui
        self._story_bg_color = (10, 10, 30) # Background color for scene
        self._story_title: str = ""         # Story title
        self._story_surface = None          # Cached pygame surface

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
        global PROJ_WIDTH, PROJ_HEIGHT
        try:
            # Use the second display (HDMI) if available
            os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

            import pygame
            # Only init display + font — NOT pygame.init() which grabs the
            # audio device and blocks aplay/TTS from working
            pygame.display.init()
            pygame.font.init()

            # Detect actual screen resolution
            info = pygame.display.Info()
            PROJ_WIDTH = info.current_w
            PROJ_HEIGHT = info.current_h
            logger.info("Projector display detected: {}x{}", PROJ_WIDTH, PROJ_HEIGHT)

            self._screen = pygame.display.set_mode(
                (PROJ_WIDTH, PROJ_HEIGHT),
                pygame.FULLSCREEN | pygame.NOFRAME,
            )
            pygame.display.set_caption("RPi Bot Projector")
            pygame.mouse.set_visible(False)

            # Scale fonts based on screen height (720p baseline)
            scale = PROJ_HEIGHT / 720
            self._font_large = pygame.font.Font(None, max(60, int(240 * scale)))
            self._font_medium = pygame.font.Font(None, max(40, int(120 * scale)))
            self._font_small = pygame.font.Font(None, max(28, int(72 * scale)))

            self._pygame_initialized = True
            logger.info("Projector pygame initialized ({}x{}, scale={:.2f})", PROJ_WIDTH, PROJ_HEIGHT, scale)
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

    def rotate(self):
        """Toggle 180-degree rotation (table vs wall projection)."""
        with self._lock:
            self._rotated = not self._rotated
            self._needs_redraw = True
        logger.info("Projector rotation: {}", "180°" if self._rotated else "0°")

    @property
    def rotated(self) -> bool:
        return self._rotated

    # ── Sketch mode controls ────────────────────────────────────────

    def set_sketch(self, name: str):
        """Set a built-in drawing to trace."""
        with self._lock:
            self._sketch_name = name
            self._sketch_custom = None
            self._sketch_step = 0
            self._needs_redraw = True

    def set_custom_sketch(self, name: str, steps: list):
        """Set a custom drawing from image upload."""
        with self._lock:
            self._sketch_name = "__custom__"
            self._sketch_custom = {"name": name, "steps": steps}
            self._sketch_step = 0
            self._needs_redraw = True

    def _get_drawing(self):
        """Get the current drawing data (built-in, pro, or custom)."""
        if self._sketch_name == "__custom__" and self._sketch_custom:
            return self._sketch_custom
        from modules.sketch_drawings import SKETCH_DRAWINGS
        from modules.sketch_pro import PRO_DRAWINGS
        return SKETCH_DRAWINGS.get(self._sketch_name) or PRO_DRAWINGS.get(self._sketch_name)

    def next_sketch_step(self):
        """Advance to next step."""
        drawing = self._get_drawing()
        if not drawing:
            return
        with self._lock:
            if self._sketch_step < len(drawing["steps"]):
                self._sketch_step += 1
                self._needs_redraw = True

    def prev_sketch_step(self):
        """Go back one step."""
        with self._lock:
            if self._sketch_step > 0:
                self._sketch_step -= 1
                self._needs_redraw = True

    def restart_sketch(self):
        """Reset to step 0."""
        with self._lock:
            self._sketch_step = 0
            self._needs_redraw = True

    def get_sketch_info(self) -> Optional[tuple]:
        """Return (name, current_step, total_steps, instruction) or None."""
        drawing = self._get_drawing()
        if not drawing:
            return None
        steps = drawing["steps"]
        step = self._sketch_step
        total = len(steps)
        instruction = steps[step]["instruction"] if step < total else "All done!"
        return (drawing["name"], step, total, instruction)

    def get_current_flashcard(self) -> Optional[tuple]:
        """Get the current flashcard (name, description)."""
        cards = FLASHCARDS.get(self._current_category, [])
        if cards and 0 <= self._current_index < len(cards):
            return cards[self._current_index]
        return None

    # ── Imagination mode ────────────────────────────────────────────

    def show_trace_image(self, pil_image, name: str = ""):
        """Display a line art image for tracing on the projector.

        Uses IMAGINE mode rendering (proven working). Passes raw data
        to render thread for thread-safe pygame surface creation.
        """
        try:
            rgb_img = pil_image.convert("RGB")
            with self._lock:
                self._pending_image_data = (rgb_img.tobytes(), rgb_img.size)
                self._imagine_prompt = f"Trace: {name}"
                self._trace_name = name
                self._mode = ProjectorMode.IMAGINE
                self._needs_redraw = True

            logger.info("Showing trace image: '{}'", name)

        except Exception as e:
            logger.error("Failed to display trace image: {}", e)

    def show_loading_message(self, prompt: str):
        """Show a loading screen while DALL-E generates the image."""
        with self._lock:
            self._loading_text = prompt
            self._mode = ProjectorMode.LOADING
            self._needs_redraw = True

    def show_generated_image(self, pil_image, prompt: str):
        """Display an AI-generated PIL Image on the projector.

        Stores raw image data; the render thread creates the pygame Surface.
        This avoids cross-thread pygame calls.
        """
        try:
            rgb_img = pil_image.convert("RGB")
            with self._lock:
                # Store raw data for the render thread to convert
                self._pending_image_data = (rgb_img.tobytes(), rgb_img.size)
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
                    elif mode == ProjectorMode.TRACE:
                        self._render_trace(pygame)
                    elif mode == ProjectorMode.SKETCH:
                        self._render_sketch(pygame)
                    elif mode == ProjectorMode.IMAGINE:
                        self._render_imagine(pygame)
                    elif mode == ProjectorMode.STORY:
                        self._render_story(pygame)
                    elif mode == ProjectorMode.LOADING:
                        self._render_loading(pygame)
                    elif mode == ProjectorMode.BLANK:
                        self._screen.fill((0, 0, 0))
                    else:
                        self._screen.fill((0, 0, 0))

                    # Rotate 180° if projecting onto table
                    if self._rotated:
                        rotated = pygame.transform.rotate(self._screen, 180)
                        self._screen.blit(rotated, (0, 0))

                    pygame.display.flip()
                    with self._lock:
                        # Keep redrawing in LOADING mode (animation) and
                        # IMAGINE mode (pending image data may arrive)
                        if self._mode not in (ProjectorMode.LOADING, ProjectorMode.IMAGINE):
                            self._needs_redraw = False

                time.sleep(0.05 if mode == ProjectorMode.LOADING else 0.05)

            except Exception as e:
                logger.debug("Projector render error: {}", e)
                time.sleep(0.5)

    def _fit_text(self, pygame, text: str, font, max_width: int, color):
        """Render text, shrinking font if it exceeds max_width."""
        surf = font.render(text, True, color)
        if surf.get_width() <= max_width:
            return surf
        # Shrink font until it fits
        size = font.get_height()
        while size > 20:
            size -= 4
            smaller = pygame.font.Font(None, size)
            surf = smaller.render(text, True, color)
            if surf.get_width() <= max_width:
                return surf
        return surf

    def _render_flashcard(self, pygame):
        """Render current flashcard — image on left, word + description on right."""
        card = self.get_current_flashcard()
        if not card:
            return

        name, description = card
        category = self._current_category
        color = CATEGORY_COLORS.get(category, (255, 255, 255))

        self._screen.fill((20, 20, 30))

        # Safe margins
        margin = int(PROJ_WIDTH * 0.03)

        # Try to load the image
        image = self._load_flashcard_image(pygame, category, name)

        if image:
            # ── Layout with image: image left, text right ──
            img_w, img_h = image.get_size()

            # Center image in left half
            img_x = (PROJ_WIDTH // 2 - img_w) // 2
            img_y = (PROJ_HEIGHT - img_h) // 2
            self._screen.blit(image, (img_x, img_y))

            # Text in right half — constrain to right half width minus margins
            text_cx = PROJ_WIDTH * 3 // 4
            text_max_w = PROJ_WIDTH // 2 - margin * 2

            # Render all text surfaces first
            cat_surf = self._fit_text(pygame, category.upper(), self._font_small, text_max_w, (150, 150, 150))
            word_surf = self._fit_text(pygame, name, self._font_large, text_max_w, color)
            desc_surf = self._fit_text(pygame, description, self._font_medium, text_max_w, (200, 200, 200))

            # Stack vertically with gap, centered on screen
            gap = int(PROJ_HEIGHT * 0.04)
            total_h = cat_surf.get_height() + gap + word_surf.get_height() + gap + desc_surf.get_height()
            top_y = (PROJ_HEIGHT - total_h) // 2

            cat_rect = cat_surf.get_rect(center=(text_cx, top_y + cat_surf.get_height() // 2))
            self._screen.blit(cat_surf, cat_rect)

            word_y = top_y + cat_surf.get_height() + gap
            word_rect = word_surf.get_rect(center=(text_cx, word_y + word_surf.get_height() // 2))
            self._screen.blit(word_surf, word_rect)

            desc_y = word_y + word_surf.get_height() + gap
            desc_rect = desc_surf.get_rect(center=(text_cx, desc_y + desc_surf.get_height() // 2))
            self._screen.blit(desc_surf, desc_rect)

        else:
            # ── Fallback: text-only centered layout ──
            text_max_w = PROJ_WIDTH - margin * 2

            cat_surf = self._fit_text(pygame, category.upper(), self._font_small, text_max_w, (150, 150, 150))
            word_surf = self._fit_text(pygame, name, self._font_large, text_max_w, color)
            desc_surf = self._fit_text(pygame, description, self._font_medium, text_max_w, (200, 200, 200))

            gap = int(PROJ_HEIGHT * 0.04)
            total_h = cat_surf.get_height() + gap + word_surf.get_height() + gap + desc_surf.get_height()
            top_y = (PROJ_HEIGHT - total_h) // 2

            cat_rect = cat_surf.get_rect(center=(PROJ_WIDTH // 2, top_y + cat_surf.get_height() // 2))
            self._screen.blit(cat_surf, cat_rect)

            word_y = top_y + cat_surf.get_height() + gap
            word_rect = word_surf.get_rect(center=(PROJ_WIDTH // 2, word_y + word_surf.get_height() // 2))
            self._screen.blit(word_surf, word_rect)

            desc_y = word_y + word_surf.get_height() + gap
            desc_rect = desc_surf.get_rect(center=(PROJ_WIDTH // 2, desc_y + desc_surf.get_height() // 2))
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

        # Build surface from pending raw data (thread-safe pygame call)
        pending = None
        with self._lock:
            if self._pending_image_data is not None:
                pending = self._pending_image_data
                self._pending_image_data = None

        if pending:
            raw, size = pending
            try:
                surface = pygame.image.fromstring(raw, size, "RGB")
                margin = int(PROJ_WIDTH * 0.03)
                max_w = PROJ_WIDTH - margin * 2
                max_h = PROJ_HEIGHT - 80
                img_w, img_h = surface.get_size()
                scale = min(max_w / img_w, max_h / img_h)
                self._generated_surface = pygame.transform.smoothscale(
                    surface, (int(img_w * scale), int(img_h * scale))
                )
                logger.info("Projector: new image rendered ({}x{})", size[0], size[1])
            except Exception as e:
                logger.error("Failed to build imagine surface: {}", e)

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

    # ── Trace rendering (line art image) ────────────────────────────

    def _render_trace(self, pygame):
        """Render uploaded line art image centered on black background."""
        self._screen.fill((0, 0, 0))

        if self._trace_surface:
            img_w, img_h = self._trace_surface.get_size()
            x = (PROJ_WIDTH - img_w) // 2
            y = (PROJ_HEIGHT - img_h) // 2
            self._screen.blit(self._trace_surface, (x, y))

    # ── Sketch rendering ───────────────────────────────────────────

    def _render_sketch(self, pygame):
        """Render step-by-step drawing for tracing."""
        drawing = self._get_drawing()
        if not drawing:
            return

        self._screen.fill((5, 5, 8))  # Near-black for table projection

        steps = drawing["steps"]
        current = self._sketch_step
        total = len(steps)

        # Drawing area — leave room for title at top and instruction at bottom
        margin = int(PROJ_WIDTH * 0.05)
        title_h = int(PROJ_HEIGHT * 0.10)
        footer_h = int(PROJ_HEIGHT * 0.15)
        draw_x = margin
        draw_y = title_h
        draw_w = PROJ_WIDTH - margin * 2
        draw_h = PROJ_HEIGHT - title_h - footer_h

        def to_px(cx, cy):
            return (int(draw_x + cx / 100 * draw_w),
                    int(draw_y + cy / 100 * draw_h))

        def scale_r(val):
            return max(2, int(val / 100 * min(draw_w, draw_h)))

        # Render completed steps (dimmed)
        for i in range(min(current, total)):
            self._draw_elements(pygame, steps[i]["elements"],
                                (50, 50, 60), to_px, scale_r)

        # Render current step (bright yellow)
        if current < total:
            self._draw_elements(pygame, steps[current]["elements"],
                                (255, 255, 80), to_px, scale_r)

        # Title — detect craft vs drawing
        is_craft = "cut" in drawing["name"].lower() or "fold" in drawing["name"].lower() or "craft" in drawing["name"].lower()
        title_text = f"Let's Build: {drawing['name']}" if is_craft else f"Let's Draw: {drawing['name']}"
        title_surf = self._fit_text(pygame, title_text, self._font_medium,
                                    PROJ_WIDTH - margin * 2, (200, 200, 220))
        title_rect = title_surf.get_rect(center=(PROJ_WIDTH // 2, title_h // 2))
        self._screen.blit(title_surf, title_rect)

        # Instruction text
        if current < total:
            instr = f"Step {current + 1}: {steps[current]['instruction']}"
            instr_surf = self._fit_text(pygame, instr, self._font_medium,
                                        PROJ_WIDTH - margin * 2, (255, 255, 80))
            instr_y = PROJ_HEIGHT - footer_h + int(footer_h * 0.25)
            instr_rect = instr_surf.get_rect(center=(PROJ_WIDTH // 2, instr_y))
            self._screen.blit(instr_surf, instr_rect)
        else:
            # Completed!
            done_surf = self._fit_text(pygame, "Great job! You did it!", self._font_large,
                                       PROJ_WIDTH - margin * 2, (80, 255, 80))
            done_rect = done_surf.get_rect(center=(PROJ_WIDTH // 2,
                                                    PROJ_HEIGHT - footer_h + int(footer_h * 0.3)))
            self._screen.blit(done_surf, done_rect)

        # Step counter
        counter = f"{min(current + 1, total)} / {total}"
        count_surf = self._font_small.render(counter, True, (90, 90, 90))
        count_rect = count_surf.get_rect(center=(PROJ_WIDTH // 2,
                                                  PROJ_HEIGHT - int(footer_h * 0.15)))
        self._screen.blit(count_surf, count_rect)

    def _draw_elements(self, pygame, elements, color, to_px, scale_r):
        """Draw a list of sketch primitives."""
        for elem in elements:
            t = elem["type"]
            # Line width scaled to screen — base width ~3px at 720p
            base_w = elem.get("width", 3)
            w = max(2, int(base_w * min(PROJ_WIDTH, PROJ_HEIGHT) / 350))

            if t == "line":
                p1 = to_px(*elem["start"])
                p2 = to_px(*elem["end"])
                pygame.draw.line(self._screen, color, p1, p2, w)

            elif t == "circle":
                cx, cy = to_px(*elem["center"])
                r = scale_r(elem["radius"])
                pygame.draw.circle(self._screen, color, (cx, cy), r, w)

            elif t == "dot":
                cx, cy = to_px(*elem["center"])
                r = scale_r(elem["radius"])
                pygame.draw.circle(self._screen, color, (cx, cy), r, 0)  # filled

            elif t == "ellipse":
                rx, ry, rw, rh = elem["rect"]
                px, py = to_px(rx, ry)
                pw = max(4, int(rw / 100 * (PROJ_WIDTH - int(PROJ_WIDTH * 0.1))))
                ph = max(4, int(rh / 100 * (PROJ_HEIGHT - int(PROJ_HEIGHT * 0.25))))
                pygame.draw.ellipse(self._screen, color,
                                    pygame.Rect(px, py, pw, ph), w)

            elif t == "arc":
                rx, ry, rw, rh = elem["rect"]
                px, py = to_px(rx, ry)
                pw = max(4, int(rw / 100 * (PROJ_WIDTH - int(PROJ_WIDTH * 0.1))))
                ph = max(4, int(rh / 100 * (PROJ_HEIGHT - int(PROJ_HEIGHT * 0.25))))
                pygame.draw.arc(self._screen, color,
                                pygame.Rect(px, py, pw, ph),
                                elem["start_angle"], elem["end_angle"], w)

            elif t == "lines":
                points = [to_px(*p) for p in elem["points"]]
                closed = elem.get("closed", False)
                if closed:
                    pygame.draw.polygon(self._screen, color, points, w)
                else:
                    pygame.draw.lines(self._screen, color, False, points, w)

            elif t == "dashed":
                # Dashed line for fold marks (crafts)
                points = [to_px(*p) for p in elem["points"]]
                dash_len = max(6, int(12 * min(PROJ_WIDTH, PROJ_HEIGHT) / 720))
                gap_len = max(4, int(8 * min(PROJ_WIDTH, PROJ_HEIGHT) / 720))
                dash_color = elem.get("color", (180, 180, 80))  # yellow-ish
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    dx = x2 - x1
                    dy = y2 - y1
                    import math as _m
                    length = _m.hypot(dx, dy)
                    if length < 1:
                        continue
                    ux, uy = dx / length, dy / length
                    drawn = 0.0
                    on = True
                    while drawn < length:
                        seg = dash_len if on else gap_len
                        seg = min(seg, length - drawn)
                        sx = x1 + ux * drawn
                        sy = y1 + uy * drawn
                        ex = x1 + ux * (drawn + seg)
                        ey = y1 + uy * (drawn + seg)
                        if on:
                            pygame.draw.line(self._screen, dash_color,
                                             (int(sx), int(sy)), (int(ex), int(ey)), w)
                        drawn += seg
                        on = not on

    # ── Story mode ────────────────────────────────────────────────

    def set_story_scene(self, scene_id: str, bg_color: tuple = (10, 10, 30), title: str = ""):
        """Set the current story scene to render on projector.

        Uses PIL draw_scene() from card_ui to render the illustration,
        then converts to pygame surface for display.
        """
        with self._lock:
            self._story_scene = scene_id
            self._story_bg_color = bg_color
            self._story_title = title
            self._story_surface = None  # Force re-render
            self._mode = ProjectorMode.STORY
            self._needs_redraw = True
        logger.info("Projector story scene: '{}' ({})", scene_id, title)

    def _render_story(self, pygame):
        """Render story illustration fullscreen on projector using PIL draw_scene.

        The draw_scene primitives use hardcoded pixel sizes designed for
        ~480×320. We render at a base resolution (480×270 for 16:9 match)
        then upscale to projector res with LANCZOS for smooth results.
        """
        # Build surface from PIL if not cached
        if self._story_surface is None and self._story_scene:
            try:
                from PIL import Image, ImageDraw
                from display.card_ui import draw_scene

                # Render at base resolution matching projector aspect ratio
                # 480×270 = 16:9 matching 1920×1080
                base_w, base_h = 480, 270
                img = Image.new("RGB", (base_w, base_h), self._story_bg_color)
                draw = ImageDraw.Draw(img)
                draw_scene(draw, self._story_scene, base_w, base_h, 0)

                # Upscale to projector resolution with LANCZOS for smooth output
                img = img.resize((PROJ_WIDTH, PROJ_HEIGHT), Image.LANCZOS)

                # Convert PIL → pygame surface
                raw = img.tobytes()
                self._story_surface = pygame.image.fromstring(raw, (PROJ_WIDTH, PROJ_HEIGHT), "RGB")
                logger.debug("Story scene rendered: {} ({}x{} → {}x{})",
                             self._story_scene, base_w, base_h, PROJ_WIDTH, PROJ_HEIGHT)
            except Exception as e:
                logger.error("Failed to render story scene '{}': {}", self._story_scene, e)

        # Draw the scene
        if self._story_surface:
            self._screen.blit(self._story_surface, (0, 0))
        else:
            self._screen.fill(self._story_bg_color)

        # Show title at bottom (subtle)
        if self._story_title:
            title_surf = self._font_small.render(self._story_title, True, (120, 120, 150))
            title_rect = title_surf.get_rect(center=(PROJ_WIDTH // 2, PROJ_HEIGHT - 30))
            self._screen.blit(title_surf, title_rect)

    # ── Cleanup ───────────────────────────────────────────────────

    def close(self):
        self.stop()

    def __del__(self):
        self.close()
