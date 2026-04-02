"""DALL-E 3 image generation for the imagination/projector feature.

Generates child-safe images from voice descriptions and returns
PIL Images for display on the HDMI projector.

Wraps every prompt with a child-safety prefix to ensure age-appropriate
content regardless of what the child asks for.
"""

import time
from io import BytesIO
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image


# Child-safety prompt wrapper
SAFETY_PREFIX = (
    "A colorful, friendly, cartoon-style children's illustration of: {prompt}. "
    "Style: bright colors, cute, age-appropriate for young children (ages 3-6), "
    "no text overlays, no scary or violent elements, warm and inviting, "
    "digital art, simple and clear composition."
)


class ImageGenerator:
    """Generate child-safe images via OpenAI DALL-E 3."""

    def __init__(self, api_key: str, save_dir: Optional[Path] = None):
        self._api_key = api_key
        self._client = None
        self._save_dir = save_dir
        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    def update_api_key(self, api_key: str):
        self._api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)

    async def generate(self, prompt: str) -> Optional[Image.Image]:
        """Generate an image from a child's description.

        Args:
            prompt: The child's raw description (e.g., "a flying dinosaur")

        Returns:
            PIL Image (1024x1024) or None on failure.
        """
        if not self._api_key:
            logger.warning("No OpenAI API key for image generation")
            return None

        self._ensure_client()

        safe_prompt = SAFETY_PREFIX.format(prompt=prompt)
        logger.info("Generating image: '{}'", prompt)

        try:
            import httpx

            response = await self._client.images.generate(
                model="dall-e-3",
                prompt=safe_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            image_url = response.data[0].url
            logger.debug("DALL-E returned URL, downloading...")

            # Download the image
            async with httpx.AsyncClient() as http:
                img_response = await http.get(image_url, timeout=30)
                img_response.raise_for_status()

            image = Image.open(BytesIO(img_response.content)).convert("RGB")
            logger.info("Image generated successfully ({}x{})", image.width, image.height)

            # Save for parent review
            if self._save_dir:
                ts = int(time.time())
                safe_name = "".join(c if c.isalnum() or c == " " else "" for c in prompt)
                safe_name = safe_name.strip().replace(" ", "_")[:50]
                save_path = self._save_dir / f"{ts}_{safe_name}.png"
                image.save(str(save_path), "PNG")
                logger.debug("Saved generated image: {}", save_path.name)

            return image

        except Exception as e:
            logger.error("Image generation failed: {}", e)
            return None
