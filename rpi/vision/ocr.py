import asyncio
from pathlib import Path

import numpy as np
from loguru import logger


class OCRReader:
    """Optical Character Recognition using PaddleOCR.

    Used for reading book pages, labels, and text that the child shows to the camera.
    Loaded on-demand to save RAM.
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self._reader = None

    def _ensure_reader(self):
        if self._reader is not None:
            return

        try:
            from paddleocr import PaddleOCR

            self._reader = PaddleOCR(
                use_angle_cls=True,
                lang=self.language,
                show_log=False,
            )
            logger.info("PaddleOCR loaded for language: {}", self.language)
        except ImportError:
            logger.warning("paddleocr not installed. OCR features disabled.")

    async def read_text(self, image: np.ndarray) -> str:
        """Extract text from an image.

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            Extracted text as a single string.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_sync, image)

    def _read_sync(self, image: np.ndarray) -> str:
        self._ensure_reader()
        if self._reader is None:
            return ""

        try:
            results = self._reader.ocr(image, cls=True)
            if not results or not results[0]:
                return ""

            # Extract text from OCR results
            lines = []
            for line in results[0]:
                text = line[1][0]  # text content
                confidence = line[1][1]  # confidence score
                if confidence > 0.5:
                    lines.append(text)

            full_text = " ".join(lines)
            logger.debug("OCR result: '{}' ({} lines)", full_text[:100], len(lines))
            return full_text

        except Exception as e:
            logger.error("OCR error: {}", e)
            return ""
