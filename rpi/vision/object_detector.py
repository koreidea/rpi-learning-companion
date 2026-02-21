import asyncio
from pathlib import Path

import numpy as np
from loguru import logger


class ObjectDetector:
    """Object detection using YOLOv8-nano.

    Loaded on-demand to save RAM when camera is not active.
    """

    def __init__(self, model_dir: Path, confidence: float = 0.4):
        self.model_dir = model_dir
        self.confidence = confidence
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO

            model_path = self.model_dir / "yolov8n.pt"
            if model_path.exists():
                self._model = YOLO(str(model_path))
            else:
                # Downloads automatically if not present
                self._model = YOLO("yolov8n.pt")
            logger.info("YOLOv8n object detector loaded.")
        except ImportError:
            logger.warning("ultralytics not installed. Object detection disabled.")

    async def detect(self, image: np.ndarray) -> list[dict]:
        """Detect objects in an image.

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            List of detections: [{"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._detect_sync, image)

    def _detect_sync(self, image: np.ndarray) -> list[dict]:
        self._ensure_model()
        if self._model is None:
            return []

        try:
            results = self._model(image, conf=self.confidence, verbose=False)
            detections = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()

                    detections.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": [round(x) for x in bbox],
                    })

            logger.debug("Detected {} objects: {}", len(detections),
                        [d["label"] for d in detections])
            return detections

        except Exception as e:
            logger.error("Object detection error: {}", e)
            return []
