import asyncio
from typing import Optional

import numpy as np
from loguru import logger


class Camera:
    """RPi Camera Module 3 interface using picamera2."""

    def __init__(self, resolution: tuple = (640, 480)):
        self.resolution = resolution
        self._camera = None

    def _ensure_camera(self):
        """Lazily initialize the camera."""
        if self._camera is not None:
            return

        try:
            from picamera2 import Picamera2
            from libcamera import controls

            self._camera = Picamera2()
            config = self._camera.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self._camera.configure(config)
            self._camera.start()

            # Enable continuous autofocus on RPi Camera Module 3
            try:
                self._camera.set_controls({
                    "AfMode": controls.AfModeEnum.Continuous,
                    "AfSpeed": controls.AfSpeedEnum.Fast,
                })
                logger.info("Autofocus enabled: continuous + fast mode")
            except Exception as e:
                logger.warning("Could not enable autofocus ({}). Camera may not support AF.", e)

            logger.info("Camera initialized: {}x{}", *self.resolution)
        except ImportError:
            logger.warning("picamera2 not available. Camera features disabled.")
        except Exception as e:
            logger.error("Camera init error: {}", e)

    async def capture(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera.

        Returns:
            RGB numpy array (H, W, 3) or None if camera unavailable.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._capture_sync)

    def _capture_sync(self) -> Optional[np.ndarray]:
        self._ensure_camera()
        if self._camera is None:
            return None

        try:
            import time

            # Trigger autofocus and wait for it to lock
            try:
                from libcamera import controls
                self._camera.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
                time.sleep(0.5)  # Give AF time to lock
            except Exception:
                pass  # AF not available, continue anyway

            frame = self._camera.capture_array()
            logger.debug("Captured frame: shape={}", frame.shape)
            return frame
        except Exception as e:
            logger.error("Camera capture error: {}", e)
            return None

    def close(self):
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None

    def __del__(self):
        self.close()
