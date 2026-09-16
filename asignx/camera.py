"""
ASignX Hand Recognition Module - Webcam Stream Manager.

Handles webcam access, high-throughput threaded frame capture, resolution
configuration, exposure control for locked 30 FPS, and graceful failure handling.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Exception raised when camera capture operations fail."""
    pass


class WebcamStream:
    """Manages OpenCV VideoCapture connection and frame streaming.

    Supports asynchronous background threaded capture and exposure optimization
    to eliminate hardware latency and lock in a fluid 30+ FPS.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = 640,
        height: Optional[int] = 480,
        target_fps: Optional[int] = 30,
        threaded: bool = True,
        exposure: Optional[float] = -5.0,
    ) -> None:
        """Initialize camera stream.

        Args:
            camera_index: Device index (default 0 for default primary webcam).
            width: Desired capture frame width (pixels). Default 640 for 30+ FPS.
            height: Desired capture frame height (pixels). Default 480 for 30+ FPS.
            target_fps: Desired capture frame rate (default 30).
            threaded: If True (default), acquires frames in a dedicated background
                thread to eliminate sensor wait latency and achieve maximum FPS.
            exposure: Camera exposure value (default -5.0 = 1/32s for locked 30 FPS).
                Pass None for driver default auto-exposure.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.threaded = threaded
        self.exposure = exposure

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running: bool = False
        self._latest_frame: Optional[np.ndarray] = None
        self._last_read_success: bool = False
        self._new_frame_event = threading.Event()

    def set_exposure(self, exposure: Optional[float]) -> bool:
        """Dynamically adjust camera exposure.

        Args:
            exposure: Target exposure in DirectShow units:
                -5.0 corresponds to ~1/32s (guarantees 30 FPS).
                -6.0 corresponds to ~1/64s (guarantees 60 FPS).
                None enables driver automatic exposure.

        Returns:
            True if settings were applied, False otherwise.
        """
        if self._cap is None or not self._cap.isOpened():
            return False

        self.exposure = exposure
        try:
            if exposure is None:
                # DirectShow auto-exposure mode
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                # DirectShow manual exposure mode
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self._cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            return True
        except Exception as e:
            logger.warning("Failed to set camera exposure: %s", e)
            return False

    def start(self) -> WebcamStream:
        """Open the camera, configure properties, and start background reader if threaded.

        Raises:
            CameraError: If the camera cannot be opened or initialized.
        """
        # On Windows, cv2.CAP_DSHOW provides fast, reliable index-based capture
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback to default backend if DirectShow is unavailable
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            raise CameraError(
                f"Failed to access webcam at index {self.camera_index}. "
                "Please verify that a camera is connected and not in use by another application."
            )

        # Minimize driver buffering to prevent lag
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Attempt to set MJPG fourcc for highest FPS support on USB webcams
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        # Set resolution if specified
        if self.width is not None and self.height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.target_fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Apply exposure optimization if requested
        if self.exposure is not None:
            self.set_exposure(self.exposure)

        # Verify stream by reading an initial test frame
        success, initial_frame = self._cap.read()
        if not success or initial_frame is None:
            self.stop()
            raise CameraError(
                f"Webcam at index {self.camera_index} opened but could not read frames."
            )

        self._latest_frame = initial_frame
        self._last_read_success = True

        if self.threaded:
            self._running = True
            self._new_frame_event.clear()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="ASignX-WebcamReader",
                daemon=True,
            )
            self._thread.start()

        return self

    def _capture_loop(self) -> None:
        """Background worker thread continuously grabbing frames from hardware."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            success, frame = self._cap.read()
            if success and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._last_read_success = True
                self._new_frame_event.set()
            else:
                time.sleep(0.001)

    @property
    def is_opened(self) -> bool:
        """Check whether webcam is currently open and active."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return actual (width, height) configured on the camera."""
        if not self.is_opened:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def fps(self) -> float:
        """Return frame rate configured on the camera."""
        if not self.is_opened:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the most recent frame from the webcam.

        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray])
        """
        if not self.is_opened:
            return False, None

        if self.threaded:
            with self._lock:
                if self._latest_frame is not None:
                    return self._last_read_success, self._latest_frame.copy()
                return False, None

        success, frame = self._cap.read()
        return success, frame

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Generator yielding incoming frames continuously at high FPS."""
        if not self.is_opened:
            self.start()

        while self.is_opened and (not self.threaded or self._running):
            if self.threaded:
                # Wait for a new frame from the background capture thread
                signaled = self._new_frame_event.wait(timeout=0.1)
                self._new_frame_event.clear()
                if not signaled and not self._running:
                    break

            success, frame = self.read()
            if not success or frame is None:
                if not self.threaded:
                    logger.warning("Failed to retrieve frame from camera stream.")
                    break
                continue
            yield frame

    def stop(self) -> None:
        """Release camera hardware resources and terminate background worker thread."""
        self._running = False
        self._new_frame_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            if self._cap.isOpened():
                self._cap.release()
            self._cap = None

    def __enter__(self) -> WebcamStream:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
