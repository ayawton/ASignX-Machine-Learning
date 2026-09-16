"""
ASignX - Hand Recognition Module.

Responsible for real-time hand detection and 21-landmark extraction
using MediaPipe Hands.
"""

import os
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



from asignx.camera import CameraError, WebcamStream
from asignx.detector import HandDetector
from asignx.models import (
    HAND_LANDMARK_NAMES,
    HandDetectionResult,
    HandLandmarks,
    Landmark,
)
from asignx.visualizer import VisualDebugger

__all__ = [
    "CameraError",
    "HandDetector",
    "HandDetectionResult",
    "HandLandmarks",
    "HAND_LANDMARK_NAMES",
    "Landmark",
    "VisualDebugger",
    "WebcamStream",
]
