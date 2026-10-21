"""
Unit tests for CameraStream and VisualDebugger.
"""

import numpy as np
import pytest

from asignx.camera import CameraError, WebcamStream
from asignx.models import (
    HandDetectionResult,
    HandLandmarks,
)
from asignx.visualizer import VisualDebugger
from tests.test_models import create_sample_landmarks


def test_invalid_camera_index_raises_camera_error() -> None:
    # Camera index 999 is almost certainly not connected
    camera = WebcamStream(camera_index=999)
    with pytest.raises(CameraError, match="Failed to access webcam"):
        camera.start()


def test_visualizer_overlay_no_hand() -> None:
    vis = VisualDebugger()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = HandDetectionResult(hand_detected=False, hand=None)

    out = vis.render_overlay(frame, res, fps=30.0)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype

    # Test disabling HUD
    vis.show_hud = False
    out_no_hud = vis.render_overlay(frame, res, fps=30.0)
    assert np.array_equal(out_no_hud, frame)


def test_visualizer_overlay_with_hand() -> None:
    vis = VisualDebugger()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hand = HandLandmarks(
        landmarks=create_sample_landmarks(21),
        handedness="Right",
        score=0.98,
    )
    res = HandDetectionResult(
        hand_detected=True,
        hand=hand,
        all_hands=[hand],
    )

    out = vis.render_overlay(frame, res, fps=28.5)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_visualizer_overlay_with_two_hands() -> None:
    vis = VisualDebugger()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left_hand = HandLandmarks(
        landmarks=create_sample_landmarks(21),
        handedness="Left",
        score=0.98,
    )
    right_hand = HandLandmarks(
        landmarks=create_sample_landmarks(21),
        handedness="Right",
        score=0.99,
    )
    res = HandDetectionResult(
        hand_detected=True,
        hand=left_hand,
        all_hands=[left_hand, right_hand],
    )

    out = vis.render_overlay(frame, res, fps=35.0)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_visualizer_overlay_with_sign_recognition_hud() -> None:
    from ml.predict import Prediction, Recognition

    vis = VisualDebugger()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hand = HandLandmarks(
        landmarks=create_sample_landmarks(21),
        handedness="Right",
        score=0.99,
    )
    res = HandDetectionResult(hand_detected=True, hand=hand, all_hands=[hand])
    rec = Recognition(
        hand="Right",
        prediction=Prediction(label="A", confidence=0.985, raw_label="A"),
        stable_label="A",
        emitted_label="A",
    )

    out_Without_rec = vis.render_overlay(frame, res, fps=30.0)
    out_with_rec = vis.render_overlay(frame, res, fps=30.0, recognitions=[rec])
    assert out_with_rec.shape == frame.shape
    assert not np.array_equal(out_Without_rec, out_with_rec)


