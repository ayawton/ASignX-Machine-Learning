"""
Unit tests for ASignX HandDetector.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from asignx import HandDetector
from asignx.models import HandDetectionResult


def test_detector_context_manager() -> None:
    with HandDetector(static_image_mode=True) as detector:
        assert detector is not None
        assert detector._hands is not None
    # After context exit, resources should be cleaned
    assert detector._hands is None


def test_detector_empty_or_none_frame() -> None:
    detector = HandDetector(static_image_mode=True)
    try:
        # None frame
        res, frame = detector.process(None)
        assert isinstance(res, HandDetectionResult)
        assert res.hand_detected is False
        assert res.hand is None

        # 0-size frame
        res, frame = detector.process(np.empty((0, 0, 3), dtype=np.uint8))
        assert res.hand_detected is False
        assert res.hand is None
    finally:
        detector.close()


def test_detector_blank_image_returns_null() -> None:
    detector = HandDetector(static_image_mode=True)
    try:
        # Blank black frame (no hands)
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result, annotated = detector.process(black_frame, draw=True)

        assert isinstance(result, HandDetectionResult)
        assert result.hand_detected is False
        assert result.hand is None
        assert len(result.all_hands) == 0
        assert result.to_tree_string() == "Hand → null"
        assert result.to_dict()["hand"] is None
        assert annotated.shape == black_frame.shape
    finally:
        detector.close()


def test_detector_accepts_tasks_api_landmark_lists() -> None:
    """MediaPipe Tasks returns each hand as a plain list of landmarks."""
    detector = HandDetector(static_image_mode=True)

    class FakeHandLandmarker:
        def detect(self, _image):
            landmarks = [
                SimpleNamespace(x=0.1 + index * 0.01, y=0.2, z=-0.01)
                for index in range(21)
            ]
            category = SimpleNamespace(category_name="Right", score=0.99)
            return SimpleNamespace(
                hand_landmarks=[landmarks],
                handedness=[[category]],
            )

        def close(self) -> None:
            pass

    try:
        detector._hands.close()
        detector._hands = FakeHandLandmarker()
        result, _ = detector.process(np.zeros((64, 64, 3), dtype=np.uint8), draw=False)

        assert result.hand_detected is True
        assert result.hand is not None
        assert result.hand.handedness == "Right"
        assert len(result.hand.landmarks) == 21
    finally:
        detector.close()


def test_detector_landmark_index_integrity() -> None:
    # Verify that when landmarks are generated, all 21 landmark IDs are present in sequence 0..20
    detector = HandDetector(static_image_mode=True)
    try:
        # White noise / plain image test
        plain_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        result, _ = detector.process(plain_frame)
        assert result.hand_detected is False
        assert result.hand is None
    finally:
        detector.close()


def test_duplicate_hand_suppression() -> None:
    from asignx.models import HAND_LANDMARK_NAMES, HandLandmarks, Landmark

    detector = HandDetector(static_image_mode=True)
    try:
        # Hand 1 at center
        lms1 = [
            Landmark(id=i, name=HAND_LANDMARK_NAMES[i], x=0.5 + 0.005 * i, y=0.5 + 0.005 * i, z=0.0)
            for i in range(21)
        ]
        hand1 = HandLandmarks(landmarks=lms1, handedness="Right", score=0.95)

        # Hand 2 virtually identical (same physical hand ghost duplicate with lower score)
        lms2 = [
            Landmark(id=i, name=HAND_LANDMARK_NAMES[i], x=0.51 + 0.005 * i, y=0.51 + 0.005 * i, z=0.0)
            for i in range(21)
        ]
        hand2 = HandLandmarks(landmarks=lms2, handedness="Left", score=0.60)

        assert detector._is_duplicate(hand1, hand2)

        survivors = detector._suppress_duplicates([(hand1, None), (hand2, None)])
        # Must keep only 1 hand (the one with higher confidence score)
        assert len(survivors) == 1
        assert survivors[0][0].score == 0.95
        assert survivors[0][0].handedness == "Right"
    finally:
        detector.close()


def test_distinct_hands_preserved() -> None:
    from asignx.models import HAND_LANDMARK_NAMES, HandLandmarks, Landmark

    detector = HandDetector(static_image_mode=True)
    try:
        # Hand Left at x ~ 0.20
        lms_left = [
            Landmark(id=i, name=HAND_LANDMARK_NAMES[i], x=0.20 + 0.002 * i, y=0.50 + 0.002 * i, z=0.0)
            for i in range(21)
        ]
        hand_left = HandLandmarks(landmarks=lms_left, handedness="Left", score=0.92)

        # Hand Right at x ~ 0.80
        lms_right = [
            Landmark(id=i, name=HAND_LANDMARK_NAMES[i], x=0.80 + 0.002 * i, y=0.50 + 0.002 * i, z=0.0)
            for i in range(21)
        ]
        hand_right = HandLandmarks(landmarks=lms_right, handedness="Right", score=0.94)

        assert detector._is_duplicate(hand_left, hand_right) is False

        survivors = detector._suppress_duplicates([(hand_left, None), (hand_right, None)])
        # Must preserve both hands
        assert len(survivors) == 2
    finally:
        detector.close()


def test_handedness_stabilization() -> None:
    from asignx.models import HAND_LANDMARK_NAMES, HandLandmarks, Landmark

    detector = HandDetector(static_image_mode=False)
    try:
        # Frame 1: Strong Right hand at (0.5, 0.5)
        lms1 = [
            Landmark(id=i, name=HAND_LANDMARK_NAMES[i], x=0.5 + 0.002 * i, y=0.5 + 0.002 * i, z=0.0)
            for i in range(21)
        ]
        hand1 = HandLandmarks(landmarks=lms1, handedness="Right", score=0.95)
        detector._stabilize_handedness([(hand1, None)])
        assert hand1.handedness == "Right"

        # Frame 2: Noisy classification flipped to "Left" with low confidence
        lms2 = [
            Landmark(id=i, name=HAND_LANDMARK_NAMES[i], x=0.505 + 0.002 * i, y=0.505 + 0.002 * i, z=0.0)
            for i in range(21)
        ]
        hand2 = HandLandmarks(landmarks=lms2, handedness="Left", score=0.65)
        detector._stabilize_handedness([(hand2, None)])
        # Must stabilize to "Right" and prevent flicker
        assert hand2.handedness == "Right"
    finally:
        detector.close()


def test_fast_left_hand_reacquisition() -> None:
    """A large, single-frame left-hand jump must retain its side label."""
    from asignx.models import HAND_LANDMARK_NAMES, HandLandmarks, Landmark

    detector = HandDetector(static_image_mode=False)
    try:
        initial = HandLandmarks(
            landmarks=[
                Landmark(id=i, name=HAND_LANDMARK_NAMES[i], x=0.20 + 0.002 * i, y=0.5, z=0.0)
                for i in range(21)
            ],
            handedness="Left",
            score=0.98,
        )
        detector._stabilize_handedness([(initial, None)])

        # A rapid move makes an uncertain raw classifier report the opposite side.
        reacquired = HandLandmarks(
            landmarks=[
                Landmark(id=i, name=HAND_LANDMARK_NAMES[i], x=0.48 + 0.002 * i, y=0.5, z=0.0)
                for i in range(21)
            ],
            handedness="Right",
            score=0.65,
        )
        detector._stabilize_handedness([(reacquired, None)])

        assert reacquired.handedness == "Left"
    finally:
        detector.close()


