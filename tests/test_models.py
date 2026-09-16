"""
Unit tests for ASignX Hand Recognition data models.
"""

import json
import numpy as np
import pytest

from asignx.models import (
    HAND_LANDMARK_NAMES,
    HandDetectionResult,
    HandLandmarks,
    Landmark,
)


def create_sample_landmarks(count: int = 21) -> list[Landmark]:
    """Helper to generate mock landmarks for testing."""
    return [
        Landmark(
            id=i,
            name=HAND_LANDMARK_NAMES[i] if i < len(HAND_LANDMARK_NAMES) else f"LM_{i}",
            x=0.05 * i,
            y=0.04 * i,
            z=-0.01 * i,
        )
        for i in range(count)
    ]


def test_landmark_creation() -> None:
    lm = Landmark(id=0, name="WRIST", x=0.5, y=0.7, z=-0.02)
    assert lm.id == 0
    assert lm.name == "WRIST"
    assert lm.x == 0.5
    assert lm.y == 0.7
    assert lm.z == -0.02
    assert lm.as_tuple() == (0.5, 0.7, -0.02)
    d = lm.to_dict()
    assert d["id"] == 0
    assert d["name"] == "WRIST"
    assert d["x"] == 0.5


def test_hand_landmarks_valid() -> None:
    lms = create_sample_landmarks(21)
    hand = HandLandmarks(landmarks=lms, handedness="Right", score=0.95)

    assert len(hand) == 21
    assert hand[0].name == "WRIST"
    assert hand[20].name == "PINKY_TIP"
    assert hand.handedness == "Right"
    assert hand.score == 0.95

    # Test numpy array conversion
    arr = hand.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (21, 3)
    assert arr.dtype == np.float32
    assert np.isclose(arr[0, 0], 0.0)

    # Test dictionary conversion
    d = hand.to_dict()
    assert d["handedness"] == "Right"
    assert len(d["landmarks"]) == 21

    # Test tree string formatting
    tree_str = hand.to_tree_string()
    assert "Hand (Right)" in tree_str
    assert "├── Landmark  0 (WRIST" in tree_str
    assert "└── Landmark 20 (PINKY_TIP" in tree_str


def test_hand_landmarks_invalid_count() -> None:
    with pytest.raises(ValueError, match="must contain exactly 21 landmarks"):
        HandLandmarks(landmarks=create_sample_landmarks(20))

    with pytest.raises(ValueError, match="must contain exactly 21 landmarks"):
        HandLandmarks(landmarks=create_sample_landmarks(22))


def test_hand_detection_result_with_hand() -> None:
    lms = create_sample_landmarks(21)
    hand = HandLandmarks(landmarks=lms, handedness="Left", score=0.99)
    res = HandDetectionResult(
        hand_detected=True,
        hand=hand,
        all_hands=[hand],
        frame_timestamp_ms=12345,
    )

    assert res.hand_detected is True
    assert res.hand is not None
    assert res.hand.handedness == "Left"

    d = res.to_dict()
    assert d["hand_detected"] is True
    assert d["hand"] is not None
    assert len(d["hand"]["landmarks"]) == 21

    json_str = res.to_json()
    parsed = json.loads(json_str)
    assert parsed["hand_detected"] is True
    assert parsed["hand"]["landmarks"][0]["name"] == "WRIST"


def test_hand_detection_result_no_hand_null() -> None:
    res = HandDetectionResult(
        hand_detected=False,
        hand=None,
        all_hands=[],
        frame_timestamp_ms=12345,
    )

    assert res.hand_detected is False
    assert res.hand is None

    # Verify dictionary serialization maps hand to None (null in JSON)
    d = res.to_dict()
    assert d["hand_detected"] is False
    assert d["hand"] is None

    json_str = res.to_json()
    parsed = json.loads(json_str)
    assert parsed["hand"] is None

    # Tree string should explicitly represent Hand -> null
    assert "null" in res.to_tree_string()


def test_hand_detection_result_two_hands() -> None:
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
        frame_timestamp_ms=54321,
    )

    assert res.hand_detected is True
    assert res.hands_count == 2
    assert len(res.hands) == 2
    assert res.left_hand is left_hand
    assert res.right_hand is right_hand

    d = res.to_dict()
    assert d["hands_count"] == 2
    assert len(d["hands"]) == 2
    assert d["left_hand"]["handedness"] == "Left"
    assert d["right_hand"]["handedness"] == "Right"

    tree_str = res.to_tree_string()
    assert "Hands (2 detected)" in tree_str
    assert "Hand (Left)" in tree_str
    assert "Hand (Right)" in tree_str


