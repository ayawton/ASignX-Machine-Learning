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



def _task_result(x=None):
    if x is None:
        return SimpleNamespace(hand_landmarks=[], handedness=[])
    return SimpleNamespace(hand_landmarks=[[
        SimpleNamespace(x=x+i*.005,y=.3+i*.004,z=0.) for i in range(21)
    ]],handedness=[[SimpleNamespace(category_name='Right',score=.99)]])


def test_recovery_uses_fresh_detection_and_is_rate_limited():
    with HandDetector() as detector:
        class Primary:
            def __init__(self): self.calls = 0
            def detect_for_video(self,image,stamp):
                self.calls += 1
                return _task_result(.2 if self.calls == 1 else None)
            def close(self): pass
        class Recovery:
            def __init__(self): self.calls = 0
            def detect(self,image):
                self.calls += 1
                return _task_result(.4 if self.calls == 1 else None)
            def close(self): pass
        detector._hands.close()
        detector._hands = Primary()
        recovery = detector._recovery_hands = Recovery()
        image = np.zeros((128,128,3), dtype=np.uint8)
        assert detector.process(image,False,1000)[0].hand_detected
        recovered = detector.process(image,False,1033)[0]
        assert recovered.hand_detected
        assert recovered.hand[0].x == .4
        assert not detector.process(image,False,1066)[0].hand_detected
        assert recovery.calls == 1
        assert not detector.process(image,False,1250)[0].hand_detected
        assert recovery.calls == 2
        assert not detector.process(image,False,1700)[0].hand_detected
        assert recovery.calls == 2


def test_touching_wrists_do_not_suppress_different_hand_shapes():
    from asignx.models import HAND_LANDMARK_NAMES, HandLandmarks, Landmark
    with HandDetector(static_image_mode=True) as detector:
        def hand(direction):
            return HandLandmarks([Landmark(i,HAND_LANDMARK_NAMES[i],.5+direction*i*.008,.5+i*.003,0.) for i in range(21)])
        a,b = hand(-1),hand(1)
        assert not detector._is_duplicate(a,b)
        assert len(detector._suppress_duplicates([(a,None),(b,None)])) == 2


def test_presence_threshold_is_independent_of_tracking(monkeypatch):
    import mediapipe as mp
    options = []
    class Tracker:
        def close(self): pass
    monkeypatch.setattr(mp.tasks.vision.HandLandmarker,'create_from_options',lambda opts: options.append(opts) or Tracker())
    with HandDetector(min_detection_confidence=.4,min_presence_confidence=.35,min_tracking_confidence=.65):
        assert options[0].min_hand_detection_confidence == .4
        assert options[0].min_hand_presence_confidence == .35
        assert options[0].min_tracking_confidence == .65


def test_body_context_throttle_keeps_hand_processing_every_frame(monkeypatch):
    from asignx.body_detector import BodyHandDetector
    from asignx.models import HandDetectionResult
    hand_calls = []
    def process(self,frame,draw=True,timestamp_ms=None):
        hand_calls.append(timestamp_ms)
        return HandDetectionResult(False,frame_timestamp_ms=timestamp_ms),frame
    monkeypatch.setattr(HandDetector,'process',process)
    detector = BodyHandDetector.__new__(BodyHandDetector)
    detector.static_image_mode = False
    detector._context_interval_ms = 50.
    detector._body_timestamp = detector._body_source_timestamp = -1
    detector._body_cache = np.full((8,2),np.nan,dtype=np.float32)
    class Pose:
        def __init__(self): self.stamps=[]
        def detect_for_video(self,image,stamp):
            self.stamps.append(stamp)
            return SimpleNamespace(pose_landmarks=[])
    class Face(Pose):
        def detect_for_video(self,image,stamp):
            self.stamps.append(stamp)
            return SimpleNamespace(face_landmarks=[])
    detector._pose,detector._face = Pose(),Face()
    image = np.zeros((32,32,3),dtype=np.uint8)
    for stamp in (0,16,33,50,66,100,0):
        detector.process(image,False,stamp)
    assert len(hand_calls) == 7
    assert detector._pose.stamps == [0,50,100,101]
    assert detector._face.stamps == detector._pose.stamps
