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


def test_dynamic_capture_hides_suppressed_static_guess(monkeypatch):
    from ml.predict import Prediction, Recognition, SignRecognizer
    from tests.test_gesture_transitions import hand
    from types import SimpleNamespace
    import asignx.visualizer as module
    predictor=SimpleNamespace(predict=lambda *a,**kw:Prediction('B',.99,'B'))
    recognizer=SignRecognizer(predictor,stable_frames=1)
    rec=recognizer.update([hand(0)],timestamp=0,suppress=True)[0]
    assert rec.suppressed and rec.emitted_label is None and rec.stable_label is None
    strings=[]
    monkeypatch.setattr(module.cv2,'putText',lambda img,text,*args,**kw:strings.append(text) or img)
    VisualDebugger()._render_sign_recognition_hud(np.zeros((480,640,3),dtype=np.uint8),[rec],{'state':'gesture'})
    assert 'Static paused' in strings
    assert 'SIGN: B' not in strings
    assert not any('STABILIZING' in text for text in strings)
    # Deliberately held letters remain available outside dynamic suppression.
    held=recognizer.update([hand(0)],timestamp=.1)[0]
    assert not held.suppressed and held.emitted_label=='B'


def test_dynamic_only_hud_waits_for_motion_not_a_static_letter(monkeypatch):
    import asignx.visualizer as module
    strings=[]
    monkeypatch.setattr(module.cv2,'putText',lambda img,text,*args,**kw:strings.append(text) or img)
    VisualDebugger()._render_sign_recognition_hud(np.zeros((480,640,3),dtype=np.uint8),[],{'state':'static'})
    assert 'Waiting for motion...' in strings


def test_dynamic_ending_pose_is_not_displayed_as_another_static_letter(monkeypatch):
    import asignx.visualizer as module
    from ml.predict import Prediction,Recognition
    strings=[]
    monkeypatch.setattr(module.cv2,'putText',lambda img,text,*args,**kw:strings.append(text) or img)
    rec=Recognition('Left',Prediction('B',.99,'B'),None,None,suppressed=True,awaiting_release=True)
    VisualDebugger()._render_sign_recognition_hud(np.zeros((480,640,3),dtype=np.uint8),[rec])
    assert 'Release pose' in strings
    assert any('GESTURE COMPLETE' in s for s in strings)
    assert 'SIGN: B' not in strings


def test_low_confidence_dynamic_guess_is_shown_as_uncertain(monkeypatch):
    import asignx.visualizer as module
    from ml.predict import Prediction
    strings=[]
    monkeypatch.setattr(module.cv2,'putText',lambda img,text,*args,**kw:strings.append(text) or img)
    canvas=np.zeros((480,640,3),dtype=np.uint8)
    VisualDebugger()._render_sign_recognition_hud(canvas,[],{
        'dynamic_prediction':Prediction(None,.47,'Z'),'dynamic_stable':None})
    assert 'Uncertain: Z (47%)' in strings
    assert not any(text.startswith('MOTION: Z') for text in strings)
    strings.clear()
    VisualDebugger()._render_sign_recognition_hud(canvas,[],{
        'dynamic_prediction':Prediction('HELLO',.93,'HELLO'),'dynamic_stable':'HELLO'})
    assert 'MOTION: HELLO (93%)' in strings


