"""Replay regressions for frame freshness, hand identity and browser feedback."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import threading

import numpy as np
import pytest

from asignx.camera import CameraError, WebcamStream
from asignx.detector import HandDetector
from ml.predict import Prediction, SignRecognizer


def camera():
    cam=WebcamStream()
    cam._cap=SimpleNamespace(isOpened=lambda:True)
    cam._running=True
    cam._latest_frame=np.ones((2,2,3),dtype=np.uint8)
    cam._last_read_success=True
    cam._frame_id=1
    cam._capture_timestamp=12.0
    return cam


def test_threaded_reads_consume_each_capture_once_and_keep_capture_time(monkeypatch):
    monkeypatch.setattr('asignx.camera.time.monotonic',lambda:12.1)
    cam=camera()
    ok,first=cam.read()
    assert ok and cam.frame_timestamp==12.0
    first[:]=0
    assert cam._latest_frame.any()
    assert cam.read()==(False,None)
    with cam._lock:
        cam._frame_id+=1;cam._capture_timestamp=12.04
    assert cam.read()[0] and cam.frame_timestamp==12.04
    cam._frame_id+=1;cam._last_read_success=False
    assert cam.read()==(False,None)


def test_stream_skips_timeouts_and_ignores_duplicate_notifications(monkeypatch):
    monkeypatch.setattr('asignx.camera.time.monotonic',lambda:12.1)
    cam=camera();calls=0
    def wait(timeout):
        nonlocal calls
        calls+=1
        if calls==1:return False
        if calls==2:return True
        if calls==3:return True  # a notification without a new capture
        cam._running=False
        return False
    cam._new_frame_event=SimpleNamespace(wait=wait,clear=lambda:None)
    frames=list(cam.stream())
    assert len(frames)==1 and calls==4


def test_stalled_capture_is_reported_instead_of_hanging_preview(monkeypatch):
    monkeypatch.setattr('asignx.camera.time.monotonic',lambda:15.0)
    cam=camera()
    with pytest.raises(CameraError,match='fresh frames'):
        cam.read()
    cam._new_frame_event=SimpleNamespace(wait=lambda timeout:False,clear=lambda:None)
    with pytest.raises(CameraError,match='fresh frames'):
        next(cam.stream())


def test_right_hand_confidence_is_not_overridden_by_old_left_track():
    d=object.__new__(HandDetector)
    d._left_track=deque([np.array([.2,.2])],maxlen=2)
    d.left_hand_motion_distance_threshold=.45
    hand=SimpleNamespace(handedness='Right',score=.99,landmarks=[SimpleNamespace(x=.2,y=.2) for _ in range(21)])
    d._reacquire_fast_left_hand([(hand,None)])
    assert hand.handedness=='Right'
    hand.score=.6
    d._reacquire_fast_left_hand([(hand,None)])
    assert hand.handedness=='Left'


def test_static_confirmation_restarts_after_capture_gap():
    predictor=SimpleNamespace(predict=lambda *a,**kw:Prediction('A',.99,'A'))
    hand=SimpleNamespace(handedness='Right',to_numpy=lambda:np.ones((21,3),dtype=np.float32))
    recognizer=SignRecognizer(predictor,stable_frames=2)
    assert recognizer.update([hand],timestamp=0)[0].stable_label is None
    assert recognizer.update([hand],timestamp=.04)[0].emitted_label=='A'
    assert recognizer.update([hand],timestamp=1)[0].stable_label is None
    assert recognizer.update([hand],timestamp=1.04)[0].emitted_label=='A'


def test_sdk_state_regressions():
    node=shutil.which('node')
    if not node:pytest.skip('Node is required for actual TypeScript SDK replay')
    major=int(subprocess.check_output([node,'--version'],text=True).strip().lstrip('v').split('.')[0])
    if major<22:pytest.skip('Node 22+ is required')
    subprocess.run([node,str(Path(__file__).with_name('sdk_live_state.mjs'))],check=True,capture_output=True,text=True)
