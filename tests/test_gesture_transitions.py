from types import SimpleNamespace
from collections import deque
import numpy as np
from ml.predict import UnifiedSignRecognizer, MotionGate, Prediction

class Static:
    def predict(self, pts, handedness=None):
        return Prediction('I', .99, 'I')

class Sequence:
    input_size = 63
    sequence_length = 30
    def __init__(self):
        self.buffer = deque(maxlen=30)
        self.calls = []
    def reset(self):
        self.buffer.clear()
    def add_frame(self, frame, timestamp=None):
        self.buffer.append(frame)
    def _buffer_features(self):
        return np.stack(self.buffer).reshape(len(self.buffer), 63)
    def predict_take(self, frames):
        from ml.predict_sequence import SequencePredictor
        return SequencePredictor.predict_take(self, frames)
    def predict_sequence(self, features):
        self.calls.append(features.copy())
        return Prediction('J', .95, 'J')

def hand(x):
    pts = np.arange(63, dtype=np.float32).reshape(21, 3) / 200
    pts[:, 0] += x
    return SimpleNamespace(handedness='Right', to_numpy=lambda: pts)

def test_completed_gesture_blocks_static_and_preserves_full_take():
    seq = Sequence()
    router = UnifiedSignRecognizer(Static(), seq, velocity_threshold=.004, stable_frames=2)
    outputs = []
    for i in range(100):
        t = i / 30
        x = min(max(t - .3, 0), 1.5) * .3
        outputs.append(router.update([hand(x)], timestamp=t))
    active = [r for r in outputs if r['state'] != 'static']
    assert active
    assert all(r['static'][0].emitted_label is None for r in active)
    assert [r['dynamic_emitted'] for r in outputs if r['dynamic_emitted']] == ['J']
    assert len(seq.calls) == 1
    assert len(seq.calls[0]) > 30
    assert outputs[-1]['static'][0].stable_label == 'I'
    assert seq.buffer.maxlen == 30

def test_finger_motion_is_not_diluted():
    gate = MotionGate(velocity_threshold=.02)
    pts = hand(0).to_numpy()
    gate.update(pts, 0)
    moved = pts.copy()
    moved[20, 0] += .04
    assert gate.update(moved, 1/30)[1]

def test_router_rejects_nonincreasing_time():
    import pytest
    router = UnifiedSignRecognizer()
    router.update([], timestamp=1)
    with pytest.raises(ValueError):
        router.update([], timestamp=1)

def real_sequence():
    import torch
    from ml.predict_sequence import SequencePredictor
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[8., 0.]])
    seq = SequencePredictor.__new__(SequencePredictor)
    seq.model = Model()
    seq.labels = ['J', 'Z']
    seq.sequence_length = 30
    seq.input_size = 63
    seq.target_fps = 15.
    seq.min_visibility = .5
    seq.max_gap_frames = 3
    seq.threshold = .8
    seq.device = 'cpu'
    seq.buffer = deque(maxlen=30)
    seq.reset()
    return seq

def test_real_sequence_preprocessing_and_resampling():
    seq = real_sequence()
    router = UnifiedSignRecognizer(Static(), seq, velocity_threshold=.004)
    emitted = []
    for i in range(100):
        t = i / 30
        x = min(max(t - .3, 0), 1.5) * .3
        r = router.update([hand(x)], timestamp=t)
        if r['dynamic_emitted']:
            emitted.append(r['dynamic_emitted'])
    assert emitted == ['J']
    assert seq.buffer.maxlen == 30

def test_tracking_loss_rejects_completed_take():
    router = UnifiedSignRecognizer(Static(), real_sequence(), velocity_threshold=.004)
    emitted = []
    for i in range(70):
        t = i / 30
        hands = [hand(min(t, 1.) * .3)] if i < 32 else []
        r = router.update(hands, timestamp=t)
        emitted.append(r['dynamic_emitted'])
    assert not any(emitted)
    assert router.state == 'static'

def test_camera_stall_discards_gesture():
    router = UnifiedSignRecognizer(Static(), real_sequence(), velocity_threshold=.004)
    for i in range(15):
        router.update([hand(i * .01)], timestamp=i / 30)
    assert router.state == 'gesture'
    result = router.update([hand(.14)], timestamp=2.)
    assert result['dynamic_emitted'] is None
    assert result['state'] == 'static'

def test_duplicate_handedness_updates_each_tracker_once_per_frame():
    from ml.predict import SignRecognizer
    low, high = hand(.8), hand(0)
    low.score, high.score = .6, .95
    router = UnifiedSignRecognizer(Static(), Sequence(), velocity_threshold=.004)
    result = router.update([low, high], timestamp=1.)
    assert len(result['static']) == 1
    np.testing.assert_array_equal(router._history[-1][1], high.to_numpy())
    assert len(router.hand_motion_gates['Right'].timestamps) == 1
    result = router.update([low, high], timestamp=1.1)
    assert not result['is_moving']
    static = SignRecognizer(Static(), motion_gate_threshold=.004)
    assert len(static.update([low, high], timestamp=1.)) == 1
    assert len(static.motion_gates['Right'].timestamps) == 1


def test_distinct_hands_keep_independent_trackers():
    left, right = hand(0), hand(.2)
    left.handedness = 'Left'
    router = UnifiedSignRecognizer(Static())
    result = router.update([left,right],timestamp=1.)
    assert len(result['static']) == 2
    assert set(router.hand_motion_gates) == {'Left','Right'}
