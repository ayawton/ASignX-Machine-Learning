"""Acceptance checks reject motion noise without claiming ASL correctness."""
import numpy as np
import pytest
from ml.gesture_acceptance import MotionAcceptance, is_negative_label
from ml.predict import UnifiedSignRecognizer, Prediction
from tests.test_gesture_transitions import hand


def frames(path, fps=30, scale=1.):
    return [(i/fps, {'Right': hand(x).to_numpy()*scale}) for i,x in enumerate(path)]


@pytest.mark.parametrize('fps', [15, 30, 60])
@pytest.mark.parametrize('scale', [.5, 1., 2.])
def test_sustained_motion_is_scale_and_frame_rate_independent(fps, scale):
    MotionAcceptance().validate(frames(np.linspace(0, .2, fps), fps, scale))


def test_stationary_jitter_and_isolated_spike_are_rejected():
    paths = [np.zeros(60), np.tile([.001, -.001],30), np.r_[np.zeros(20), .3, np.zeros(39)],
             np.r_[np.zeros(30), np.full(30,.3)]]
    for path in paths:
        with pytest.raises(ValueError, match='sustained motion'):
            MotionAcceptance().validate(frames(path))


def test_depth_noise_and_missing_hand_jump_do_not_count_as_motion():
    depth = frames(np.zeros(30))
    for i, (_, hands) in enumerate(depth): hands['Right'][:,2] += i*.03
    missing = frames(np.r_[np.zeros(15), np.full(15,.5)])
    missing[15] = (15/30, {})
    for observations in (depth, missing):
        with pytest.raises(ValueError): MotionAcceptance().validate(observations)


@pytest.mark.parametrize('label', ['BACKGROUND','background','IDLE','no_gesture','No Gesture','no-gesture'])
def test_reserved_negative_classes_are_rejected(label):
    assert is_negative_label(label)


def test_ordinary_vocabulary_labels_remain_available():
    assert not any(is_negative_label(label) for label in ['J','Z','HELLO','NO','LOVE'])


def test_high_confidence_does_not_override_motion_or_background_rejection():
    class Predictor:
        input_size = 63
        def __init__(self): self.calls = 0
        def reset(self): pass
        def predict_take(self, frames):
            self.calls += 1
            return Prediction('IDLE', .999, 'IDLE')
    predictor = Predictor()
    router = UnifiedSignRecognizer(sequence_predictor=predictor, velocity_threshold=.00001)
    results = []
    for i in range(90):
        # Small noise can cross the entry gate, but cannot pass full-take acceptance.
        x = (.001 if i%2 else -.001) if i<30 else 0
        results.append(router.update([hand(x)], timestamp=i/30))
    assert predictor.calls == 0
    assert not any(r['dynamic_emitted'] for r in results)
    assert any(r['dynamic_rejected'] for r in results)
    router = UnifiedSignRecognizer(sequence_predictor=predictor, velocity_threshold=.004)
    results = [router.update([hand(min(max(i/30-.3,0),1.)*.3)],timestamp=i/30) for i in range(90)]
    assert predictor.calls == 1
    assert not any(r['dynamic_emitted'] for r in results)
    assert any(r['dynamic_rejected']=='Background / no gesture' for r in results)


def test_invalid_motion_thresholds_are_not_silently_accepted():
    for value in [0, -1, float('nan'), float('inf')]:
        with pytest.raises(ValueError): MotionAcceptance(min_excursion=value)


def test_sequence_predictor_suppresses_a_learned_negative_class(tmp_path):
    import torch
    from ml.sequence_model import SignLSTM, save_sequence_checkpoint
    from ml.predict_sequence import SequencePredictor
    class IdleModel(torch.nn.Module):
        def forward(self, features): return torch.tensor([[20., 0.]])
    path=tmp_path/'model.pth'
    save_sequence_checkpoint(path, SignLSTM(2, hidden_size=4, num_layers=1), ['IDLE','J'])
    predictor=SequencePredictor(path, confidence_threshold=0)
    predictor.model=IdleModel()
    result=predictor.predict_sequence(np.ones((30,63), dtype=np.float32))
    assert result.raw_label=='IDLE' and result.confidence>.99 and result.label is None


def test_standalone_camera_uses_completion_instead_of_rolling_predictions(monkeypatch, capsys):
    import sys
    from types import SimpleNamespace
    import asignx
    import ml.predict_sequence as module
    class Predictor:
        input_size=63
        calls=0
        def __init__(self,*args): pass
        def reset(self): pass
        def predict(self,*args,**kwargs):
            raise AssertionError('Live CLI must not classify arbitrary rolling windows')
        def predict_take(self, frames):
            self.calls += 1
            return Prediction('J', .99, 'J')
    class Camera:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def set_exposure(self,value): pass
        def stream(self):
            for i in range(100):
                self.frame_timestamp=i/30
                yield np.zeros((64,64,3),dtype=np.uint8)
    class Detector:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def process(self, frame, **kwargs):
            t=kwargs['timestamp_ms']/1000
            return SimpleNamespace(all_hands=[hand(min(max(t-.3,0),1.5)*.3)]),frame
    monkeypatch.setattr(sys,'argv',['predict_sequence','--motion-threshold','.004'])
    monkeypatch.setattr(module,'SequencePredictor',Predictor)
    monkeypatch.setattr(asignx,'WebcamStream',Camera)
    monkeypatch.setattr(asignx,'HandDetector',Detector)
    monkeypatch.setattr(module.cv2,'imshow',lambda *args:None)
    monkeypatch.setattr(module.cv2,'putText',lambda *args:None)
    monkeypatch.setattr(module.cv2,'waitKey',lambda *args:-1)
    monkeypatch.setattr(module.cv2,'destroyAllWindows',lambda:None)
    module.main()
    assert capsys.readouterr().out.count('[SIGN] J')==1
