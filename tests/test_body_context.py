"""Body-context contracts across capture, storage, training and inference."""
from types import SimpleNamespace
import numpy as np
import pytest
from ml.body_context import BODY_NUM_FEATURES, context_features, resample_context
from ml.video import LandmarkSequenceCapture
from ml.sequence_io import SequenceSample, save_sequence_file, load_sequence_file, read_sequence_dir
from ml.predict_sequence import SequencePredictor
from ml.sequence_model import SignLSTM, save_sequence_checkpoint


def hands(offset=0.):
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.linspace(.4, .55, 21) + offset
    points[:, 1] = np.linspace(.4, .5, 21)
    return [SimpleNamespace(handedness="Right", to_numpy=lambda: points)]


def body():
    return np.array([[.5,.15],[.5,.25],[.3,.4],[.7,.4],[.2,.6],[.8,.6]], dtype=np.float32)


def test_translation_scale_invariance_and_movement():
    first = context_features(hands(), body())
    translated = [SimpleNamespace(handedness=h.handedness, to_numpy=lambda h=h: h.to_numpy()*2+.1) for h in hands()]
    np.testing.assert_allclose(first, context_features(translated, body()*2+.1), atol=2e-6)
    moved = context_features(hands(.1), body())
    np.testing.assert_allclose(first[:126], moved[:126], atol=2e-6)
    assert moved[-3] > first[-3]  # right wrist/body travel survives
    assert not first[:63].any()  # missing left slot stays zero
    assert np.isfinite(context_features(hands(), None)).all()
    assert not context_features(hands(), None)[126:].any()


def test_resampling_does_not_invent_visibility():
    visible = context_features(hands(), body())
    missing = context_features([], None)
    out = resample_context(np.stack([visible, missing]), 3)
    assert not out[1,126:].any()
    assert not out[1,:126].any()


def test_capture_roundtrip_and_live_contract(tmp_path):
    capture = LandmarkSequenceCapture('both', body_context=True)
    for i in range(30):
        capture.add_hands(hands(i*.002), body())
    features, handedness, metadata = capture.finish()
    assert features.shape == (30, BODY_NUM_FEATURES)
    path = save_sequence_file(tmp_path/'body.npz', SequenceSample('HELLO','visit1',handedness,features,metadata))
    np.testing.assert_array_equal(load_sequence_file(path).features, features)
    ckpt = tmp_path/'body.pth'
    save_sequence_checkpoint(ckpt, SignLSTM(2,input_size=150,hidden_size=8,num_layers=1), ['HELLO','HELP'])
    predictor = SequencePredictor(ckpt, confidence_threshold=0)
    for i in range(30):
        predictor.add_frame(context_features(hands(i*.002),body()), timestamp=i/15)
    np.testing.assert_allclose(predictor._buffer_features(), features, atol=1e-6)
    assert predictor.predict_sequence(features).label in ('HELLO','HELP')
    predictor.reset()
    for i in range(30):
        predictor.add_frame(context_features(hands(),None), timestamp=i/15)
    assert predictor.predict() is None
    save_sequence_file(tmp_path/'old.npz', SequenceSample('J','visit2','Right',np.ones((30,126))))
    with pytest.raises(ValueError,match='width'):
        read_sequence_dir(tmp_path)


def test_missing_shoulders_reject_recording():
    capture = LandmarkSequenceCapture('both', body_context=True)
    for i in range(10):
        capture.add_hands(hands(), None)
    with pytest.raises(ValueError,match='Shoulders'):
        capture.finish()


def test_body_sequence_trains_without_hand_only_augmentation(tmp_path):
    from ml.train_sequence import train_sequence, SequenceTrainConfig
    for label in ('HELLO','HELP'):
        for session in range(3):
            features = np.stack([context_features(hands((session+i)*.003),body()) for i in range(30)])
            features[:,0] = session*.001 + (0.01 if label == 'HELP' else 0)
            save_sequence_file(tmp_path/'data'/label/f'{session}.npz',SequenceSample(label,f'{label}{session}','Right',features,{'target_fps':15.}))
    results = train_sequence(str(tmp_path/'data'),str(tmp_path/'model.pth'),SequenceTrainConfig(epochs=1, hidden_size=8,num_layers=1))
    assert results['labels'] == ['HELLO','HELP']
    assert SequencePredictor(tmp_path/'model.pth').input_size == 150


def test_body_recorder_settings_and_saved_contract(tmp_path):
    from ml.dynamic_recording import CaptureSettings, TakeRecorder, validate_output
    settings = CaptureSettings(tmp_path, 'HELLO', body_context=True, countdown_seconds=0, min_frames=2)
    recorder = TakeRecorder(settings)
    recorder.start(0)
    for i in range(4):
        recorder.update(hands(i*.01), i/10, body())
    sample = load_sequence_file(recorder.save(.4))
    assert sample.features.shape == (30, 150)
    assert sample.metadata['body_context'] is True
    with pytest.raises(ValueError,match='different'):
        validate_output(tmp_path, 'both', 30, False)
    with pytest.raises(ValueError,match='both'):
        CaptureSettings(tmp_path/'other','HELLO',hand='Right',body_context=True).validate()


def test_unified_router_uses_body_features(tmp_path):
    from ml.predict import UnifiedSignRecognizer
    ckpt = tmp_path/'body.pth'
    save_sequence_checkpoint(ckpt,SignLSTM(2,input_size=150,hidden_size=8,num_layers=1),['HELLO','HELP'])
    router = UnifiedSignRecognizer(sequence_predictor=SequencePredictor(ckpt, confidence_threshold=0), velocity_threshold=.004)
    emitted = []
    for i in range(100):
        t = i/30
        x = min(max(t-.3,0),1.5)*.3
        result = router.update(hands(x),timestamp=t,body_context=body())
        if result['dynamic_emitted']:
            emitted.append(result['dynamic_emitted'])
    assert len(emitted) == 1
    assert emitted[0] in ('HELLO','HELP')
    assert router.state == 'static'
