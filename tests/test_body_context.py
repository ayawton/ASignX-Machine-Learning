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


@pytest.mark.parametrize('width', [150, 180])
def test_completed_take_trims_only_boundary_tracking_gaps(tmp_path, width):
    from ml.body_context import temple_features
    path = tmp_path/'model.pth'
    save_sequence_checkpoint(path, SignLSTM(2, input_size=width, hidden_size=8,
                                          num_layers=1), ['HELLO', 'HELP'])
    predictor = SequencePredictor(path, confidence_threshold=0)
    extended_body = np.concatenate([body(), [[.4, .2], [.6, .2]]]).astype(np.float32)
    valid = (temple_features(hands(), extended_body) if width == 180
             else context_features(hands(), body()))
    missing = np.zeros(width, dtype=np.float32)
    # Pre-roll/settling can contain no tracked hand. Retain enough visible data.
    take = [missing]*5 + [valid]*20 + [missing]*5
    result = predictor.predict_take([(i/15, row) for i, row in enumerate(take)])
    assert result.label in ('HELLO', 'HELP')
    assert predictor.buffer.maxlen == 30 and not predictor.buffer
    # Equal-sized loss during the actual sign must still fail.
    interior_gap = [valid]*12 + [missing]*5 + [valid]*13
    with pytest.raises(ValueError, match='gap is too long'):
        predictor.predict_take([(i/15, row) for i, row in enumerate(interior_gap)])
    # Boundary trimming must not turn a mostly missing take into a valid one.
    mostly_missing = [missing]*20 + [valid]*10
    with pytest.raises(ValueError, match='Insufficient visible'):
        predictor.predict_take([(i/15, row) for i, row in enumerate(mostly_missing)])


@pytest.mark.parametrize('width', [150, 180])
def test_resampling_cannot_change_tracking_gap_duration(tmp_path, width):
    from ml.body_context import temple_features
    path=tmp_path/'model.pth'
    save_sequence_checkpoint(path, SignLSTM(2, input_size=width, hidden_size=8,
                                          num_layers=1), ['HELLO','HELP'])
    predictor=SequencePredictor(path, confidence_threshold=0)
    references=np.concatenate([body(), [[.4,.2],[.6,.2]]]).astype(np.float32)
    valid=(temple_features(hands(),references) if width==180 else context_features(hands(),body()))
    # 2 missing observations at 15 FPS are allowed. Stretching to 30 model
    # timesteps previously made that same physical dropout exceed 3 steps.
    short=np.stack([valid]*12); short[5:7]=0
    prediction=predictor.predict_take([(i/15,row) for i,row in enumerate(short)])
    assert prediction.label in ('HELLO','HELP')
    assert not predictor.buffer and predictor.buffer.maxlen==30
    # Downsampling cannot hide an actually excessive original tracking gap.
    long=np.stack([valid]*60); long[28:33]=0
    with pytest.raises(ValueError, match='gap is too long'):
        predictor.predict_sequence(long)


def test_tracking_rejection_identifies_missing_references(tmp_path):
    from ml.body_context import temple_features
    path=tmp_path/'model.pth'
    save_sequence_checkpoint(path, SignLSTM(2,input_size=180,hidden_size=8,num_layers=1),['HELLO','HELP'])
    predictor=SequencePredictor(path)
    # Shoulders and hand exist; the head references do not.
    points=np.concatenate([body(), [[np.nan,np.nan],[np.nan,np.nan]]]).astype(np.float32)
    row=temple_features(hands(),points)
    with pytest.raises(ValueError, match='face references 0%'):
        predictor.predict_take([(i/15,row) for i in range(15)])


def test_timestamp_interpolation_does_not_turn_slow_camera_into_missing_hands():
    from ml.body_context import resample_context_timestamps
    times=np.arange(15,dtype=np.float64)/12  # observed 12 FPS; checkpoint samples at 15
    rows=np.stack([context_features(hands(float(t)*.02),body()) for t in times])
    prepared=resample_context_timestamps(rows,times,15,.2)
    assert prepared.shape==(18,150)
    assert prepared[:,126:].reshape(len(prepared),-1,3)[:,7,2].all()
    np.testing.assert_allclose(prepared[:,147],np.arange(len(prepared))/15*.02/.4+rows[0,147],atol=1e-6)
    assert np.isfinite(prepared).all()


def test_timestamp_resampling_preserves_actual_loss_and_rejects_camera_stalls():
    from ml.body_context import resample_context_timestamps
    times=np.arange(10)/15
    rows=np.stack([context_features(hands(),body()) for _ in times])
    rows[4]=context_features([],None)
    source=rows.copy()
    # Large monotonic clock values must not smear visibility into adjacent frames.
    result=resample_context_timestamps(rows,380000+times,15,.2)
    np.testing.assert_allclose(result,source,atol=1e-6)
    np.testing.assert_array_equal(rows,source)
    with pytest.raises(ValueError,match='Camera gap'):
        resample_context_timestamps(rows,[0,.067,.134,.5,.567,.634,.701,.768,.835,.902],15,.2)
    with pytest.raises(ValueError,match='timestamps must increase'):
        resample_context_timestamps(rows,np.zeros(10),15,.2)


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
