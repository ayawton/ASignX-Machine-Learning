"""Temple references provide location evidence without imposing sign-specific rules."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from ml.body_context import context_features, resample_context, TEMPLE_NUM_FEATURES, TEMPLE_CONTEXT_VERSION
from ml.video import LandmarkSequenceCapture
from ml.sequence_io import SequenceSample,save_sequence_file,load_sequence_file,read_sequence_dir
from ml.sequence_model import SignLSTM,save_sequence_checkpoint,load_sequence_checkpoint
from ml.predict_sequence import SequencePredictor
from ml.dynamic_recording import CaptureSettings,TakeRecorder,validate_output
from tests.test_body_context import hands,body


def references():
    return np.concatenate((body(),np.array([[.38,.15],[.62,.15]],dtype=np.float32)))


def test_legacy_prefix_and_head_position_are_distinct_features():
    refs=references()
    original=context_features(hands(),refs,temple_context=True)
    assert original.shape==(180,)
    np.testing.assert_array_equal(original[:150],context_features(hands(),body()))
    moved=refs.copy(); moved[6:8,1] += .25
    other=context_features(hands(),moved,temple_context=True)
    np.testing.assert_array_equal(original[:150],other[:150])
    assert not np.allclose(original[150:],other[150:])
    assert not original[:63].any()  # absent hand remains masked
    assert not original[156:168].any()
    assert original[168:].reshape(-1,3)[:,2].all()


def test_head_offsets_are_translation_and_scale_invariant():
    first=context_features(hands(),references(),temple_context=True)
    changed=[SimpleNamespace(handedness=h.handedness,to_numpy=lambda h=h:h.to_numpy()*1.5+.2) for h in hands()]
    np.testing.assert_allclose(first,context_features(changed,references()*1.5+.2,temple_context=True),atol=2e-6)


def test_missing_face_does_not_fabricate_head_geometry():
    refs=references(); refs[6:8]=np.nan
    sample=context_features(hands(),refs,temple_context=True)
    assert np.isfinite(sample).all()
    assert not sample[150:].any()
    visible=context_features(hands(),references(),temple_context=True)
    interpolated=resample_context(np.stack([visible,sample]),3)
    assert not interpolated[1,150:].any()
    with pytest.raises(ValueError,match='eight'):
        context_features(hands(),body(),temple_context=True)


def test_capture_storage_and_checkpoint_contract(tmp_path):
    recorder=TakeRecorder(CaptureSettings(tmp_path/'data','HELLO',temple_context=True,countdown_seconds=0,min_frames=2))
    recorder.start(0)
    for i in range(12): recorder.update(hands(i*.001),i/15,references())
    sample=load_sequence_file(recorder.save(1))
    assert sample.features.shape==(30,180)
    assert sample.metadata['temple_context']
    with pytest.raises(ValueError,match='different'):
        validate_output(tmp_path/'data','both',30,True)
    path=tmp_path/'model.pth'
    save_sequence_checkpoint(path,SignLSTM(2,input_size=180,hidden_size=8,num_layers=1),['HELLO','THANK_YOU'])
    predictor=SequencePredictor(path,confidence_threshold=0)
    assert predictor.metadata['preprocessing']==TEMPLE_CONTEXT_VERSION
    assert predictor.predict_take([(i/15,context_features(hands(i*.001),references(),temple_context=True)) for i in range(30)]).label in predictor.labels
    with pytest.raises(ValueError,match='visible|gap'):
        predictor.predict_take([(i/15,context_features(hands(),None,temple_context=True)) for i in range(30)])


def test_missing_temples_reject_recording():
    capture=LandmarkSequenceCapture('both',temple_context=True)
    for i in range(10): capture.add_hands(hands(),np.concatenate((body(),np.full((2,2),np.nan))))
    with pytest.raises(ValueError,match='Temple'):
        capture.finish()


def test_unified_recognition_dispatches_temple_model(tmp_path):
    from ml.predict import UnifiedSignRecognizer
    path=tmp_path/'model.pth'
    save_sequence_checkpoint(path,SignLSTM(2,input_size=180,hidden_size=8,num_layers=1),['HELLO','THANK_YOU'])
    router=UnifiedSignRecognizer(sequence_predictor=SequencePredictor(path,confidence_threshold=0),velocity_threshold=.004)
    outputs=[]
    for i in range(100):
        t=i/30
        result=router.update(hands(min(max(t-.3,0),1.5)*.3),timestamp=t,body_context=references())
        if result['dynamic_emitted']: outputs.append(result['dynamic_emitted'])
    assert len(outputs)==1


def test_temple_dataset_trains_and_does_not_mix_legacy(tmp_path):
    from ml.train_sequence import train_sequence,SequenceTrainConfig
    for label in ('HELLO','THANK_YOU'):
        for session in range(3):
            features=np.stack([context_features(hands((i+session)*.003),references(),temple_context=True) for i in range(30)])
            features[:,0] = session*.001 + (.01 if label=='THANK_YOU' else 0)
            save_sequence_file(tmp_path/'data'/label/f'{session}.npz',SequenceSample(label,f'{label}{session}','Right',features,{'target_fps':15.}))
    results=train_sequence(str(tmp_path/'data'),str(tmp_path/'model.pth'),SequenceTrainConfig(epochs=1,hidden_size=8,num_layers=1))
    assert results['labels']==['HELLO','THANK_YOU']
    assert SequencePredictor(tmp_path/'model.pth').input_size==180
    save_sequence_file(tmp_path/'data'/'old.npz',SequenceSample('HELLO','legacy','Right',np.ones((30,150))))
    with pytest.raises(ValueError,match='width'):
        read_sequence_dir(tmp_path/'data')


def test_background_is_not_emitted(tmp_path):
    import torch
    path=tmp_path/'background.pth'
    save_sequence_checkpoint(path,SignLSTM(2,input_size=180,hidden_size=8,num_layers=1),['BACKGROUND','HELLO'])
    predictor=SequencePredictor(path,confidence_threshold=0)
    class Background(torch.nn.Module):
        def forward(self,x): return torch.tensor([[20.,0.]])
    predictor.model=Background()
    result=predictor.predict_take([(i/15,context_features(hands(),references(),temple_context=True)) for i in range(30)])
    assert result.raw_label=='BACKGROUND'
    assert result.label is None


def test_detector_extracts_temple_proxies_from_existing_face_mesh(monkeypatch):
    from asignx.body_detector import BodyHandDetector
    from asignx.detector import HandDetector
    from asignx.models import HandDetectionResult
    monkeypatch.setattr(HandDetector,'process',lambda self,frame,draw=True,timestamp_ms=None:(HandDetectionResult(False,frame_timestamp_ms=timestamp_ms),frame))
    detector=BodyHandDetector.__new__(BodyHandDetector)
    detector.static_image_mode=False
    detector._body_timestamp=detector._body_source_timestamp=-1
    detector._context_interval_ms=50
    detector._body_cache=np.full((8,2),np.nan,dtype=np.float32)
    pose=[SimpleNamespace(x=.2+i*.01,y=.5,visibility=1.,presence=1.) for i in range(33)]
    face=[SimpleNamespace(x=.5,y=.2) for i in range(478)]
    face[127]=SimpleNamespace(x=.35,y=.15)
    face[356]=SimpleNamespace(x=.65,y=.15)
    detector._pose=SimpleNamespace(detect_for_video=lambda image,stamp:SimpleNamespace(pose_landmarks=[pose]))
    detector._face=SimpleNamespace(detect_for_video=lambda image,stamp:SimpleNamespace(face_landmarks=[face]))
    result,_=detector.process(np.zeros((32,32,3),dtype=np.uint8),False,0)
    np.testing.assert_allclose(result.body_context[6:8],[[.35,.15],[.65,.15]])
    assert result.body_context.shape==(8,2)


def test_existing_body_checkpoint_ignores_extra_reference_points():
    path=Path('ml/models/sequence_body_v1.pth')
    if not path.exists(): pytest.skip('Existing checkpoint unavailable')
    predictor=SequencePredictor(path,confidence_threshold=0)
    assert predictor.input_size==150
    frames=[(i/15,context_features(hands(i*.001),references())) for i in range(30)]
    assert predictor.predict_take(frames).label in predictor.labels


def test_video_import_rejects_insufficient_temple_visibility(tmp_path,monkeypatch):
    import ml.prepare_videos as module
    import asignx.body_detector as detector_module
    for label in ('HELLO','THANK_YOU'):
        directory=tmp_path/'videos'/label
        directory.mkdir(parents=True)
        for i in range(3): (directory/f'{i}.mp4').write_bytes(f'{label}{i}'.encode())
    class Detector:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
    monkeypatch.setattr(detector_module,'BodyHandDetector',Detector)
    monkeypatch.setattr(module,'extract_video_landmarks',lambda *args,**kwargs:(np.ones((30,180),dtype=np.float32),'Right',{'detection_ratio':1.,'temple_detection_ratio':.1}))
    with pytest.raises(RuntimeError,match='temples visible'):
        module.prepare_video_dataset(tmp_path/'videos',tmp_path/'out',temple_context=True)
    assert not list((tmp_path/'out').rglob('*.npz'))


@pytest.mark.parametrize('module_name,required',[
    ('ml.collect_sequence',['--label','HELLO']),
    ('ml.prepare_videos',['--input','videos']),
])
@pytest.mark.parametrize('flag,temples',[
    ('--body-context',True),('--legacy-body-context',False),('--temple-context',True),
])
def test_standard_body_cli_includes_temples_and_legacy_is_explicit(module_name,required,flag,temples,monkeypatch):
    import importlib,sys
    module=importlib.import_module(module_name)
    captured={}
    name='collect_sequence' if module_name.endswith('collect_sequence') else 'prepare_video_dataset'
    monkeypatch.setattr(module,name,lambda *args,**kwargs:captured.update(kwargs))
    monkeypatch.setattr(sys,'argv',['test',*required,flag])
    module.main()
    assert captured['body_context'] is True
    assert captured['temple_context'] is temples
