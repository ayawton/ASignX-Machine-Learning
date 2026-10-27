import json
from types import SimpleNamespace
import numpy as np
import pytest
from asignx.live_diagnostics import LiveDiagnostics


def result(visible=True):
    body=np.ones((8,2))
    if not visible: body[6:8]=np.nan
    return SimpleNamespace(all_hands=[SimpleNamespace(handedness='Left')] if visible else [],
                           body_context=body)


def record(logger,index,visible=True,info=None):
    logger.record(result(visible), info, capture_time=100+index/15,
                  processed_time=100+index/15+.05, detector_ms=35., recognition_ms=5.)


def test_diagnostics_distinguishes_tracking_timing_and_emissions(tmp_path):
    logger=LiveDiagnostics(tmp_path,{'sampling_fps':15})
    assert logger.mark(ord('9'))
    record(logger,0)
    record(logger,1,False,{'dynamic_rejected':'tracking gap'})
    record(logger,2,info={'dynamic_emitted':'HELLO','dynamic_evaluated':True,
                         'dynamic_prediction':SimpleNamespace(raw_label='HELLO',confidence=.9)})
    logger.close(); logger.close()
    summary=json.loads((logger.root/'summary.json').read_text())
    metrics=summary['by_target']['HELLO']
    assert metrics['processed_fps']==pytest.approx(15)
    assert metrics['hand_visibility']==pytest.approx(2/3)
    assert metrics['shoulder_visibility']==1
    assert metrics['face_reference_visibility']==pytest.approx(2/3)
    assert metrics['frame_age_ms_p95']==pytest.approx(50)
    assert metrics['rejections']=={'tracking gap':1}
    assert metrics['emissions']=={'HELLO':1}
    assert len(summary['events'])==2
    logged=[json.loads(line) for line in (logger.root/'frames.jsonl').read_text().splitlines()]
    assert len(logged)==3
    assert not any('landmarks' in row or 'image' in row or 'body_context' in row for row in logged)


def test_repeated_blocks_do_not_inflate_capture_gaps(tmp_path):
    logger=LiveDiagnostics(tmp_path)
    for index,key in enumerate('778877'):
        logger.mark(ord(key)); record(logger,index)
    logger.close()
    report=json.loads((logger.root/'summary.json').read_text())
    assert report['by_target']['J']['processed_fps']==pytest.approx(15)
    assert report['by_target']['J']['capture_intervals_over_200ms']==0


def test_empty_session_and_absent_body_context(tmp_path):
    empty=LiveDiagnostics(tmp_path); empty.close()
    assert json.loads((empty.root/'summary.json').read_text())['overall']=={'frames':0}
    logger=LiveDiagnostics(tmp_path)
    assert not logger.mark(ord('q'))
    logger.record(SimpleNamespace(all_hands=[]),None,capture_time=1,processed_time=1.1,
                  detector_ms=100,recognition_ms=0)
    logger.close()
    assert json.loads((logger.root/'summary.json').read_text())['overall']['shoulder_visibility'] is None


def test_static_candidates_and_confirmations_are_logged_separately(tmp_path):
    from ml.predict import Prediction, Recognition
    logger=LiveDiagnostics(tmp_path)
    rec=Recognition('Left',Prediction('B',.99,'B'),None,None,suppressed=True)
    record(logger,0,info={'static':[rec]})
    rec=Recognition('Left',Prediction('B',.99,'B'),'B','B')
    record(logger,1,info={'static':[rec]})
    logger.close()
    rows=[json.loads(line) for line in (logger.root/'frames.jsonl').read_text().splitlines()]
    assert rows[0]['static_candidates']==['B'] and rows[0]['static_emitted']==[]
    assert rows[0]['static_suppressed']
    report=json.loads((logger.root/'summary.json').read_text())
    assert report['overall']['static_emissions']=={'B':1}
    assert report['overall']['emissions']=={}


def test_main_camera_diagnostic_controls_save_report_without_camera(tmp_path,monkeypatch):
    import main
    class Camera:
        resolution=(64,64)
        exposure=None
        def __init__(self,**kwargs): pass
        def start(self): pass
        def stop(self): pass
        def set_exposure(self,value): pass
        def stream(self):
            for i in range(2):
                self.frame_timestamp=main.time.monotonic()
                yield np.zeros((64,64,3),dtype=np.uint8)
    class Detector:
        def __init__(self,**kwargs): pass
        def process(self,frame,**kwargs): return SimpleNamespace(all_hands=[],hand_detected=False),frame
        def close(self): pass
    class Visualizer:
        def render_overlay(self,frame,**kwargs): return frame
    monkeypatch.setattr(main,'WebcamStream',Camera)
    monkeypatch.setattr(main,'HandDetector',Detector)
    monkeypatch.setattr(main,'VisualDebugger',Visualizer)
    for name in ['namedWindow','imshow','putText','destroyAllWindows']:
        monkeypatch.setattr(main.cv2,name,lambda *a,**kw:None)
    keys=iter([ord('9'),ord('q')]); monkeypatch.setattr(main.cv2,'waitKey',lambda _:next(keys))
    main.run_pipeline(main.parse_args(['--diagnostics-dir',str(tmp_path),
                                      '--diagnostic-person-id','pilot_user_001']))
    summary=json.loads(next(tmp_path.glob('*/summary.json')).read_text())
    assert summary['overall']['frames']==2
    assert summary['by_target']['HELLO']['frames']==1
    assert summary['metadata']['person_id']=='pilot_user_001'


def test_opt_in_input_capture_can_reproduce_model_prediction(tmp_path):
    import torch,hashlib
    from ml.sequence_model import SignLSTM,save_sequence_checkpoint
    from ml.predict_sequence import SequencePredictor
    from ml.replay_gesture_inputs import replay
    from tests.test_body_context import hands,body
    from ml.body_context import context_features
    model=tmp_path/'model.pth'
    save_sequence_checkpoint(model,SignLSTM(2,input_size=150,hidden_size=8,num_layers=1),['HELLO','J'])
    values=np.stack([context_features(hands(i*.002),body()) for i in range(30)])
    frames=[(i/15,row) for i,row in enumerate(values)]
    expected=SequencePredictor(model).predict_take(frames)
    logger=LiveDiagnostics(tmp_path,{'sequence_model_sha256':hashlib.sha256(model.read_bytes()).hexdigest()})
    logger.mark(ord('9'))
    record(logger,0,info={'completed_take':frames,'dynamic_prediction':expected,
                          'dynamic_emitted':expected.label,'dynamic_evaluated':True})
    logger.close()
    report=replay(logger.root,model)
    assert report['takes'][0]['reproduces_recorded_prediction']
    assert report['takes'][0]['confidence']==pytest.approx(expected.confidence)
    assert report['takes'][0]['target']=='HELLO'
    with np.load(next((logger.root/'gesture_inputs').glob('*.npz')),allow_pickle=False) as data:
        np.testing.assert_array_equal(data['frames'],values)
    # Recording metadata cannot silently select a different checkpoint.
    model.write_bytes(model.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='differs'):
        replay(logger.root,model)


def test_capture_inputs_requires_diagnostics_directory():
    import main
    with pytest.raises(SystemExit):main.parse_args(['--capture-gesture-inputs'])


def test_form_rejection_replay_includes_original_motion_observations(tmp_path):
    import hashlib,torch
    from ml.sequence_model import SignLSTM,save_sequence_checkpoint
    from ml.replay_gesture_inputs import replay
    from tests.test_gesture_forms import pose
    model=tmp_path/'model.pth'
    save_sequence_checkpoint(model,SignLSTM(2,hidden_size=4,num_layers=1),['J','Z'])
    values=pose(()).reshape(-1)
    frames=[(i/15,values.copy()) for i in range(30)]
    motion=[(i/15,{'Right':pose(())+np.array([i*.02,0,0])}) for i in range(30)]
    logger=LiveDiagnostics(tmp_path,{'sequence_model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
                                    'sequence_confidence':0.,'gesture_form_policy':'j_z_salute_form_v1'})
    record(logger,0,info={'completed_take':frames,'completed_motion_take':motion,
                         'dynamic_rejected':'form rejected'})
    logger.close()
    with np.load(next((logger.root/'gesture_inputs').glob('*.npz')),allow_pickle=False) as data:
        np.testing.assert_array_equal(data['motion_timestamps'],np.arange(30)/15)
        assert np.isnan(data['motion_hands'][:,0]).all()
        np.testing.assert_allclose(data['motion_hands'][:,1],np.stack([hands['Right'] for _,hands in motion]))
    report=replay(logger.root,model)
    assert report['takes'][0]['accepted_label'] is None
    assert 'requires sustained' in report['takes'][0]['tracking_or_background_reason']
    assert report['takes'][0]['reproduces_recorded_emission']
