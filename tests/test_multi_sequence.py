from types import SimpleNamespace
import numpy as np
import pytest
from ml.predict import Prediction, UnifiedSignRecognizer
from ml.predict_sequence import SequencePredictor, SequencePredictorGroup
from tests.test_gesture_transitions import hand


class Predictor:
    def __init__(self,width,prediction,fail=False):
        self.input_size=width
        self.labels=['J','Z'] if width==126 else ['Hello','Thank You']
        self.prediction=prediction
        self.fail=fail
        self.frames=[]
    def reset(self): pass
    def predict_take(self,frames):
        self.frames=frames
        if self.fail: raise ValueError('lost tracking')
        return self.prediction


@pytest.mark.parametrize('left,right,expected,raw',[
    (Prediction('J',.85,'J'),Prediction(None,.7,'Hello'),'J','J'),
    (Prediction(None,.5,'Z'),Prediction('Hello',.9,'Hello'),'Hello','Hello'),
    (Prediction('J',.99,'J'),Prediction('Hello',.81,'Hello'),None,'AMBIGUOUS'),
    (Prediction(None,.7,'J'),Prediction(None,.7,'Hello'),None,'UNKNOWN'),
])
def test_disagreement_does_not_compare_uncalibrated_scores(left,right,expected,raw):
    group=SequencePredictorGroup({'letters':Predictor(126,left),'words':Predictor(150,right)})
    frame=group.prepare_frame([hand(0)],None)
    result=group.predict_take([(0,frame)])
    assert result.label==expected
    assert result.raw_label==raw
    assert group.predictors['letters'].frames[0][1].shape==(2,21,3)
    assert group.predictors['words'].frames[0][1].shape==(150,)
    assert group.input_size==150


def test_tracking_failure_is_isolated_to_one_model():
    group=SequencePredictorGroup({'letters':Predictor(126,Prediction('Z',.9,'Z')),
                                  'words':Predictor(150,None,fail=True)})
    assert group.predict_take([(0,group.prepare_frame([hand(0)],None))]).label=='Z'
    assert group.last_predictions['words'] is None


def test_models_share_capture_and_settling_without_double_emission():
    group=SequencePredictorGroup({'letters':Predictor(126,Prediction('J',.9,'J')),
                                  'words':Predictor(150,Prediction(None,.5,'Hello'))})
    router=UnifiedSignRecognizer(sequence_predictor=group,velocity_threshold=.004)
    emissions=[]
    for i in range(100):
        t=i/30
        result=router.update([hand(min(max(t-.3,0),1.5)*.3)],timestamp=t)
        if result['dynamic_emitted']: emissions.append(result['dynamic_emitted'])
    assert emissions==['J']
    assert len(group.predictors['letters'].frames)==len(group.predictors['words'].frames)
    assert router.state=='static'


def test_real_checkpoints_load_together():
    from pathlib import Path
    letters=Path('ml/models/sequence_both_hands_v1.pth')
    words=Path('ml/models/sequence_body_v1.pth')
    if not letters.exists() or not words.exists():
        pytest.skip('Local checkpoints unavailable')
    group=SequencePredictorGroup({'letters':SequencePredictor(letters),'words':SequencePredictor(words)})
    assert {'J','Z','Hello','Thank You'} <= set(group.labels)
    assert group.input_size==150
