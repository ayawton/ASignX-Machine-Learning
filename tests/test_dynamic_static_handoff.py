from types import SimpleNamespace
import numpy as np
from ml.predict import Prediction,SignRecognizer,UnifiedSignRecognizer
from tests.test_gesture_transitions import hand,Sequence


class Static:
    label='B'
    def predict(self,*args,**kwargs):return Prediction(self.label,.99,self.label)


class Hello(Sequence):
    accepted=True
    def predict_sequence(self,features):
        self.calls.append(features.copy())
        return Prediction('HELLO' if self.accepted else None,.99,'HELLO')


def run_gesture(router,both=False):
    outputs=[]
    for i in range(100):
        t=i/30
        moving=hand(min(max(t-.3,0),1.5)*.3)
        hands=[moving]
        if both:
            still=hand(0);still.handedness='Left';hands.append(still)
        outputs.append(router.update(hands,timestamp=t))
    return outputs


def test_hello_does_not_emit_ending_B_but_B_works_after_hand_release():
    router=UnifiedSignRecognizer(Static(),Hello(),velocity_threshold=.004,
        stable_frames=2,release_frames=3,hold_static_after_gesture=True)
    outputs=run_gesture(router)
    accepted=next(i for i,r in enumerate(outputs) if r['dynamic_emitted']=='HELLO')
    assert not any(r['static'][0].emitted_label for r in outputs[accepted:])
    assert outputs[-1]['static'][0].awaiting_release
    assert outputs[-1]['static'][0].prediction.raw_label=='B'
    for i in range(100,104):router.update([],timestamp=i/30)
    fresh=[router.update([hand(.45)],timestamp=i/30) for i in range(104,110)]
    assert [r['static'][0].emitted_label for r in fresh if r['static'][0].emitted_label]==['B']


def test_different_intentional_letter_can_rearm_without_hands_leaving():
    predictor=Static();recognizer=SignRecognizer(predictor,stable_frames=3,release_frames=3)
    recognizer.update([hand(0)],timestamp=0,suppress=True,consume_hands={'Right'})
    assert recognizer.update([hand(0)],timestamp=.1)[0].awaiting_release
    predictor.label='A'
    outputs=[recognizer.update([hand(0)],timestamp=i/10)[0] for i in range(2,5)]
    assert outputs[-1].emitted_label=='A' and not outputs[-1].awaiting_release
    predictor.label='B'
    outputs=[recognizer.update([hand(0)],timestamp=i/10)[0] for i in range(5,8)]
    assert outputs[-1].emitted_label=='B'


def test_brief_tracking_loss_does_not_rearm_the_ending_pose():
    recognizer=SignRecognizer(Static(),stable_frames=1,release_frames=3)
    recognizer.update([hand(0)],timestamp=0,suppress=True,consume_hands={'Right'})
    recognizer.update([],timestamp=.1)
    result=recognizer.update([hand(0)],timestamp=.2)[0]
    assert result.awaiting_release and result.emitted_label is None


def test_only_the_moving_hands_ending_pose_is_consumed():
    class TwoHandHello(Hello):
        input_size=126
        def predict_take(self,frames):return Prediction('HELLO',.99,'HELLO')
    class PerHandStatic:
        def predict(self,*args,handedness=None):
            label='A' if handedness=='Left' else 'B'
            return Prediction(label,.99,label)
    router=UnifiedSignRecognizer(PerHandStatic(),TwoHandHello(),velocity_threshold=.004,
        stable_frames=2,hold_static_after_gesture=True)
    outputs=run_gesture(router,both=True)
    final={r.hand:r for r in outputs[-1]['static']}
    assert final['Right'].awaiting_release
    assert not final['Left'].awaiting_release and final['Left'].stable_label=='A'


def test_unaccepted_dynamic_candidate_does_not_consume_static_B():
    sequence=Hello();sequence.accepted=False
    router=UnifiedSignRecognizer(Static(),sequence,velocity_threshold=.004,
        stable_frames=2,hold_static_after_gesture=True)
    outputs=run_gesture(router)
    assert not any(r['dynamic_emitted'] for r in outputs)
    assert outputs[-1]['static'][0].stable_label=='B'
    assert not outputs[-1]['static'][0].awaiting_release
