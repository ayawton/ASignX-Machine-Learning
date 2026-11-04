from types import SimpleNamespace
import numpy as np
import pytest
from ml.gesture_forms import GestureFormAcceptance,extended_fingers
from ml.predict import UnifiedSignRecognizer,Prediction


def pose(extended=()):
    points=np.zeros((21,3),dtype=np.float32)
    points[1:5]=[[-.3,.2,0],[-.5,.4,0],[-.6,.5,0],[-.7,.6,0]]
    for finger,start in enumerate([5,9,13,17]):
        x=-.35+finger*.25
        points[start]=[x,.7,0]
        if finger in extended:
            points[start+1:start+4]=[[x,1.2,0],[x,1.55,0],[x,1.8,0]]
        else:points[start+1:start+4]=[[x,1.05,0],[x,.88,.1],[x,.7,.08]]
    return points


def frames(extended,near=True):
    observations=[]
    for i in range(30):
        row=np.zeros(180,dtype=np.float32);row[:63]=pose(extended).reshape(-1)
        row[150:156]=[0,0,1,1,0,1]
        offsets=row[156:].reshape(2,2,2,3)
        offsets[0,:,1]=[.1 if near else 1.5,.1,1]
        observations.append((i/30,row))
    return observations


@pytest.mark.parametrize('extended', [(),(0,),(3,),(0,1,2,3)])
def test_extension_flags_follow_finger_geometry(extended):
    assert np.flatnonzero(extended_fingers(pose(extended))).tolist()==list(extended)


@pytest.mark.parametrize('label,fingers',[('J',(3,)),('Z',(0,)),('HELLO',(0,1,2,3))])
def test_expected_shapes_are_accepted(label,fingers):
    GestureFormAcceptance().validate(label,frames(fingers))


@pytest.mark.parametrize('label',['J','Z','HELLO'])
def test_closed_fist_is_rejected_despite_label(label):
    with pytest.raises(ValueError,match='requires sustained'):
        GestureFormAcceptance().validate(label,frames(()))


def test_finger_evidence_is_scale_rotation_and_mirror_invariant():
    original=pose((3,))
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=np.float32)
    for scale in [.1,1.,5.]:
        for mirror in [-1,1]:
            transformed=original@rotation*scale*np.array([mirror,1,1])+3
            assert extended_fingers(transformed).tolist()==[False,False,False,True]


def test_a_brief_correct_pose_cannot_validate_a_fist_take():
    observations=frames(())
    observations[5][1][:63]=pose((3,)).reshape(-1)
    with pytest.raises(ValueError):GestureFormAcceptance().validate('J',observations)


def test_idle_second_hand_cannot_supply_the_moving_fists_shape():
    observations=frames(())
    motion=[]
    for i,(stamp,row) in enumerate(observations):
        row[63:126]=pose((3,)).reshape(-1)
        motion.append((stamp,{'Left':pose(())+np.array([i*.1,0,0]),'Right':pose((3,))}))
    with pytest.raises(ValueError):
        GestureFormAcceptance().validate('J',observations,motion_frames=motion)


def test_hello_requires_early_open_hand_and_visible_face_reference():
    validator=GestureFormAcceptance()
    missing=frames((0,1,2,3))
    for _,row in missing:row[150:156]=0
    for observations in [frames((0,1,2,3),near=False),missing]:
        with pytest.raises(ValueError,match='forehead'):
            validator.validate('HELLO',observations)
    late=frames((0,1,2,3),near=False)
    for _,row in late[15:]:row[156:].reshape(2,2,2,3)[0,:,1]=[.1,.1,1]
    with pytest.raises(ValueError,match='forehead'):validator.validate('HELLO',late)


def test_fast_salute_needs_consecutive_start_evidence_without_a_hold():
    validator=GestureFormAcceptance()
    for count in [1,2]:
        observations=frames((0,1,2,3),near=False)
        for _,row in observations[:count]:row[156:].reshape(2,2,2,3)[0,:,1]=[.1,.1,1]
        if count==1:
            with pytest.raises(ValueError,match='forehead'):validator.validate('HELLO',observations)
        else:validator.validate('HELLO',observations)


def test_high_confidence_cannot_bypass_form_check_and_router_recovers():
    class Predictor:
        input_size=63
        def reset(self):pass
        def predict_take(self,frames):return Prediction('J',.999,'J')
    for fingers,expected in [((),[]),((3,),['J'])]:
        router=UnifiedSignRecognizer(sequence_predictor=Predictor(),velocity_threshold=.004,
                                     form_acceptance=GestureFormAcceptance())
        outputs=[]
        for i in range(100):
            pts=pose(fingers)*.1+np.array([min(max(i/30-.3,0),1.)*.3,0,0])
            outputs.append(router.update([SimpleNamespace(handedness='Right',to_numpy=lambda pts=pts:pts)],timestamp=i/30))
        assert [r['dynamic_emitted'] for r in outputs if r['dynamic_emitted']]==expected
        assert router.state=='static' and not router._gesture
        if not fingers:assert any('extended pinky' in (r['dynamic_rejected'] or '') for r in outputs)
