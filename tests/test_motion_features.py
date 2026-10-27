import numpy as np
import pytest
import torch
from ml.motion_features import MotionDeltaFeatures
from ml.sequence_model import SignLSTM, save_sequence_checkpoint, load_sequence_checkpoint


@pytest.mark.parametrize('width', [63, 126, 150, 180])
def test_stationary_has_zero_motion_and_change_retains_direction(width):
    features=torch.ones(2, 10, width)
    layer=MotionDeltaFeatures(width)
    assert not layer(features).any()
    features[:,:,1]=torch.linspace(1,2,10)
    forward=layer(features)
    assert torch.allclose(forward[:,1:,1],torch.ones(2,9))
    reversed_motion=layer(features.flip(1))
    assert torch.allclose(reversed_motion[:,1:],-forward[:,1:].flip(1),atol=1e-6)
    assert not forward[:,0].any()


def test_missing_hand_and_body_references_do_not_fabricate_motion():
    features=torch.ones(1,5,180)
    features[:,2,:63]=0
    features[:,2,144:147]=0
    motion=MotionDeltaFeatures(180)(features)
    assert not motion[:,2:4,:63].any()
    assert not motion[:,2:4,144:147].any()
    assert not motion[:,:,128::3].any()  # visibility is not displacement
    assert torch.isfinite(motion).all()


@pytest.mark.parametrize('backbone', ['lstm','tcn_attention'])
@pytest.mark.parametrize('width,kinematics,cosine', [(63,True,True),(126,False,False),(180,False,True)])
def test_motion_model_roundtrip_and_gradients(tmp_path,backbone,width,kinematics,cosine):
    model=SignLSTM(3,input_size=width,hidden_size=8,num_layers=1,dropout=0,
                   backbone=backbone,use_kinematics=kinematics,cosine_head=cosine,
                   use_motion_deltas=True)
    features=torch.ones(2,6,width,requires_grad=True)
    output=model(features)
    assert output.shape==(2,3) and torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(features.grad).all()
    path=tmp_path/'motion.pth'; model.eval()
    expected=model(features).detach()
    save_sequence_checkpoint(path,model,['HELLO','J','Z'],sequence_length=6)
    loaded,labels,checkpoint=load_sequence_checkpoint(path)
    torch.testing.assert_close(expected,loaded(features))
    assert loaded.config['use_motion_deltas']


def test_legacy_checkpoint_without_motion_flag_still_loads(tmp_path):
    path=tmp_path/'legacy.pth'
    model=SignLSTM(2,input_size=180,hidden_size=8,num_layers=1)
    save_sequence_checkpoint(path,model,['HELLO','J'])
    checkpoint=torch.load(path,weights_only=False)
    checkpoint['model_config'].pop('use_motion_deltas')
    torch.save(checkpoint,path)
    loaded,_,_=load_sequence_checkpoint(path)
    assert loaded.motion_features is None
    assert loaded(torch.ones(1,30,180)).shape==(1,2)
