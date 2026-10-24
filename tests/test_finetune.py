"""Warm-start weights, baseline eligibility and source checkpoint protection."""
import csv
import hashlib
import numpy as np
import pytest
import torch
from ml.data_io import CSV_COLUMNS
from ml.model import SignMLP, save_checkpoint, load_checkpoint
from ml.preprocessing import normalize_landmarks, PREPROCESSING_VERSION, STATIC_HAND_CANONICAL
from ml.train import train, TrainConfig


def test_finetune_loads_weights_and_keeps_epoch_zero_eligible(tmp_path):
    data=tmp_path/'data.csv'
    with data.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(CSV_COLUMNS)
        for i in range(12):
            w.writerow(['A' if i<6 else 'B',f'session-{i}','Right',PREPROCESSING_VERSION,
                        *normalize_landmarks(np.random.default_rng(i).normal(size=(21,3)))])
    source=tmp_path/'initial.pth';output=tmp_path/'fine.pth'
    initial=SignMLP(2,hidden_sizes=(8,),static_hand_preprocessing=STATIC_HAND_CANONICAL)
    save_checkpoint(source,initial,['A','B'])
    before=source.read_bytes()
    config=TrainConfig(epochs=1,hidden_sizes=(8,),learning_rate=1e-30,weight_decay=0,
                       initial_checkpoint=str(source),evaluate_test=False)
    result=train(str(data),str(output),config)
    assert result['best_epoch']==0 and result['test'] is None
    loaded,_,metadata=load_checkpoint(output)
    for name,value in initial.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[name],rtol=0,atol=0)
    assert metadata['training']['initial_checkpoint_sha256']==hashlib.sha256(before).hexdigest()
    assert source.read_bytes()==before
    with pytest.raises(ValueError,match='separate output'):
        train(str(data),str(source),config)
    save_checkpoint(source,initial,['B','A'])
    with pytest.raises(ValueError,match='must match'):
        train(str(data),str(tmp_path/'wrong.pth'),config)
