"""Stable output paths, automatic backups and protected failed retraining."""
import json
from pathlib import Path
import numpy as np
import pytest
from ml.sequence_io import SequenceSample,save_sequence_file
from ml.train_sequence import train_sequence,SequenceTrainConfig
from ml.sequence_model import load_sequence_checkpoint


def add_take(root,label,index):
    seed=sum(map(ord,label))*100+index
    features=np.random.default_rng(seed).normal(size=(6,63)).astype(np.float32)
    save_sequence_file(root/label/f'{index}.npz',SequenceSample(label,f'{label}{index}','Right',features,{'target_fps':15.}))


def dataset(root):
    for label in ('J','Z'):
        for i in range(3): add_take(root,label,i)


def config(**kwargs):
    return SequenceTrainConfig(epochs=1,hidden_size=4,num_layers=1,backbone='lstm',augment=False,**kwargs)


def artifacts(path):
    return {suffix:path.with_suffix(suffix).read_bytes() for suffix in ('.pth','.splits.json','.metrics.json')}


def test_changed_dataset_reuses_model_path_and_backs_up_old_artifacts(tmp_path):
    data=tmp_path/'data'; output=tmp_path/'model.pth'; dataset(data)
    train_sequence(str(data),str(output),config())
    before=artifacts(output)
    add_take(data,'J',3)
    train_sequence(str(data),str(output),config())
    after=artifacts(output)
    assert json.loads(after['.splits.json'])['dataset_sha256'] != json.loads(before['.splits.json'])['dataset_sha256']
    assert sum(sum(counts.values()) for counts in json.loads(after['.splits.json'])['counts'].values())==7
    backup=list((tmp_path/'backups'/'model').iterdir())
    assert len(backup)==1
    for suffix,content in before.items(): assert (backup[0]/output.with_suffix(suffix).name).read_bytes()==content
    assert load_sequence_checkpoint(output)[2]['dataset_sha256']==json.loads(after['.splits.json'])['dataset_sha256']


def test_unchanged_data_reuses_splits(tmp_path):
    data=tmp_path/'data'; output=tmp_path/'model.pth'; dataset(data)
    train_sequence(str(data),str(output),config())
    splits=output.with_suffix('.splits.json').read_bytes()
    train_sequence(str(data),str(output),config())
    assert output.with_suffix('.splits.json').read_bytes()==splits


def test_added_labels_are_in_all_splits(tmp_path):
    data=tmp_path/'data'; output=tmp_path/'model.pth'; dataset(data)
    train_sequence(str(data),str(output),config())
    for i in range(3): add_take(data,'HELLO',i)
    train_sequence(str(data),str(output),config())
    assert set(load_sequence_checkpoint(output)[1])=={'J','Z','HELLO'}
    manifest=json.loads(output.with_suffix('.splits.json').read_text())
    assert all(set(counts)=={'J','Z','HELLO'} for counts in manifest['counts'].values())


def test_strict_mode_keeps_old_artifacts_untouched(tmp_path):
    data=tmp_path/'data'; output=tmp_path/'model.pth'; dataset(data)
    train_sequence(str(data),str(output),config())
    before=artifacts(output)
    add_take(data,'J',3)
    with pytest.raises(ValueError,match='strict-splits'):
        train_sequence(str(data),str(output),config(strict_splits=True))
    assert artifacts(output)==before
    assert not (tmp_path/'backups').exists()


def test_training_failure_does_not_replace_model_or_manifest(tmp_path,monkeypatch):
    import ml.train_sequence as module
    data=tmp_path/'data'; output=tmp_path/'model.pth'; dataset(data)
    train_sequence(str(data),str(output),config())
    before=artifacts(output)
    add_take(data,'J',3)
    def fail(*args,**kwargs): raise RuntimeError('simulated training failure')
    monkeypatch.setattr(module,'evaluate_sequence_model',fail)
    with pytest.raises(RuntimeError,match='simulated'):
        train_sequence(str(data),str(output),config())
    assert artifacts(output)==before
    assert not list(tmp_path.glob('.model-training-*'))


def test_invalid_unchanged_manifest_is_not_silently_reset(tmp_path):
    data=tmp_path/'data'; output=tmp_path/'model.pth'; dataset(data)
    train_sequence(str(data),str(output),config())
    path=output.with_suffix('.splits.json')
    manifest=json.loads(path.read_text()); manifest['splits']['train']=[]
    path.write_text(json.dumps(manifest))
    before=artifacts(output)
    with pytest.raises(ValueError,match='invalid for unchanged'):
        train_sequence(str(data),str(output),config())
    assert artifacts(output)==before
