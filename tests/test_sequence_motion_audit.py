import numpy as np
import pytest
from ml.audit_sequence_motion import variants,audit
from tests.test_temple_context import references
from tests.test_body_context import hands
from ml.body_context import context_features


def test_variants_preserve_source_and_expose_frame_order():
    source=np.stack([context_features(hands(i*.01),references(),temple_context=True) for i in range(30)])
    saved=source.copy()
    changed=variants(source)
    np.testing.assert_array_equal(source,saved)
    assert all(x.shape==source.shape for x in changed.values())
    np.testing.assert_array_equal(changed['frozen_first'],np.repeat(source[:1],30,axis=0))
    np.testing.assert_array_equal(changed['reversed'],source[::-1])
    np.testing.assert_array_equal(changed['first_half_stretched'][-1],source[14])
    np.testing.assert_array_equal(changed['shuffled'],variants(source)['shuffled'])
    assert not np.array_equal(changed['shuffled'],source)
    assert set(np.unique(changed['first_half_stretched'][:,126:].reshape(30,-1,3)[:,:,2])) <= {0,1}


def test_audit_refuses_to_claim_held_out_when_dataset_changed(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import ml.audit_sequence_motion as module
    sample=SimpleNamespace(features=np.ones((30,180),dtype=np.float32),label='J',session_id='s')
    monkeypatch.setattr(module,'read_sequence_dir',lambda path:[sample])
    monkeypatch.setattr(module,'load_sequence_checkpoint',lambda path:(None,['J','Hello'],{'dataset_sha256':'old'}))
    monkeypatch.setattr(module,'dataset_sequence_digest',lambda samples:'changed')
    with pytest.raises(ValueError,match='Dataset differs'):
        audit('data','model',tmp_path/'report')
    assert not (tmp_path/'report').exists()
