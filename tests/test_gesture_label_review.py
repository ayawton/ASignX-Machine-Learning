import hashlib,json
from pathlib import Path
import tkinter as tk
import numpy as np
import pytest
from ml.review_gesture_labels import project_frame,load_review_takes,load_saved_backgrounds,save_decision,LabelReviewer
from ml.body_context import temple_features
from tests.test_body_context import hands,body


def features():
    references=np.concatenate([body(),[[.4,.2],[.6,.2]]]).astype(np.float32)
    row=temple_features(hands(),references)
    return row,references


def make_run(root):
    folder=root/'gesture_inputs';folder.mkdir(parents=True)
    row,_=features()
    for i in range(3):
        np.savez_compressed(folder/f'take_{i:06d}.npz',frames=np.stack([row]*3),
                            timestamps=np.arange(3)/15,present=np.ones(3,dtype=bool),
                            metadata=json.dumps({'target':'REST','emitted':'HELLO','confidence':.9}))
    return root


def test_hand_projection_recovers_original_relative_coordinates():
    row,references=features()
    reconstructed_body,temples,reconstructed_hands=project_frame(row)
    origin=references[2:4].mean(axis=0);scale=np.linalg.norm(references[2]-references[3])
    expected=(hands()[0].to_numpy()[:,:2]-origin)/scale
    np.testing.assert_allclose(reconstructed_hands[1],expected,atol=2e-6)
    np.testing.assert_allclose(temples,(references[6:8]-origin)/scale,atol=2e-6)
    assert not np.isfinite(reconstructed_hands[0]).any()


def test_absent_geometry_is_not_replaced_by_an_invented_hand():
    _,_,points=project_frame(np.zeros(180,dtype=np.float32))
    assert not np.isfinite(points).any()
    with pytest.raises(ValueError):project_frame(np.ones(126))


def test_hello_review_includes_rejected_captures_without_changing_sources(tmp_path):
    root=make_run(tmp_path/'run')
    path=root/'gesture_inputs'/'hello_rejected.npz'
    row,_=features()
    np.savez_compressed(path,frames=np.stack([row]*3),timestamps=np.arange(3)/15,
                        present=np.ones(3,dtype=bool),
                        metadata=json.dumps({'target':'HELLO','emitted':None}))
    before=path.read_bytes()
    assert len(load_review_takes(root))==3
    with pytest.raises(ValueError):load_review_takes(root,target='HELLO')
    takes=load_review_takes(root,target='HELLO',include_unaccepted=True)
    assert len(takes)==1 and takes[0]['metadata']['hide_model']
    assert path.read_bytes()==before


def test_selected_capture_review_can_label_J_and_Z_without_changing_sources(tmp_path):
    root=make_run(tmp_path/'run')
    takes=load_review_takes(root,selected_files={'take_000000.npz','take_000001.npz'})
    assert len(takes)==2
    before={t['path']:t['path'].read_bytes() for t in takes}
    output=tmp_path/'separate-review'/'labels.json';output.parent.mkdir()
    save_decision(output,takes[0],'J');save_decision(output,takes[1],'Z')
    assert {d['review_label'] for d in json.loads(output.read_text())['decisions'].values()}=={'J','Z'}
    assert all(p.read_bytes()==v for p,v in before.items())


def test_pilot_review_buttons_save_all_supported_labels(tmp_path):
    root=tk.Tk();root.withdraw()
    try:
        run=make_run(tmp_path/'run')
        app=LabelReviewer(root,run,pilot_vocabulary=True)
        app.mark('J');app.mark('Z');app.mark('UNSURE')
        assert {d['review_label'] for d in json.loads(app.output.read_text())['decisions'].values()}=={'J','Z','UNSURE'}
    finally:root.destroy()


def test_review_is_a_sidecar_and_preserves_sources(tmp_path):
    root=make_run(tmp_path/'run');takes=load_review_takes(root)
    before={t['path']:t['path'].read_bytes() for t in takes}
    output=root/'label_review.json'
    save_decision(output,takes[0],'HELLO')
    save_decision(output,takes[1],'BACKGROUND')
    save_decision(output,takes[2],'UNSURE')
    review=json.loads(output.read_text())
    assert {r['review_label'] for r in review['decisions'].values()}=={'HELLO','BACKGROUND','UNSURE'}
    assert all(path.read_bytes()==value for path,value in before.items())
    changed=dict(takes[0],sha256='changed')
    with pytest.raises(ValueError,match='source file changed'):save_decision(output,changed,'BACKGROUND')


def test_reviewer_controls_save_and_advance_without_camera(tmp_path):
    root=tk.Tk();root.withdraw()
    try:
        app=LabelReviewer(root,make_run(tmp_path/'run'))
        root.update_idletasks()
        app.draw()
        for label in ['HELLO','BACKGROUND','UNSURE']:app.mark(label)
        assert len(json.loads(app.output.read_text())['decisions'])==3
        assert not app.playing
    finally:root.destroy()


def test_saved_background_review_preserves_data_and_hides_model_outputs(tmp_path):
    from ml.sequence_io import SequenceSample,save_sequence_file
    row,_=features()
    folder=tmp_path/'data'
    for label in ['HELLO','BACKGROUND']:
        save_sequence_file(folder/label/'take.npz',SequenceSample(label,label,'Right',
            np.stack([row]*30),{'duration_seconds':2.,'hand_selection':'both','target_fps':15.}))
    before={p:p.read_bytes() for p in folder.rglob('*.npz')}
    takes=load_saved_backgrounds(folder)
    assert len(takes)==1 and takes[0]['metadata']['hide_model']
    assert takes[0]['metadata']['emitted'] is None
    assert takes[0]['times'][-1]==2.
    output=tmp_path/'review.json';save_decision(output,takes[0],'HELLO')
    assert json.loads(output.read_text())['decisions']['take.npz']['original_target']=='BACKGROUND'
    assert all(path.read_bytes()==value for path,value in before.items())
