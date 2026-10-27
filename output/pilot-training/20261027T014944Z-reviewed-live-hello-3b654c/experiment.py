"""Reproducible development experiment; preserves sources and active launchers.

Run from the repository root using runpy.run_path(..., run_name='__main__').
Reviewed live captures become training data, so their post-training scores are
fit checks only. Existing validation/test assignments stay fixed; test is unused.
"""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import torch

from ml.body_context import resample_context, resample_context_timestamps
from ml.predict_sequence import SequencePredictor
from ml.sequence_dataset import read_sequence_dir, save_sequence_manifest, load_sequence_manifest
from ml.sequence_io import SequenceSample, save_sequence_file, load_sequence_file
from ml.train_sequence import SequenceTrainConfig, train_sequence


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_capture(path):
    with np.load(path, allow_pickle=False) as data:
        values=data['frames'].copy()
        values[~data['present']]=0
        stamps=data['timestamps'].copy()
    return stamps, values


def prepare_capture(path, predictor):
    stamps,values=load_capture(path)
    features=resample_context_timestamps(values,stamps,predictor.target_fps,
                                        predictor.max_gap_frames/predictor.target_fps)
    predictor.reset()
    try:
        for row in features:predictor.add_frame(row)
        features=predictor._buffer_features(trim_boundary_gaps=True)
        if len(features)<10:raise ValueError('Too few sampled gesture frames')
        return resample_context(features,predictor.sequence_length)
    finally:predictor.reset()


def prediction_row(predictor,path,truth=None,live=False):
    try:
        if live:
            stamps,values=load_capture(path)
            prediction=predictor.predict_take(list(zip(stamps,values)))
        else:prediction=predictor.predict_sequence(load_sequence_file(path).features)
        accepted=prediction.label
        return {'file':path.name,'truth':truth,'accepted':accepted,
                'raw':prediction.raw_label,'confidence':prediction.confidence,
                'correct':(accepted is None if truth=='BACKGROUND' else accepted==truth)
                           if truth else None}
    except ValueError as error:
        return {'file':path.name,'truth':truth,'accepted':None,'error':str(error),
                'correct':truth=='BACKGROUND' if truth else None}


def main():
    torch.set_num_threads(4)
    base=Path('output/pilot-training/20261027T012722Z-reviewed-training-381cbe')
    active=Path('output/pilot-training/20261026T235757Z-training-v2-f5ac65/sequence_training_v2.pth')
    live=Path('output/live-diagnostics/20261027T013734Z-ba69e7')
    review_path=live/'label_review.json'
    review=json.loads(review_path.read_text())['decisions']
    assert Counter(d['review_label'] for d in review.values())=={'HELLO':13,'BACKGROUND':5}
    run=Path('output/pilot-training')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-reviewed-live-hello-'+uuid4().hex[:6])
    run.mkdir(parents=True,exist_ok=False)
    (run/'experiment.py').write_bytes(Path(__file__).read_bytes())
    sources=sorted((base/'sequences').rglob('*.npz'))
    protected={str(p):sha(p) for p in [active,review_path,*sources,*sorted((live/'gesture_inputs').glob('*.npz'))]}
    samples=read_sequence_dir(base/'sequences')
    manifest=load_sequence_manifest(base/'sequence_reviewed_training.splits.json',samples)
    assignments={str(p.relative_to(base/'sequences')):split
                 for split,indices in manifest['splits'].items() for i in indices for p in [sources[i]]}
    for p in sources:
        dest=run/'sequences'/p.relative_to(base/'sequences')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    predictor=SequencePredictor(active)
    imported=[]
    for name,decision in sorted(review.items()):
        source=live/'gesture_inputs'/name
        assert sha(source)==decision['sha256']
        assert Path(decision['source']).resolve()==source.resolve()
        label=decision['review_label']
        stamps,_=load_capture(source)
        features=prepare_capture(source,predictor)
        # Verify the stored model input preserves the live preparation result.
        direct=predictor.predict_take(list(zip(*load_capture(source))))
        prepared=predictor.predict_sequence(features)
        assert direct.raw_label==prepared.raw_label and abs(direct.confidence-prepared.confidence)<1e-6
        relative=Path(label)/('LIVE_'+name)
        sample=SequenceSample(label,'live_'+live.name+'_'+label,'Left',features,{
            'person_id':'pilot_user_001','recording_visit_id':live.name,
            'source':str(source.resolve()),'source_sha256':sha(source),
            'review_sha256':sha(review_path),'target_fps':predictor.target_fps,
            'duration_seconds':float(stamps[-1]-stamps[0]),
            'preparation':'timestamp_interpolation_v1 then body-context resample to 30',
            'usage':'training_only; same session captures not independent evaluation'})
        save_sequence_file(run/'sequences'/relative,sample)
        assignments[str(relative)]='train'
        imported.append({'file':str(relative),'source':str(source),'label':label})
    newpaths=sorted((run/'sequences').rglob('*.npz'))
    newsamples=read_sequence_dir(run/'sequences')
    splits={s:[i for i,p in enumerate(newpaths) if assignments[str(p.relative_to(run/'sequences'))]==s]
            for s in ['train','validation','test']}
    candidate=run/'sequence_reviewed_live_hello.pth'
    save_sequence_manifest(candidate.with_suffix('.splits.json'),newsamples,splits,42)
    config=SequenceTrainConfig(**json.loads((base/'preparation.json').read_text())['config'])
    config.strict_splits=True;config.evaluate_test=False
    preparation={'active_checkpoint':str(active),'base_dataset':str(base/'sequences'),
                 'imported':imported,'counts':dict(Counter(s.label for s in newsamples)),
                 'old_partitions_preserved':True,'all_imports_train_only':True,
                 'original_test_not_evaluated':True,'config':asdict(config),'protected_sha256':protected}
    (run/'preparation.json').write_text(json.dumps(preparation,indent=2))
    print('EXPERIMENT',run,flush=True)
    metrics=train_sequence(str(run/'sequences'),str(candidate),config)
    external=Path('dist/data/vocabulary_v1/evaluation')
    external_review=json.loads(Path('output/label-reviews/evaluation-background-20261027/label_review.json').read_text())['decisions']
    results={}
    for tag,model in [('active',active),('candidate',candidate)]:
        p=SequencePredictor(model)
        development=[]
        for path in sorted(external.rglob('*.npz')):
            sample=load_sequence_file(path)
            truth=sample.label
            if truth=='BACKGROUND':
                decision=external_review[path.name]
                assert sha(path)==decision['sha256']
                truth=decision['review_label']
            development.append(prediction_row(p,path,truth))
        fit=[prediction_row(p,live/'gesture_inputs'/name,d['review_label'],live=True)
             for name,d in sorted(review.items())]
        jz=[]
        for source in sorted((live/'gesture_inputs').glob('*.npz')):
            with np.load(source,allow_pickle=False) as data:meta=json.loads(str(data['metadata']))
            if meta['target'] in ['J','Z']:
                row=prediction_row(p,source,live=True);row.update(target=meta['target'],recorded_emitted=meta['emitted']);jz.append(row)
        results[tag]={'reviewed_development':development,'training_fit_only':fit,
                      'jz_untrained_replay':jz,
                      'development_hello_correct':sum(r['correct'] for r in development if r['truth']=='HELLO'),
                      'development_background_correct':sum(r['correct'] for r in development if r['truth']=='BACKGROUND'),
                      'fit_hello_correct':sum(r['correct'] for r in fit if r['truth']=='HELLO'),
                      'fit_background_correct':sum(r['correct'] for r in fit if r['truth']=='BACKGROUND'),
                      'jz_previous_accepted_retained':sum(r['accepted']==r['recorded_emitted'] for r in jz if r['recorded_emitted']),
                      'jz_emissions':dict(Counter(r['accepted'] for r in jz if r['accepted']))}
    assert all(sha(p)==value for p,value in protected.items())
    summary={'run':str(run),'candidate':str(candidate),'candidate_sha256':sha(candidate),
             'best_epoch':metrics['best_epoch'],'results':results,
             'sources_unchanged':True,'launchers_unchanged':True,
             'limitations':['Reviewed live session now belongs to training; fit counts are not accuracy.',
                            'Older external clips are reused development data, not a fresh final test.',
                            'All data is same signer; a new live session remains necessary.',
                            'No original test split was evaluated.']}
    (run/'summary.json').write_text(json.dumps(summary,indent=2))
    print('RESULT',json.dumps({'run':str(run),'results':{k:{n:v for n,v in d.items() if not isinstance(v,list)} for k,d in results.items()}}),flush=True)


if __name__=='__main__':main()
