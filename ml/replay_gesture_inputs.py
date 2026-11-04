"""Replay diagnostic landmark takes without opening a camera or retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ml.predict_sequence import SequencePredictor
from ml.gesture_acceptance import is_negative_label


def replay(run, model, output=None):
    root=Path(run); model=Path(model)
    paths=sorted((root/'gesture_inputs').glob('*.npz'))
    if not paths:
        raise ValueError('No captured gesture inputs in this diagnostic run')
    predictor=None
    model_hash=hashlib.sha256(model.read_bytes()).hexdigest()
    rows=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            stamps=data['timestamps']; values=data['frames']; present=data['present']
            metadata=json.loads(str(data['metadata']))
            motion=None
            if 'motion_timestamps' in data and 'motion_hands' in data:
                motion_stamps=data['motion_timestamps'];motion_points=data['motion_hands']
                if motion_points.shape!=(len(motion_stamps),2,21,3) or not np.isfinite(motion_stamps).all() or np.any(np.diff(motion_stamps)<=0):
                    raise ValueError('Invalid saved form-check motion observations')
                motion=[(float(t),{name:points[i] for i,name in enumerate(('Left','Right')) if np.isfinite(points[i]).all()})
                        for t,points in zip(motion_stamps,motion_points)]
        settings=metadata['settings']
        expected_hash=settings.get('sequence_model_sha256')
        if expected_hash and model_hash!=expected_hash:
            raise ValueError('Replay checkpoint differs from the captured checkpoint')
        if predictor is None:
            predictor=SequencePredictor(model)
        if not (len(stamps)==len(values)==len(present)) or not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0):
            raise ValueError(f'Invalid replay timeline: {path.name}')
        predictor.threshold=float(settings.get('sequence_confidence',.8))
        frames=[(float(t),v if exists else None) for t,v,exists in zip(stamps,values,present)]
        raw=confidence=emitted=None; reason=None
        try:
            prediction=predictor.predict_take(frames)
            raw=prediction.raw_label; confidence=prediction.confidence
            emitted=prediction.label
            policy=settings.get('gesture_form_policy')
            if policy:
                from ml.gesture_forms import GestureFormAcceptance
                validator=GestureFormAcceptance()
                if policy!=validator.policy_version:raise ValueError('Unsupported gesture form policy')
                if emitted:
                    if motion is None:raise ValueError('Missing motion observations for exact form-check replay')
                    try:validator.validate(emitted,frames,motion_frames=motion)
                    except ValueError as error:reason=str(error);emitted=None
            if is_negative_label(raw): reason='Background / no gesture'
        except ValueError as error:
            reason=str(error);emitted=None
        expected=metadata.get('raw_prediction')
        rows.append({'file':path.name,'target':metadata['target'],
                     'input_observations':len(stamps),'duration_seconds':float(stamps[-1]-stamps[0]) if len(stamps) else 0,
                     'raw_prediction':raw,'confidence':confidence,'accepted_label':emitted,
                     'tracking_or_background_reason':reason,'recorded_raw_prediction':expected,
                     'reproduces_recorded_prediction':raw==expected if expected is not None else None,
                     'reproduces_recorded_emission':emitted==metadata.get('emitted'),
                     'recorded_rejection':metadata.get('rejected')})
    report={'model':str(model.resolve()),'model_sha256':model_hash,'takes':rows,
            'limitations':['Replays model preparation/classification, not the motion guard.',
                           'Target labels describe user intent, not verified ASL correctness.',
                           'Diagnostic inputs are not automatically training examples.']}
    target=Path(output) if output else root/'replay.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output')
    args=parser.parse_args()
    print(json.dumps(replay(args.run,args.model,args.output),indent=2))


if __name__=='__main__': main()
