"""Diagnose sequence shortcuts without changing model weights or recordings."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from ml.body_context import BODY_FEATURE_WIDTHS,resample_context
from ml.sequence_dataset import read_sequence_dir,load_sequence_manifest,dataset_sequence_digest
from ml.sequence_model import load_sequence_checkpoint
from ml.video import resample_sequence


def variants(sequence,seed=42):
    """Alter prepared sequences only; these are diagnostics, not training samples."""
    sequence=np.asarray(sequence,dtype=np.float32)
    length=len(sequence)
    if length<2: raise ValueError('Motion audit requires at least two steps')
    half=sequence[:max(2,length//2)]
    stretched=(resample_context(half,length) if sequence.shape[1] in BODY_FEATURE_WIDTHS
               else resample_sequence(half,length,interpolate=True))
    order=np.random.default_rng(seed).permutation(length)
    return {
        'original':sequence.copy(),
        'frozen_first':np.repeat(sequence[:1],length,axis=0),
        'frozen_middle':np.repeat(sequence[length//2:length//2+1],length,axis=0),
        'reversed':sequence[::-1].copy(),
        'first_half_stretched':stretched,
        'shuffled':sequence[order].copy(),
    }


@torch.inference_mode()
def audit(data_dir,model_path,output,split='test',threshold=.8):
    if not 0<=threshold<=1: raise ValueError('Threshold must be between zero and one')
    if split not in ('train','validation','test','all'): raise ValueError('Invalid split')
    samples=read_sequence_dir(data_dir)
    model,labels,checkpoint=load_sequence_checkpoint(model_path)
    digest=dataset_sequence_digest(samples)
    matches=digest==checkpoint.get('dataset_sha256')
    if split=='all':
        indices=list(range(len(samples)))
    else:
        if not matches: raise ValueError('Dataset differs from checkpoint. Use original data for held-out evaluation, or --split all for exploratory diagnostics only.')
        manifest=load_sequence_manifest(Path(model_path).with_suffix('.splits.json'),samples)
        indices=manifest['splits'][split]
    length=int(checkpoint.get('sequence_length',30))
    width=int(model.config['input_size'])
    if any(s.features.shape!=(length,width) or s.label not in labels for s in samples):
        raise ValueError('Dataset shape or labels do not match checkpoint')
    rows=[]
    for index in indices:
        sample=samples[index]
        transformed=variants(sample.features,seed=42+index)
        names=list(transformed)
        batch=torch.from_numpy(np.stack(list(transformed.values())))
        probabilities=torch.softmax(model(batch),dim=1).numpy()
        for name,features,probs in zip(names,transformed.values(),probabilities):
            best=int(probs.argmax())
            rows.append({'session_id':sample.session_id,'true_label':sample.label,'split':split,
                         'variant':name,'prediction':labels[best],'confidence':float(probs[best]),
                         'true_label_confidence':float(probs[labels.index(sample.label)]),
                         'accepts_original_label':bool(labels[best]==sample.label and probs[best]>=threshold),
                         'visible_hand_fraction':float(np.any(features[:,:min(width,126)]!=0,axis=1).mean())})
    summary=[]
    for label in sorted({r['true_label'] for r in rows}):
        originals={r['session_id']:r for r in rows if r['true_label']==label and r['variant']=='original'}
        for name in transformed:
            selected=[r for r in rows if r['true_label']==label and r['variant']==name]
            paired=[r for r in selected if originals[r['session_id']]['accepts_original_label']]
            summary.append({'label':label,'variant':name,'recordings':len(selected),
                            'accepted_as_original_label':sum(r['accepts_original_label'] for r in selected),
                            'originally_accepted':len(paired),
                            'still_accepted_after_change':sum(r['accepts_original_label'] for r in paired),
                            'mean_true_label_confidence':float(np.mean([r['true_label_confidence'] for r in selected]))})
    report={'model':str(model_path),'data_dir':str(data_dir),'split':split,'threshold':threshold,
            'dataset_matches_checkpoint':matches,'selected_recordings':len(indices),
            'feature_width':width,'labels':labels,'summary':summary,
            'limitations':['Raw classifier outputs: bypasses live motion gating and visibility rejection.',
                           'Held-out recordings follow saved session splits; they are not necessarily unseen signers.',
                           'Reversed or truncated examples are not automatically invalid ASL gestures.',
                           'Confidence is a model score, not a calibrated probability of ASL correctness.',
                           'For hand-only trajectory-anchored formats, reversal does not recompute the original raw anchor.']}
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (root/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Sequence motion audit','',f'Model: `{model_path}`; split: **{split}**; threshold: {threshold:.0%}.',
           '','Counts show predictions accepted as the recording label. Altered sequences are diagnostic probes, not labeled test accuracy.',
           '','| Label | Input | Accepted / recordings | Mean score for recording label |',
           '|---|---|---:|---:|']
    for r in summary:
        lines.append(f"| {r['label']} | {r['variant']} | {r['accepted_as_original_label']} / {r['recordings']} | {r['mean_true_label_confidence']:.1%} |")
    lines+=['','Limitations:',*['- '+x for x in report['limitations']]]
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',default='output/sequence-motion-audit')
    parser.add_argument('--split',choices=('train','validation','test','all'),default='test')
    parser.add_argument('--threshold',type=float,default=.8)
    args=parser.parse_args()
    audit(args.data_dir,args.model,args.output,args.split,args.threshold)


if __name__=='__main__': main()
