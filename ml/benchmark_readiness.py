"""Prepare and audit an explicit, signer-separated static benchmark.

Anonymous signer IDs are supported. Unknown sessions remain unassigned until
someone can establish which recordings belong to the same real person.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from ml.benchmark import GROUP_COLUMNS, load_group_plan
from ml.data_io import read_samples
from ml.dataset import validate_splits
from ml.preprocessing import STATIC_HAND_CANONICAL

PLAN_COLUMNS = [*GROUP_COLUMNS, 'identity_status', 'identity_evidence', 'label', 'samples', 'left_samples', 'right_samples']


def prepare(data, output, identity_plan=None):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    samples = read_samples(data)
    sessions = defaultdict(list)
    for s in samples:
        sessions[s.session_id].append(s)
    plan = Path(identity_plan) if identity_plan else root/'identity-plan.csv'
    if not plan.exists():
        if identity_plan:
            raise FileNotFoundError(plan)
        with plan.open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, PLAN_COLUMNS)
            w.writeheader()
            for session, rows in sorted(sessions.items()):
                hands = Counter(s.handedness for s in rows)
                w.writerow(dict(session_id=session, identity_status='unknown', label=rows[0].label,
                                samples=len(rows), left_samples=hands['Left'], right_samples=hands['Right']))
    with plan.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    unknown = [r.get('session_id', '') for r in rows if r.get('identity_status') != 'verified'
               or not (r.get('signer_id') or '').strip() or not (r.get('identity_evidence') or '').strip()]
    problems = []
    if unknown:
        problems.append(f'{len(unknown)} sessions lack verified anonymous signer identity and evidence')
    splits = None
    try:
        splits, _ = load_group_plan(plan, samples)
        validate_splits(samples, splits, static_hand_preprocessing=STATIC_HAND_CANONICAL)
    except ValueError as error:
        problems.append(str(error))
    coverage = {}
    if splits is not None:
        mapping = {r['session_id']: r for r in rows}
        for split, ids in splits.items():
            per_label = {}
            for label in sorted({s.label for s in samples}):
                subset = [samples[i] for i in ids if samples[i].label == label]
                signers = {mapping[s.session_id].get('signer_id', '').strip() for s in subset} - {''}
                hands = Counter(s.handedness for s in subset)
                per_label[label] = {'samples': len(subset), 'signers': len(signers), 'handedness': dict(hands)}
                minimum = 3 if split == 'train' else 2
                if len(signers) < minimum:
                    problems.append(f'{split}/{label}: needs at least {minimum} verified signers')
                if not hands['Left'] or not hands['Right']:
                    problems.append(f'{split}/{label}: missing Left or Right examples')
            coverage[split] = per_label
    result = {'dataset_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest(),
              'identity_plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
              'samples': len(samples), 'sessions': len(sessions), 'identity_plan': str(plan),
              'unknown_identity_sessions': len(unknown), 'benchmark_ready': not problems,
              'problems': problems, 'coverage': coverage,
              'criteria': 'Verified anonymous identities/evidence; signer/source/session-disjoint splits; canonical duplicate checks; all labels and both hands per split; at least 3 training and 2 validation/test signers per label. These are collection targets, not statistical reliability guarantees.',
              'test_policy': 'Collect a new untouched test from different people; historical test is development-known. Do not assign invented distinct signers to unknown imported sessions.'}
    (root/'readiness.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='ml/data/landmarks.csv')
    parser.add_argument('--output', default='output/signer-benchmark')
    parser.add_argument('--identity-plan')
    args = parser.parse_args()
    result = prepare(args.data, args.output, args.identity_plan)
    print(json.dumps(result, indent=2))
    return 0 if result['benchmark_ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
