"""Readiness must distinguish anonymous verified people from unknown imports."""
import csv
from types import SimpleNamespace

import numpy as np

from ml.benchmark_readiness import PLAN_COLUMNS, prepare
from ml.data_io import CSV_COLUMNS
from ml.preprocessing import normalize_landmarks, PREPROCESSING_VERSION


def dataset(tmp_path):
    path=tmp_path/'data.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(CSV_COLUMNS)
        for person in range(7):
            for label in ['A','B']:
                for side in ['Left','Right']:
                    points=np.random.default_rng(person*10+ord(label)+(side=='Right')*100).normal(size=(21,3))
                    writer.writerow([label,f'{person}-{label}',side,PREPROCESSING_VERSION,*normalize_landmarks(points)])
    return path


def write_verified(plan):
    with plan.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    for row in rows:
        person=int(row['session_id'].split('-')[0])
        row.update(group_id=f'person-{person}',signer_id=f'person-{person}',
                   split='train' if person<3 else 'validation' if person<5 else 'test',
                   identity_status='verified',identity_evidence='supervised collection register')
    with plan.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,PLAN_COLUMNS);w.writeheader();w.writerows(rows)
    return rows


def test_unknown_identities_do_not_create_a_ready_benchmark(tmp_path):
    data=dataset(tmp_path);out=tmp_path/'benchmark'
    result=prepare(data,out)
    assert not result['benchmark_ready'] and result['unknown_identity_sessions']==14
    plan=out/'identity-plan.csv'
    original=plan.read_bytes()
    prepare(data,out)
    assert plan.read_bytes()==original
    with plan.open(newline='') as f:rows=list(csv.DictReader(f))
    assert all(not r['signer_id'] and not r['split'] for r in rows)


def test_verified_anonymous_signers_pass_but_cross_split_reuse_fails(tmp_path):
    data=dataset(tmp_path);out=tmp_path/'benchmark'
    prepare(data,out);plan=out/'identity-plan.csv';rows=write_verified(plan)
    result=prepare(data,out)
    assert result['benchmark_ready'] and result['coverage']['test']['A']['signers']==2
    rows[-1]['signer_id']=rows[0]['signer_id']
    with plan.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,PLAN_COLUMNS);w.writeheader();w.writerows(rows)
    result=prepare(data,out)
    assert not result['benchmark_ready']
    assert any('leaks' in p for p in result['problems'])
