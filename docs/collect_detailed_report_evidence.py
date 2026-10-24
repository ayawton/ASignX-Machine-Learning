"""Inspect frozen report inputs and compare static models on one saved split."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from ml.data_io import read_samples
from ml.dataset import LandmarkDataset, load_manifest, validate_splits
from ml.evaluate import evaluate_model
from ml.model import load_checkpoint
from ml.sequence_model import load_sequence_checkpoint

OUT = ROOT / 'output/system-report-2026-10-22'
OUT.mkdir(parents=True, exist_ok=True)
def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

data = ROOT / 'ml/data/landmarks.csv'
before = sha(data)
counts, hands, versions, sessions = Counter(), Counter(), Counter(), set()
with data.open(newline='') as stream:
    reader = csv.DictReader(stream)
    schema = reader.fieldnames
    for row in reader:
        counts[row['label']] += 1
        hands[row['handedness']] += 1
        versions[row['preprocessing']] += 1
        sessions.add(row['session_id'])
run = ROOT / 'ml/models/manual_runs/20261022T071741Z-87dde74ec0df'
snapshot = run / 'landmarks.csv'
manifest = load_manifest(run / 'sign_classifier.splits.json', snapshot)
samples = read_samples(snapshot)
validate_splits(samples, manifest['splits'], static_hand_preprocessing='left_x_reflection_v1')
evidence = {
    'report_date': '2026-10-22', 'timezone': 'Asia/Singapore',
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'dataset': {'sha256': before, 'rows': sum(counts.values()), 'class_counts': dict(sorted(counts.items())),
                'handedness': dict(hands), 'sessions': len(sessions), 'schema': schema, 'preprocessing': dict(versions)},
    'snapshot_sha256': sha(snapshot), 'manifest_sha256': sha(run / 'sign_classifier.splits.json'),
    'current_split_counts': {k:len(v) for k,v in manifest['splits'].items()},
    'models': {}, 'paired_evaluation': {},
    'evaluation_note': 'Fresh, unthresholded offline predictions on the manual-run frozen validation/test split. Development-known session split, not verified signer-independent.',
}
torch.set_num_threads(4)
for path in sorted((ROOT / 'ml/models').rglob('*.pth')):
    loader = load_sequence_checkpoint if path.name == 'sequence_classifier.pth' else load_checkpoint
    model, labels, checkpoint = loader(path)
    key = str(path.relative_to(ROOT)).replace('\\', '/')
    evidence['models'][key] = {'sha256': sha(path), 'parameters':sum(p.numel() for p in model.parameters()),
        'labels':labels, 'config':model.config, 'training':checkpoint.get('training'),
        'dataset_sha256':checkpoint.get('dataset_sha256'), 'preprocessing':checkpoint.get('preprocessing'),
        'sequence_length':checkpoint.get('sequence_length'),
        'split_counts':{k:len(v) for k,v in checkpoint.get('splits',{}).items()},
        'saved_metrics':read(path.with_suffix('.metrics.json'))}
    if path.name != 'sequence_classifier.pth':
        metrics = {}
        for partition in ['validation', 'test']:
            ds = LandmarkDataset(samples, manifest['splits'][partition], labels, static_hand_preprocessing=model.static_hand_preprocessing)
            metrics[partition] = evaluate_model(model, DataLoader(ds, batch_size=1024), labels)
        evidence['paired_evaluation'][key] = metrics
        print(key, {k:round(v['macro_f1'],6) for k,v in metrics.items()}, flush=True)
references = list((ROOT/'ml').glob('*.py')) + list((ROOT/'asignx').glob('*.py')) + [
    ROOT/'main.py', ROOT/'web_client/asignx_client_sdk.ts', ROOT/'requirements.txt',
    ROOT/'output/current-model-comparison/comparison.json', ROOT/'output/data-repair/latest-correction.json',
    ROOT/'output/recognition-reevaluation/summary.json', ROOT/'ASignX_TestRecognition.bat',
    ROOT/'output/system-cleanup/cleanup-record.json', ROOT/'output/model-cleanup/README.md']
original = ROOT/'archive/data-corrections/remove-wrong-M-20261021T150522127835Z/landmarks.before.csv'
old_manifest = read(ROOT/'ml/models/sign_classifier_canonical.splits.json')
assert sha(original) == old_manifest['dataset_sha256']
with original.open(newline='') as stream:
    original_sessions = [r['session_id'] for r in csv.DictReader(stream)]
old_groups = {k:{original_sessions[i] for i in v} for k,v in old_manifest['splits'].items()}
evidence['cross_benchmark_overlap'] = {
    part:{oldpart:sum(samples[i].session_id in groups for i in indices)
          for oldpart,groups in old_groups.items()}
    for part,indices in manifest['splits'].items()
}
evidence['overlap_note'] = 'Counts of new-split rows whose recording/import session belonged to each canonical historical partition. A new test split is not automatically held out from the old model.'
evidence['source_hashes'] = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in references if p.exists()}
assert sha(data) == before, 'Main dataset changed during review'
assert evidence['snapshot_sha256'] == before, 'Snapshot differs from current main data'
evidence['dataset_unchanged_during_review'] = True
(OUT/'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print('Evidence:', OUT/'evidence.json')
