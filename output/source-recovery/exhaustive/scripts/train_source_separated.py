"""Build an evidence-only source holdout and train one isolated static candidate.

Run only after both exhaustive recovery caches are complete. No input dataset,
deployed checkpoint, signer identity or linguistic label is changed.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ml.data_io import read_samples
from ml.dataset import save_manifest, validate_splits
from ml.evaluate import classification_metrics
from ml.model import load_checkpoint
from ml.preprocessing import STATIC_HAND_CANONICAL, canonicalize_static_features
from ml.train import TrainConfig, train

RECOVERY = ROOT / 'output/source-recovery'
CACHE = RECOVERY / 'exhaustive'
OUT = ROOT / 'output/source-separated-training/20261024-source-holdout'
ALLOCATION = {'archive4': 'train', 'manual': 'train', 'archive5': 'validation', 'archive3': 'test'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    torch.set_num_threads(2)
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    data = ROOT / 'ml/data/landmarks.csv'
    retained = ROOT / 'ml/models/sign_classifier_canonical.pth'
    assert digest(data) == protocol['dataset_sha256']
    assert digest(retained) == protocol['retained_checkpoint_sha256']
    assert protocol['allocation'] == ALLOCATION
    assert not (OUT / 'model.pth').exists(), 'Use a new experiment directory for another run'
    protected = {str(p.relative_to(ROOT)): digest(p) for p in
                 [data, *sorted((ROOT / 'ml/models').rglob('*.pth'))]}
    samples = read_samples(data)
    sources = defaultdict(set)
    image_counts = Counter()
    subject_folders = defaultdict(set)
    cache_metadata = {}
    for family in ('archive3', 'archive5'):
        state = json.loads((CACHE / f'{family}-progress.json').read_text(encoding='utf-8'))
        assert state['complete'] and state['dataset_sha256'] == protocol['dataset_sha256']
        with sqlite3.connect(f'file:{(CACHE / (family + ".sqlite3")).as_posix()}?mode=ro', uri=True) as db:
            meta = json.loads(db.execute("SELECT value FROM meta WHERE key='inputs'").fetchone()[0])
            assert meta['dataset_sha256'] == protocol['dataset_sha256']
            assert db.execute('SELECT count(*) FROM images').fetchone()[0] == state['total_images']
            cache_metadata[family] = {'inputs': meta, 'progress': state,
                                      'status_counts': dict(db.execute('SELECT status,count(*) FROM images GROUP BY status'))}
            for row, folder, count in db.execute('SELECT csv_row,subject_folder,count(*) FROM matches GROUP BY csv_row,subject_folder'):
                assert 2 <= row <= len(samples) + 1
                sources[row].add(family)
                image_counts[(row, family)] += count
                if folder:
                    subject_folders[row].add(folder)
    with (RECOVERY / 'session-provenance.csv').open(newline='', encoding='utf-8') as f:
        known = {r['session_id']: r for r in csv.DictReader(f)}
    for row, sample in enumerate(samples, 2):
        evidence = known[sample.session_id]
        assert evidence['dataset_sha256'] == protocol['dataset_sha256']
        if evidence['evidence'] == 'exact_import_session_hash':
            assert evidence['source_path'].endswith('archive (4).zip')
            sources[row].add('archive4')
        elif evidence['evidence'] == 'archived_capture_manifest_session_and_frame_count':
            sources[row].add('manual')

    # Unknown and multi-family rows are never silently assigned by session name.
    reasons = {}
    assigned = {}
    for row in range(2, len(samples) + 2):
        families = sources[row]
        if len(families) == 1:
            assigned[row] = next(iter(families))
        else:
            reasons[row] = 'unresolved_source' if not families else 'ambiguous_source_family'
    session_sources = defaultdict(set)
    for row, family in assigned.items():
        session_sources[samples[row - 2].session_id].add(ALLOCATION[family])
    for row in list(assigned):
        if len(session_sources[samples[row - 2].session_id]) > 1:
            reasons[row] = 'original_session_crosses_source_partitions'
            del assigned[row]
    fingerprints = defaultdict(list)
    for row, family in assigned.items():
        s = samples[row - 2]
        key = canonicalize_static_features(s.features, s.handedness, STATIC_HAND_CANONICAL).tobytes()
        fingerprints[key].append(row)
    for rows in fingerprints.values():
        labels = {samples[r - 2].label for r in rows}
        splits = {ALLOCATION[assigned[r]] for r in rows}
        if len(labels) > 1 or len(splits) > 1:
            for row in rows:
                reasons[row] = 'canonical_duplicate_conflicting_label_or_partition'
                del assigned[row]

    with (OUT / 'row-ledger.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['current_csv_row', 'label', 'session_id', 'matched_source_families',
                         'archive3_subject_folders', 'archive3_image_matches', 'archive5_image_matches',
                         'candidate_split', 'excluded_reason', 'signer_id', 'signer_status'])
        for row, s in enumerate(samples, 2):
            writer.writerow([row, s.label, s.session_id, ';'.join(sorted(sources[row])),
                             ';'.join(sorted(subject_folders[row])), image_counts[(row, 'archive3')],
                             image_counts[(row, 'archive5')], ALLOCATION[assigned[row]] if row in assigned else '',
                             reasons.get(row, ''), '', 'unknown'])
    with data.open(newline='', encoding='utf-8') as f, (OUT / 'dataset.csv').open('w', newline='', encoding='utf-8') as dest:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
        writer.writeheader()
        candidate_rows = []
        for row, values in enumerate(reader, 2):
            if row in assigned:
                writer.writerow(values)
                candidate_rows.append(row)
    subset = read_samples(OUT / 'dataset.csv')
    splits = {split: [i for i, row in enumerate(candidate_rows) if ALLOCATION[assigned[row]] == split]
              for split in ('train', 'validation', 'test')}
    assert len(subset) == len(candidate_rows)
    assert len({s.label for s in subset}) == 24
    validate_splits(subset, splits, static_hand_preprocessing=STATIC_HAND_CANONICAL)
    for split in splits:
        assert len({subset[i].label for i in splits[split]}) == 24
    model_path = OUT / 'model.pth'
    save_manifest(model_path.with_suffix('.splits.json'), OUT / 'dataset.csv', subset, splits,
                  42, static_hand_preprocessing=STATIC_HAND_CANONICAL)
    manifest = json.loads(model_path.with_suffix('.splits.json').read_text(encoding='utf-8'))
    manifest.update({'source_family_separated': True, 'signer_independence_verified': False,
                     'source_allocation': ALLOCATION, 'protocol_sha256': digest(OUT / 'protocol.json')})
    model_path.with_suffix('.splits.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    preparation = {'protocol': protocol, 'cache_metadata': cache_metadata,
                   'included_rows': len(subset), 'excluded_rows': len(samples) - len(subset),
                   'excluded_reasons': dict(Counter(reasons.values())),
                   'source_counts': dict(Counter(assigned.values())),
                   'split_counts': {split: len(ids) for split, ids in splits.items()},
                   'per_class_counts': manifest['counts'],
                   'handedness_counts': {split: dict(Counter(subset[i].handedness for i in ids)) for split, ids in splits.items()},
                   'protected_inputs': protected, 'dataset_sha256': digest(OUT / 'dataset.csv')}
    write_json(OUT / 'preparation.json', preparation)
    write_json(OUT / 'status.json', {'status': 'training', 'started': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({k: v for k, v in preparation.items() if k not in ('protected_inputs', 'cache_metadata')}, indent=2), flush=True)
    del samples
    config = TrainConfig(epochs=50, evaluate_test=False)
    assert json.loads(json.dumps(asdict(config))) == protocol['train_config']
    metrics = train(str(OUT / 'dataset.csv'), str(model_path), config)
    # The generic trainer conservatively labels explicit manifests as legacy.
    # Record the stronger, independently audited source-family scope here.
    metrics['benchmark'] = 'source_family_holdout_not_verified_signer_independent'
    write_json(model_path.with_suffix('.metrics.json'), metrics)
    write_json(OUT / 'status.json', {'status': 'evaluating_selected_checkpoint'})
    model, labels, checkpoint = load_checkpoint(model_path)
    evaluation = {}
    for split in ('validation', 'test'):
        ids = splits[split]
        targets, predictions, confidences = [], [], []
        with torch.inference_mode():
            for start in range(0, len(ids), 512):
                batch = ids[start:start + 512]
                x = torch.from_numpy(np.stack([canonicalize_static_features(subset[i].features, subset[i].handedness,
                                                                           model.static_hand_preprocessing) for i in batch]))
                confidence, pred = model(x).softmax(1).max(1)
                predictions.extend(pred.tolist())
                confidences.extend(confidence.tolist())
                targets.extend(labels.index(subset[i].label) for i in batch)
        evaluation[split] = classification_metrics(targets, predictions, labels)
        with (OUT / f'{split}-predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['current_csv_row', 'label', 'prediction', 'confidence', 'correct', 'session_id', 'handedness'])
            for i, pred, confidence in zip(ids, predictions, confidences):
                s = subset[i]
                writer.writerow([candidate_rows[i], s.label, labels[pred], confidence, int(s.label == labels[pred]), s.session_id, s.handedness])
    assert evaluation['validation']['confusion_matrix'] == metrics['validation']['confusion_matrix']
    unchanged = {p: digest(ROOT / p) == value for p, value in protected.items()}
    assert all(unchanged.values())
    results = {'best_epoch': checkpoint['training']['best_epoch'], 'evaluation': evaluation,
               'protected_inputs_unchanged': unchanged, 'checkpoint_sha256': digest(model_path)}
    write_json(OUT / 'results.json', results)
    lines = ['# Source-separated static model experiment', '',
             'One new 50-epoch candidate trained from scratch across all 24 static letters. '
             'Archive (4), including its archive.zip alias, and the local M captures are training-only; '
             'image-recovered archive (5) rows are validation-only; image-recovered archive (3) rows are test-only. '
             'The allocation was written before training and was not chosen using candidate test scores.', '',
             f'Included {len(subset):,} source-supported rows; excluded {len(reasons):,} unresolved, ambiguous or conflicting rows from this experiment only. '
             f'Train {len(splits["train"]):,}; validation {len(splits["validation"]):,}; test {len(splits["test"]):,}. '
             'Every partition contains all 24 letters. Source families and original sessions do not cross partitions, '
             'and the existing canonical duplicate/conflicting-label validator passes.', '',
             '| Metric | Validation: archive (5) | Test: archive (3) |', '|---|---:|---:|']
    for key, name in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1')]:
        lines.append(f'| {name} | {evaluation["validation"][key]:.2%} | {evaluation["test"][key]:.2%} |')
    lines += ['', f'Selected epoch **{results["best_epoch"]} of 50**, using validation macro F1, with loss as the tie-breaker. '
              'Test inference ran only after checkpoint selection. No current model weights were used to initialize this candidate, '
              'because the retained model has already trained on some rows of the new holdout.', '',
              '| Letter | Validation rows | Validation F1 | Test rows | Test precision | Test recall | Test F1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        v, t = evaluation['validation']['per_class'][label], evaluation['test']['per_class'][label]
        lines.append(f'| {label} | {v["support"]:,} | {v["f1"]:.2%} | {t["support"]:,} | {t["precision"]:.2%} | {t["recall"]:.2%} | {t["f1"]:.2%} |')
    lines += ['', '## Interpretation and limits', '',
              'This is a source-transfer diagnostic, not verified signer-independent evaluation. No real or fictional signer IDs were assigned. '
              'Image matches reproduce the label, handedness and all 63 stored coordinates within 1e-6; multiple matching images remain recorded. '
              'The candidate excludes rows that cannot be linked by that evidence. This creates a recoverability-selected subset, '
              'not a representative random sample of either complete archive. Only one validation source and one test source are available. '
              'Transformed image overlap and unknown shared people across different archives are not ruled out. '
              'Some holdout rows and source examples were examined during earlier diagnostics; this is a development benchmark, '
              'not an untouched external test.', '',
              'These scores cannot be directly compared with the historical 56.62% accuracy / 63.07% macro F1: '
              'the test rows and training distribution changed. The retained checkpoint is also contaminated for this new holdout, '
              'so it is not presented as an unseen-data baseline. This run establishes a reproducible source-separated baseline for future candidates; '
              'it does not establish that this candidate improves the deployed model.', '',
              'The original dataset, labels and all existing .pth checkpoints were verified unchanged. '
              'The candidate was saved separately and was not deployed. J/Z dynamic recognition was not retrained.', '',
              '[Protocol](protocol.json), [preparation and coverage](preparation.json), [all-row inclusion/exclusion ledger](row-ledger.csv), '
              '[results and confusion matrices](results.json), [training history](model.metrics.json), '
              '[validation predictions](validation-predictions.csv), [test predictions](test-predictions.csv).']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(OUT / 'status.json', {'status': 'complete', 'best_epoch': results['best_epoch'], 'evaluation': evaluation})
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
