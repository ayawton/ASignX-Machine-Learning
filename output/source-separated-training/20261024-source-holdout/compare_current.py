"""Score the retained and candidate models on identical rows; audit prior exposure."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'ml/model.py').exists())
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ml.data_io import read_samples
from ml.model import load_checkpoint
from ml.evaluate import classification_metrics
from ml.preprocessing import canonicalize_static_features

OUT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def main():
    torch.set_num_threads(2)
    paths = {'current': ROOT / 'ml/models/sign_classifier_canonical.pth', 'candidate': OUT / 'model.pth'}
    before = {str(p.relative_to(ROOT)): sha(p) for p in [*paths.values(), ROOT / 'ml/data/landmarks.csv', OUT / 'dataset.csv']}
    models = {name: load_checkpoint(path) for name, path in paths.items()}
    current, labels, retained_meta = models['current']
    assert models['candidate'][1] == labels
    candidate_results = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    assert sha(paths['current']) == protocol['retained_checkpoint_sha256']
    assert sha(paths['candidate']) == candidate_results['checkpoint_sha256']
    original = ROOT / 'archive/data-corrections/remove-wrong-M-20261021T150522127835Z/landmarks.before.csv'
    assert sha(original) == retained_meta['dataset_sha256']
    print('Auditing the retained model original training/validation/test exposure...', flush=True)
    old = read_samples(original)
    membership = {i: split for split, ids in retained_meta['splits'].items() for i in ids}
    assert sorted(membership) == list(range(len(old)))
    exact = defaultdict(set)
    canonical = defaultdict(set)
    for i, s in enumerate(old):
        exact[(s.label, s.session_id, s.handedness, s.features.tobytes())].add(membership[i])
        f = canonicalize_static_features(s.features, s.handedness, current.static_hand_preprocessing)
        canonical[f.tobytes()].add(membership[i])
    del old
    samples = read_samples(OUT / 'dataset.csv')
    manifest = json.loads((OUT / 'model.splits.json').read_text(encoding='utf-8'))
    assert sha(OUT / 'dataset.csv') == manifest['dataset_sha256']
    with (OUT / 'row-ledger.csv').open(newline='', encoding='utf-8') as f:
        original_rows = [int(r['current_csv_row']) for r in csv.DictReader(f) if r['candidate_split']]
    assert len(samples) == len(original_rows)
    result = {'models': {name: {'checkpoint': str(p.relative_to(ROOT)), 'sha256': sha(p)} for name, p in paths.items()},
              'original_retained_dataset': str(original.relative_to(ROOT)),
              'original_retained_dataset_sha256': retained_meta['dataset_sha256'], 'splits': {}}
    for split in ('validation', 'test'):
        ids = manifest['splits'][split]
        truth = [labels.index(samples[i].label) for i in ids]
        exposure = []
        original_partition = []
        for i in ids:
            s = samples[i]
            original_partition.append(exact.get((s.label, s.session_id, s.handedness, s.features.tobytes()), set()))
            f = canonicalize_static_features(s.features, s.handedness, current.static_hand_preprocessing)
            exposure.append(canonical.get(f.tobytes(), set()))
        # A secondary comparison excludes both fitted and selection-exposed rows.
        # Source/person overlap and previous diagnostic test use still remain.
        shared_test = [j for j, parts in enumerate(original_partition)
                       if parts == {'test'} and not exposure[j] & {'train', 'validation'}]
        rec = {'samples': len(ids), 'retained_original_partition_counts': dict(Counter(';'.join(sorted(p)) or 'unmatched' for p in original_partition)),
               'retained_training_geometry_overlap': sum('train' in p for p in exposure),
               'retained_validation_geometry_overlap': sum('validation' in p for p in exposure),
               'shared_historical_test_rows': len(shared_test),
               'shared_historical_test_label_counts': dict(Counter(samples[ids[j]].label for j in shared_test)),
               'models': {}}
        predicted = {}
        print(f'Scoring {split}: {len(ids)} rows; shared historical-test subset {len(shared_test)}...', flush=True)
        for name, (model, model_labels, _) in models.items():
            pred, confidence = [], []
            with torch.inference_mode():
                for start in range(0, len(ids), 512):
                    x = torch.from_numpy(np.stack([canonicalize_static_features(samples[i].features, samples[i].handedness,
                                                                              model.static_hand_preprocessing) for i in ids[start:start + 512]]))
                    c, p = model(x).softmax(1).max(1)
                    pred.extend(p.tolist()); confidence.extend(c.tolist())
            metrics = classification_metrics(truth, pred, labels)
            if name == 'candidate':
                assert metrics == candidate_results['evaluation'][split]
            rec['models'][name] = {'full': metrics,
                                  'shared_historical_test': classification_metrics([truth[j] for j in shared_test], [pred[j] for j in shared_test], labels) if shared_test else None}
            predicted[name] = (pred, confidence)
        a, b = predicted['current'][0], predicted['candidate'][0]
        rec['candidate_fixes_current_errors'] = sum(p != t and q == t for p, q, t in zip(a, b, truth))
        rec['candidate_breaks_current_correct'] = sum(p == t and q != t for p, q, t in zip(a, b, truth))
        with (OUT / f'current-comparison-{split}-predictions.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['current_csv_row', 'label', 'session_id', 'handedness', 'retained_original_partitions',
                        'retained_train_geometry_seen', 'shared_historical_test_subset', 'current_prediction', 'candidate_prediction',
                        'current_confidence', 'candidate_confidence'])
            shared_set = set(shared_test)
            for j, i in enumerate(ids):
                s = samples[i]
                w.writerow([original_rows[i], s.label, s.session_id, s.handedness, ';'.join(sorted(original_partition[j])),
                            int('train' in exposure[j]), int(j in shared_set), labels[a[j]], labels[b[j]],
                            predicted['current'][1][j], predicted['candidate'][1][j]])
        result['splits'][split] = rec
    assert all(sha(ROOT / p) == value for p, value in before.items())
    result['protected_inputs_unchanged'] = True
    write_json(OUT / 'current-comparison.json', result)
    lines = ['# Current model versus source-separated candidate', '',
             'Both saved checkpoints were scored on exactly the same candidate validation/test rows, using each model\'s recorded handedness preprocessing. '
             'No retraining, threshold fitting or checkpoint selection was performed for this comparison.', '',
             '| Partition | Metric | Current deployed model | New candidate | Candidate change (pp) |',
             '|---|---|---:|---:|---:|']
    for split, rec in result['splits'].items():
        for metric in ('accuracy', 'macro_f1'):
            a = rec['models']['current']['full'][metric]; b = rec['models']['candidate']['full'][metric]
            lines.append(f'| {split.title()} ({rec["samples"]:,}) | {metric} | {a:.2%} | {b:.2%} | {(b-a)*100:+.2f} |')
    lines += ['', '## Exposure audit', '',
              'The full comparison favors the current model because it trained on part of these candidate holdouts. '
              'The audit uses its original dataset, verified against the dataset SHA-256 embedded in its checkpoint, '
              'and its actual saved split indices. It also checks exact canonical geometry against all original training rows.', '',
              '| Candidate partition | Current original train rows | Current original validation rows | Current original test rows | Training-geometry overlap |',
              '|---|---:|---:|---:|---:|']
    for split, rec in result['splits'].items():
        counts = rec['retained_original_partition_counts']
        lines.append(f'| {split} | {counts.get("train",0):,} | {counts.get("validation",0):,} | {counts.get("test",0):,} | {rec["retained_training_geometry_overlap"]:,} |')
    lines += ['', '## Shared historical-test subset', '',
              'This secondary view keeps only rows assigned to test by the current model\'s original manifest, '
              'excluding any canonical geometry found in its training or validation. These rows also belong to the candidate holdout. '
              'It removes direct fitted-row and validation-selection exposure, but the current model still trained on other examples '
              'from these source families and potentially the same people. The historical test has also been inspected in prior development. '
              'This is a diagnostic comparison, not proof of independent-person generalization.', '',
              '| Candidate partition / shared subset | Metric | Current | Candidate | Candidate change (pp) |',
              '|---|---|---:|---:|---:|---:|']
    for split, rec in result['splits'].items():
        for metric in ('accuracy', 'macro_f1'):
            a = rec['models']['current']['shared_historical_test'][metric]; b = rec['models']['candidate']['shared_historical_test'][metric]
            lines.append(f'| {split} ({rec["shared_historical_test_rows"]:,}; {len(rec["shared_historical_test_label_counts"])} labels) | {metric} | {a:.2%} | {b:.2%} | {(b-a)*100:+.2f} |')
    lines += ['', 'Macro F1 always uses the 24 model labels. Read shared-subset label coverage before comparing it with full-partition scores.', '',
              '## All-letter full test comparison', '',
              '| Letter | Test rows | Current F1 | Candidate F1 | Candidate change (pp) |', '|---|---:|---:|---:|---:|']
    rec = result['splits']['test']
    for label in labels:
        a = rec['models']['current']['full']['per_class'][label]; b = rec['models']['candidate']['full']['per_class'][label]
        lines.append(f'| {label} | {a["support"]:,} | {a["f1"]:.2%} | {b["f1"]:.2%} | {(b["f1"]-a["f1"])*100:+.2f} |')
    lines += ['', f'On the full test, the candidate fixes {rec["candidate_fixes_current_errors"]:,} current-model errors '
              f'and breaks {rec["candidate_breaks_current_correct"]:,} previously correct predictions. '
              'The source-transfer benchmark is nearly all Left-handed in test and does not establish balanced performance across both hands.', '',
              'Dataset and both checkpoints were verified unchanged. This comparison does not promote either checkpoint.', '',
              '[Machine-readable scores, confusion matrices and exposure counts](current-comparison.json), '
              '[combined validation predictions](current-comparison-validation-predictions.csv), '
              '[combined test predictions](current-comparison-test-predictions.csv), [comparison script](compare_current.py).']
    (OUT / 'current-comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({s: {'exposure': v['retained_original_partition_counts'],
                         'full': {n: {k: z['full'][k] for k in ('accuracy','macro_f1')} for n,z in v['models'].items()},
                         'shared_test': {n: {k: z['shared_historical_test'][k] for k in ('accuracy','macro_f1')} for n,z in v['models'].items()},
                         'shared_count': v['shared_historical_test_rows'], 'shared_labels': len(v['shared_historical_test_label_counts'])}
                      for s,v in result['splits'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
