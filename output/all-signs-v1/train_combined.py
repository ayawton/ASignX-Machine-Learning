"""Build a combined development dataset, preserving existing split assignments."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import shutil

from ml.sequence_io import read_sequence_dir
from ml.sequence_dataset import load_sequence_manifest, save_sequence_manifest
from ml.train_sequence import SequenceTrainConfig, train_sequence
from ml.predict_sequence import SequencePredictor


def main():
    root = Path('output/all-signs-v1')
    dataset = root / 'sequences'
    sources = [
        (Path('output/pilot-training/20261027T014944Z-reviewed-live-hello-3b654c/sequences'),
         Path('output/pilot-training/20261027T014944Z-reviewed-live-hello-3b654c/sequence_reviewed_live_hello.splits.json'), 'existing'),
        (Path('ml/data/sequences_words'), Path('ml/models/sequence_words_candidate.splits.json'), 'words'),
    ]
    assignments = {}
    provenance = []
    for folder, manifest_path, prefix in sources:
        samples = read_sequence_dir(folder)
        manifest = load_sequence_manifest(manifest_path, samples)
        paths = sorted(folder.rglob('*.npz'))
        for split, indices in manifest['splits'].items():
            for index in indices:
                source = paths[index]
                relative = Path(samples[index].label) / (prefix + '_' + source.name)
                dest = dataset / relative
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
                else:
                    shutil.copy2(source, dest)
                assignments[relative.as_posix()] = split
                provenance.append({'source': str(source), 'destination': str(relative), 'sha256': digest, 'split': split})
    paths = sorted(dataset.rglob('*.npz'))
    assert len(paths) == len(assignments), 'Unexpected files in combined dataset'
    samples = read_sequence_dir(dataset)
    assert {s.label for s in samples} == {'BACKGROUND', 'HELLO', 'J', 'Z', 'THANK_YOU', 'PLEASE', 'HELP'}
    assert {s.features.shape for s in samples} == {(30, 180)}
    splits = {split: [i for i, p in enumerate(paths) if assignments[p.relative_to(dataset).as_posix()] == split]
              for split in ('train', 'validation', 'test')}
    # Seed the candidate's manifest from preserved assignments before training.
    model = Path('ml/models/sequence_all_signs_v1.pth')
    assert not model.exists(), 'Use a new output for a later experiment'
    save_sequence_manifest(model.with_suffix('.splits.json'), samples, splits, 42)
    config = SequenceTrainConfig(use_motion_deltas=True, strict_splits=True, evaluate_test=False)
    preparation = {'counts': dict(Counter(s.label for s in samples)), 'sources': provenance,
                   'config': asdict(config), 'note': 'Old and word-candidate partitions preserved. Reused development data; not signer-independent. Test partition not evaluated.'}
    (root / 'preparation.json').write_text(json.dumps(preparation, indent=2), encoding='utf-8')
    results = train_sequence(str(dataset), str(model), config)
    predictor = SequencePredictor(str(model))
    assert set(predictor.labels) == {s.label for s in samples}
    assert predictor.input_size == 180
    print('READY', model, predictor.labels, flush=True)
    print('VALIDATION', json.dumps(results['validation']), flush=True)


if __name__ == '__main__':
    main()
