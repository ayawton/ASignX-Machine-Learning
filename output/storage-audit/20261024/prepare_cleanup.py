"""Verify the owner's approved first-pass cleanup and retain exact restore recipes."""
from pathlib import Path
import hashlib
import json
import stat
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / 'ml/data/landmarks.csv').exists())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def member_sha(z, name):
    with z.open(name) as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    plan = json.loads((OUT / 'cleanup-candidates.json').read_text(encoding='utf-8'))
    datasets = {r['path']: r for r in json.loads((OUT / 'datasets-and-backups.json').read_text())['datasets']}
    csv_archives = json.loads((OUT / 'archive-csv-hashes.json').read_text())
    restore = {}
    # Verify each frozen version inside a retained capture ZIP before removing a loose copy.
    for record in csv_archives:
        with zipfile.ZipFile(ROOT / record['zip']) as z:
            assert member_sha(z, record['member']) == record['sha256']
    targets, skipped, zip_checks = [], [], {}
    for entry in plan['recommended_first_pass_files']:
        rel = entry['path']
        path = ROOT / rel
        assert path.resolve().is_relative_to(ROOT) and path.absolute() == path.resolve()
        assert stat.S_ISREG(path.stat().st_mode) and not path.is_symlink()
        assert path.stat().st_size == entry['bytes'], f'Candidate changed since audit: {rel}'
        if entry['category'] in {'Duplicate current CSV', 'Duplicate interim backups'}:
            expected = datasets[rel]['sha256']
            assert sha(path) == expected
            archived = next(r for r in csv_archives if r['sha256'] == expected)
            restore[rel] = dict(archived, bytes=entry['bytes'], archive_sha256=sha(ROOT / archived['zip']))
        if entry['category'] == 'Distribution ZIP copies':
            mismatches = []
            checked = 0
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    counterpart = path.parent / info.filename
                    assert counterpart.resolve().is_relative_to(path.parent.resolve())
                    if not counterpart.is_file() or counterpart.stat().st_size != info.file_size:
                        mismatches.append(info.filename)
                    elif sha(counterpart) != member_sha(z, info.filename):
                        mismatches.append(info.filename)
                    checked += 1
            zip_checks[rel] = {'members_checked': checked, 'mismatches': mismatches}
            if mismatches:
                skipped.append(dict(entry, skip_reason='Archive contains content different from extracted package'))
                continue
        targets.append(dict(entry, sha256=sha(path)))
    (OUT / 'snapshot-restoration.json').write_text(json.dumps(restore, indent=2), encoding='utf-8')
    target_paths = {(ROOT / r['path']).resolve() for r in targets}
    # Protect the live source, all remaining ML data, models, tests and packaged tools.
    protected = {}
    roots = [ROOT / name for name in ('asignx', 'ml', 'web_client', 'tests', 'docs', 'dist',
                                     'output/source-recovery', 'output/signer-benchmark',
                                     'output/datasets', 'output/python314-migration/onnx/models',
                                     'output/extractor-reviewed/ASignX_DataExtractor',
                                     'output/csv-merger/ASignX_CSVMerger')]
    files = [p for directory in roots for p in directory.rglob('*') if p.is_file()]
    files += [p for p in ROOT.iterdir() if p.is_file()]
    for p in files:
        if p.resolve() not in target_paths:
            protected[p.relative_to(ROOT).as_posix()] = sha(p)
    execution = {'workspace': str(ROOT), 'delete': targets, 'skipped': skipped,
                 'protected_sha256': protected, 'distribution_zip_checks': zip_checks,
                 'restore_manifest': 'output/storage-audit/20261024/snapshot-restoration.json',
                 'scope': 'Only audited first-pass files. No Git/LFS cleanup or derived snapshot deletion.'}
    (OUT / 'execution-plan.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
    print(json.dumps({'verified_delete_files': len(targets), 'delete_bytes': sum(r['bytes'] for r in targets),
                      'protected_files': len(protected), 'restore_recipes': len(restore),
                      'skipped': skipped, 'distribution_zip_checks': zip_checks}, indent=2), flush=True)


if __name__ == '__main__':
    main()
