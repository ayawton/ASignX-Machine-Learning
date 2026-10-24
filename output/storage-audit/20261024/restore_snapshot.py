"""Restore one removed dataset copy from its retained, verified capture ZIP.

List recipes: python output/storage-audit/20261024/restore_snapshot.py --list
Restore:      python output/storage-audit/20261024/restore_snapshot.py RELATIVE_PATH
Existing files are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / 'ml').is_dir() and (p / 'asignx').is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    recipes = json.loads((OUT / 'snapshot-restoration.json').read_text(encoding='utf-8'))
    if args.list or not args.path:
        print('\n'.join(recipes))
        return
    key = args.path.replace('\\', '/')
    if key not in recipes:
        parser.error('Only exact paths recorded in the restoration manifest can be restored')
    rec = recipes[key]
    target = ROOT / key
    assert target.resolve().is_relative_to(ROOT)
    if target.exists():
        raise FileExistsError(f'Will not overwrite: {target}')
    archive = ROOT / rec['zip']
    with archive.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == rec['archive_sha256'], 'Capture archive changed'
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.snapshot-restore-', dir=target.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            with zipfile.ZipFile(archive) as z, z.open(rec['member']) as source:
                shutil.copyfileobj(source, f)
        except BaseException:
            f.close(); temporary.unlink(missing_ok=True)
            raise
    try:
        with temporary.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == rec['sha256']
        assert temporary.stat().st_size == rec['bytes']
        # Windows rename fails if the destination exists, preserving the no-overwrite rule.
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Restored verified snapshot: {target}')


if __name__ == '__main__':
    main()
