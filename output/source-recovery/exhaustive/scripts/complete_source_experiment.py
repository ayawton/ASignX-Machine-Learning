"""Wait for authorized recovery workers, then audit, prepare and train once."""
from pathlib import Path
import json
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'output/source-recovery/exhaustive'
workers = [('archive3-' + folder, ['--subject-folder', folder]) for folder in 'ABCDE'] + [('archive5', ['--archive', 'archive5'])]
while True:
    all_done = True
    progress = {}
    live = [p.info['cmdline'] or [] for p in psutil.process_iter(['cmdline'])]
    for name, flags in workers:
        path = CACHE / f'{name}-progress.json'
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            state = {}
        progress[name] = {k: state.get(k) for k in ('processed_images', 'total_images', 'unique_current_rows_matched', 'complete')}
        if not state.get('complete'):
            all_done = False
            assert any('tmp/recover_sources_exhaustive.py' in cmd and all(flag in cmd for flag in flags) for cmd in live), f'{name}: worker stopped before completion; inspect its log'
    print(json.dumps(progress), flush=True)
    if all_done:
        break
    time.sleep(30)
for script in ('tmp/finalize_source_recovery.py', 'tmp/train_source_separated.py'):
    subprocess.run([sys.executable, '-u', script], cwd=ROOT, check=True)
