import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ml.data_io import CSV_COLUMNS, read_samples
from ml.merge_landmarks import preview_merge, commit_merge, digest
from ml.preprocessing import PREPROCESSING_VERSION, normalize_landmarks


def row(seed=0, label='A', session='source1'):
    features = normalize_landmarks(np.random.default_rng(seed).normal(size=(21, 3)))
    return [label, session, 'Right', PREPROCESSING_VERSION, *map(str, features)]


def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    return path


def test_priority_backup_report_and_repeated_import(tmp_path):
    main = write(tmp_path / 'main.csv', [row()])
    # Preserve original precision, line endings and even a missing final newline.
    main.write_bytes(main.read_bytes().rstrip(b'\r\n'))
    original = main.read_bytes()
    incoming = write(tmp_path / 'new.csv', [row(), row(0, 'B', 'source2'),
                     row(1, session='other-device'), row(1, session='other-device')])
    untouched = incoming.read_bytes()
    preview = preview_merge(main, incoming)
    assert preview.counts == {'duplicate': 2, 'label conflict': 1, 'append': 1}
    assert main.read_bytes() == original
    backup, report = commit_merge(preview)
    assert backup.read_bytes() == original
    assert main.read_bytes().startswith(original)
    assert incoming.read_bytes() == untouched
    samples = read_samples(main)
    assert len(samples) == 2
    assert samples[1].session_id == 'other-device'
    audit = json.loads(report.read_text())
    assert audit['expected_output_sha256'] == digest(main.read_bytes())
    assert len(audit['decisions']) == 4
    assert not preview_merge(main, incoming).accepted
    assert not Path(str(main) + '.lock').exists()


def test_incoming_conflicts_are_order_independent(tmp_path):
    main = write(tmp_path / 'main.csv', [row()])
    rows = [row(1, 'B', 'b'), row(1, 'C', 'c'), row(2, 'B', 'source1'),
            row(3, 'B', 'mixed'), row(4, 'C', 'mixed'), row(5, 'D', 'good')]
    incoming = write(tmp_path / 'new.csv', rows)
    preview = preview_merge(main, incoming)
    assert preview.counts == {'label conflict': 2, 'session conflict': 3, 'append': 1}
    write(incoming, list(reversed(rows)))
    assert preview_merge(main, incoming).counts == preview.counts


@pytest.mark.parametrize('which', ['main', 'incoming'])
def test_changed_preview_is_refused(tmp_path, which):
    main = write(tmp_path / 'main.csv', [row()])
    incoming = write(tmp_path / 'new.csv', [row(1)])
    preview = preview_merge(main, incoming)
    path = main if which == 'main' else incoming
    path.write_bytes(path.read_bytes() + b'\n')
    original = main.read_bytes()
    with pytest.raises(ValueError, match='changed'):
        commit_merge(preview)
    assert main.read_bytes() == original
    assert not list(tmp_path.glob('*.backup-*'))


def test_lock_is_respected_and_preserved(tmp_path):
    main = write(tmp_path / 'main.csv', [row()])
    incoming = write(tmp_path / 'new.csv', [row(1)])
    preview = preview_merge(main, incoming)
    lock = Path(str(main) + '.lock')
    lock.write_text('another writer')
    with pytest.raises(ValueError, match='locked'):
        commit_merge(preview)
    assert lock.read_text() == 'another writer'
    assert len(read_samples(main)) == 1


def test_invalid_incoming_skipped_but_invalid_main_stops(tmp_path):
    bad = row(1)
    bad[4] = 'nan'
    main = write(tmp_path / 'main.csv', [row()])
    incoming = write(tmp_path / 'new.csv', [bad, row(2), row(3)[:-1]])
    preview = preview_merge(main, incoming)
    assert preview.counts == {'invalid': 2, 'append': 1}
    write(main, [bad])
    with pytest.raises(ValueError, match='Main CSV row 2'):
        preview_merge(main, incoming)


def test_schema_version_and_same_file_refused(tmp_path):
    main = write(tmp_path / 'main.csv', [row()])
    with pytest.raises(ValueError, match='different'):
        preview_merge(main, main)
    incoming = tmp_path / 'new.csv'
    incoming.write_text('label,x0\nA,0\n')
    with pytest.raises(ValueError, match='header'):
        preview_merge(main, incoming)
    bad = row(1)
    bad[3] = 'old'
    write(incoming, [bad])
    with pytest.raises(ValueError, match='preprocessing'):
        preview_merge(main, incoming)


def test_replace_failure_keeps_original_and_backup(tmp_path, monkeypatch):
    main = write(tmp_path / 'main.csv', [row()])
    incoming = write(tmp_path / 'new.csv', [row(1)])
    original = main.read_bytes()
    def fail(*args):
        raise OSError('disk failure')
    monkeypatch.setattr('ml.merge_landmarks.os.replace', fail)
    with pytest.raises(OSError, match='disk failure'):
        commit_merge(preview_merge(main, incoming))
    assert main.read_bytes() == original
    assert next(tmp_path.glob('*.backup-*')).read_bytes() == original
    assert not list(tmp_path.glob('.merge-*'))
    assert not Path(str(main) + '.lock').exists()


def test_empty_main_and_signed_zero_duplicate(tmp_path):
    main = write(tmp_path / 'main.csv', [])
    first = row()
    signed = first.copy()
    signed[4:7] = ['-0.0', '0', '-0']
    incoming = write(tmp_path / 'new.csv', [first, signed])
    preview = preview_merge(main, incoming)
    assert preview.counts == {'append': 1, 'duplicate': 1}
    commit_merge(preview)
    assert len(read_samples(main)) == 1


def test_gui_preview_confirm_and_merge(tmp_path, monkeypatch):
    import tkinter as tk
    from ml.csv_merger_gui import MergerApp

    main = write(tmp_path / 'main.csv', [row()])
    incoming = write(tmp_path / 'new.csv', [row(1)])
    root = tk.Tk()
    root.withdraw()
    try:
        app = MergerApp(root)
        app.main.set(str(main))
        app.incoming.set(str(incoming))
        assert str(app.merge_button['state']) == 'disabled'
        app.start_preview()
        # Wait for the worker through its queue, then exercise the UI completion path.
        event = app.events.get(timeout=5)
        app.events.put(event)
        app.poll()
        assert len(app.preview.accepted) == 1
        assert str(app.merge_button['state']) == 'normal'
        monkeypatch.setattr('ml.csv_merger_gui.messagebox.askyesno', lambda *args: True)
        app.start_merge()
        event = app.events.get(timeout=5)
        app.events.put(event)
        app.poll()
        assert len(read_samples(main)) == 2
        assert app.preview is None
        assert 'Merge completed' in app.status.get()
        assert str(app.merge_button['state']) == 'disabled'
    finally:
        root.destroy()
