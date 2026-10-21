"""Publication, rollback and interruption guarantees for the manual studio."""
import csv
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from ml.data_io import CSV_COLUMNS, read_samples
from ml.manual_recording import RecordingBatch, RecordingWorkspace, process_batch
from ml.preprocessing import PREPROCESSING_VERSION, normalize_landmarks


def points(seed):
    rng = np.random.default_rng(seed)
    result = rng.uniform(0.2, 0.7, (21, 3)).astype(np.float32)
    result[:, 2] *= 0.1
    return result


class Detector:
    def process(self, frame, draw=False):
        value = int(frame[0, 0, 0])
        hand = SimpleNamespace(handedness='Right', to_numpy=lambda: points(value))
        return SimpleNamespace(all_hands=[hand]), frame

    def close(self):
        pass


@pytest.fixture
def workspace(tmp_path):
    value = RecordingWorkspace(tmp_path / 'recording')
    yield value
    value.close()


def complete(workspace, label='A', offset=0):
    batch = RecordingBatch(workspace, label, 'Right', 10)
    for session in range(3):
        for n in range(10):
            batch.add_frame(session, np.full((12, 12, 3), 1 + offset + session * 10 + n, dtype=np.uint8))
        batch.finish_session(session)
    return batch


def existing_csv(path):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        writer.writerow(['B', 'old-session', 'Right', PREPROCESSING_VERSION, *normalize_landmarks(points(99))])


def test_incomplete_batches_never_publish_and_are_cleaned(workspace, tmp_path):
    batch = RecordingBatch(workspace, 'A', 'Right', 10)
    batch.add_frame(0, np.zeros((12, 12, 3), dtype=np.uint8))
    target = tmp_path / 'landmarks.csv'
    with pytest.raises(ValueError, match='all three'):
        process_batch(batch, target, detector_factory=Detector)
    assert not target.exists()
    assert workspace.cleanup_abandoned() == 1
    assert not batch.path.exists()


def test_success_archives_frames_before_removing_working_batch(workspace, tmp_path):
    target = tmp_path / 'landmarks.csv'
    existing_csv(target)
    original = target.read_bytes()
    batch = complete(workspace)
    result = process_batch(batch, target, detector_factory=Detector)
    samples = read_samples(target)
    assert len(samples) == 31
    assert len({s.session_id for s in samples if s.label == 'A'}) == 3
    assert next(s for s in samples if s.label == 'B').session_id == 'old-session'
    assert not batch.path.exists()
    from pathlib import Path
    assert Path(result['csv_backup']).read_bytes() == original
    with zipfile.ZipFile(result['archive']) as archive:
        assert archive.testzip() is None
        assert len([name for name in archive.namelist() if name.endswith('.png')]) == 30
        assert 'manifest.json' in archive.namelist()


def test_failed_quality_check_preserves_csv_and_recordings(workspace, tmp_path):
    target = tmp_path / 'landmarks.csv'
    existing_csv(target)
    original = target.read_bytes()
    batch = complete(workspace)
    class Missing(Detector):
        def process(self, frame, draw=False):
            return SimpleNamespace(all_hands=[]), frame
    with pytest.raises(ValueError, match='Session 1'):
        process_batch(batch, target, detector_factory=Missing)
    assert target.read_bytes() == original
    assert len(list(batch.path.rglob('*.png'))) == 30


def test_backup_failure_never_changes_dataset(workspace, tmp_path, monkeypatch):
    target = tmp_path / 'landmarks.csv'
    existing_csv(target)
    original = target.read_bytes()
    batch = complete(workspace)
    monkeypatch.setattr(zipfile.ZipFile, 'testzip', lambda self: 'broken.png')
    with pytest.raises(OSError, match='verification'):
        process_batch(batch, target, detector_factory=Detector)
    assert target.read_bytes() == original
    assert batch.path.exists()


def test_retry_after_publication_does_not_duplicate_rows(workspace, tmp_path, monkeypatch):
    batch = complete(workspace)
    target = tmp_path / 'landmarks.csv'
    discard = batch.discard
    monkeypatch.setattr(batch, 'discard', lambda: (_ for _ in ()).throw(OSError('interrupted cleanup')))
    with pytest.raises(OSError, match='interrupted'):
        process_batch(batch, target, detector_factory=Detector)
    published = target.read_bytes()
    monkeypatch.setattr(batch, 'discard', discard)
    result = process_batch(batch, target, detector_factory=Detector)
    assert result['recovered']
    assert target.read_bytes() == published
    assert not batch.path.exists()


def test_replace_preserves_other_labels_and_retakes_reset_session(workspace, tmp_path):
    target = tmp_path / 'landmarks.csv'
    existing_csv(target)
    with target.open('a', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerow(['C', 'keep-session', 'Right', PREPROCESSING_VERSION, *normalize_landmarks(points(100))])
    batch = complete(workspace, 'B')
    batch.retake(1)
    assert not batch.data['sessions'][1]['complete']
    assert batch.data['sessions'][1]['frames'] == 0
    assert not batch.session_path(1).exists()
    for n in range(10):
        batch.add_frame(1, np.full((12, 12, 3), 11 + n, dtype=np.uint8))
    batch.finish_session(1)
    process_batch(batch, target, replace=True, detector_factory=Detector)
    assert len(read_samples(target)) == 31
    assert next(s for s in read_samples(target) if s.label == 'C').session_id == 'keep-session'
    assert all(s.session_id != 'old-session' for s in read_samples(target))


def test_cleanup_is_confined_and_second_gui_cannot_take_lock(workspace, tmp_path):
    outside = tmp_path / 'keep'
    outside.mkdir()
    with pytest.raises(ValueError, match='outside'):
        workspace.remove_batch(outside)
    assert outside.exists()
    with pytest.raises(RuntimeError, match='Another recording'):
        RecordingWorkspace(workspace.root)


def test_concurrent_dataset_change_is_not_overwritten(workspace, tmp_path):
    target = tmp_path / 'landmarks.csv'
    existing_csv(target)
    batch = complete(workspace)
    changed = None
    def edit_dataset(message):
        nonlocal changed
        if message.startswith('Extracted session 1'):
            changed = target.read_bytes() + b'\n'
            target.write_bytes(changed)
    with pytest.raises(ValueError, match='changed during processing'):
        process_batch(batch, target, detector_factory=Detector, progress=edit_dataset)
    assert target.read_bytes() == changed
    assert batch.path.exists()


def test_recorded_dataset_trains_current_canonical_model(workspace, tmp_path):
    from ml.train import TrainConfig, train
    target = tmp_path / 'landmarks.csv'
    process_batch(complete(workspace, 'A'), target, detector_factory=Detector)
    process_batch(complete(workspace, 'B', offset=30), target, detector_factory=Detector)
    before = target.read_bytes()
    result = train(str(target), str(tmp_path / 'model.pth'), TrainConfig(epochs=1, batch_size=8))
    assert result['labels'] == ['A', 'B']
    assert (tmp_path / 'model.pth').exists()
    assert target.read_bytes() == before


def test_gui_capture_requires_three_sessions_and_close_discards(tmp_path, monkeypatch):
    import tkinter as tk
    import ml.manual_training_gui as gui
    monkeypatch.setattr(gui, 'PROJECT', tmp_path)
    root = tk.Tk()
    root.withdraw()
    app = gui.ManualTrainingApp(root)
    try:
        assert str(app.process_button['state']) == 'disabled'
        app.label.set('A')
        app.samples.set(10)
        app.new_batch()
        batch_path = app.batch.path
        class Camera:
            def read(self):
                return True, np.zeros((12, 12, 3), dtype=np.uint8)
            def stop(self):
                pass
        app.camera = Camera()
        app.detector = Detector()
        clock = [100.0]
        monkeypatch.setattr(gui.time, 'monotonic', lambda: clock[0])
        root.update_idletasks()
        for session in range(3):
            app.session.set(str(session + 1))
            app.select_session()
            app.record()
            clock[0] += 4
            for _ in range(10):
                clock[0] += 0.2
                app.tick()
            assert app.batch.data['sessions'][session]['complete']
            assert str(app.process_button['state']) == ('normal' if session == 2 else 'disabled')
        monkeypatch.setattr(gui.messagebox, 'askyesno', lambda *args: True)
        app.close()
        assert not batch_path.exists()
        assert not (tmp_path / 'ml/data/landmarks.csv').exists()
    finally:
        try:
            if root.winfo_exists():
                app.close_camera()
                app.workspace.close()
                root.destroy()
        except tk.TclError:
            pass


def test_gui_training_uses_snapshot_and_versioned_model(tmp_path, workspace, monkeypatch):
    import os
    from pathlib import Path
    import time
    import tkinter as tk
    import ml.manual_training_gui as gui
    source = tmp_path / 'source.csv'
    process_batch(complete(workspace, 'A'), source, detector_factory=Detector)
    process_batch(complete(workspace, 'B', offset=30), source, detector_factory=Detector)
    original = source.read_bytes()
    project = str(Path(__file__).resolve().parent.parent)
    monkeypatch.setenv('PYTHONPATH', project + os.pathsep + os.environ.get('PYTHONPATH', ''))
    monkeypatch.setattr(gui, 'PROJECT', tmp_path / 'app')
    root = tk.Tk()
    root.withdraw()
    app = gui.ManualTrainingApp(root)
    try:
        app.csv.set(str(source))
        app.epochs.set(1)
        app.train()
        deadline = time.monotonic() + 45
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        assert not app.busy, 'Training subprocess did not finish'
        assert app.model, app.log.get('1.0', 'end')
        model = Path(app.model)
        assert model.exists()
        assert (model.parent / 'training.log').exists()
        assert (model.parent / 'landmarks.csv').read_bytes() == original
        assert source.read_bytes() == original
        assert app.training_progress.get() == 'Training complete'
        app.close()
    finally:
        try:
            if root.winfo_exists():
                app.stop_training()
                app.workspace.close()
                root.destroy()
        except tk.TclError:
            pass
