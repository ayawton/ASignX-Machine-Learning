"""Dynamic studio state transitions and standalone dependency contract."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ml.dynamic_recording import CaptureSettings, TakeRecorder, validate_output
from ml.sequence_io import SequenceSample, load_sequence_file, save_sequence_file


def hand(x):
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0, 0.2, 21) + x
    points[:, 1] = np.linspace(0, 0.1, 21)
    return SimpleNamespace(handedness="Right", to_numpy=lambda: points)


def test_countdown_save_and_discard_are_distinct_sessions(tmp_path):
    recorder = TakeRecorder(CaptureSettings(tmp_path, "J", min_frames=2, target_fps=10))
    recorder.start(100)
    recorder.update([hand(0)], 102)
    assert recorder.state == "Preparing"
    assert list(tmp_path.rglob("*.npz")) == []
    recorder.update([hand(0)], 103)
    recorder.update([hand(0.1)], 103.2)
    first = recorder.save(103.3)
    assert recorder.state == "Ready"
    sample = load_sequence_file(first)
    assert sample.label == "J"
    assert sample.features.shape == (30, 126)
    assert sample.features[-1, 63] > sample.features[0, 63]
    recorder.start(104)
    recorder.discard()
    recorder.update([hand(0)], 110)
    assert recorder.state == "Ready"
    assert list(tmp_path.rglob("*.npz")) == [first]
    recorder.start(111)
    recorder.update([hand(0)], 114)
    recorder.update([hand(0.2)], 114.2)
    second = recorder.save(114.3)
    assert load_sequence_file(second).session_id != sample.session_id


def test_auto_save_and_rejected_take_can_be_retried(tmp_path):
    settings = CaptureSettings(tmp_path, "Z", min_frames=2, countdown_seconds=0, max_seconds=1)
    recorder = TakeRecorder(settings)
    recorder.start(0)
    recorder.update([], 0)
    with pytest.raises(ValueError, match="Hand detected"):
        recorder.update([], 1)
    assert recorder.state == "Ready"
    assert not list(tmp_path.rglob("*.npz"))
    recorder.start(2)
    recorder.update([hand(0)], 2)
    path = recorder.update([hand(0.1)], 3)
    assert path.exists()
    assert recorder.state == "Ready"
    assert recorder.update([hand(0.3)], 4) is None
    assert len(list(tmp_path.rglob("*.npz"))) == 1


def test_output_rejects_equal_width_but_different_hand_contract(tmp_path):
    save_sequence_file(tmp_path / "right.npz", SequenceSample(
        "J", "take1", "Right", np.ones((30, 63)), {"hand_selection": "Right"}))
    validate_output(tmp_path, "Right", 30)
    with pytest.raises(ValueError, match="different hand"):
        validate_output(tmp_path, "Left", 30)


@pytest.mark.parametrize("changes", [
    {"label": " "}, {"target_fps": float("nan")}, {"max_seconds": float("inf")},
    {"min_detection_ratio": 0}, {"countdown_seconds": -1}, {"sequence_length": 1},
    {"target_fps": 1, "max_seconds": 1},
])
def test_bad_settings_fail_before_opening_camera(tmp_path, changes):
    with pytest.raises(ValueError):
        replace(CaptureSettings(tmp_path, "J"), **changes).validate()


def test_gui_dependencies_do_not_import_torch():
    # Enforce the real portable-app contract in a fresh interpreter.
    script = """
import sys
class NoTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Capture/import must work without PyTorch')
sys.meta_path.insert(0, NoTorch())
import ml.dynamic_studio_gui
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_progress_and_cancellation_keep_completed_files(tmp_path, monkeypatch):
    import ml.prepare_videos as preparer
    source = tmp_path / "videos"
    for label in ("J", "Z"):
        (source / label).mkdir(parents=True)
        for take in range(3):
            (source / label / f"{take}.mp4").write_bytes(f"{label}-{take}".encode())

    class Detector:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(preparer, "HandDetector", Detector)
    monkeypatch.setattr(preparer, "extract_video_landmarks", lambda *a, **k: (
        np.ones((30, 126)), "Right", {"detection_ratio": 1, "hand_selection": "both", "target_fps": 15.0}))
    progress = []
    output = tmp_path / "output"
    with pytest.raises(InterruptedError, match="Completed sequences were kept"):
        preparer.prepare_video_dataset(source, output,
            progress=lambda *args: progress.append(args),
            cancelled=lambda: len(list(output.rglob("*.npz"))) >= 1)
    assert progress[0][:2] == (0, 6)
    assert len(list(output.rglob("*.npz"))) == 1
    result = preparer.prepare_video_dataset(source, output, progress=lambda *args: progress.append(args))
    assert len(result) == 6
    assert progress[-1] == (6, 6, "Import complete")
