"""Webcam sequence capture controls and saved training data."""

from types import SimpleNamespace

import numpy as np
import pytest

import ml.collect_sequence as collector
from ml.sequence_dataset import load_sequence_file, read_sequence_dir
from ml.train_sequence import SequenceTrainConfig, train_sequence
from ml.video import LandmarkSequenceCapture


def _hand(x: float, side: str = "Right"):
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[:, 0] = x + np.linspace(0.0, 0.2, 21)
    landmarks[:, 1] = np.linspace(0.0, 0.1, 21)
    return SimpleNamespace(handedness=side, to_numpy=lambda: landmarks.copy())


def test_complete_take_keeps_motion_and_rejects_poor_detection() -> None:
    capture = LandmarkSequenceCapture("both")
    capture.add_hands([_hand(0.1)])
    capture.add_hands([])
    capture.add_hands([_hand(0.3)])
    features, _, metadata = capture.finish(6, min_frames=3, min_detection_ratio=0.5)
    assert features.shape == (6, 126)
    assert metadata["frames_with_hand"] == 2
    assert np.isfinite(features).all()
    assert features[-1, 63] > features[0, 63]

    with pytest.raises(ValueError, match="Hand detected in only"):
        capture.finish(6, min_frames=3, min_detection_ratio=0.8)


def test_space_saves_two_unique_sessions_and_r_discards(tmp_path, monkeypatch) -> None:
    # First take is discarded; two subsequent takes are saved without reopening
    # the camera or typing the label again.
    keys = iter([32, -1, -1, ord("r"), 32, -1, -1, 32, 32, -1, -1, 32])
    moments = iter(np.arange(100, dtype=float) * 0.11)

    class Camera:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self):
            for _ in range(12):
                yield np.zeros((32, 32, 3), dtype=np.uint8)

    class Detector:
        def __init__(self, **kwargs):
            self.index = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def process(self, frame, draw):
            self.index += 1
            return SimpleNamespace(all_hands=[_hand(0.1 + 0.01 * self.index)]), frame

    monkeypatch.setattr(collector, "WebcamStream", Camera)
    monkeypatch.setattr(collector, "HandDetector", Detector)
    monkeypatch.setattr(collector.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(collector.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(collector.cv2, "waitKey", lambda _: next(keys))
    monkeypatch.setattr(collector.cv2, "destroyAllWindows", lambda: None)

    output = tmp_path / "sequences"
    saved = collector.collect_sequence(
        "J",
        output_dir=output,
        min_frames=2,
        target_fps=10,
        takes=2,
    )
    assert len(saved) == 2
    assert saved[0] != saved[1]
    assert not (tmp_path / "landmarks.csv").exists()
    samples = read_sequence_dir(output)
    assert {sample.label for sample in samples} == {"J"}
    assert len({sample.session_id for sample in samples}) == 2
    assert all(sample.features.shape == (30, 126) for sample in samples)
    assert all(sample.metadata["capture_method"] == "manual_webcam" for sample in samples)
    assert load_sequence_file(saved[0]).metadata["detection_ratio"] == 1.0


def test_captured_takes_can_train_sequence_model(tmp_path) -> None:
    data_dir = tmp_path / "sequences"
    for label, direction in (("J", 1), ("Z", -1)):
        for take in range(3):
            capture = LandmarkSequenceCapture("both")
            for step in range(12):
                capture.add_hands([_hand(0.3 + direction * step * (0.01 + take * 0.001))])
            collector._save_take(
                capture,
                label=label,
                output_dir=data_dir,
                sequence_length=30,
                target_fps=15.0,
                min_frames=10,
                min_detection_ratio=0.5,
                duration_seconds=1.0,
            )
    output = tmp_path / "sequence_classifier.pth"
    result = train_sequence(
        str(data_dir),
        str(output),
        SequenceTrainConfig(epochs=1, batch_size=2, hidden_size=8, num_layers=1, dropout=0.0),
    )
    assert output.exists()
    assert result["labels"] == ["J", "Z"]
    assert "accuracy" in result["test"]
