"""Camera-free checks for moving-sign preprocessing and sequence storage."""

from types import SimpleNamespace

import numpy as np

from ml.predict_sequence import SequencePredictor
import ml.prepare_videos as video_preparer
from ml.preprocessing import (
    DUAL_NUM_FEATURES,
    normalize_landmark_sequence,
)
from ml.sequence_dataset import SequenceSample, load_sequence_file, save_sequence_file
from ml.sequence_model import SignLSTM, save_sequence_checkpoint
from ml.video import extract_video_landmarks, resample_sequence


def _hand(x_offset: float, side: str):
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = x_offset + np.linspace(0.0, 0.2, 21)
    points[:, 1] = np.linspace(0.0, 0.1, 21)
    return SimpleNamespace(handedness=side, to_numpy=lambda: points.copy())


def test_clip_normalization_preserves_wrist_motion() -> None:
    clip = np.stack([_hand(0.1 + step * 0.05, "Right").to_numpy() for step in range(4)])
    features = normalize_landmark_sequence(clip)
    wrists = features[:, :3]
    assert np.allclose(wrists[0], 0.0)
    assert np.all(np.diff(wrists[:, 0]) > 0)


def test_video_extraction_keeps_timing_and_canonical_two_hand_slots() -> None:
    class Detector:
        def __init__(self):
            self.index = 0

        def process(self, frame, draw, timestamp_ms):
            step = self.index
            self.index += 1
            hands = [] if step == 1 else [_hand(0.2 + step * 0.02, "Right"), _hand(0.7, "Left")]
            return SimpleNamespace(all_hands=hands), frame

    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
    features, handedness, metadata = extract_video_landmarks(
        frames,
        detector=Detector(),
        hand_selection="both",
        target_length=4,
        return_metadata=True,
    )
    assert features.shape == (4, DUAL_NUM_FEATURES)
    assert np.isfinite(features).all()
    assert handedness in {"Left", "Right"}
    assert metadata["frames_with_hand"] == 3
    # A missed middle frame is interpolated, not deleted or turned into a zero timestep.
    assert not np.allclose(features[1], 0.0)


def test_sequence_storage_and_predictor_checkpoint(tmp_path) -> None:
    features = np.ones((6, 63), dtype=np.float32)
    sample = SequenceSample("WAVE", "wave-1", "Right", features, {"source": "x.mp4"})
    path = tmp_path / "wave.npz"
    save_sequence_file(path, sample)
    loaded = load_sequence_file(path)
    assert loaded.label == "WAVE"
    np.testing.assert_array_equal(loaded.features, features)

    model_path = tmp_path / "model.pth"
    save_sequence_checkpoint(
        model_path,
        SignLSTM(num_classes=2, hidden_size=8, num_layers=1, dropout=0.0),
        ["J", "Z"],
        sequence_length=6,
    )
    predictor = SequencePredictor(model_path, confidence_threshold=0.0)
    prediction = predictor.predict_sequence(features)
    assert prediction.raw_label in {"J", "Z"}


def test_resample_video_timeline_interpolates_endpoints() -> None:
    sequence = np.arange(3, dtype=np.float32)[:, None]
    result = resample_sequence(sequence, 5, interpolate=True)
    assert result.shape == (5, 1)
    assert result[0, 0] == 0
    assert result[-1, 0] == 2


def test_prepare_labeled_video_folders_is_reproducible(tmp_path, monkeypatch) -> None:
    source = tmp_path / "videos"
    output = tmp_path / "sequences"
    for label in ("J", "Z"):
        (source / label).mkdir(parents=True)
        for take in range(3):
            (source / label / f"take{take}.mp4").write_bytes(f"{label}-{take}".encode())

    class Detector:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_extract(path, **kwargs):
        length = kwargs["target_length"]
        width = 126 if kwargs["hand_selection"] == "both" else 63
        metadata = {
            "detection_ratio": 1.0,
            "target_fps": kwargs["target_fps"],
            "hand_selection": kwargs["hand_selection"],
        }
        return np.ones((length, width), dtype=np.float32), "Right", metadata

    monkeypatch.setattr(video_preparer, "HandDetector", Detector)
    monkeypatch.setattr(video_preparer, "extract_video_landmarks", fake_extract)
    first = video_preparer.prepare_video_dataset(source, output)
    second = video_preparer.prepare_video_dataset(source, output)
    assert len(first) == len(second) == 6
