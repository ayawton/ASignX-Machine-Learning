"""Comprehensive tests for the rebuilt ASignX ML architecture and dataset extractor."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import torch

from asignx.models import HandDetectionResult, HandLandmarks, Landmark
from ml.data_io import CSV_COLUMNS
from ml.export_onnx import export_sequence_checkpoint, export_static_checkpoint
from ml.extract_dataset import (
    ExtractedCandidate,
    canonicalize_label,
    check_bone_plausibility,
    extract_and_clean_dataset,
    filter_statistical_outliers_mad,
    prune_near_duplicates,
)
from ml.model import SignMLP, SignResMLP, TorchKinematicEnricher, load_checkpoint, save_checkpoint
from ml.predict import MotionGate, SignPredictor, UnifiedSignRecognizer
from ml.preprocessing import (
    PREPROCESSING_VERSION,
    augment_landmarks_3d,
    augment_sequence_3d,
    compute_kinematic_features,
    normalize_holistic_landmarks,
    normalize_landmarks,
)
from ml.sequence_dataset import SequenceSample, save_sequence_file
from ml.sequence_model import SignLSTM, SignTCNAttention
from ml.train import TrainConfig, train
from ml.train_sequence import SequenceTrainConfig, train_sequence


def _make_plausible_hand(
    handedness: str = "Right", offset: float = 0.0, spread: float = 1.0
) -> HandLandmarks:
    """Create an anatomically plausible 21-point hand skeleton."""
    pts = [
        (0.5, 0.8, 0.0),  # 0: wrist
        # Thumb (1..4)
        (0.44, 0.75, -0.01),
        (0.39, 0.69, -0.02),
        (0.35, 0.63, -0.02),
        (0.31 - 0.02 * spread, 0.58, -0.03),
        # Index (5..8)
        (0.45, 0.65, 0.0),
        (0.44, 0.55, -0.01),
        (0.44, 0.47, -0.01),
        (0.44, 0.39 - 0.03 * spread, -0.02),
        # Middle (9..12)
        (0.50, 0.63, 0.0),
        (0.50, 0.52, -0.01),
        (0.50, 0.43, -0.01),
        (0.50, 0.35 - 0.03 * spread, -0.02),
        # Ring (13..16)
        (0.55, 0.65, 0.0),
        (0.55, 0.55, -0.01),
        (0.55, 0.47, -0.01),
        (0.55, 0.40 - 0.02 * spread, -0.02),
        # Pinky (17..20)
        (0.60, 0.68, 0.0),
        (0.61, 0.60, -0.01),
        (0.62, 0.53, -0.01),
        (0.63 + 0.02 * spread, 0.47, -0.01),
    ]
    landmarks = [
        Landmark(
            id=i,
            name=f"LM_{i}",
            x=float(np.clip(x + offset, 0.01, 0.99)),
            y=float(np.clip(y + offset * 0.5, 0.01, 0.99)),
            z=float(z),
        )
        for i, (x, y, z) in enumerate(pts)
    ]
    return HandLandmarks(landmarks=landmarks, handedness=handedness)


def test_kinematic_feature_enricher_and_augmentation() -> None:
    hand = _make_plausible_hand("Right")
    norm63 = normalize_landmarks(hand)
    assert norm63.shape == (63,)

    # Single-hand 63 -> 91 kinematic enrichment
    kin91 = compute_kinematic_features(norm63)
    assert kin91.shape == (91,)
    assert np.all(np.isfinite(kin91))

    # PyTorch TorchKinematicEnricher matches NumPy enricher
    enricher = TorchKinematicEnricher()
    torch_kin91 = enricher(torch.from_numpy(norm63).unsqueeze(0)).squeeze(0).numpy()
    np.testing.assert_allclose(kin91, torch_kin91, atol=1e-5)

    # Sequence 126 -> 182 kinematic enrichment
    seq126 = np.tile(np.concatenate([norm63, norm63]), (10, 1))
    seq182 = compute_kinematic_features(seq126)
    assert seq182.shape == (10, 182)

    # 3D geometric landmark and sequence augmentation
    rng = np.random.default_rng(123)
    aug63 = augment_landmarks_3d(norm63, rng=rng)
    assert aug63.shape == (63,)
    assert np.allclose(aug63[:3], 0.0, atol=1e-6)  # Wrist remains anchored at origin

    aug_seq = augment_sequence_3d(seq126, rng=rng)
    assert aug_seq.shape == (10, 126)

    # Holistic 6-keypoint upper-body + hand normalization (81 dims)
    upper_body = np.array(
        [
            [0.50, 0.30, 0.0],  # nose
            [0.50, 0.38, 0.0],  # mouth center
            [0.35, 0.60, 0.0],  # left shoulder
            [0.65, 0.60, 0.0],  # right shoulder
            [0.30, 0.85, 0.0],  # left elbow
            [0.70, 0.85, 0.0],  # right elbow
        ],
        dtype=np.float32,
    )
    holistic = normalize_holistic_landmarks(hand.to_numpy(), upper_body)
    assert holistic.shape == (81,)
    assert np.all(np.isfinite(holistic))


def test_resmlp_cosine_prototype_and_zero_retrain_addition(tmp_path: Path) -> None:
    model = SignResMLP(num_classes=2, hidden_sizes=(64, 32))
    x = torch.randn(4, 63)
    logits = model(x)
    assert logits.shape == (4, 2)

    # Zero-retrain registration of a new sign prototype
    support_samples = torch.randn(5, 63)
    model.add_sign_prototype(support_samples)
    assert model.config["num_classes"] == 3
    assert model(x).shape == (4, 3)

    # Checkpoint save/load with class_centroids and SignPredictor.register_new_sign
    ckpt_path = tmp_path / "static_resmlp.pth"
    centroids = {"A": [0.0] * 63, "B": [0.1] * 63, "C": [0.2] * 63}
    save_checkpoint(ckpt_path, model, ["A", "B", "C"], class_centroids=centroids)

    loaded_model, labels, raw = load_checkpoint(ckpt_path)
    assert labels == ["A", "B", "C"]
    assert "A" in raw["class_centroids"]

    predictor = SignPredictor(ckpt_path, confidence_threshold=0.0)
    hand_d = _make_plausible_hand("Right", spread=1.5).to_numpy()
    predictor.register_new_sign("D", [hand_d, hand_d, hand_d])
    assert "D" in predictor.labels
    assert predictor.model.config["num_classes"] == 4


def test_tcn_attention_sequence_model_and_training(tmp_path: Path) -> None:
    tcn_standalone = SignTCNAttention(num_classes=2, input_size=126, hidden_size=64, num_layers=2)
    seq = torch.randn(3, 30, 126)
    assert tcn_standalone(seq).shape == (3, 2)

    model = SignLSTM(
        num_classes=2,
        input_size=126,
        hidden_size=64,
        num_layers=2,
        backbone="tcn_attention",
    )
    logits = model(seq)
    assert logits.shape == (3, 2)

    # Zero-retrain dynamic sign addition
    model.add_sign_prototype(torch.randn(4, 30, 126))
    assert model.config["num_classes"] == 3
    assert model(seq).shape == (3, 3)

    # Full sequence training pipeline with TCN+Attention, AdamW, CosineAnnealingLR, and augmentation
    seq_dir = tmp_path / "sequences"
    out_model = tmp_path / "seq_tcn.pth"
    rng = np.random.default_rng(7)
    base_h = normalize_landmarks(_make_plausible_hand("Right"))
    base_frame = np.concatenate([np.zeros(63, dtype=np.float32), base_h])

    for sign_idx, sign in enumerate(["HELLO", "THANK_YOU"]):
        for sess in range(3):
            for take in range(2):
                clip = (
                    np.tile(base_frame, (16, 1))
                    + rng.normal(0, 0.02, (16, 126)).astype(np.float32)
                )
                clip[:, 63] += np.linspace(0, 0.2 * (sign_idx + 1), 16, dtype=np.float32)
                sample = SequenceSample(
                    label=sign,
                    session_id=f"{sign}_s{sess}",
                    handedness="Right",
                    features=clip,
                )
                save_sequence_file(seq_dir / sign / f"{sign}_s{sess}_t{take}.npz", sample)

    metrics = train_sequence(
        seq_dir,
        out_model,
        SequenceTrainConfig(
            epochs=3,
            batch_size=4,
            hidden_size=32,
            backbone="tcn_attention",
            augment=True,
        ),
    )
    assert out_model.exists()
    assert "test" in metrics

    # Export sequence checkpoint to Web bundle
    seq_export = export_sequence_checkpoint(out_model, tmp_path / "web" / "sequence.onnx")
    assert seq_export["meta"].exists()
    assert seq_export["weights"].exists()


def test_motion_gate_and_unified_sign_recognizer(tmp_path: Path) -> None:
    gate = MotionGate(velocity_threshold=0.04)
    still_hand = _make_plausible_hand("Right", offset=0.0)
    fast_hand = _make_plausible_hand("Right", offset=0.15)

    _, moving1 = gate.update(still_hand.to_numpy())
    _, moving2 = gate.update(still_hand.to_numpy())
    _, moving3 = gate.update(fast_hand.to_numpy())
    assert not moving1
    assert not moving2
    assert moving3

    model = SignMLP(num_classes=2, hidden_sizes=(32, 16))
    ckpt_path = tmp_path / "static.pth"
    save_checkpoint(ckpt_path, model, ["A", "B"])

    static_pred = SignPredictor(ckpt_path, confidence_threshold=0.0)
    unified = UnifiedSignRecognizer(
        static_predictor=static_pred,
        sequence_predictor=None,
        velocity_threshold=0.04,
        stable_frames=1,
    )

    res_still1 = unified.update([still_hand])
    res_still2 = unified.update([still_hand])
    assert not res_still2["is_moving"]
    assert res_still2["static"][0].stable_label is not None

    res_fast = unified.update([fast_hand])
    assert res_fast["is_moving"]
    assert res_fast["static"][0].stable_label is None


def test_flexible_static_dataset_extractor_and_4pass_cleaner(tmp_path: Path) -> None:
    assert canonicalize_label("a") == "A"
    assert canonicalize_label("sign_b") == "B"
    assert canonicalize_label("Class-Hello") == "HELLO"

    base_vec = normalize_landmarks(_make_plausible_hand("Right"))
    assert check_bone_plausibility(base_vec)

    collapsed = np.zeros(63, dtype=np.float32)
    assert not check_bone_plausibility(collapsed)

    # Statistical MAD outlier filter rejects extreme pose
    rng = np.random.default_rng(42)
    inliers = [
        ExtractedCandidate(
            label="A",
            hand="Right",
            features=normalize_landmarks(_make_plausible_hand("Right", spread=1.0 + 0.02 * i)),
            file_hash=f"h{i}",
            source_path=f"img_{i}.jpg",
        )
        for i in range(10)
    ]
    outlier = ExtractedCandidate(
        label="A",
        hand="Right",
        features=(base_vec + 2.5).astype(np.float32),
        file_hash="hout",
        source_path="outlier.jpg",
    )
    kept_mad, rejected_mad = filter_statistical_outliers_mad(inliers + [outlier], mad_z_threshold=3.5)
    assert rejected_mad == 1
    assert len(kept_mad) == 10

    # Near-duplicate epsilon-ball pruning
    dup_candidate = ExtractedCandidate(
        label="A",
        hand="Right",
        features=inliers[0].features.copy(),
        file_hash="hdup",
        source_path="dup.jpg",
    )
    pruned, near_dups, _ = prune_near_duplicates([inliers[0], dup_candidate], epsilon=0.015)
    assert near_dups == 1
    assert len(pruned) == 1

    # End-to-end extraction from a nested .zip archive and non-destructive append to landmarks.csv
    raw_dir = tmp_path / "raw_nested"
    (raw_dir / "train" / "a").mkdir(parents=True)
    (raw_dir / "test" / "Sign_B").mkdir(parents=True)

    dummy_img = np.full((64, 64, 3), 128, dtype=np.uint8)
    for idx in range(6):
        img_a = dummy_img.copy()
        img_a[0, 0, 0] = idx
        cv2.imwrite(str(raw_dir / "train" / "a" / f"img_{idx}.png"), img_a)

        img_b = dummy_img.copy()
        img_b[0, 0, 1] = idx + 10
        cv2.imwrite(str(raw_dir / "test" / "Sign_B" / f"img_{idx}.png"), img_b)

    copy_src = raw_dir / "train" / "a" / "img_0.png"
    cv2.imwrite(str(raw_dir / "train" / "a" / "img_0_copy.png"), cv2.imread(str(copy_src)))

    zip_path = tmp_path / "dataset_upload.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_path in raw_dir.rglob("*.png"):
            zf.write(file_path, file_path.relative_to(raw_dir))

    class _FakeDetector:
        def __init__(self) -> None:
            self.call_count = 0

        def process(self, bgr_frame: np.ndarray, draw: bool = False):
            self.call_count += 1
            spread = 0.2 * (self.call_count % 7) + 0.05 * self.call_count
            hand = _make_plausible_hand("Right", offset=0.005 * self.call_count, spread=spread)
            return HandDetectionResult(hand_detected=True, all_hands=[hand]), bgr_frame

        def close(self) -> None:
            pass

    csv_path = tmp_path / "data" / "landmarks.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerow(
            [
                "EXISTING",
                "s_init",
                "Right",
                PREPROCESSING_VERSION,
                *[f"{v:.8f}" for v in base_vec],
            ]
        )

    staging_dir = tmp_path / "staged"
    report = extract_and_clean_dataset(
        input_path=zip_path,
        output_csv=csv_path,
        staging_dir=staging_dir,
        duplicate_epsilon=0.001,
        min_sessions=3,
        detector=_FakeDetector(),
    )

    assert report.exact_duplicates_skipped == 1
    assert report.samples_appended == 12
    assert report.per_label_counts == {"A": 6, "B": 6}
    assert (staging_dir / "A").exists()
    assert (staging_dir / "B").exists()

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 13
    assert rows[0]["label"] == "EXISTING"
    a_sessions = {r["session_id"] for r in rows if r["label"] == "A"}
    # One unknown-origin import remains one group; random buckets are not independent.
    assert len(a_sessions) == 1
    assert any("fewer than 3 source groups" in note for note in report.warnings)

    # Verify closed-fist sign 'A' (where MCP knuckle 5 is the farthest point = 1.0) passes bone plausibility
    from ml.extract_dataset import infer_label_from_path

    fist_Vec = base_vec.copy().reshape(21, 3)
    fist_Vec[5] = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    fist_Vec[6] = np.array([0.0, -0.82, 0.0], dtype=np.float32)
    fist_Vec[7] = np.array([0.0, -0.65, 0.0], dtype=np.float32)
    fist_Vec[8] = np.array([0.0, -0.50, 0.0], dtype=np.float32)
    assert check_bone_plausibility(fist_Vec.reshape(-1))

    # Verify single-sign folder selection (e.g. user selects ".../A" directly) and flat-folder selection
    single_folder = tmp_path / "A"
    single_folder.mkdir()
    img1 = single_folder / "001.jpg"
    img2 = single_folder / "img_02.jpg"
    img1.write_bytes(b"x")
    img2.write_bytes(b"y")
    assert infer_label_from_path(img1, single_folder, root_label_hint="A")[0] == "A"
    assert infer_label_from_path(img2, single_folder, root_label_hint="A")[0] == "A"

    flat_folder = tmp_path / "flat_ds"
    flat_folder.mkdir()
    flat_a = flat_folder / "A_test.jpg"
    flat_b = flat_folder / "B_test.jpg"
    flat_a.write_bytes(b"a")
    flat_b.write_bytes(b"b")
    assert infer_label_from_path(flat_a, flat_folder, root_label_hint="flat_ds")[0] == "A"
    assert infer_label_from_path(flat_b, flat_folder, root_label_hint="flat_ds")[0] == "B"


def test_train_static_and_export_web_bundle(tmp_path: Path) -> None:
    csv_path = tmp_path / "landmarks.csv"
    ckpt_path = tmp_path / "sign_classifier.pth"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for sign_idx, label in enumerate(["A", "B"]):
            for sess in range(3):
                for sample_idx in range(4):
                    hand = _make_plausible_hand(
                        "Right",
                        spread=0.4 + sign_idx * 0.8 + 0.09 * sess + 0.02 * sample_idx,
                    )
                    vec = normalize_landmarks(hand)
                    writer.writerow(
                        [
                            label,
                            f"{label}_s{sess}",
                            "Right",
                            PREPROCESSING_VERSION,
                            *[f"{v:.8f}" for v in vec],
                        ]
                    )

    metrics = train(
        csv_path,
        ckpt_path,
        TrainConfig(epochs=4, batch_size=8, hidden_sizes=(32, 16), augment=True),
    )
    assert ckpt_path.exists()
    assert "test" in metrics

    onnx_target = tmp_path / "web" / "sign_classifier.onnx"
    artifacts = export_static_checkpoint(ckpt_path, onnx_target)
    assert artifacts["meta"].exists()
    assert artifacts["weights"].exists()

    meta = json.loads(artifacts["meta"].read_text(encoding="utf-8"))
    assert meta["labels"] == ["A", "B"]
    assert "A" in meta["class_centroids"]
    assert len(meta["class_centroids"]["A"]) == 63

    from ml.extractor_gui import inspect_existing_csv

    summary = inspect_existing_csv(csv_path)
    assert "24 samples across 2 sign classes" in summary
    assert "does not exist yet" in inspect_existing_csv(tmp_path / "missing.csv")

