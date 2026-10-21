"""Static input contract parity, checkpoint compatibility, and training integration."""
import csv
import json
import subprocess
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ml.data_io import CSV_COLUMNS, Sample, read_samples
from ml.dataset import LandmarkDataset, validate_splits
from ml.export_onnx import export_static_checkpoint
from ml.model import SignMLP, load_checkpoint, save_checkpoint
from ml.predict import SignPredictor, SignRecognizer
from ml.preprocessing import (
    PREPROCESSING_VERSION, STATIC_HAND_CANONICAL, STATIC_HAND_LEGACY,
    canonicalize_static_features, normalize_landmarks,
)
from ml.train import TrainConfig, _compute_class_centroids, train


def hand(seed=0):
    return np.random.default_rng(seed).normal(size=(21, 3)).astype(np.float32)


def mirrored(points):
    result = points.copy()
    result[:, 0] *= -1
    return result


def test_mirror_parity_orientation_and_csv_features_unchanged():
    right = normalize_landmarks(hand())
    left = normalize_landmarks(mirrored(hand()))
    before = left.copy()
    np.testing.assert_array_equal(canonicalize_static_features(left, "Left"), right)
    np.testing.assert_array_equal(left, before)
    np.testing.assert_array_equal(canonicalize_static_features(right, "Right"), right)
    np.testing.assert_array_equal(canonicalize_static_features(left, None, STATIC_HAND_LEGACY), left)
    # Tilt/direction are intentionally retained, rather than palm alignment.
    rotated = hand() @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    assert not np.allclose(canonicalize_static_features(normalize_landmarks(rotated), "Right"), right)


@pytest.mark.parametrize("label", [None, "", "left", "Unknown"])
def test_canonical_requires_valid_handedness(label):
    with pytest.raises(ValueError, match="requires Left or Right"):
        canonicalize_static_features(normalize_landmarks(hand()), label)


def test_dataset_augmentation_and_centroids_use_canonical_coordinates():
    right = normalize_landmarks(hand())
    left = normalize_landmarks(mirrored(hand()))
    samples = [Sample("A", "r", "Right", right), Sample("A", "l", "Left", left)]
    dataset = LandmarkDataset(samples, [0, 1], ["A"], static_hand_preprocessing=STATIC_HAND_CANONICAL)
    np.testing.assert_array_equal(dataset.raw_features[0], dataset.raw_features[1])
    centroids = _compute_class_centroids(samples, [0, 1], ["A"], STATIC_HAND_CANONICAL)
    np.testing.assert_allclose(centroids["A"], right, atol=1e-6)
    augmented = LandmarkDataset(
        samples, [0, 1], ["A"], augment=True, seed=2,
        static_hand_preprocessing=STATIC_HAND_CANONICAL,
    )
    # Identical RNG states must augment corresponding left/right hands identically.
    a, _ = augmented[0]
    augmented.rng = np.random.default_rng(2)
    b, _ = augmented[1]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_mirrored_copies_cannot_leak_across_splits():
    samples = [
        Sample("A", "r", "Right", normalize_landmarks(hand())),
        Sample("A", "l", "Left", normalize_landmarks(mirrored(hand()))),
        Sample("A", "other", "Right", normalize_landmarks(hand(1))),
    ]
    splits = {"train": [0], "validation": [1], "test": [2]}
    validate_splits(samples, splits)
    with pytest.raises(ValueError, match="Identical features"):
        validate_splits(samples, splits, static_hand_preprocessing=STATIC_HAND_CANONICAL)


def test_predictor_live_registration_and_export_contract(tmp_path, monkeypatch):
    path = tmp_path / "model.pth"
    model = SignMLP(2, hidden_sizes=(8,), static_hand_preprocessing=STATIC_HAND_CANONICAL)
    save_checkpoint(path, model, ["A", "B"])
    predictor = SignPredictor(path, confidence_threshold=0)
    inputs = []
    hook = predictor.model.register_forward_pre_hook(lambda model, args: inputs.append(args[0].clone()))
    a = predictor.predict(hand(), "Right")
    b = predictor.predict(mirrored(hand()), "Left")
    assert a == b
    torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires Left or Right"):
        predictor.predict(hand())
    live_hand = SimpleNamespace(handedness="Left", to_numpy=lambda: mirrored(hand()))
    result = SignRecognizer(predictor, stable_frames=1).update([live_hand])
    assert result[0].prediction == a
    hook.remove()
    predictor.register_new_sign("C", [mirrored(hand())], handedness="Left")
    np.testing.assert_allclose(predictor.metadata["class_centroids"]["C"], normalize_landmarks(hand()), atol=1e-6)
    loaded, _, _ = load_checkpoint(path)
    assert loaded.static_hand_preprocessing == STATIC_HAND_CANONICAL
    # Exercise real metadata/weight export without depending on optional ONNX packages.
    def no_onnx(*args, **kwargs):
        raise ImportError("Test TorchScript fallback")
    monkeypatch.setattr(torch.onnx, "export", no_onnx)
    artifacts = export_static_checkpoint(path, tmp_path / "export.onnx")
    meta = json.loads(artifacts["meta"].read_text())
    assert meta["static_hand_preprocessing"] == STATIC_HAND_CANONICAL
    assert json.loads(artifacts["weights"].read_text())["metadata"] == meta


def test_old_checkpoint_keeps_original_inputs(tmp_path):
    path = tmp_path / "old.pth"
    save_checkpoint(path, SignMLP(2, hidden_sizes=(8,)), ["A", "B"])
    payload = torch.load(path, weights_only=True)
    del payload["model_config"]["static_hand_preprocessing"]
    torch.save(payload, path)
    predictor = SignPredictor(path, confidence_threshold=0)
    assert predictor.model.static_hand_preprocessing == STATIC_HAND_LEGACY
    captured = []
    predictor.model.register_forward_pre_hook(lambda model, args: captured.append(args[0].clone()))
    predictor.predict(mirrored(hand()), "Left")
    np.testing.assert_array_equal(captured[0].numpy()[0], normalize_landmarks(mirrored(hand())))
    predictor.predict(hand())  # No handedness needed for old checkpoints.


def test_small_training_and_cli_evaluation_preserve_csv(tmp_path):
    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for i in range(12):
            label = "A" if i < 6 else "B"
            side = "Left" if i % 2 else "Right"
            writer.writerow([label, f"session-{i}", side, PREPROCESSING_VERSION, *normalize_landmarks(hand(i))])
    original = path.read_bytes()
    output = tmp_path / "model.pth"
    train(str(path), str(output), TrainConfig(epochs=1, batch_size=4, hidden_sizes=(8,)))
    model, _, checkpoint = load_checkpoint(output)
    assert model.static_hand_preprocessing == STATIC_HAND_CANONICAL
    samples = read_samples(path)
    expected = _compute_class_centroids(samples, checkpoint["splits"]["train"], ["A", "B"], STATIC_HAND_CANONICAL)
    assert checkpoint["class_centroids"] == expected
    metrics_path = tmp_path / "evaluated.json"
    import sys
    subprocess.run([
        sys.executable, "-m", "ml.evaluate", "--data", str(path), "--model", str(output),
        "--output", str(metrics_path),
    ], check=True, capture_output=True, text=True)
    metrics = json.loads(metrics_path.read_text())
    trained_metrics = json.loads(output.with_suffix(".metrics.json").read_text())
    assert metrics["confusion_matrix"] == trained_metrics["test"]["confusion_matrix"]
    assert metrics["loss"] == pytest.approx(trained_metrics["test"]["loss"], abs=1e-6)
    assert path.read_bytes() == original


def test_browser_normalization_matches_python():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node 22+ is needed for the browser/Python parity check")
    major = int(subprocess.check_output([node, "--version"], text=True).strip().lstrip("v").split(".")[0])
    if major < 22:
        pytest.skip("Node 22+ is needed for TypeScript stripping")
    sdk = Path(__file__).resolve().parents[1] / "web_client" / "asignx_client_sdk.ts"
    # Load actual SDK preprocessing through Node's built-in TS stripper. The
    # rest of this SDK uses parameter properties requiring a separate TS build.
    script = """
import fs from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(input.sdk, 'utf8').split('export function computeKinematicFeatures91D')[0];
const js = stripTypeScriptTypes(source);
const sdk = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'));
const points = input.points.map(([x,y,z]) => ({x,y,z}));
const meta = {static_hand_preprocessing: 'left_x_reflection_v1'};
let rejected = false;
try {sdk.prepareStaticHandLandmarks(points, undefined, meta);} catch {rejected = true;}
console.log(JSON.stringify({
    canonical: Array.from(sdk.prepareStaticHandLandmarks(points, 'Left', meta)),
    legacy: Array.from(sdk.prepareStaticHandLandmarks(points, 'Left', {})),
    rejected,
}));
"""
    points = mirrored(hand())
    process = subprocess.run(
        [node, "--input-type=module", "-e", script],
        input=json.dumps({"sdk": str(sdk), "points": points.tolist()}),
        capture_output=True, text=True, check=True,
    )
    result = json.loads(process.stdout)
    np.testing.assert_allclose(result["canonical"], normalize_landmarks(hand()), atol=1e-6)
    np.testing.assert_allclose(result["legacy"], normalize_landmarks(points), atol=1e-6)
    assert result["rejected"]
