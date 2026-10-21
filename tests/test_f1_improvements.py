"""Regression checks for benchmark honesty, F1 selection, rejection and timing."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ml.audit_f1 import rejection_metrics
from ml.benchmark import GROUP_COLUMNS, class_weights, load_group_plan, selection_score
from ml.calibrate_f1 import calibrate, load_calibration
from ml.model import SignMLP, save_checkpoint
from ml.predict import MotionGate, UnifiedSignRecognizer
from ml.predict_sequence import SequencePredictor
from ml.sequence_model import SignLSTM, save_sequence_checkpoint


def test_global_group_plan_keeps_signer_across_labels_and_rejects_leakage(tmp_path):
    samples = [SimpleNamespace(session_id=f"{p}-{label}", label=label) for p in range(3) for label in ("A", "B")]
    path = tmp_path / "groups.csv"
    rows = [{"session_id": s.session_id, "group_id": s.session_id.split('-')[0],
             "signer_id": s.session_id.split('-')[0], "split": ("train", "validation", "test")[int(s.session_id[0])]}
            for s in samples]
    def write():
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, GROUP_COLUMNS); writer.writeheader(); writer.writerows(rows)
    write()
    splits, meta = load_group_plan(path, samples)
    assert splits == {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}
    assert meta["global_identity_groups"]
    rows[1]["split"] = "test"; write()
    with pytest.raises(ValueError, match="leaks"):
        load_group_plan(path, samples)
    rows[1]["split"] = "train"; rows[0]["group_id"] = ""; write()
    with pytest.raises(ValueError, match="missing identity"):
        load_group_plan(path, samples)


def test_macro_f1_selection_weights_and_rejections():
    a, b = {"macro_f1": .8, "loss": .2}, {"macro_f1": .9, "loss": .3}
    assert selection_score(b, "macro_f1") > selection_score(a, "macro_f1")
    assert selection_score(a, "loss") > selection_score(b, "loss")
    counts = torch.tensor([1., 9.])
    assert class_weights(counts, "none") is None
    assert class_weights(counts, "sqrt")[0] < class_weights(counts, "inverse")[0]
    # Rejecting one of two correct known signs is a false negative, not 100% F1.
    score = rejection_metrics([0, 1], [0, 1], [.9, .6], .8, ["A", "B"])
    assert score["macro_f1"] == .5
    assert score["accepted_accuracy"] == 1
    # An explicitly labeled unknown accepted as A counts against A precision.
    unknown = rejection_metrics([0, 1, -1], [0, 1, 0], [.9, .9, .7], 0, ["A", "B"])
    rejected = rejection_metrics([0, 1, -1], [0, 1, 0], [.9, .9, .7], .8, ["A", "B"])
    assert rejected["macro_f1"] == 1
    assert unknown["macro_f1"] < rejected["macro_f1"]


def test_calibration_refuses_test_and_wrong_checkpoint(tmp_path):
    cache = tmp_path / "predictions.npz"
    model = tmp_path / "m.pth"; model.write_bytes(b"model")
    import hashlib
    def write(split):
        np.savez(cache, logits=[[3., 0.], [0., 3.]], targets=[0, 1], labels=["A", "B"],
                 split=split, model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), dataset_sha256="x")
    write("test")
    with pytest.raises(ValueError, match="validation"):
        calibrate(cache, tmp_path / "calibration.json")
    write("validation")
    output = tmp_path / "calibration.json"
    result = calibrate(cache, output)
    assert not result["unknown_rejection_validated"]
    assert load_calibration(output, model, ["A", "B"]) == 0
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="different checkpoint"):
        load_calibration(output, model, ["A", "B"])


def sequence_predictor(tmp_path):
    path = tmp_path / "s.pth"
    save_sequence_checkpoint(path, SignLSTM(2, hidden_size=8, num_layers=1), ["J", "Z"], sequence_length=30)
    return SequencePredictor(path, confidence_threshold=0)


def hand(t=0):
    p = np.zeros((21, 3), dtype=np.float32)
    p[:, 0] = np.linspace(0, .2, 21) + t * .2
    p[:, 1] = np.linspace(0, .1, 21)
    return p


def test_live_sequence_sampling_is_frame_rate_independent_and_rejects_gaps(tmp_path):
    predictor = sequence_predictor(tmp_path)
    windows = []
    for fps in (15, 30, 60):
        predictor.reset()
        for t in np.arange(0, 2, 1/fps):
            predictor.add_frame(hand(t), timestamp=float(t))
        assert len(predictor.buffer) == 30
        windows.append(predictor._buffer_features())
    np.testing.assert_allclose(windows[0], windows[1], atol=1e-5)
    np.testing.assert_allclose(windows[0], windows[2], atol=1e-5)
    predictor.reset()
    for t in range(30):
        predictor.add_frame(None, timestamp=t/15)
    assert predictor.predict() is None
    predictor.reset()
    for t in range(30):
        predictor.add_frame(None if 10 <= t <= 15 else hand(), timestamp=t/15)
    assert predictor.predict() is None
    predictor.add_frame(hand(), timestamp=10)
    assert len(predictor.buffer) == 1  # camera stall clears stale motion


def test_motion_uses_timestamps_and_both_hands(tmp_path):
    speeds = []
    for fps in (15, 30, 60):
        gate = MotionGate(velocity_threshold=.004)
        for t in np.arange(0, 1, 1/fps):
            speed, moving = gate.update(hand(t), timestamp=float(t))
        assert moving
        speeds.append(speed)
    np.testing.assert_allclose(speeds, speeds[0], rtol=1e-4)
    recognizer = UnifiedSignRecognizer(velocity_threshold=.004)
    def hands(t):
        return [SimpleNamespace(handedness="Left", to_numpy=lambda: hand()),
                SimpleNamespace(handedness="Right", to_numpy=lambda: hand(t))]
    recognizer.update(hands(0), timestamp=0)
    assert recognizer.update(hands(.1), timestamp=.1)["is_moving"]


def test_identity_training_skips_test_and_preserves_csv(tmp_path, monkeypatch):
    from ml.data_io import CSV_COLUMNS
    from ml.preprocessing import PREPROCESSING_VERSION, normalize_landmarks
    import ml.train as trainer
    data, plan, model = tmp_path / "data.csv", tmp_path / "groups.csv", tmp_path / "model.pth"
    rows = []
    with data.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(CSV_COLUMNS)
        for person in range(3):
            for label in ("A", "B"):
                session = f"p{person}-{label}"
                points = np.random.default_rng(len(rows)).normal(size=(21, 3)).astype(np.float32)
                writer.writerow([label, session, "Right", PREPROCESSING_VERSION, *normalize_landmarks(points)])
                rows.append(dict(session_id=session, group_id=f"person-{person}", signer_id=f"person-{person}",
                                 split=("train", "validation", "test")[person]))
    before = data.read_bytes()
    with plan.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, GROUP_COLUMNS); writer.writeheader(); writer.writerows(rows)
    calls = []
    original = trainer._evaluate
    def spy(*args):
        calls.append(args[1]); return original(*args)
    monkeypatch.setattr(trainer, "_evaluate", spy)
    metrics = trainer.train(str(data), str(model), trainer.TrainConfig(epochs=1, hidden_sizes=(8,),
                            group_plan=str(plan), evaluate_test=False))
    assert len(calls) == 1
    assert metrics["test"] is None
    assert metrics["benchmark"] == "global_identity_groups"
    assert data.read_bytes() == before
    assert json.loads(model.with_suffix(".splits.json").read_text())["global_identity_groups"]
    with pytest.raises(ValueError, match="explicit --group-plan"):
        trainer.train(str(data), str(model), trainer.TrainConfig(epochs=1))


def test_capture_preserves_real_identity_metadata(tmp_path):
    from ml.collect_sequence import _save_take
    from ml.video import LandmarkSequenceCapture
    from ml.sequence_io import load_sequence_file
    capture = LandmarkSequenceCapture("both")
    for i in range(3):
        capture.add_hands([SimpleNamespace(handedness="Right", to_numpy=lambda i=i: hand(i/15))])
    path = _save_take(capture, label="J", output_dir=tmp_path, sequence_length=30, target_fps=15,
                      min_frames=2, min_detection_ratio=.5, duration_seconds=.2,
                      provenance={"signer_id":"real-person-1", "recording_id":"visit-1", "split":"train"})
    meta = load_sequence_file(path).metadata
    assert meta["signer_id"] == "real-person-1"
    assert meta["split"] == "train"


def test_browser_motion_and_release_match_python():
    import shutil, subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for SDK execution")
    script = """
import fs from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
const source = stripTypeScriptTypes(fs.readFileSync(process.argv[1], 'utf8'));
const sdk = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const meta={labels:['A','B']};
const points=Array.from({length:21},(_,i)=>({x:i*.01,y:i*.005,z:0}));
const state=new sdk.ASignLiveStabilizer(meta,.8,2,2);
const strong=new Float32Array([5,0]), weak=new Float32Array([0,0]);
state.update('Right',points,strong,0);
const emitted=state.update('Right',points,strong,1/30);
const rejected=state.update('Right',points,weak,2/30);
state.update('Right',points,weak,3/30);
state.update('Right',points,strong,4/30);
const repeated=state.update('Right',points,strong,5/30);
const speeds=[];
for(const fps of [15,30,60]) {
 const s=new sdk.ASignLiveStabilizer(meta,.8,2,2,.004);
 let result;
 for(let i=0;i<fps;i++) result=s.update('Right', points.map(p=>({...p,x:p.x+.2*i/fps})),strong,i/fps);
 speeds.push(result.velocity);
}
console.log(JSON.stringify({emitted,rejected,repeated,speeds}));
"""
    path = Path(__file__).resolve().parents[1] / "web_client/asignx_client_sdk.ts"
    process = subprocess.run([node, "--input-type=module", "-e", script, str(path)], capture_output=True, text=True, check=True)
    result = json.loads(process.stdout)
    assert result["emitted"]["emittedLabel"] == "A"
    assert result["rejected"]["stableLabel"] is None
    assert result["repeated"]["emittedLabel"] == "A"
    np.testing.assert_allclose(result["speeds"], .2 / 30, rtol=1e-5)
