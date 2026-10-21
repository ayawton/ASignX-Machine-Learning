"""Paired static ablations; select on validation only and evaluate one winner.

Existing production checkpoints are never overwritten. Historical session splits
remain a diagnostic benchmark, not a new-signer generalization claim.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from ml.audit_f1 import audit
from ml.train import TrainConfig, train


def run(data, manifest, output, epochs=12):
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(manifest).read_text())
    if plan["dataset_sha256"] != hashlib.sha256(Path(data).read_bytes()).hexdigest():
        raise ValueError("Experiment requires a matching frozen manifest")
    variants = {
        "reference": {},
        "sqrt_weights": {"class_weighting": "sqrt"},
        "no_weights": {"class_weighting": "none"},
        "mild_augmentation": {"augmentation_scale": 0.5},
        "no_smoothing": {"label_smoothing": 0.0},
    }
    results = []
    for name, changes in variants.items():
        path = root / name / "model.pth"; path.parent.mkdir(parents=True, exist_ok=True)
        config = TrainConfig(epochs=epochs, batch_size=512, evaluate_test=False, **changes)
        config_data = asdict(config)
        config_file = path.parent / "experiment-config.json"
        if config_file.exists():
            if json.loads(config_file.read_text()) != json.loads(json.dumps(config_data)):
                raise ValueError(f"Refusing to reuse different experiment: {name}")
        else:
            config_file.write_text(json.dumps(config_data, indent=2))
        metric_path = path.with_suffix(".metrics.json")
        if metric_path.exists():
            metrics = json.loads(metric_path.read_text())
        else:
            shutil.copyfile(manifest, path.with_suffix(".splits.json"))
            started = time.perf_counter()
            metrics = train(data, str(path), config)
            print(f"Completed {name}: {time.perf_counter()-started:.1f}s", flush=True)
        results.append({"name": name, "path": str(path), "config": config_data,
                        "best_epoch": metrics["best_epoch"], "validation_macro_f1": metrics["validation"]["macro_f1"],
                        "validation_loss": metrics["validation"]["loss"]})
        (root / "progress.json").write_text(json.dumps(results, indent=2))
    winner = max(results, key=lambda r: (r["validation_macro_f1"], -r["validation_loss"]))
    # Test is consulted after variant/epoch selection, never used to choose settings.
    test = audit(data, winner["path"], root / "selected-test", "test")
    validation = audit(data, winner["path"], root / "selected-validation", "validation")
    summary = {"benchmark": "historical_session_split_not_signer_independent", "variants": results,
               "selected": winner["name"], "selection": "validation_macro_f1_then_loss",
               "selected_test": test["metrics"], "selected_validation": validation["metrics"],
               "production_models_unchanged": True}
    (root / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selected":winner["name"],"test_macro_f1":test["metrics"]["macro_f1"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml/data/landmarks.csv")
    parser.add_argument("--manifest", default="ml/models/sign_classifier_canonical.splits.json")
    parser.add_argument("--output", default="output/f1-review/experiments")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args(); torch.set_num_threads(2)
    run(args.data, args.manifest, args.output, args.epochs)
