"""Choose rejection on validation only; count rejected known signs as misses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.audit_f1 import rejection_metrics


def calibrate(prediction_file, output, negatives=None):
    with np.load(prediction_file, allow_pickle=False) as data:
        if str(data["split"]) != "validation":
            raise ValueError("Threshold fitting accepts validation predictions only, never test")
        logits, targets, labels = data["logits"], data["targets"], data["labels"].tolist()
        model_hash, dataset_hash = str(data["model_sha256"]), str(data["dataset_sha256"])
    negative_count = 0
    if negatives:
        # Negative examples must be independently labeled as outside the vocabulary.
        with np.load(negatives, allow_pickle=False) as data:
            if str(data["split"]) != "validation" or str(data["model_sha256"]) != model_hash or data["labels"].tolist() != labels:
                raise ValueError("Negative validation predictions must match model and labels")
            if not np.all(data["targets"] == -1):
                raise ValueError("Negative targets must all be -1")
            logits = np.concatenate([logits, data["logits"]])
            negative_count = len(data["targets"])
            targets = np.concatenate([targets, data["targets"]])
    probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    predictions, confidence = probs.argmax(1), probs.max(1)
    sweep = [rejection_metrics(targets, predictions, confidence, t, labels) for t in np.linspace(0, 1, 201)]
    best = max(sweep, key=lambda r: (r["macro_f1"], r["coverage"]))
    result = {"version": 1, "model_sha256": model_hash, "dataset_sha256": dataset_hash, "labels": labels,
              "fit_split": "validation", "threshold": best["threshold"], "validation": best,
              "negative_examples": negative_count, "unknown_rejection_validated": negative_count > 0,
              "warning": "Known-sign threshold optimization cannot validate unknown-gesture rejection without representative negatives",
              "sweep": sweep}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("threshold", "validation", "negative_examples")}))
    return result


def load_calibration(path, model_path, labels):
    result = json.loads(Path(path).read_text())
    if result.get("version") != 1 or result.get("fit_split") != "validation" or result.get("labels") != labels:
        raise ValueError("Invalid calibration contract")
    if result.get("model_sha256") != hashlib.sha256(Path(model_path).read_bytes()).hexdigest():
        raise ValueError("Calibration belongs to a different checkpoint")
    threshold = result["threshold"]
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid calibrated threshold")
    return threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-predictions")
    args = parser.parse_args(); calibrate(args.predictions, args.output, args.negative_predictions)
