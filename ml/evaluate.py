"""Classification metrics shared by static and sequence training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def classification_metrics(
    targets: list[int], predictions: list[int], labels: list[str]
) -> dict:
    if not targets or len(targets) != len(predictions):
        raise ValueError("Evaluation needs nonempty, equally sized targets and predictions")
    size = len(labels)
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (np.asarray(targets), np.asarray(predictions)), 1)
    true_positive = matrix.diagonal().astype(float)
    predicted = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_model(model, loader, labels: list[str], device: str = "cpu") -> dict:
    import torch
    from torch import nn

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            logits = model(features)
            loss += float(criterion(logits, target).item())
            targets.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(targets, predictions, labels)
    metrics["loss"] = loss / len(targets)
    return metrics


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from ml.data_io import read_samples
    from ml.dataset import LandmarkDataset, load_manifest
    from ml.model import load_checkpoint

    parser = argparse.ArgumentParser(description="Evaluate a static sign checkpoint")
    parser.add_argument("--data", default="ml/data/landmarks.csv")
    parser.add_argument("--model", default="ml/models/sign_classifier.pth")
    parser.add_argument("--manifest")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    manifest_path = Path(args.manifest) if args.manifest else Path(args.model).with_suffix(".splits.json")
    manifest = load_manifest(manifest_path, args.data)
    model, labels, checkpoint = load_checkpoint(args.model, args.device)
    if checkpoint.get("dataset_sha256") != manifest.get("dataset_sha256") or checkpoint.get("splits") != manifest.get("splits"):
        raise ValueError("Checkpoint and split manifest do not match")
    samples = read_samples(args.data)
    dataset = LandmarkDataset(samples, manifest["splits"][args.split], labels)
    metrics = evaluate_model(
        model, DataLoader(dataset, batch_size=args.batch_size), labels, args.device
    )
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
