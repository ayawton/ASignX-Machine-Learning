"""Train a static-pose MLP with fixed session-disjoint splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.data_io import read_samples
from ml.dataset import (
    LandmarkDataset,
    create_splits,
    dataset_digest,
    load_manifest,
    save_manifest,
)
from ml.evaluate import classification_metrics
from ml.model import SignMLP, load_checkpoint, save_checkpoint


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 50
    hidden_sizes: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    device: str = "cpu"


@torch.inference_mode()
def _evaluate(model, loader, labels, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = 0.0
    targets, predictions = [], []
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        logits = model(features)
        loss += float(criterion(logits, target).item())
        targets.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(targets, predictions, labels)
    metrics["loss"] = loss / len(targets)
    return metrics


def train(
    data_path: str = "ml/data/landmarks.csv",
    output: str = "ml/models/sign_classifier.pth",
    config: TrainConfig | None = None,
):
    config = config or TrainConfig()
    if config.batch_size < 1 or config.epochs < 1 or config.learning_rate <= 0 or not math.isfinite(config.learning_rate):
        raise ValueError("Batch size, epochs and learning rate must be positive and finite")
    output_path = Path(output)
    if output_path.suffix.lower() != ".pth":
        raise ValueError("Model output must have a .pth extension")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    samples = read_samples(data_path)
    labels = sorted({sample.label for sample in samples})
    manifest_path = output_path.with_suffix(".splits.json")
    if manifest_path.exists():
        manifest = load_manifest(manifest_path, data_path)
        splits = manifest["splits"]
        print(f"Reusing fixed split manifest: {manifest_path}")
    else:
        splits = create_splits(samples, config.validation_fraction, config.test_fraction, config.seed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_manifest(manifest_path, data_path, samples, splits, config.seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = {name: LandmarkDataset(samples, indices, labels) for name, indices in splits.items()}
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }
    model = SignMLP(len(labels), config.hidden_sizes, config.dropout).to(config.device)
    counts = torch.bincount(datasets["train"].targets, minlength=len(labels)).float()
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / (len(labels) * counts.clamp_min(1))).to(config.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = math.inf
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for features, targets in loaders["train"]:
            features, targets = features.to(config.device), targets.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; inspect data and learning rate")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(targets)
            count += len(targets)
        validation = _evaluate(model, loaders["validation"], labels, config.device)
        record = {"epoch": epoch, "train_loss": total / count, "validation": validation}
        history.append(record)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_checkpoint(
                output_path,
                model,
                labels,
                training={"config": asdict(config), "best_epoch": epoch},
                dataset_sha256=dataset_digest(data_path),
                splits=splits,
            )
    model, _, checkpoint = load_checkpoint(output_path, config.device)
    results = {
        "labels": labels,
        "best_epoch": checkpoint["training"]["best_epoch"],
        "validation": history[checkpoint["training"]["best_epoch"] - 1]["validation"],
        "test": _evaluate(model, loaders["test"], labels, config.device),
        "history": history,
    }
    output_path.with_suffix(".metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml/data/landmarks.csv")
    parser.add_argument("--output", default="ml/models/sign_classifier.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    values = vars(args).copy()
    data, output = values.pop("data"), values.pop("output")
    values["hidden_sizes"] = tuple(values["hidden_sizes"])
    train(data, output, TrainConfig(**values))


if __name__ == "__main__":
    main()
