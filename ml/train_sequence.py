"""Train and evaluate the dynamic-gesture LSTM."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.evaluate import classification_metrics
from ml.sequence_dataset import (
    SequenceDataset,
    create_sequence_splits,
    dataset_sequence_digest,
    load_sequence_manifest,
    read_sequence_dir,
    save_sequence_manifest,
)
from ml.sequence_model import SignLSTM, load_sequence_checkpoint, save_sequence_checkpoint


@dataclass
class SequenceTrainConfig:
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 50
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    pooling: str = "mean"
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    device: str = "cpu"


@torch.inference_mode()
def evaluate_sequence_model(
    model: SignLSTM, loader: DataLoader, labels: list[str], device: str = "cpu"
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        logits = model(features)
        total_loss += float(criterion(logits, target).item())
        targets.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(targets, predictions, labels)
    metrics["loss"] = total_loss / len(targets)
    return metrics


def train_sequence(
    data_dir: str = "ml/data/sequences",
    output: str = "ml/models/sequence_classifier.pth",
    config: SequenceTrainConfig | None = None,
) -> dict[str, Any]:
    config = config or SequenceTrainConfig()
    if (
        config.batch_size < 1
        or config.epochs < 1
        or not math.isfinite(config.learning_rate)
        or config.learning_rate <= 0
    ):
        raise ValueError("Batch size, epochs and learning rate must be positive and finite")
    output_path = Path(output)
    if output_path.suffix.lower() != ".pth":
        raise ValueError("Model output must have a .pth extension")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    samples = read_sequence_dir(data_dir)
    labels = sorted({sample.label for sample in samples})
    input_size = samples[0].features.shape[1]
    sequence_length = samples[0].features.shape[0]
    manifest_path = output_path.with_suffix(".splits.json")
    if manifest_path.exists():
        manifest = load_sequence_manifest(manifest_path, samples)
        splits = manifest["splits"]
        print(f"Reusing fixed split manifest: {manifest_path}")
    else:
        splits = create_sequence_splits(
            samples,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=config.seed,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_sequence_manifest(manifest_path, samples, splits, config.seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Class counts by split: {manifest['counts']}")

    datasets = {
        split: SequenceDataset(samples, indices, labels) for split, indices in splits.items()
    }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    model = SignLSTM(
        num_classes=len(labels),
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        pooling=config.pooling,
    ).to(config.device)
    train_targets = datasets["train"].targets
    counts = torch.bincount(train_targets, minlength=len(labels)).float()
    weights = counts.sum() / (len(labels) * counts.clamp_min(1.0))
    criterion = nn.CrossEntropyLoss(weight=weights.to(config.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = math.inf
    history: list[dict[str, Any]] = []
    dataset_hash = dataset_sequence_digest(samples)
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for features, targets in loaders["train"]:
            features, targets = features.to(config.device), targets.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; inspect data and learning rate")
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(targets)
            train_count += len(targets)
        validation = evaluate_sequence_model(model, loaders["validation"], labels, config.device)
        if not math.isfinite(validation["loss"]):
            raise ValueError("Non-finite validation loss; inspect data and learning rate")
        record = {
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "validation": validation,
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{config.epochs} train_loss={record['train_loss']:.4f} "
            f"validation_loss={validation['loss']:.4f} "
            f"validation_accuracy={validation['accuracy']:.3f}"
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_sequence_checkpoint(
                output_path,
                model,
                labels,
                sequence_length=sequence_length,
                training={"config": asdict(config), "best_epoch": epoch},
                dataset_sha256=dataset_hash,
                splits=splits,
            )

    best_model, _, checkpoint = load_sequence_checkpoint(output_path, config.device)
    test_metrics = evaluate_sequence_model(best_model, loaders["test"], labels, config.device)
    results = {
        "labels": labels,
        "best_epoch": checkpoint["training"]["best_epoch"],
        "validation": history[checkpoint["training"]["best_epoch"] - 1]["validation"],
        "test": test_metrics,
        "history": history,
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps({"test": test_metrics}, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="ml/data/sequences")
    parser.add_argument("--output", default="ml/models/sequence_classifier.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--pooling", choices=("mean", "last"), default="mean")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    values = vars(args).copy()
    data_dir = values.pop("data_dir")
    output = values.pop("output")
    train_sequence(data_dir, output, SequenceTrainConfig(**values))


if __name__ == "__main__":
    main()
