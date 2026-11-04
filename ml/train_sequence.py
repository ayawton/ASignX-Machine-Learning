"""Train and evaluate the dynamic-gesture classifier (LSTM or TCN-Attention) with 3D sequence augmentation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from uuid import uuid4
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
    SEQUENCE_MANIFEST_VERSION,
    create_sequence_splits,
    dataset_sequence_digest,
    load_sequence_manifest,
    read_sequence_dir,
    save_sequence_manifest,
)
from ml.sequence_model import SignLSTM, load_sequence_checkpoint, save_sequence_checkpoint
from ml.benchmark import class_weights, load_group_plan, selection_score


@dataclass
class SequenceTrainConfig:
    batch_size: int = 16
    learning_rate: float = 0.0015
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    epochs: int = 50
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    pooling: str = "mean"
    backbone: str = "tcn_attention"
    augment: bool = True
    use_kinematics: bool = True
    cosine_head: bool = True
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    device: str = "cpu"
    selection_metric: str = "macro_f1"
    class_weighting: str = "inverse"
    group_plan: str | None = None
    evaluate_test: bool = True
    strict_splits: bool = False
    early_stopping_patience: int = 10
    use_motion_deltas: bool = False


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
    """Train to staging, then back up and replace the stable output on success."""
    target = Path(output)
    if target.suffix.lower() != ".pth":
        raise ValueError("Model output must have a .pth extension")
    target.parent.mkdir(parents=True, exist_ok=True)
    suffixes = (".pth", ".metrics.json", ".splits.json")
    print(f"Training output: {target}")
    with tempfile.TemporaryDirectory(prefix=f".{target.stem}-training-", dir=target.parent) as directory:
        staged = Path(directory) / target.name
        previous_manifest = target.with_suffix(".splits.json")
        if previous_manifest.exists():
            shutil.copy2(previous_manifest, staged.with_suffix(".splits.json"))
        results = _train_sequence(data_dir, str(staged), config)
        # Verify the trained checkpoint before touching the current output.
        load_sequence_checkpoint(staged)
        existing = [target.with_suffix(suffix) for suffix in suffixes if target.with_suffix(suffix).exists()]
        if existing:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
            backup = target.parent / "backups" / target.stem / stamp
            backup.mkdir(parents=True, exist_ok=False)
            for path in existing:
                shutil.copy2(path, backup / path.name)
            print(f"Previous model and reports backed up to: {backup}")
        # Publish reports first and checkpoint last. Each replacement is atomic;
        # a running recognition process can continue using its loaded old model.
        for suffix in (".splits.json", ".metrics.json", ".pth"):
            os.replace(staged.with_suffix(suffix), target.with_suffix(suffix))
    print(f"Model ready: {target}; restart recognition to load the updated model")
    return results


def _train_sequence(
    data_dir: str = "ml/data/sequences",
    output: str = "ml/models/sequence_classifier.pth",
    config: SequenceTrainConfig | None = None,
) -> dict[str, Any]:
    config = config or SequenceTrainConfig()
    if config.selection_metric not in ("macro_f1", "loss") or not 0 <= config.label_smoothing < 1:
        raise ValueError("Invalid selection metric or label smoothing")
    class_weights(torch.ones(2), config.class_weighting)
    if (
        config.batch_size < 1
        or config.epochs < 1
        or config.early_stopping_patience < 0
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
    rates = {float(s.metadata["target_fps"]) for s in samples if s.metadata.get("target_fps") is not None}
    if len(rates) > 1 or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Sequence training requires one positive sampling FPS contract")
    sampling_fps = next(iter(rates), 15.0)
    manifest_path = output_path.with_suffix(".splits.json")
    splits = None
    if manifest_path.exists():
        try:
            manifest = load_sequence_manifest(manifest_path, samples)
            splits = manifest["splits"]
            print(f"Reusing fixed split manifest: {manifest_path}")
        except ValueError as err:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if config.strict_splits:
                raise ValueError("Existing benchmark changed; --strict-splits requires the original dataset or a new output path") from err
            if previous.get("version") != SEQUENCE_MANIFEST_VERSION:
                raise ValueError("Unsupported saved split manifest; inspect it before retraining") from err
            if previous.get("dataset_sha256") == dataset_sequence_digest(samples):
                raise ValueError("Saved split manifest is invalid for unchanged data; inspect it before retraining") from err
            if previous.get("global_identity_groups") or config.group_plan:
                raise ValueError("Identity-group benchmarks require an explicitly reviewed new split/output path") from err
            print("Dataset changed: rebuilding session-disjoint train/validation/test splits.")
            print("Evaluation split changed; scores are not directly comparable with the previous run.")

    if splits is None:
        splits = load_group_plan(config.group_plan, samples)[0] if config.group_plan else create_sequence_splits(
            samples,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=config.seed,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_sequence_manifest(manifest_path, samples, splits, config.seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.group_plan:
        splits, group_metadata = load_group_plan(config.group_plan, samples)
        if manifest.get("group_plan_sha256") not in (None, group_metadata["group_plan_sha256"]):
            raise ValueError("Identity plan changed; use a new benchmark output path")
        from ml.sequence_dataset import validate_sequence_splits
        validate_sequence_splits(samples, splits)
        save_sequence_manifest(manifest_path, samples, splits, config.seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(group_metadata)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elif manifest.get("global_identity_groups"):
        raise ValueError("This benchmark requires its explicit --group-plan on every training run")
    print(f"Class counts by split: {manifest['counts']}")

    datasets = {
        split: SequenceDataset(
            samples,
            indices,
            labels,
            augment=(split == "train" and config.augment),
            seed=config.seed,
        )
        for split, indices in splits.items()
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
        backbone=config.backbone,
        use_kinematics=config.use_kinematics,
        cosine_head=config.cosine_head,
        use_motion_deltas=config.use_motion_deltas,
    ).to(config.device)

    train_targets = datasets["train"].targets
    counts = torch.bincount(train_targets, minlength=len(labels)).float()
    weights = class_weights(counts, config.class_weighting)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(config.device) if weights is not None else None,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs),
        eta_min=min(1e-5, config.learning_rate * 0.1),
    )

    best_loss = math.inf
    best_score = -math.inf
    best_epoch = 0
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
        scheduler.step()
        validation = evaluate_sequence_model(model, loaders["validation"], labels, config.device)
        if not math.isfinite(validation["loss"]):
            raise ValueError("Non-finite validation loss; inspect data and learning rate")
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss / train_count,
            "validation": validation,
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{config.epochs} train_loss={record['train_loss']:.4f} "
            f"validation_loss={validation['loss']:.4f} "
            f"validation_accuracy={validation['accuracy']:.3f}"
        )
        score = selection_score(validation, config.selection_metric)
        if score > best_score or (score == best_score and validation["loss"] < best_loss):
            best_score = score
            best_loss = validation["loss"]
            best_epoch = epoch
            save_sequence_checkpoint(
                output_path,
                model,
                labels,
                sequence_length=sequence_length,
                training={"config": asdict(config), "best_epoch": epoch, "sampling_fps": sampling_fps,
                          "sampling_fps_assumed": not bool(rates)},
                dataset_sha256=dataset_hash,
                splits=splits,
            )

        if config.early_stopping_patience and epoch-best_epoch >= config.early_stopping_patience:
            print(f'Early stopping after {config.early_stopping_patience} epochs without validation improvement')
            break

    best_model, _, checkpoint = load_sequence_checkpoint(output_path, config.device)
    test_metrics = evaluate_sequence_model(best_model, loaders["test"], labels, config.device) if config.evaluate_test else None
    results = {
        "labels": labels,
        "best_epoch": checkpoint["training"]["best_epoch"],
        "validation": history[checkpoint["training"]["best_epoch"] - 1]["validation"],
        "test": test_metrics,
        "selection_metric": config.selection_metric,
        "benchmark": "global_identity_groups" if config.group_plan else "legacy_session_groups_not_signer_independent",
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
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Stop after this many epochs without validation improvement; 0 disables")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--pooling", choices=("mean", "last"), default="mean")
    parser.add_argument("--backbone", choices=("tcn_attention", "lstm"), default="tcn_attention")
    parser.add_argument('--motion-deltas', dest='use_motion_deltas', action='store_true',
                        help='Append signed temporal changes; experimental, creates a new model configuration')
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selection-metric", choices=("macro_f1", "loss"), default="macro_f1")
    parser.add_argument("--class-weighting", choices=("none", "sqrt", "inverse"), default="inverse")
    parser.add_argument("--group-plan")
    parser.add_argument("--strict-splits", action="store_true",
                        help="Refuse dataset changes instead of refreshing saved session splits")
    parser.add_argument("--no-test", dest="evaluate_test", action="store_false")
    args = parser.parse_args()
    values = vars(args).copy()
    data_dir = values.pop("data_dir")
    output = values.pop("output")
    train_sequence(data_dir, output, SequenceTrainConfig(**values))


if __name__ == "__main__":
    main()
