"""Train a static-pose ResMLP with 3D augmentation, AdamW, Cosine Annealing, and Label Smoothing."""

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
    validate_splits,
)
from ml.preprocessing import STATIC_HAND_CANONICAL, STATIC_HAND_LEGACY, canonicalize_static_features
from ml.evaluate import classification_metrics
from ml.model import SignMLP, load_checkpoint, save_checkpoint
from ml.benchmark import class_weights, load_group_plan, selection_score


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 0.002
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    epochs: int = 50
    hidden_sizes: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    augment: bool = True
    use_kinematics: bool = True
    residual: bool = True
    cosine_head: bool = True
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    device: str = "cpu"
    static_hand_preprocessing: str = STATIC_HAND_CANONICAL
    selection_metric: str = "macro_f1"
    class_weighting: str = "inverse"
    augmentation_scale: float = 1.0
    group_plan: str | None = None
    evaluate_test: bool = True
    initial_checkpoint: str | None = None


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


def _compute_class_centroids(
    samples, train_indices: list[int], labels: list[str],
    static_hand_preprocessing: str = STATIC_HAND_LEGACY,
) -> dict[str, list[float]]:
    """Compute per-class median normalized landmark vector (63-d) for ASignLearn coaching."""
    by_label: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    for idx in train_indices:
        s = samples[idx]
        by_label[s.label].append(canonicalize_static_features(
            s.features, s.handedness, static_hand_preprocessing,
        ))
    centroids: dict[str, list[float]] = {}
    for label, feats in by_label.items():
        if feats:
            med = np.median(np.stack(feats), axis=0).astype(np.float32)
            centroids[label] = [round(float(x), 6) for x in med.tolist()]
    return centroids


def train(
    data_path: str = "ml/data/landmarks.csv",
    output: str = "ml/models/sign_classifier_canonical.pth",
    config: TrainConfig | None = None,
):
    config = config or TrainConfig()
    if config.selection_metric not in ("macro_f1", "loss") or not 0 <= config.label_smoothing < 1:
        raise ValueError("Invalid selection metric or label smoothing")
    if not math.isfinite(config.augmentation_scale) or config.augmentation_scale < 0:
        raise ValueError("augmentation_scale must be nonnegative and finite")
    class_weights(torch.ones(2), config.class_weighting)
    if (
        config.batch_size < 1
        or config.epochs < 1
        or config.learning_rate <= 0
        or not math.isfinite(config.learning_rate)
    ):
        raise ValueError("Batch size, epochs and learning rate must be positive and finite")
    output_path = Path(output)
    if output_path.suffix.lower() != ".pth":
        raise ValueError("Model output must have a .pth extension")
    if config.initial_checkpoint and Path(config.initial_checkpoint).resolve() == output_path.resolve():
        raise ValueError("Fine-tuning requires a separate output path; preserve the initial checkpoint")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    samples = read_samples(data_path)
    # Fail before writing any artifacts when handedness/contract is unusable.
    for sample in samples:
        canonicalize_static_features(sample.features, sample.handedness, config.static_hand_preprocessing)
    labels = sorted({sample.label for sample in samples})
    manifest_path = output_path.with_suffix(".splits.json")
    splits = None
    if manifest_path.exists():
        try:
            manifest = load_manifest(manifest_path, data_path)
            if manifest.get("static_hand_preprocessing", config.static_hand_preprocessing) != config.static_hand_preprocessing:
                raise ValueError("Static preprocessing contract changed; use a new model/manifest path")
            splits = manifest["splits"]
            print(f"Reusing fixed split manifest: {manifest_path}")
        except ValueError as err:
            raise ValueError("Existing benchmark changed; use a new output path and an explicitly reviewed split") from err

    if splits is None:
        splits = (load_group_plan(config.group_plan, samples)[0] if config.group_plan else
                  create_splits(samples, config.validation_fraction, config.test_fraction, config.seed))
        validate_splits(samples, splits, static_hand_preprocessing=config.static_hand_preprocessing)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_manifest(manifest_path, data_path, samples, splits, config.seed,
                      static_hand_preprocessing=config.static_hand_preprocessing)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if config.group_plan:
        splits, group_metadata = load_group_plan(config.group_plan, samples)
        if manifest.get("group_plan_sha256") not in (None, group_metadata["group_plan_sha256"]):
            raise ValueError("Identity plan changed; use a new benchmark output path")
        validate_splits(samples, splits, static_hand_preprocessing=config.static_hand_preprocessing)
        save_manifest(manifest_path, data_path, samples, splits, config.seed,
                      static_hand_preprocessing=config.static_hand_preprocessing)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(group_metadata)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elif manifest.get("global_identity_groups"):
        raise ValueError("This benchmark requires its explicit --group-plan on every training run")

    validate_splits(samples, splits, static_hand_preprocessing=config.static_hand_preprocessing)
    datasets = {
        name: LandmarkDataset(
            samples,
            indices,
            labels,
            augment=(name == "train" and config.augment),
            seed=config.seed,
            static_hand_preprocessing=config.static_hand_preprocessing,
            augmentation_scale=config.augmentation_scale,
        )
        for name, indices in splits.items()
    }
    class_centroids = _compute_class_centroids(
        samples, splits["train"], labels, config.static_hand_preprocessing,
    )
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
    model = SignMLP(
        len(labels),
        config.hidden_sizes,
        config.dropout,
        use_kinematics=config.use_kinematics,
        residual=config.residual,
        cosine_head=config.cosine_head,
        static_hand_preprocessing=config.static_hand_preprocessing,
    ).to(config.device)
    initialization = {}
    if config.initial_checkpoint:
        initial_model, initial_labels, _ = load_checkpoint(config.initial_checkpoint, config.device)
        if initial_labels != labels or initial_model.config != model.config:
            raise ValueError("Initial checkpoint labels, architecture and preprocessing must match the training configuration")
        model.load_state_dict(initial_model.state_dict())
        initialization = {"initial_checkpoint": str(config.initial_checkpoint),
                          "initial_checkpoint_sha256": dataset_digest(config.initial_checkpoint),
                          "mode": "whole_model_fine_tuning"}
        del initial_model

    counts = torch.bincount(datasets["train"].targets, minlength=len(labels)).float()
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

    best_score = -math.inf
    best_loss = math.inf
    history = []
    initial_validation = None
    if config.initial_checkpoint:
        initial_validation = _evaluate(model, loaders["validation"], labels, config.device)
        best_score = selection_score(initial_validation, config.selection_metric)
        best_loss = initial_validation['loss']
        save_checkpoint(output_path, model, labels,
                        training={"config": asdict(config), "best_epoch": 0, **initialization},
                        dataset_sha256=dataset_digest(data_path), splits=splits,
                        class_centroids=class_centroids)
        print(f"Initial checkpoint validation macro F1={initial_validation['macro_f1']:.5f}; epoch 0 remains eligible")
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
        scheduler.step()
        validation = _evaluate(model, loaders["validation"], labels, config.device)
        if not math.isfinite(validation["loss"]):
            raise ValueError("Non-finite validation loss; inspect data and learning rate")
        current_lr = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": total / count,
            "validation": validation,
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{config.epochs} lr={current_lr:.5f} train_loss={total / count:.4f} "
            f"val_loss={validation['loss']:.4f} val_acc={validation['accuracy']:.3f} "
            f"val_f1={validation['macro_f1']:.3f}"
        )
        score = selection_score(validation, config.selection_metric)
        if score > best_score or (score == best_score and validation["loss"] < best_loss):
            best_score = score
            best_loss = validation["loss"]
            save_checkpoint(
                output_path,
                model,
                labels,
                training={"config": asdict(config), "best_epoch": epoch, **initialization},
                dataset_sha256=dataset_digest(data_path),
                splits=splits,
                class_centroids=class_centroids,
            )
    model, _, checkpoint = load_checkpoint(output_path, config.device)
    results = {
        "labels": labels,
        "best_epoch": checkpoint["training"]["best_epoch"],
        "validation": (initial_validation if checkpoint["training"]["best_epoch"] == 0 else
                       history[checkpoint["training"]["best_epoch"] - 1]["validation"]),
        "initial_validation": initial_validation,
        "test": _evaluate(model, loaders["test"], labels, config.device) if config.evaluate_test else None,
        "selection_metric": config.selection_metric,
        "benchmark": "global_identity_groups" if config.group_plan else "legacy_session_groups_not_signer_independent",
        "history": history,
    }
    output_path.with_suffix(".metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml/data/landmarks.csv")
    parser.add_argument("--output", default="ml/models/sign_classifier_canonical.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selection-metric", choices=("macro_f1", "loss"), default="macro_f1")
    parser.add_argument("--class-weighting", choices=("none", "sqrt", "inverse"), default="inverse")
    parser.add_argument("--augmentation-scale", type=float, default=1.0)
    parser.add_argument("--group-plan")
    parser.add_argument("--no-test", dest="evaluate_test", action="store_false")
    parser.add_argument("--initial-checkpoint", help="Fine-tune matching weights; requires a separate output path")
    parser.add_argument(
        "--static-hand-preprocessing", choices=(STATIC_HAND_CANONICAL, STATIC_HAND_LEGACY),
        default=STATIC_HAND_CANONICAL,
        help="Mirror Left landmarks into Right coordinates (default); none uses legacy inputs",
    )
    args = parser.parse_args()
    values = vars(args).copy()
    data, output = values.pop("data"), values.pop("output")
    values["hidden_sizes"] = tuple(values["hidden_sizes"])
    train(data, output, TrainConfig(**values))


if __name__ == "__main__":
    main()
