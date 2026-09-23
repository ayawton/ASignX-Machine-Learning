"""Static-pose MLP and self-describing checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ml.preprocessing import NUM_FEATURES, PREPROCESSING_VERSION


class SignMLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes < 2 or not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("Need at least two classes and positive hidden sizes")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.config = {
            "num_classes": num_classes,
            "hidden_sizes": tuple(hidden_sizes),
            "dropout": dropout,
        }
        layers: list[nn.Module] = []
        previous = NUM_FEATURES
        for size in hidden_sizes:
            layers.extend((nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)))
            previous = size
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def save_checkpoint(
    path: str | Path,
    model: SignMLP,
    labels: list[str],
    *,
    training: dict[str, Any] | None = None,
    dataset_sha256: str | None = None,
    splits: dict[str, list[int]] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "architecture": "mlp",
            "preprocessing": PREPROCESSING_VERSION,
            "num_features": NUM_FEATURES,
            "model_config": model.config,
            "labels": list(labels),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training": training or {},
            "dataset_sha256": dataset_sha256,
            "splits": splits,
        },
        output,
    )


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[SignMLP, list[str], dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("architecture") != "mlp"
        or checkpoint.get("preprocessing") != PREPROCESSING_VERSION
        or checkpoint.get("num_features") != NUM_FEATURES
    ):
        raise ValueError("Unsupported checkpoint architecture or preprocessing")
    labels = checkpoint.get("labels")
    config = checkpoint.get("model_config", {})
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or len(set(labels)) != len(labels)
        or not all(isinstance(label, str) and label.strip() for label in labels)
        or config.get("num_classes") != len(labels)
    ):
        raise ValueError("Invalid checkpoint label mapping")
    model = SignMLP(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, labels, checkpoint
