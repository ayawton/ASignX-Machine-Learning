"""LSTM classifier and self-describing sequence checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ml.preprocessing import NUM_FEATURES, SEQUENCE_PREPROCESSING_VERSION

DEFAULT_SEQUENCE_LENGTH = 30


class SignLSTM(nn.Module):
    """LSTM classifier for dynamic gestures and moving signs."""

    def __init__(
        self,
        num_classes: int,
        input_size: int = NUM_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Need at least two classes")
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("input_size, hidden_size and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        if pooling not in ("mean", "last"):
            raise ValueError("pooling must be 'mean' or 'last'")
        self.config = {
            "num_classes": num_classes,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "pooling": pooling,
        }
        self.pooling = pooling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(features)
        pooled = encoded.mean(dim=1) if self.pooling == "mean" else encoded[:, -1]
        return self.classifier(pooled)


def save_sequence_checkpoint(
    path: str | Path,
    model: SignLSTM,
    labels: list[str],
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    training: dict[str, Any] | None = None,
    dataset_sha256: str | None = None,
    splits: dict[str, list[int]] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "architecture": "lstm",
            "preprocessing": SEQUENCE_PREPROCESSING_VERSION,
            "num_features": model.config["input_size"],
            "sequence_length": int(sequence_length),
            "model_config": model.config,
            "labels": list(labels),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "training": training or {},
            "dataset_sha256": dataset_sha256,
            "splits": splits,
        },
        output,
    )


def load_sequence_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[SignLSTM, list[str], dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    preprocessing = checkpoint.get("preprocessing")
    if (
        checkpoint.get("architecture") != "lstm"
        or checkpoint.get("format_version") != 2
        or preprocessing != SEQUENCE_PREPROCESSING_VERSION
    ):
        raise ValueError("Unsupported checkpoint architecture, version, or preprocessing")
    labels = checkpoint.get("labels")
    config = checkpoint.get("model_config", {})
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or not all(isinstance(label, str) and label.strip() for label in labels)
        or len(set(labels)) != len(labels)
        or config.get("num_classes") != len(labels)
    ):
        raise ValueError("Invalid checkpoint label mapping")
    model = SignLSTM(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, labels, checkpoint
