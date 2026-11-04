"""Dynamic gesture classifiers (TCN-Attention & LSTM) with kinematic enrichment and prototype heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ml.model import CosinePrototypeHead, TorchKinematicEnricher
from ml.motion_features import MotionDeltaFeatures
from ml.body_context import BODY_CONTEXT_VERSION, TEMPLE_CONTEXT_VERSION, sequence_contract
from ml.preprocessing import (
    DUAL_KINEMATIC_NUM_FEATURES,
    DUAL_NUM_FEATURES,
    KINEMATIC_NUM_FEATURES,
    NUM_FEATURES,
    SEQUENCE_PREPROCESSING_VERSION,
)

DEFAULT_SEQUENCE_LENGTH = 30


class TemporalConvBlock(nn.Module):
    """Dilated 1D Temporal Convolutional Residual Block."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(self.act(self.norm1(out)))
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(self.act(self.norm2(out)))
        return residual + out


class SignTCNAttention(nn.Module):
    """Temporal Convolutional Network + Multi-Head Self-Attention for dynamic signs."""

    def __init__(
        self,
        num_classes: int,
        input_size: int = NUM_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        *,
        use_kinematics: bool = True,
        cosine_head: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Need at least two classes")
        self.use_kinematics = bool(use_kinematics) and input_size in (NUM_FEATURES, DUAL_NUM_FEATURES)
        self.cosine_head = bool(cosine_head)
        self.enricher = TorchKinematicEnricher() if self.use_kinematics else None
        if self.use_kinematics:
            in_dim = KINEMATIC_NUM_FEATURES if input_size == NUM_FEATURES else DUAL_KINEMATIC_NUM_FEATURES
        else:
            in_dim = input_size

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.tcn_blocks = nn.ModuleList(
            [
                TemporalConvBlock(hidden_size, kernel_size=3, dilation=2 ** i, dropout=dropout)
                for i in range(num_layers)
            ]
        )
        num_heads = 4 if hidden_size % 4 == 0 else (2 if hidden_size % 2 == 0 else 1)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_pool_weights = nn.Linear(hidden_size, 1)

        embed_dim = 64
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.head = (
            CosinePrototypeHead(embed_dim, num_classes)
            if self.cosine_head
            else nn.Linear(embed_dim, num_classes)
        )
        self.config = {
            "num_classes": num_classes,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "backbone": "tcn_attention",
            "use_kinematics": self.use_kinematics,
            "cosine_head": self.cosine_head,
        }

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        x = self.enricher(features) if self.enricher is not None else features
        h = self.input_proj(x)
        for block in self.tcn_blocks:
            h = block(h)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.attn_norm(h + attn_out)
        # Attention-weighted temporal pooling across frames
        scores = torch.softmax(self.attn_pool_weights(h), dim=1)
        pooled = (h * scores).sum(dim=1)
        return self.proj(pooled)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(features))


class SignLSTM(nn.Module):
    """Dynamic gesture sequence classifier supporting both LSTM and TCN-Attention backbones."""

    def __init__(
        self,
        num_classes: int,
        input_size: int = NUM_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        pooling: str = "mean",
        *,
        backbone: str = "lstm",
        use_kinematics: bool = True,
        cosine_head: bool = True,
        use_motion_deltas: bool = False,
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
        if backbone not in ("lstm", "tcn_attention"):
            raise ValueError("backbone must be 'lstm' or 'tcn_attention'")

        self.pooling = pooling
        self.backbone = backbone
        self.use_kinematics = bool(use_kinematics) and input_size in (NUM_FEATURES, DUAL_NUM_FEATURES)
        self.cosine_head = bool(cosine_head)
        self.use_motion_deltas = bool(use_motion_deltas)
        self.motion_features = MotionDeltaFeatures(input_size) if self.use_motion_deltas else None
        self.config = {
            "num_classes": num_classes,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "pooling": pooling,
            "backbone": backbone,
            "use_kinematics": self.use_kinematics,
            "cosine_head": self.cosine_head,
            "use_motion_deltas": self.use_motion_deltas,
        }

        self.enricher = TorchKinematicEnricher() if self.use_kinematics else None
        if self.use_kinematics:
            in_dim = KINEMATIC_NUM_FEATURES if input_size == NUM_FEATURES else DUAL_KINEMATIC_NUM_FEATURES
        else:
            in_dim = input_size

        if self.use_motion_deltas:
            in_dim += input_size

        if backbone == "tcn_attention":
            self.tcn_in = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )
            self.tcn_blocks = nn.ModuleList(
                [
                    TemporalConvBlock(hidden_size, kernel_size=3, dilation=2 ** i, dropout=dropout)
                    for i in range(num_layers)
                ]
            )
            num_heads = 4 if hidden_size % 4 == 0 else (2 if hidden_size % 2 == 0 else 1)
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_size)
            self.lstm = None
            directions = 1
        else:
            self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            self.tcn_in = None
            self.tcn_blocks = None
            self.attn = None
            self.attn_norm = None
            directions = 2 if bidirectional else 1

        embed_dim = 64
        if not self.use_kinematics and not self.cosine_head and backbone == "lstm":
            # Legacy v2 mode compatibility
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * directions, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
            self.embed_proj = None
            self.head = None
        else:
            self.embed_proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * directions, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head = (
                CosinePrototypeHead(embed_dim, num_classes)
                if self.cosine_head
                else nn.Linear(embed_dim, num_classes)
            )
            self.classifier = None

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode temporal sequence (B, T, F) into a latent vector (B, 64)."""
        x = self.enricher(features) if self.enricher is not None else features
        if self.motion_features is not None:
            x = torch.cat((x, self.motion_features(features)), dim=-1)
        if self.backbone == "tcn_attention":
            assert self.tcn_in is not None and self.tcn_blocks is not None
            assert self.attn is not None and self.attn_norm is not None
            h = self.tcn_in(x)
            for block in self.tcn_blocks:
                h = block(h)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            encoded = self.attn_norm(h + attn_out)
        else:
            assert self.lstm is not None
            encoded, _ = self.lstm(x)

        pooled = encoded.mean(dim=1) if self.pooling == "mean" else encoded[:, -1]
        if self.embed_proj is not None:
            return self.embed_proj(pooled)
        assert self.classifier is not None
        return self.classifier[:-1](pooled)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.classifier is not None and self.embed_proj is None:
            x = self.enricher(features) if self.enricher is not None else features
            if self.motion_features is not None:
                x = torch.cat((x, self.motion_features(features)), dim=-1)
            assert self.lstm is not None
            encoded, _ = self.lstm(x)
            pooled = encoded.mean(dim=1) if self.pooling == "mean" else encoded[:, -1]
            return self.classifier(pooled)
        embeddings = self.encode(features)
        assert self.head is not None
        return self.head(embeddings)

    @torch.no_grad()
    def add_sign_prototype(self, sample_sequences: torch.Tensor) -> None:
        """Compute mean embedding prototype from (N, T, F) sequences and add a new class."""
        if not isinstance(self.head, CosinePrototypeHead):
            raise ValueError("Instant prototype addition requires cosine_head=True")
        self.eval()
        embeddings = self.encode(sample_sequences)
        normed = F.normalize(embeddings, p=2, dim=-1, eps=1e-6)
        mean_proto = normed.mean(dim=0)
        self.head.append_prototype(mean_proto)
        self.config["num_classes"] = self.head.num_classes


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
            "preprocessing": sequence_contract(model.config["input_size"]),
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
        or preprocessing not in (SEQUENCE_PREPROCESSING_VERSION, BODY_CONTEXT_VERSION, TEMPLE_CONTEXT_VERSION)
    ):
        raise ValueError("Unsupported checkpoint architecture, version, or preprocessing")
    labels = checkpoint.get("labels")
    config = dict(checkpoint.get("model_config", {}))
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or not all(isinstance(label, str) and label.strip() for label in labels)
        or len(set(labels)) != len(labels)
        or config.get("num_classes") != len(labels)
    ):
        raise ValueError("Invalid checkpoint label mapping")

    state_dict = checkpoint["state_dict"]
    if any(k.startswith("classifier.") for k in state_dict) and "use_kinematics" not in config:
        config["use_kinematics"] = False
        config["cosine_head"] = False
        config["backbone"] = "lstm"

    if sequence_contract(config.get("input_size", NUM_FEATURES)) != preprocessing:
        raise ValueError("Body context checkpoint contract does not match input size")
    model = SignLSTM(**config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, labels, checkpoint
