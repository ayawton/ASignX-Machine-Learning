"""Static-pose ResMLP with kinematic feature enrichment, cosine prototype head, and checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ml.preprocessing import (
    DUAL_KINEMATIC_NUM_FEATURES,
    DUAL_NUM_FEATURES,
    FINGERTIP_PAIRS,
    JOINT_TRIPLETS,
    KINEMATIC_NUM_FEATURES,
    NUM_FEATURES,
    PREPROCESSING_VERSION,
    STATIC_HAND_LEGACY,
    STATIC_HAND_CANONICAL,
)


class TorchKinematicEnricher(nn.Module):
    """Differentiable, ONNX-exportable kinematic feature enricher (63 -> 91 dims per hand)."""

    def __init__(self) -> None:
        super().__init__()
        p_idx = torch.tensor([t[0] for t in JOINT_TRIPLETS], dtype=torch.long)
        j_idx = torch.tensor([t[1] for t in JOINT_TRIPLETS], dtype=torch.long)
        c_idx = torch.tensor([t[2] for t in JOINT_TRIPLETS], dtype=torch.long)
        self.register_buffer("parent_idx", p_idx, persistent=False)
        self.register_buffer("joint_idx", j_idx, persistent=False)
        self.register_buffer("child_idx", c_idx, persistent=False)

        tip_a = torch.tensor([p[0] for p in FINGERTIP_PAIRS], dtype=torch.long)
        tip_b = torch.tensor([p[1] for p in FINGERTIP_PAIRS], dtype=torch.long)
        self.register_buffer("tip_a", tip_a, persistent=False)
        self.register_buffer("tip_b", tip_b, persistent=False)

    def _enrich_single_hand(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        flat = x.reshape(-1, 21, 3)

        # 1. 15 Finger Joint Flexion Angles (cosines)
        v1 = flat[:, self.parent_idx] - flat[:, self.joint_idx]
        v2 = flat[:, self.child_idx] - flat[:, self.joint_idx]
        dot = (v1 * v2).sum(dim=-1)
        denom = (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)).clamp_min(1e-6)
        angles = (dot / denom).clamp(-1.0, 1.0)

        # 2. 10 Pairwise Fingertip Distances
        diffs = flat[:, self.tip_a] - flat[:, self.tip_b]
        distances = torch.norm(diffs, dim=-1)

        # 3. 3D Unit Palm Normal Vector
        v_index = flat[:, 5] - flat[:, 0]
        v_pinky = flat[:, 17] - flat[:, 0]
        cross = torch.linalg.cross(v_index, v_pinky, dim=-1)
        normals = cross / torch.norm(cross, dim=-1, keepdim=True).clamp_min(1e-6)

        enriched = torch.cat([x.reshape(-1, NUM_FEATURES), angles, distances, normals], dim=-1)
        return enriched.reshape(*orig_shape, KINEMATIC_NUM_FEATURES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        width = features.shape[-1]
        if width == NUM_FEATURES:
            return self._enrich_single_hand(features)
        if width == DUAL_NUM_FEATURES:
            left = self._enrich_single_hand(features[..., :NUM_FEATURES])
            right = self._enrich_single_hand(features[..., NUM_FEATURES:])
            return torch.cat([left, right], dim=-1)
        raise ValueError(f"Unsupported feature width for kinematic enricher: {width}")


class ResidualMLPBlock(nn.Module):
    """Pre-norm Residual MLP block with LayerNorm and GELU."""

    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CosinePrototypeHead(nn.Module):
    """Metric-learning cosine similarity classifier supporting instant prototype addition."""

    def __init__(self, in_features: int, num_classes: int, init_scale: float = 16.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)).log())

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        norm_x = F.normalize(embeddings, p=2, dim=-1, eps=1e-6)
        norm_w = F.normalize(self.weight, p=2, dim=-1, eps=1e-6)
        scale = self.log_scale.exp().clamp(1.0, 50.0)
        return scale * F.linear(norm_x, norm_w)

    @torch.no_grad()
    def append_prototype(self, prototype_vector: torch.Tensor) -> None:
        """Append a new class prototype vector without retraining existing classes."""
        proto = F.normalize(prototype_vector.reshape(1, self.in_features), p=2, dim=-1, eps=1e-6)
        new_weight = torch.cat([self.weight.data, proto.to(self.weight.device)], dim=0)
        self.num_classes = new_weight.shape[0]
        self.weight = nn.Parameter(new_weight)


class SignMLP(nn.Module):
    """Kinematic Residual MLP with Cosine Prototype Head (and legacy MLP compatibility)."""

    def __init__(
        self,
        num_classes: int,
        hidden_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        *,
        use_kinematics: bool = True,
        residual: bool = True,
        cosine_head: bool = True,
        static_hand_preprocessing: str = STATIC_HAND_LEGACY,
    ) -> None:
        super().__init__()
        if num_classes < 2 or not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("Need at least two classes and positive hidden sizes")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")

        self.use_kinematics = bool(use_kinematics)
        self.residual = bool(residual)
        self.cosine_head = bool(cosine_head)
        if static_hand_preprocessing not in (STATIC_HAND_LEGACY, STATIC_HAND_CANONICAL):
            raise ValueError(f"Unsupported static hand preprocessing: {static_hand_preprocessing}")
        # Forward expects already prepared features; callers use this contract.
        self.static_hand_preprocessing = static_hand_preprocessing
        self.config = {
            "num_classes": num_classes,
            "hidden_sizes": tuple(hidden_sizes),
            "dropout": dropout,
            "use_kinematics": self.use_kinematics,
            "residual": self.residual,
            "cosine_head": self.cosine_head,
            "static_hand_preprocessing": self.static_hand_preprocessing,
        }

        self.enricher = TorchKinematicEnricher() if self.use_kinematics else None
        in_dim = KINEMATIC_NUM_FEATURES if self.use_kinematics else NUM_FEATURES

        if not self.use_kinematics and not self.residual and not self.cosine_head:
            # Legacy sequential mode for older v1 checkpoints
            layers: list[nn.Module] = []
            previous = in_dim
            for size in hidden_sizes:
                layers.extend((nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)))
                previous = size
            layers.append(nn.Linear(previous, num_classes))
            self.network = nn.Sequential(*layers)
            self.encoder = None
            self.head = None
        else:
            enc_layers: list[nn.Module] = []
            previous = in_dim
            for size in hidden_sizes:
                enc_layers.extend(
                    (
                        nn.Linear(previous, size),
                        nn.LayerNorm(size),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
                if self.residual:
                    enc_layers.append(ResidualMLPBlock(size, dropout=dropout))
                previous = size
            self.encoder = nn.Sequential(*enc_layers)
            self.head = (
                CosinePrototypeHead(previous, num_classes)
                if self.cosine_head
                else nn.Linear(previous, num_classes)
            )
            self.network = None

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Return latent embedding representation before the classification head."""
        x = self.enricher(features) if self.enricher is not None else features
        if self.encoder is not None:
            return self.encoder(x)
        # Fallback for legacy sequential network: run all layers except final Linear
        assert self.network is not None
        return self.network[:-1](x)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.network is not None and self.encoder is None:
            x = self.enricher(features) if self.enricher is not None else features
            return self.network(x)
        embeddings = self.encode(features)
        assert self.head is not None
        return self.head(embeddings)

    @torch.no_grad()
    def add_sign_prototype(self, sample_features: torch.Tensor) -> None:
        """Compute mean embedding prototype from (N, 63) sample features and add a new class."""
        if not isinstance(self.head, CosinePrototypeHead):
            raise ValueError("Instant prototype addition requires cosine_head=True")
        self.eval()
        embeddings = self.encode(sample_features)
        normed = F.normalize(embeddings, p=2, dim=-1, eps=1e-6)
        mean_proto = normed.mean(dim=0)
        self.head.append_prototype(mean_proto)
        self.config["num_classes"] = self.head.num_classes


# Alias for explicit architectural reference
SignResMLP = SignMLP


def save_checkpoint(
    path: str | Path,
    model: SignMLP,
    labels: list[str],
    *,
    training: dict[str, Any] | None = None,
    dataset_sha256: str | None = None,
    splits: dict[str, list[int]] | None = None,
    class_centroids: dict[str, list[float]] | None = None,
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
            "class_centroids": class_centroids or {},
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
    config = dict(checkpoint.get("model_config", {}))
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or len(set(labels)) != len(labels)
        or not all(isinstance(label, str) and label.strip() for label in labels)
        or config.get("num_classes") != len(labels)
    ):
        raise ValueError("Invalid checkpoint label mapping")

    state_dict = checkpoint["state_dict"]
    # Auto-detect legacy v1 checkpoints that only had 'network.*' keys
    if any(k.startswith("network.") for k in state_dict) and "use_kinematics" not in config:
        config["use_kinematics"] = False
        config["residual"] = False
        config["cosine_head"] = False

    model = SignMLP(**config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, labels, checkpoint
