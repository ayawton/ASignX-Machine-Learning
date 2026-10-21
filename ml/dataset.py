"""Leakage-safe splits for static landmark samples."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.data_io import Sample, read_samples
from ml.preprocessing import (
    PREPROCESSING_VERSION, STATIC_HAND_LEGACY, canonicalize_static_features,
)

SPLITS = ("train", "validation", "test")


def dataset_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_splits(
    samples: list[Sample],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than 1")
    labels = sorted({sample.label for sample in samples})
    if len(labels) < 2:
        raise ValueError("Collect at least two sign classes")
    sessions: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        sessions.setdefault(sample.session_id, []).append(index)
    by_label: dict[str, list[str]] = {label: [] for label in labels}
    for session, indices in sessions.items():
        session_labels = {samples[index].label for index in indices}
        if len(session_labels) != 1:
            raise ValueError("Each recording session must contain only one label")
        by_label[session_labels.pop()].append(session)
    rng = np.random.default_rng(seed)
    splits = {name: [] for name in SPLITS}
    for label, label_sessions in sorted(by_label.items()):
        if len(label_sessions) < 3:
            raise ValueError(
                f"{label}: need at least 3 independent sessions, got {len(label_sessions)}"
            )
        label_sessions = sorted(label_sessions)
        rng.shuffle(label_sessions)
        validation_count = max(1, round(len(label_sessions) * validation_fraction))
        test_count = max(1, round(len(label_sessions) * test_fraction))
        if validation_count + test_count >= len(label_sessions):
            validation_count = test_count = 1
        allocation = {
            "validation": label_sessions[:validation_count],
            "test": label_sessions[validation_count : validation_count + test_count],
            "train": label_sessions[validation_count + test_count :],
        }
        for split, selected in allocation.items():
            for session in selected:
                splits[split].extend(sessions[session])
    for indices in splits.values():
        indices.sort()
    validate_splits(samples, splits)
    counts = Counter(sample.label for sample in samples)
    if min(counts.values()) * 2 < max(counts.values()):
        warnings.warn(f"Static class imbalance: {dict(counts)}; collect similar counts per class")
    return splits


def validate_splits(
    samples: list[Sample], splits: dict[str, list[int]],
    *, static_hand_preprocessing: str = STATIC_HAND_LEGACY,
) -> None:
    if set(splits) != set(SPLITS):
        raise ValueError("Manifest must contain train, validation and test")
    flat = [index for indices in splits.values() for index in indices]
    if sorted(flat) != list(range(len(samples))) or any(type(index) is not int for index in flat):
        raise ValueError("Splits must partition all rows exactly once")
    labels = {sample.label for sample in samples}
    session_owner: dict[str, str] = {}
    feature_owner: dict[bytes, str] = {}
    feature_labels: dict[bytes, str] = {}
    for split, indices in splits.items():
        if {samples[index].label for index in indices} != labels:
            raise ValueError(f"{split} must include every class")
        for index in indices:
            sample = samples[index]
            owner = session_owner.setdefault(sample.session_id, split)
            if owner != split:
                raise ValueError("Recording session leaks across splits")
            fingerprint = canonicalize_static_features(
                sample.features, sample.handedness, static_hand_preprocessing,
            ).tobytes()
            prior_label = feature_labels.setdefault(fingerprint, sample.label)
            if prior_label != sample.label:
                raise ValueError("Identical canonical features have conflicting labels; review original evidence")
            feature_split = feature_owner.setdefault(fingerprint, split)
            if feature_split != split:
                raise ValueError("Identical features across splits; remove copied/duplicate recordings")


def save_manifest(
    path: str | Path,
    csv_path: str | Path,
    samples: list[Sample],
    splits: dict[str, list[int]],
    seed: int,
    *, static_hand_preprocessing: str = STATIC_HAND_LEGACY,
) -> None:
    validate_splits(samples, splits, static_hand_preprocessing=static_hand_preprocessing)
    payload = {
        "version": 1,
        "preprocessing": PREPROCESSING_VERSION,
        "static_hand_preprocessing": static_hand_preprocessing,
        "dataset_sha256": dataset_digest(csv_path),
        "seed": seed,
        "labels": sorted({sample.label for sample in samples}),
        "splits": splits,
        "counts": {
            split: dict(Counter(samples[index].label for index in indices))
            for split, indices in splits.items()
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: str | Path, csv_path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("preprocessing") != PREPROCESSING_VERSION:
        raise ValueError("Incompatible split manifest")
    if payload.get("dataset_sha256") != dataset_digest(csv_path):
        raise ValueError("Dataset changed since splitting; train a new model and manifest")
    samples = read_samples(csv_path)
    validate_splits(samples, payload["splits"], static_hand_preprocessing=payload.get(
        "static_hand_preprocessing", STATIC_HAND_LEGACY))
    if payload.get("labels") != sorted({sample.label for sample in samples}):
        raise ValueError("Manifest label mapping does not match dataset")
    return payload


class LandmarkDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        indices: list[int],
        labels: list[str],
        *,
        augment: bool = False,
        seed: int = 42,
        static_hand_preprocessing: str = STATIC_HAND_LEGACY,
        augmentation_scale: float = 1.0,
    ) -> None:
        mapping = {label: index for index, label in enumerate(labels)}
        self.raw_features = np.stack(
            [canonicalize_static_features(
                samples[index].features, samples[index].handedness, static_hand_preprocessing,
            ) for index in indices]
        ).astype(np.float32)
        self.features = torch.from_numpy(self.raw_features)
        self.targets = torch.tensor(
            [mapping[samples[index].label] for index in indices], dtype=torch.long
        )
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        self.augmentation_scale = augmentation_scale

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        if self.augment and self.rng.random() < 0.75:
            from ml.preprocessing import augment_landmarks_3d

            scale = self.augmentation_scale
            aug = augment_landmarks_3d(self.raw_features[index], rng=self.rng,
                                      max_z_deg=18 * scale, max_tilt_deg=10 * scale,
                                      jitter_std=0.004 * scale)
            return torch.from_numpy(aug), self.targets[index]
        return self.features[index], self.targets[index]

