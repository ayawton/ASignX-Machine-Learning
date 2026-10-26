"""Sequence storage, leakage-safe splits, and PyTorch dataset adapters."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.body_context import sequence_contract
from ml.sequence_io import SequenceSample, save_sequence_file, load_sequence_file, read_sequence_dir

from ml.preprocessing import (
    DUAL_NUM_FEATURES,
    NUM_FEATURES,
    SEQUENCE_PREPROCESSING_VERSION,
)

SPLITS = ("train", "validation", "test")
SEQUENCE_MANIFEST_VERSION = 2


def dataset_sequence_digest(samples: Iterable[SequenceSample]) -> str:
    digest = hashlib.sha256()
    ordered = sorted(
        samples,
        key=lambda sample: (
            sample.label,
            sample.session_id,
            hashlib.sha256(sample.features.tobytes()).hexdigest(),
        ),
    )
    for sample in ordered:
        digest.update(sample.label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.session_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.features.tobytes())
    return digest.hexdigest()


def create_sequence_splits(
    samples: list[SequenceSample],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than 1")
    labels = sorted({sample.label for sample in samples})
    if len(labels) < 2:
        raise ValueError("Collect at least two sign classes")

    by_session: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        by_session.setdefault(sample.session_id, []).append(index)
    for session, indices in by_session.items():
        if len({samples[index].label for index in indices}) != 1:
            raise ValueError(f"Each recording session must contain only one label: {session}")

    by_label: dict[str, list[str]] = {label: [] for label in labels}
    for session, indices in by_session.items():
        by_label[samples[indices[0]].label].append(session)
    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {name: [] for name in SPLITS}
    for label, sessions in sorted(by_label.items()):
        if len(sessions) < 3:
            raise ValueError(
                f"{label}: need at least 3 independent sessions, got {len(sessions)}"
            )
        sessions = sorted(sessions)
        rng.shuffle(sessions)
        validation_count = max(1, round(len(sessions) * validation_fraction))
        test_count = max(1, round(len(sessions) * test_fraction))
        if validation_count + test_count >= len(sessions):
            validation_count = test_count = 1
        assignments = {
            "validation": sessions[:validation_count],
            "test": sessions[validation_count : validation_count + test_count],
            "train": sessions[validation_count + test_count :],
        }
        for split, selected_sessions in assignments.items():
            for session in selected_sessions:
                splits[split].extend(by_session[session])
    for indices in splits.values():
        indices.sort()
    validate_sequence_splits(samples, splits)

    counts = Counter(sample.label for sample in samples)
    if min(counts.values()) * 2 < max(counts.values()):
        warnings.warn(f"Sequence class imbalance: {dict(counts)}; collect similar counts per class")
    return splits


def validate_sequence_splits(
    samples: list[SequenceSample], splits: dict[str, list[int]]
) -> None:
    if set(splits) != set(SPLITS):
        raise ValueError("Manifest must contain train, validation and test")
    flat = [index for indices in splits.values() for index in indices]
    if sorted(flat) != list(range(len(samples))) or any(type(index) is not int for index in flat):
        raise ValueError("Splits must partition all samples exactly once")
    expected_labels = {sample.label for sample in samples}
    session_splits: dict[str, str] = {}
    feature_splits: dict[bytes, str] = {}
    for split, indices in splits.items():
        if {samples[index].label for index in indices} != expected_labels:
            raise ValueError(f"{split} must include every class")
        for index in indices:
            sample = samples[index]
            previous = session_splits.setdefault(sample.session_id, split)
            if previous != split:
                raise ValueError("Recording session leaks across splits")
            fingerprint = sample.features.tobytes()
            prior_split = feature_splits.setdefault(fingerprint, split)
            if prior_split != split:
                raise ValueError("Identical features across splits; remove copied/duplicate recordings")


def save_sequence_manifest(
    path: str | Path,
    samples: list[SequenceSample],
    splits: dict[str, list[int]],
    seed: int,
) -> None:
    validate_sequence_splits(samples, splits)
    payload = {
        "version": SEQUENCE_MANIFEST_VERSION,
        "preprocessing": sequence_contract(samples[0].features.shape[1]),
        "dataset_sha256": dataset_sequence_digest(samples),
        "seed": seed,
        "labels": sorted({sample.label for sample in samples}),
        "splits": splits,
        "counts": {
            split: dict(Counter(samples[index].label for index in indices))
            for split, indices in splits.items()
        },
    }
    manifest = Path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sequence_manifest(
    manifest_path: str | Path, samples: list[SequenceSample]
) -> dict[str, Any]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (
        payload.get("version") != SEQUENCE_MANIFEST_VERSION
        or payload.get("preprocessing") != (sequence_contract(samples[0].features.shape[1]))
    ):
        raise ValueError("Incompatible sequence split manifest version or preprocessing")
    if payload.get("dataset_sha256") != dataset_sequence_digest(samples):
        raise ValueError("Dataset contents changed since manifest was created; retrain")
    if payload.get("labels") != sorted({sample.label for sample in samples}):
        raise ValueError("Manifest label mapping does not match dataset")
    validate_sequence_splits(samples, payload["splits"])
    return payload


class SequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[SequenceSample],
        indices: list[int],
        labels: list[str],
        *,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        label_to_index = {label: index for index, label in enumerate(labels)}
        self.raw_features = np.stack(
            [samples[index].features for index in indices]
        ).astype(np.float32)
        self.features = torch.from_numpy(self.raw_features)
        self.targets = torch.tensor(
            [label_to_index[samples[index].label] for index in indices], dtype=torch.long
        )
        # Hand-only spatial augmentation cannot rotate mixed body/mask columns.
        self.augment = bool(augment) and self.raw_features.shape[-1] in (63, 126)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augment and self.rng.random() < 0.75:
            from ml.preprocessing import augment_sequence_3d

            aug = augment_sequence_3d(self.raw_features[index], rng=self.rng)
            return torch.from_numpy(aug), self.targets[index]
        return self.features[index], self.targets[index]
