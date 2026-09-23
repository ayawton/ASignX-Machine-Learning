"""CSV storage for static landmark samples."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.preprocessing import (
    FEATURE_COLUMNS,
    PREPROCESSING_VERSION,
    normalize_landmarks,
    validate_features,
)

CSV_COLUMNS = ["label", "session_id", "handedness", "preprocessing", *FEATURE_COLUMNS]


@dataclass
class Sample:
    label: str
    session_id: str
    handedness: str
    features: np.ndarray


def read_samples(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    session_labels: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError("Unexpected CSV schema; collect data with ml.collect_data")
        for row_number, row in enumerate(reader, start=2):
            try:
                label = row["label"].strip()
                session = row["session_id"].strip()
                if not label or not session:
                    raise ValueError
                if row["preprocessing"] != PREPROCESSING_VERSION:
                    raise ValueError("Incompatible preprocessing version")
                previous = session_labels.setdefault(session, label)
                if previous != label:
                    raise ValueError("Each recording session must contain only one label")
                features = validate_features(
                    np.asarray([float(row[column]) for column in FEATURE_COLUMNS], dtype=np.float32)
                )
                samples.append(Sample(label, session, row["handedness"].strip(), features))
            except (KeyError, TypeError, AttributeError) as error:
                raise ValueError(f"Malformed CSV row {row_number}: {error}") from error
            except ValueError as error:
                raise ValueError(f"Row {row_number}: {error}") from error
    if not samples:
        raise ValueError("Dataset is empty; collect labeled hands first")
    return samples


class SampleWriter:
    """Append one labeled recording session and flush every sample."""

    def __init__(self, path: str | Path, label: str, session_id: str) -> None:
        self.path = Path(path)
        self.label = label.strip()
        self.session_id = session_id.strip()
        if not self.label or not self.session_id:
            raise ValueError("Label and session_id must not be empty")
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="", encoding="utf-8") as existing:
                reader = csv.DictReader(existing)
                if reader.fieldnames != CSV_COLUMNS:
                    raise ValueError("Cannot append to incompatible CSV schema")
                for row in reader:
                    if row["session_id"] == self.session_id and row["label"] != self.label:
                        raise ValueError("Session ID already belongs to another label")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_COLUMNS)
        if is_new:
            self._writer.writeheader()
            self._handle.flush()

    def add(self, landmarks: np.ndarray, handedness: str = "") -> np.ndarray:
        features = normalize_landmarks(landmarks)
        row = {
            "label": self.label,
            "session_id": self.session_id,
            "handedness": handedness,
            "preprocessing": PREPROCESSING_VERSION,
            **dict(zip(FEATURE_COLUMNS, features.tolist())),
        }
        self._writer.writerow(row)
        self._handle.flush()
        return features

    def __enter__(self) -> "SampleWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._handle.close()
