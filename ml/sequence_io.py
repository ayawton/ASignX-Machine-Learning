"""NumPy-only sequence storage shared by capture, import, and training."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from ml.body_context import BODY_CONTEXT_VERSION, BODY_NUM_FEATURES, TEMPLE_CONTEXT_VERSION, TEMPLE_NUM_FEATURES, BODY_FEATURE_WIDTHS, sequence_contract
from ml.preprocessing import DUAL_NUM_FEATURES, NUM_FEATURES, SEQUENCE_PREPROCESSING_VERSION


@dataclass
class SequenceSample:
    label: str
    session_id: str
    handedness: str
    features: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = self.label.strip()
        self.session_id = self.session_id.strip()
        self.handedness = self.handedness.strip()
        self.features = np.asarray(self.features, dtype=np.float32)
        if not self.label or not self.session_id:
            raise ValueError("Sequence label and session_id must not be empty")
        if (
            self.features.ndim != 2
            or self.features.shape[0] < 1
            or self.features.shape[1] not in (NUM_FEATURES, DUAL_NUM_FEATURES, *BODY_FEATURE_WIDTHS)
        ):
            raise ValueError(
                f"Features must have shape (T, {NUM_FEATURES}) or "
                f"(T, {DUAL_NUM_FEATURES}), (T, {BODY_NUM_FEATURES}) or (T, {TEMPLE_NUM_FEATURES}), got {self.features.shape}"
            )
        if not np.isfinite(self.features).all():
            raise ValueError("Sequence features must be finite")


def save_sequence_file(output_path: str | Path, sample: SequenceSample) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Sequence files must use the .npz extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=sample.features,
        label=sample.label,
        session_id=sample.session_id,
        handedness=sample.handedness,
        preprocessing=sequence_contract(sample.features.shape[1]),
        metadata=json.dumps(sample.metadata, sort_keys=True),
    )
    return path


def load_sequence_file(path: str | Path) -> SequenceSample:
    with np.load(Path(path), allow_pickle=False) as data:
        preprocessing = str(data["preprocessing"]) if "preprocessing" in data else ""
        if preprocessing not in (SEQUENCE_PREPROCESSING_VERSION, BODY_CONTEXT_VERSION, TEMPLE_CONTEXT_VERSION):
            raise ValueError(
                f"Incompatible preprocessing version {preprocessing!r}; "
                f"expected a supported hand/body contract, including {TEMPLE_CONTEXT_VERSION!r}. Regenerate videos "
                "with python -m ml.prepare_videos"
            )
        width = data["features"].shape[-1]
        if sequence_contract(width) != preprocessing:
            raise ValueError("Body context feature width and preprocessing contract disagree")
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
        return SequenceSample(
            label=str(data["label"]),
            session_id=str(data["session_id"]),
            handedness=str(data["handedness"]),
            features=data["features"].astype(np.float32),
            metadata=metadata,
        )


def read_sequence_dir(directory: str | Path) -> list[SequenceSample]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Sequence directory not found: {root}")
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz sequence files found in {root}")
    samples = [load_sequence_file(path) for path in paths]
    widths = {sample.features.shape[1] for sample in samples}
    lengths = {sample.features.shape[0] for sample in samples}
    if len(widths) != 1:
        raise ValueError("All sequence samples must use the same one/two-hand feature width")
    if len(lengths) != 1:
        raise ValueError("All sequence samples must have the same sequence length")
    return samples


