"""Small prediction types shared by static and sequence inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ml.model import load_checkpoint
from ml.preprocessing import normalize_landmarks


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    raw_label: str


class SignPredictor:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.8,
        device: str = "cpu",
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        self.model, self.labels, self.metadata = load_checkpoint(model_path, device)
        self.threshold = confidence_threshold
        self.device = device

    @torch.inference_mode()
    def predict(self, landmarks) -> Prediction:
        features = torch.from_numpy(normalize_landmarks(landmarks)).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(features), dim=1)
        confidence, index = probabilities.max(dim=1)
        score = float(confidence.item())
        raw_label = self.labels[int(index.item())]
        return Prediction(raw_label if score >= self.threshold else None, score, raw_label)


class PredictionStabilizer:
    """Confirm consecutive frames and emit a held prediction only once."""

    def __init__(self, stable_frames: int = 5, release_frames: int = 10) -> None:
        if stable_frames < 1 or release_frames < 1:
            raise ValueError("Stable and release frame counts must be positive")
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self.candidate: str | None = None
        self.count = 0
        self.missing = 0
        self.last_emitted: str | None = None

    def update(self, label: str | None) -> tuple[str | None, str | None]:
        if label is None:
            self.missing += 1
            self.candidate, self.count = None, 0
            if self.missing >= self.release_frames:
                self.last_emitted = None
            return None, None
        self.missing = 0
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = label, 1
        stable = label if self.count >= self.stable_frames else None
        emitted = None
        if stable is not None and stable != self.last_emitted:
            emitted = stable
            self.last_emitted = stable
        return stable, emitted


@dataclass(frozen=True)
class Recognition:
    hand: str
    prediction: Prediction
    stable_label: str | None
    emitted_label: str | None


class SignRecognizer:
    """Keep independent static-sign stabilization state for each hand."""

    def __init__(
        self, predictor: SignPredictor, stable_frames: int = 5, release_frames: int = 10
    ) -> None:
        self.predictor = predictor
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self.states: dict[str, PredictionStabilizer] = {}

    def update(self, hands: list) -> list[Recognition]:
        results: list[Recognition] = []
        present: set[str] = set()
        for index, hand in enumerate(hands):
            key = hand.handedness or f"Hand_{index + 1}"
            present.add(key)
            state = self.states.setdefault(
                key, PredictionStabilizer(self.stable_frames, self.release_frames)
            )
            try:
                prediction = self.predictor.predict(hand.to_numpy())
            except ValueError:
                state.update(None)
                continue
            stable, emitted = state.update(prediction.label)
            results.append(Recognition(key, prediction, stable, emitted))
        for key in list(self.states):
            if key not in present:
                self.states[key].update(None)
        return results
