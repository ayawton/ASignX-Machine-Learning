"""Sliding-window inference for moving signs and gestures."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ml.predict import Prediction, PredictionStabilizer
from ml.preprocessing import (
    DUAL_NUM_FEATURES,
    fill_landmark_gaps,
    NUM_FEATURES,
    normalize_landmark_sequence,
)
from ml.sequence_model import DEFAULT_SEQUENCE_LENGTH, load_sequence_checkpoint
from ml.video import resample_sequence


class SequencePredictor:
    """Run a trained sequence model over a fixed sliding landmark window."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.8,
        device: str = "cpu",
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        self.model, self.labels, self.metadata = load_sequence_checkpoint(model_path, device)
        self.sequence_length = int(
            self.metadata.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)
        )
        self.input_size = int(self.model.config["input_size"])
        if self.input_size not in (NUM_FEATURES, DUAL_NUM_FEATURES):
            raise ValueError(f"Unsupported sequence feature width: {self.input_size}")
        self.threshold = confidence_threshold
        self.device = device
        self.buffer: deque[tuple[str, np.ndarray, np.ndarray | None]] = deque(
            maxlen=self.sequence_length
        )

    def reset(self) -> None:
        self.buffer.clear()

    def add_frame(self, landmarks: np.ndarray | None) -> None:
        """Append raw landmarks, preprocessed features, or a missing frame.

        One-hand checkpoints accept ``(21, 3)`` or ``(63,)``. Two-hand
        checkpoints accept canonical Left/Right ``(2, 21, 3)`` or ``(126,)``.
        """
        hands = self.input_size // NUM_FEATURES
        if landmarks is None:
            raw = np.full((hands, 21, 3), np.nan, dtype=np.float32)
            self.buffer.append(("raw", raw, np.zeros(hands, dtype=bool)))
            return
        array = np.asarray(landmarks, dtype=np.float32)
        if array.shape == (self.input_size,):
            if not np.isfinite(array).all():
                raise ValueError("Sequence features must be finite")
            self.buffer.append(("features", array, None))
            return
        expected = (21, 3) if hands == 1 else (2, 21, 3)
        if array.shape != expected:
            raise ValueError(
                f"Expected raw landmarks {expected} or ({self.input_size},) features, "
                f"got {array.shape}"
            )
        if hands == 1:
            array = array[None, :, :]
        valid = np.isfinite(array).all(axis=(1, 2))
        self.buffer.append(("raw", array, valid))

    def _buffer_features(self) -> np.ndarray:
        if all(kind == "raw" for kind, _, _ in self.buffer):
            raw = np.stack([value for _, value, _ in self.buffer])
            valid = np.stack([mask for _, _, mask in self.buffer])
            try:
                raw, valid = fill_landmark_gaps(raw, valid)
                return normalize_landmark_sequence(raw, valid=valid)
            except ValueError:
                return np.zeros((len(self.buffer), self.input_size), dtype=np.float32)
        rows: list[np.ndarray] = []
        for kind, value, _ in self.buffer:
            if kind != "features":
                raise ValueError("Do not mix raw landmarks and preprocessed features in one window")
            rows.append(value)
        return np.stack(rows).astype(np.float32)

    @torch.inference_mode()
    def predict(self, min_filled_ratio: float = 0.5) -> Prediction | None:
        if not 0 < min_filled_ratio <= 1:
            raise ValueError("min_filled_ratio must be in (0, 1]")
        required = max(1, int(np.ceil(self.sequence_length * min_filled_ratio)))
        if len(self.buffer) < required:
            return None
        sequence = self._buffer_features()
        if sequence.shape[0] < self.sequence_length:
            padding = np.zeros(
                (self.sequence_length - sequence.shape[0], self.input_size), dtype=np.float32
            )
            sequence = np.vstack((padding, sequence))
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)
        confidence, index = probabilities.max(dim=1)
        score = float(confidence.item())
        raw_label = self.labels[int(index.item())]
        return Prediction(
            label=raw_label if score >= self.threshold else None,
            confidence=score,
            raw_label=raw_label,
        )

    def predict_sequence(self, sequence: np.ndarray) -> Prediction:
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim == 3 or array.ndim == 4:
            hands = self.input_size // NUM_FEATURES
            if array.ndim == 3 and hands == 1:
                array = normalize_landmark_sequence(array)
            elif array.ndim == 4 and array.shape[1] == hands:
                array = normalize_landmark_sequence(array)
            else:
                raise ValueError("Raw sequence hand count does not match the checkpoint")
        if array.ndim != 2 or array.shape[1] != self.input_size:
            raise ValueError(f"Expected sequence shape (T, {self.input_size}), got {array.shape}")
        array = resample_sequence(array, self.sequence_length, interpolate=True)
        self.reset()
        for row in array:
            self.add_frame(row)
        prediction = self.predict(min_filled_ratio=1.0)
        assert prediction is not None
        return prediction


def _canonical_hands(hands: list[Any], both: bool, selection: str) -> np.ndarray | None:
    if both:
        output = np.full((2, 21, 3), np.nan, dtype=np.float32)
        found = False
        for slot, label in enumerate(("Left", "Right")):
            hand = next((item for item in hands if item.handedness == label), None)
            if hand is not None:
                output[slot] = hand.to_numpy()
                found = True
        return output if found else None
    if selection == "single":
        hand = hands[0] if len(hands) == 1 else None
    else:
        hand = next((item for item in hands if item.handedness == selection), None)
    return hand.to_numpy() if hand is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ml/models/sequence_classifier.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--hand", choices=("single", "Left", "Right"), default="single")
    parser.add_argument("--stable-windows", type=int, default=3)
    parser.add_argument("--release-windows", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from asignx import HandDetector, WebcamStream

    predictor = SequencePredictor(args.model, args.confidence, args.device)
    stabilizer = PredictionStabilizer(args.stable_windows, args.release_windows)
    both = predictor.input_size == DUAL_NUM_FEATURES
    with WebcamStream(camera_index=args.camera, threaded=True, exposure=-5.0) as camera:
        with HandDetector(max_num_hands=2) as detector:
            for frame in camera.stream():
                result, annotated = detector.process(frame, draw=True)
                predictor.add_frame(_canonical_hands(result.all_hands, both, args.hand))
                prediction = predictor.predict()
                label = prediction.label if prediction else None
                stable, emitted = stabilizer.update(label)
                status = "collecting"
                if prediction is not None:
                    status = f"{prediction.raw_label} {prediction.confidence:.0%}"
                if stable:
                    status = f"SIGN: {stable}"
                cv2.putText(
                    annotated,
                    status,
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if stable else (0, 255, 255),
                    2,
                )
                if emitted:
                    print(f"[SIGN] {emitted}")
                cv2.imshow("ASignX - Moving Sign Recognition", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    predictor.reset()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
