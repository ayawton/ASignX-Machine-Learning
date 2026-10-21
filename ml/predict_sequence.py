"""Sliding-window inference for moving signs and gestures."""

from __future__ import annotations

import argparse
import time
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
        self.target_fps = float(self.metadata.get("training", {}).get("sampling_fps", 15.0))
        if not np.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("Invalid checkpoint sampling FPS")
        self._last_timestamp: float | None = None
        self._next_sample: float | None = None
        self.sampled = False
        self.min_visibility = 0.5
        self.max_gap_frames = 3
        self.buffer: deque[tuple[str, np.ndarray, np.ndarray | None]] = deque(
            maxlen=self.sequence_length
        )

    def reset(self) -> None:
        self.buffer.clear()
        self._last_timestamp = self._next_sample = None
        self.sampled = False

    def add_frame(self, landmarks: np.ndarray | None, timestamp: float | None = None) -> None:
        """Append raw landmarks, preprocessed features, or a missing frame.

        One-hand checkpoints accept ``(21, 3)`` or ``(63,)``. Two-hand
        checkpoints accept canonical Left/Right ``(2, 21, 3)`` or ``(126,)``.
        """
        self.sampled = False
        if timestamp is not None:
            if not np.isfinite(timestamp) or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
                raise ValueError("Live sequence timestamps must increase and be finite")
            self._last_timestamp = timestamp
            interval = 1.0 / self.target_fps
            if self._next_sample is None:
                self._next_sample = timestamp
            if timestamp + 1e-8 < self._next_sample:
                return
            # Camera stalls must not retain an old gesture or synthesize unlimited motion.
            if timestamp - self._next_sample > interval * self.max_gap_frames:
                self.buffer.clear()
                self._next_sample = timestamp
            while timestamp - self._next_sample >= interval - 1e-8:
                self._append_frame(None)
                self._next_sample += interval
            self._next_sample += interval
        self._append_frame(landmarks)
        self.sampled = True

    def _append_frame(self, landmarks):
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
            frame_visible = valid.any(axis=1)
            if frame_visible.mean() < self.min_visibility:
                raise ValueError("Insufficient visible hands")
            for hand in range(valid.shape[1]):
                if not valid[:, hand].any():
                    continue
                longest = run = 0
                for visible in valid[:, hand]:
                    run = 0 if visible else run + 1
                    longest = max(longest, run)
                if longest > self.max_gap_frames:
                    raise ValueError("Hand tracking gap is too long")
            raw, valid = fill_landmark_gaps(raw, valid)
            return normalize_landmark_sequence(raw, valid=valid)
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
        try:
            sequence = self._buffer_features()
        except ValueError:
            return None
        if not np.any(np.abs(sequence) > 1e-8):
            return None
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
        if prediction is None:
            raise ValueError("Sequence has insufficient valid hand evidence")
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
    parser.add_argument("--exposure", default="auto", help="Camera exposure: 'auto' (default) or a manual value such as -5.0")
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--hand", choices=("single", "Left", "Right"), default="single")
    parser.add_argument("--stable-windows", type=int, default=3)
    parser.add_argument("--release-windows", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        exposure = None if args.exposure.lower() == "auto" else float(args.exposure)
        if exposure is not None and not np.isfinite(exposure):
            raise ValueError
    except ValueError:
        parser.error("--exposure must be 'auto' or a finite number")

    from asignx import HandDetector, WebcamStream

    predictor = SequencePredictor(args.model, args.confidence, args.device)
    stabilizer = PredictionStabilizer(args.stable_windows, args.release_windows)
    both = predictor.input_size == DUAL_NUM_FEATURES
    with WebcamStream(camera_index=args.camera, threaded=True, exposure=exposure) as camera:
        if exposure is None:
            # Explicitly restore auto mode: a previous run may have left the driver in manual mode.
            camera.set_exposure(None)
        with HandDetector(max_num_hands=2) as detector:
            for frame in camera.stream():
                result, annotated = detector.process(frame, draw=True)
                predictor.add_frame(_canonical_hands(result.all_hands, both, args.hand), timestamp=time.monotonic())
                if not predictor.sampled:
                    cv2.imshow("ASignX - Moving Sign Recognition", annotated)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                        break
                    continue
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
