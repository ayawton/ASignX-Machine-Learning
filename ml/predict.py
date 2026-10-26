"""Static and unified motion-gated sign inference, OOD rejection, and instant sign registration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ml.model import load_checkpoint, save_checkpoint
from ml.preprocessing import canonicalize_static_features, normalize_landmarks


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    raw_label: str


class SignPredictor:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.8,
        device: str = "cpu",
        *, calibration_path: str | Path | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        self.model_path = Path(model_path)
        self.model, self.labels, self.metadata = load_checkpoint(model_path, device)
        self.threshold = confidence_threshold
        if calibration_path:
            from ml.calibrate_f1 import load_calibration
            self.threshold = load_calibration(calibration_path, model_path, self.labels)
        self.device = device

    @torch.inference_mode()
    def predict(self, landmarks: np.ndarray, handedness: str | None = None) -> Prediction:
        if handedness is None:
            handedness = getattr(landmarks, "handedness", None)
        prepared = canonicalize_static_features(
            normalize_landmarks(landmarks), handedness, self.model.static_hand_preprocessing,
        )
        features = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
        logits = self.model(features)
        probabilities = torch.softmax(logits, dim=1)
        confidence, index = probabilities.max(dim=1)
        score = float(confidence.item())
        raw_label = self.labels[int(index.item())]
        return Prediction(raw_label if score >= self.threshold else None, score, raw_label)

    @torch.no_grad()
    def register_new_sign(
        self,
        new_label: str,
        landmark_samples: list[np.ndarray] | np.ndarray,
        save_to: str | Path | None = None,
        *, handedness: str | list[str] | None = None,
    ) -> list[str]:
        """Add a prototype from original raw/normalized samples, with a hand label per sample."""
        clean_label = new_label.strip()
        if not clean_label or clean_label in self.labels:
            raise ValueError(f"Label {clean_label!r} is empty or already exists in {self.labels}")
        norm_rows: list[np.ndarray] = []
        if isinstance(handedness, list) and len(handedness) != len(landmark_samples):
            raise ValueError("Provide one handedness label per registration sample")
        for index, sample in enumerate(landmark_samples):
            arr = np.asarray(sample, dtype=np.float32)
            hand = handedness[index] if isinstance(handedness, list) else handedness
            features = normalize_landmarks(arr) if arr.shape == (21, 3) else arr
            norm_rows.append(canonicalize_static_features(
                features, hand, self.model.static_hand_preprocessing,
            ))
        if not norm_rows:
            raise ValueError("Provide at least one sample to register a new sign")
        batch = torch.from_numpy(np.stack(norm_rows)).to(self.device)
        self.model.add_sign_prototype(batch)
        self.labels.append(clean_label)

        centroids = dict(self.metadata.get("class_centroids", {}))
        med = np.median(np.stack(norm_rows), axis=0).astype(np.float32)
        centroids[clean_label] = [round(float(x), 6) for x in med.tolist()]
        self.metadata["class_centroids"] = centroids

        target_path = Path(save_to) if save_to is not None else self.model_path
        save_checkpoint(
            target_path,
            self.model,
            self.labels,
            training=self.metadata.get("training"),
            dataset_sha256=self.metadata.get("dataset_sha256"),
            splits=self.metadata.get("splits"),
            class_centroids=centroids,
        )
        return list(self.labels)


def _unique_hands(hands: list[Any]) -> list[Any]:
    """Keep one detection per named hand so each tracker advances once per frame."""
    selected: dict[str, Any] = {}
    for hand in hands:
        label = hand.handedness
        if label in ("Left", "Right"):
            previous = selected.get(label)
            if previous is None or getattr(hand, "score", 0.0) > getattr(previous, "score", 0.0):
                selected[label] = hand
    emitted: set[str] = set()
    result = []
    for hand in hands:
        label = hand.handedness
        if label not in selected:
            result.append(hand)
        elif label not in emitted:
            result.append(selected[label])
            emitted.add(label)
    return result


class MotionGate:
    """Sliding-window hand velocity tracker to route static vs. dynamic signs and suppress transition blur."""

    def __init__(self, window_size: int = 8, velocity_threshold: float = 0.025) -> None:
        if window_size < 2 or velocity_threshold < 0:
            raise ValueError("window_size must be >= 2 and velocity_threshold >= 0")
        self.window_size = window_size
        self.velocity_threshold = velocity_threshold
        self.history: deque[np.ndarray] = deque(maxlen=window_size * 16)
        self.timestamps: deque[float] = deque(maxlen=window_size * 16)

    def reset(self) -> None:
        self.history.clear()
        self.timestamps.clear()

    def update(self, landmarks: np.ndarray | None, timestamp: float | None = None) -> tuple[float, bool]:
        """Return time-averaged speed of the fastest tracked point and motion state."""
        if landmarks is None:
            self.reset()
            return 0.0, False
        pts = np.asarray(landmarks, dtype=np.float32)
        if pts.shape != (21, 3) or not np.isfinite(pts).all():
            self.reset()
            return 0.0, False
        # Track wrist + 5 fingertips in image space
        tracked = pts[[0, 4, 8, 12, 16, 20]]
        stamp = timestamp if timestamp is not None else (self.timestamps[-1] + 1 / 30 if self.timestamps else 0.0)
        if not np.isfinite(stamp) or (self.timestamps and stamp <= self.timestamps[-1]):
            raise ValueError("Motion timestamps must increase and be finite")
        if self.timestamps and stamp - self.timestamps[-1] > 0.5:
            self.reset()
        self.history.append(tracked)
        self.timestamps.append(stamp)
        # Fixed-duration history instead of a camera-FPS-dependent span.
        while len(self.timestamps) > 2 and stamp - self.timestamps[0] > (self.window_size - 1) / 30:
            self.timestamps.popleft(); self.history.popleft()
        if len(self.history) < 2:
            return 0.0, False
        diffs = [
            float(np.linalg.norm(self.history[i] - self.history[i - 1], axis=1).max())
            for i in range(1, len(self.history))
        ]
        velocity = float(np.sum(diffs) / (30 * (self.timestamps[-1] - self.timestamps[0])))
        return velocity, velocity >= self.velocity_threshold


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
    velocity: float = 0.0
    is_moving: bool = False


class SignRecognizer:
    """Keep independent static-sign stabilization and optional motion-gating state for each hand."""

    def __init__(
        self,
        predictor: SignPredictor,
        stable_frames: int = 5,
        release_frames: int = 10,
        *,
        motion_gate_threshold: float | None = None,
    ) -> None:
        self.predictor = predictor
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self.motion_gate_threshold = motion_gate_threshold
        self.states: dict[str, PredictionStabilizer] = {}
        self.motion_gates: dict[str, MotionGate] = {}
        self._last_timestamp: float | None = None

    def update(self, hands: list[Any], timestamp: float | None = None, *, suppress: bool = False) -> list[Recognition]:
        hands = _unique_hands(hands)
        if timestamp is not None:
            if not np.isfinite(timestamp) or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
                raise ValueError("Recognition timestamps must increase and be finite")
            if self._last_timestamp is not None and timestamp - self._last_timestamp > 0.5:
                self.states.clear()
                self.motion_gates.clear()
            self._last_timestamp = timestamp
        results: list[Recognition] = []
        present: set[str] = set()
        for index, hand in enumerate(hands):
            key = hand.handedness or f"Hand_{index + 1}"
            present.add(key)
            state = self.states.setdefault(
                key, PredictionStabilizer(self.stable_frames, self.release_frames)
            )
            pts = hand.to_numpy()
            velocity, is_moving = 0.0, False
            if self.motion_gate_threshold is not None:
                gate = self.motion_gates.setdefault(
                    key, MotionGate(velocity_threshold=self.motion_gate_threshold)
                )
                velocity, is_moving = gate.update(pts, timestamp)

            try:
                prediction = self.predictor.predict(pts, handedness=hand.handedness)
            except ValueError:
                state.update(None)
                continue

            # Suppress static confirmation when hand is mid-sweep if motion gate is active
            effective_label = None if is_moving or suppress else prediction.label
            stable, emitted = state.update(effective_label)
            results.append(
                Recognition(
                    hand=key,
                    prediction=prediction,
                    stable_label=stable,
                    emitted_label=emitted,
                    velocity=velocity,
                    is_moving=is_moving,
                )
            )
        for key in list(self.states):
            if key not in present:
                self.states[key].update(None)
                if key in self.motion_gates:
                    self.motion_gates[key].reset()
        return results


class UnifiedSignRecognizer:
    """Unified Motion-Gated Router combining Static (SignMLP) and Dynamic (SignLSTM/TCN) recognition."""

    def __init__(
        self,
        static_predictor: SignPredictor | None = None,
        sequence_predictor: Any | None = None,
        *,
        velocity_threshold: float = 0.022,
        stable_frames: int = 5,
        release_frames: int = 10,
    ) -> None:
        self.static_recognizer = (
            SignRecognizer(
                static_predictor,
                stable_frames=stable_frames,
                release_frames=release_frames,
                motion_gate_threshold=velocity_threshold,
            )
            if static_predictor is not None
            else None
        )
        self.sequence_predictor = sequence_predictor
        self.hand_motion_gates: dict[str, MotionGate] = {}
        self.velocity_threshold = velocity_threshold
        self.last_motion_at: float | None = None
        self._last_timestamp: float | None = None
        self.state = "static"
        self.pre_roll_seconds = 0.2
        self.settle_seconds = 0.4
        self.max_gesture_seconds = 10.0
        self._history: deque[tuple[float, np.ndarray | None]] = deque(maxlen=1600)
        self._gesture: list[tuple[float, np.ndarray | None]] = []
        self._started_at: float | None = None
        self._result = None
        self._result_at: float | None = None
        self._discard_gesture = False

    def update(self, hands: list[Any], timestamp: float | None = None, *, body_context=None) -> dict[str, Any]:
        hands = _unique_hands(hands)
        timestamp = time.perf_counter() if timestamp is None else timestamp
        if not np.isfinite(timestamp) or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
            raise ValueError("Recognition timestamps must increase and be finite")
        if self._last_timestamp is not None and timestamp - self._last_timestamp > 0.5:
            self.state = "static"
            self.last_motion_at = None
            self._history.clear()
            self._gesture.clear()
            self._result = None
            self._discard_gesture = False
            for gate in self.hand_motion_gates.values():
                gate.reset()
            if self.sequence_predictor is not None:
                self.sequence_predictor.reset()
        self._last_timestamp = timestamp
        present = set()
        velocity = 0.0
        # Lower exit threshold prevents switching routes around the entry threshold.
        threshold = self.velocity_threshold * (0.6 if self.state != "static" else 1.0)
        for index, hand in enumerate(hands):
            key = hand.handedness or f"Hand_{index}"
            present.add(key)
            gate = self.hand_motion_gates.setdefault(key, MotionGate(velocity_threshold=self.velocity_threshold))
            speed, _ = gate.update(hand.to_numpy(), timestamp)
            velocity = max(velocity, speed)
        for key, gate in self.hand_motion_gates.items():
            if key not in present:
                gate.reset()
        is_moving = bool(hands) and velocity >= threshold
        if is_moving:
            self.last_motion_at = timestamp

        dynamic_emitted = None
        dynamic_stable = None
        finalized = False
        if self.sequence_predictor is not None:
            from ml.predict_sequence import _canonical_hands
            from ml.preprocessing import DUAL_NUM_FEATURES

            from ml.body_context import BODY_NUM_FEATURES, TEMPLE_NUM_FEATURES, BODY_FEATURE_WIDTHS, context_features
            both = self.sequence_predictor.input_size in (DUAL_NUM_FEATURES, *BODY_FEATURE_WIDTHS)
            if hasattr(self.sequence_predictor, "prepare_frame"):
                raw = self.sequence_predictor.prepare_frame(hands, body_context)
            else:
                raw = (context_features(hands, body_context, temple_context=self.sequence_predictor.input_size == TEMPLE_NUM_FEATURES) if self.sequence_predictor.input_size in BODY_FEATURE_WIDTHS
                       else _canonical_hands(hands, both, "single"))
            raw = raw.copy() if raw is not None else None
            self._history.append((timestamp, raw))
            while self._history and timestamp - self._history[0][0] > self.pre_roll_seconds:
                self._history.popleft()
            if self.state == "static" and is_moving:
                self.state = "gesture"
                self._started_at = timestamp
                self._gesture = list(self._history)
                self._result = None
                self._discard_gesture = False
            elif self.state != "static":
                self._gesture.append((timestamp, raw))

            if self.state != "static":
                self.state = "gesture" if is_moving else "settling"
                quiet = timestamp - self.last_motion_at >= self.settle_seconds
                timed_out = timestamp - self._started_at >= self.max_gesture_seconds
                if quiet or timed_out:
                    finalized = True
                    # Replay the completed take at the checkpoint sampling rate, then
                    # resample it to the training sequence length. Missing samples
                    # still pass through the predictor's visibility/gap checks.
                    predictor = self.sequence_predictor
                    try:
                        if quiet and not self._discard_gesture:
                            self._result = predictor.predict_take(self._gesture)
                            self._result_at = timestamp
                            dynamic_stable = self._result.label
                            dynamic_emitted = self._result.label
                    except ValueError:
                        self._result = None
                    self._gesture.clear()
                    self.state = "static" if quiet else "settling"
                    if timed_out and not quiet:
                        # Abort an excessively long take; wait for quiet before rearming.
                        self._started_at = timestamp
                        self._discard_gesture = True

        # Block static confirmation through settling and on the finalization frame.
        suppress = self.state != "static" or finalized
        static_results = (
            self.static_recognizer.update(hands, timestamp, suppress=suppress)
            if self.static_recognizer is not None else []
        )
        visible_result = self._result if self._result_at is not None and timestamp - self._result_at <= 0.6 else None
        return {
            "mode": "dynamic" if suppress or is_moving else "static",
            "state": self.state,
            "velocity": velocity,
            "is_moving": is_moving,
            "static": static_results,
            "dynamic_prediction": visible_result,
            "dynamic_stable": dynamic_stable or (visible_result.label if visible_result else None),
            "dynamic_emitted": dynamic_emitted,
            "dynamic_evaluated": finalized and self._result is not None,
            "dynamic_candidates": getattr(self.sequence_predictor, "last_predictions", {}),
        }



def main() -> None:
    from main import parse_args, run_pipeline

    args = parse_args()
    if not args.sign_model:
        args.sign_model = "ml/models/sign_classifier_canonical.pth"
    run_pipeline(args)


if __name__ == "__main__":
    main()
