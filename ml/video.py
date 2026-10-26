"""Video decoding and trajectory-preserving landmark extraction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ml.preprocessing import fill_landmark_gaps, normalize_landmark_sequence

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
HAND_SELECTIONS = ("single", "Left", "Right", "both")


class LandmarkSequenceCapture:
    """Accumulate detected hands in frame order for one complete gesture."""

    def __init__(self, hand_selection: str = "both", *, body_context: bool = False, temple_context: bool = False) -> None:
        body_context = body_context or temple_context
        self.temple_context = temple_context
        if hand_selection not in HAND_SELECTIONS:
            raise ValueError(f"hand_selection must be one of {HAND_SELECTIONS}")
        if body_context and hand_selection != "both":
            raise ValueError("Body context requires both hand slots")
        self.body_context = body_context
        self._context_frames = []
        self.hand_selection = hand_selection
        self.hand_count = 2 if hand_selection == "both" else 1
        self._frames: list[np.ndarray] = []
        self._valid: list[np.ndarray] = []
        self._handedness: Counter[str] = Counter()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def frames_with_hand(self) -> int:
        return sum(bool(mask.any()) for mask in self._valid)

    @property
    def detection_ratio(self) -> float:
        return self.frames_with_hand / self.frame_count if self.frame_count else 0.0

    @property
    def temple_detection_ratio(self):
        if not self.temple_context or not self._context_frames:
            return 0.0
        return float(np.mean([np.all(frame[150:156].reshape(2, 3)[:, 2] == 1)
                              for frame in self._context_frames]))

    def add_hands(self, hands: Sequence[Any], body_context=None) -> None:
        """Store one sampled frame, including an empty one on detection loss."""
        frame = np.full((self.hand_count, 21, 3), np.nan, dtype=np.float32)
        valid = np.zeros(self.hand_count, dtype=bool)
        if self.hand_selection == "both":
            selected = [
                next((hand for hand in hands if hand.handedness == label), None)
                for label in ("Left", "Right")
            ]
        else:
            selected = [_select_one(hands, self.hand_selection)]
        for slot, hand in enumerate(selected):
            if hand is None:
                continue
            landmarks = np.asarray(hand.to_numpy(), dtype=np.float32)
            if landmarks.shape != (21, 3) or not np.isfinite(landmarks).all():
                raise ValueError("Detected hand must contain 21 finite 3D landmarks")
            frame[slot] = landmarks
            valid[slot] = True
            self._handedness[hand.handedness or "Unknown"] += 1
        if self.body_context:
            from ml.body_context import context_features
            self._context_frames.append(context_features(hands, body_context, temple_context=self.temple_context))
        self._frames.append(frame)
        self._valid.append(valid)

    def finish(
        self,
        target_length: int = 30,
        *,
        min_frames: int = 2,
        min_detection_ratio: float = 0.5,
        target_fps: float | None = 15.0,
        source: str = "<frames>",
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        """Validate and convert the complete take using the video feature contract."""
        if min_frames < 1 or target_length < 1:
            raise ValueError("min_frames and target_length must be positive")
        if not 0 < min_detection_ratio <= 1:
            raise ValueError("min_detection_ratio must be in (0, 1]")
        if self.frame_count < min_frames:
            raise ValueError(
                f"Too few sampled frames: {self.frame_count}; need at least {min_frames}"
            )
        if self.detection_ratio < min_detection_ratio:
            raise ValueError(
                f"Hand detected in only {self.detection_ratio:.1%} of sampled frames; "
                f"need at least {min_detection_ratio:.1%}"
            )
        raw = np.stack(self._frames)
        valid = np.stack(self._valid)
        raw, valid = fill_landmark_gaps(raw, valid)
        if self.body_context:
            from ml.body_context import resample_context
            context = np.stack(self._context_frames)
            if np.mean(context[:, 126:].reshape(len(context), -1, 3)[:, 2:4, 2].all(axis=1)) < min_detection_ratio:
                raise ValueError("Shoulders not visible often enough; keep upper body in view")
            if self.temple_context and self.temple_detection_ratio < min_detection_ratio:
                raise ValueError("Temple references not visible often enough; keep face in view")
            normalized = resample_context(context, target_length)
        else:
            normalized = normalize_landmark_sequence(raw, valid=valid)
            normalized = resample_sequence(normalized, target_length, interpolate=True)
        dominant = self._handedness.most_common(1)[0][0] if self._handedness else ""
        metadata = {
            "source": source,
            "decoded_frames": self.frame_count,
            "frames_with_hand": self.frames_with_hand,
            "detection_ratio": self.detection_ratio,
            "hand_selection": self.hand_selection,
            "body_context": self.body_context,
            "temple_context": self.temple_context,
            "temple_detection_ratio": self.temple_detection_ratio if self.temple_context else None,
            "feature_width": int(normalized.shape[1]),
            "target_fps": target_fps,
        }
        return normalized.astype(np.float32), dominant, metadata


def extract_frames(
    video_path: str | Path,
    target_fps: float | None = 15.0,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Decode a video, optionally sampling it at a lower frame rate."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    if target_fps is not None and (not np.isfinite(target_fps) or target_fps <= 0):
        raise ValueError("target_fps must be positive and finite")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video file: {path}")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        effective_fps = source_fps if target_fps is None else min(float(target_fps), source_fps)
        next_sample_time = 0.0
        frame_index = 0
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_time = frame_index / source_fps
            if frame_time + (0.5 / source_fps) >= next_sample_time:
                frames.append(frame)
                next_sample_time += 1.0 / effective_fps
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames could be read from video: {path}")
    return frames


def resample_sequence(
    sequence: np.ndarray,
    target_length: int = 30,
    *,
    interpolate: bool = False,
) -> np.ndarray:
    """Convert a ``(T, D)`` sequence to a fixed length.

    The compatibility default uniformly downsamples long clips and zero-pads
    short clips. Video dataset preparation uses ``interpolate=True`` to map the
    complete gesture to a fixed timeline without adding artificial stillness.
    """
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D sequence (T, D), got shape {array.shape}")
    if target_length < 1:
        raise ValueError(f"target_length must be positive, got {target_length}")
    if array.shape[0] < 1:
        raise ValueError("Cannot resample an empty sequence")
    if array.shape[0] == target_length:
        return array.copy()
    if interpolate and array.shape[0] > 1:
        old_axis = np.linspace(0.0, 1.0, array.shape[0])
        new_axis = np.linspace(0.0, 1.0, target_length)
        return np.stack(
            [np.interp(new_axis, old_axis, array[:, column]) for column in range(array.shape[1])],
            axis=1,
        ).astype(np.float32)
    if array.shape[0] > target_length:
        indices = np.linspace(0, array.shape[0] - 1, target_length).astype(int)
        return array[indices].copy()
    padding = np.zeros((target_length - array.shape[0], array.shape[1]), dtype=np.float32)
    return np.vstack((array, padding))


def _select_one(hands: Sequence[Any], selection: str) -> Any | None:
    if selection == "single":
        return hands[0] if len(hands) == 1 else None
    return next((hand for hand in hands if hand.handedness == selection), None)


def extract_video_landmarks(
    video_or_frames: str | Path | Sequence[np.ndarray],
    detector: Any | None = None,
    target_fps: float | None = 15.0,
    hand_selection: str = "single",
    target_length: int | None = 30,
    *,
    interpolate: bool = True,
    return_metadata: bool = False,
    body_context: bool = False,
    temple_context: bool = False,
) -> tuple[np.ndarray, str] | tuple[np.ndarray, str, dict[str, Any]]:
    """Extract a normalized landmark trajectory from a video or frame list.

    ``hand_selection='both'`` produces 126 features in stable Left/Right slots;
    the other modes produce 63. Brief missed detections are interpolated, so
    timing is never compressed by silently dropping frames.
    """
    body_context = body_context or temple_context
    if hand_selection not in HAND_SELECTIONS:
        raise ValueError(f"hand_selection must be one of {HAND_SELECTIONS}")
    if isinstance(video_or_frames, (str, Path)):
        frames = extract_frames(video_or_frames, target_fps=target_fps)
        source = str(Path(video_or_frames))
    else:
        frames = list(video_or_frames)
        source = "<frames>"
        if not frames:
            raise ValueError("Frames list must not be empty")

    owns_detector = detector is None
    if detector is None:
        from asignx import HandDetector

        if body_context:
            from asignx.body_detector import BodyHandDetector
            detector = BodyHandDetector(max_num_hands=2)
        else:
            detector = HandDetector(max_num_hands=2)

    capture = LandmarkSequenceCapture(hand_selection, body_context=body_context, temple_context=temple_context)
    fps = float(target_fps or 30.0)
    try:
        for frame_index, frame in enumerate(frames):
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            result, _ = detector.process(frame, draw=False, timestamp_ms=timestamp_ms)
            capture.add_hands(result.all_hands, getattr(result, "body_context", None))
    finally:
        if owns_detector:
            detector.close()

    if capture.frames_with_hand == 0:
        raise ValueError(f"No usable hand landmarks detected in {source}")
    if target_length is None:
        target_length = capture.frame_count
    if interpolate:
        normalized, dominant, metadata = capture.finish(
            target_length,
            min_frames=1,
            min_detection_ratio=1 / capture.frame_count,
            target_fps=target_fps,
            source=source,
        )
    else:
        # Preserve the original padding/downsampling option for existing callers.
        normalized, dominant, metadata = capture.finish(
            capture.frame_count,
            min_frames=1,
            min_detection_ratio=1 / capture.frame_count,
            target_fps=target_fps,
            source=source,
        )
        normalized = resample_sequence(normalized, target_length, interpolate=False)
    if return_metadata:
        return normalized.astype(np.float32), dominant, metadata
    return normalized.astype(np.float32), dominant
