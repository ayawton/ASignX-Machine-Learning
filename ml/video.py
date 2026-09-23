"""Video decoding and trajectory-preserving landmark extraction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ml.preprocessing import fill_landmark_gaps, NUM_FEATURES, normalize_landmark_sequence

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
HAND_SELECTIONS = ("single", "Left", "Right", "both")


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
) -> tuple[np.ndarray, str] | tuple[np.ndarray, str, dict[str, Any]]:
    """Extract a normalized landmark trajectory from a video or frame list.

    ``hand_selection='both'`` produces 126 features in stable Left/Right slots;
    the other modes produce 63. Missed detections remain zero feature rows, so
    timing is never compressed by silently dropping frames.
    """
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

        detector = HandDetector(max_num_hands=2)

    hand_count = 2 if hand_selection == "both" else 1
    raw = np.full((len(frames), hand_count, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((len(frames), hand_count), dtype=bool)
    handedness: Counter[str] = Counter()
    fps = float(target_fps or 30.0)
    try:
        for frame_index, frame in enumerate(frames):
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            result, _ = detector.process(frame, draw=False, timestamp_ms=timestamp_ms)
            hands = result.all_hands
            if hand_selection == "both":
                for slot, label in enumerate(("Left", "Right")):
                    hand = next((item for item in hands if item.handedness == label), None)
                    if hand is not None:
                        raw[frame_index, slot] = hand.to_numpy()
                        valid[frame_index, slot] = True
                        handedness[label] += 1
            else:
                hand = _select_one(hands, hand_selection)
                if hand is not None:
                    raw[frame_index, 0] = hand.to_numpy()
                    valid[frame_index, 0] = True
                    handedness[hand.handedness or "Unknown"] += 1
    finally:
        if owns_detector:
            detector.close()

    if not valid.any():
        raise ValueError(f"No usable hand landmarks detected in {source}")
    original_valid = valid.copy()
    raw, filled_valid = fill_landmark_gaps(raw, valid)
    normalized = normalize_landmark_sequence(raw, valid=filled_valid)
    if hand_count == 1:
        normalized = normalized[:, :NUM_FEATURES]
    if target_length is not None:
        normalized = resample_sequence(
            normalized, target_length=target_length, interpolate=interpolate
        )
    dominant = handedness.most_common(1)[0][0] if handedness else ""
    metadata = {
        "source": source,
        "decoded_frames": len(frames),
        "frames_with_hand": int(original_valid.any(axis=1).sum()),
        "detection_ratio": float(original_valid.any(axis=1).mean()),
        "hand_selection": hand_selection,
        "feature_width": int(normalized.shape[1]),
        "target_fps": target_fps,
    }
    if return_metadata:
        return normalized.astype(np.float32), dominant, metadata
    return normalized.astype(np.float32), dominant
