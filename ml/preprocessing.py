"""Shared, versioned landmark feature contracts."""

from __future__ import annotations

import numpy as np

PREPROCESSING_VERSION = "wrist_max_radius_v1"
SEQUENCE_PREPROCESSING_VERSION = "sequence_anchor_trajectory_v2"
NUM_FEATURES = 63
DUAL_NUM_FEATURES = 126
FEATURE_COLUMNS = [f"{axis}{index}" for index in range(21) for axis in "xyz"]


def _landmark_array(landmarks: np.ndarray) -> np.ndarray:
    array = np.asarray(landmarks, dtype=np.float32)
    if array.shape != (21, 3) or not np.isfinite(array).all():
        raise ValueError("Expected 21 finite (x, y, z) landmarks")
    return array


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a static pose around its wrist while preserving orientation."""
    array = _landmark_array(landmarks)
    centered = array - array[0]
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 1e-8:
        raise ValueError("Degenerate hand: landmarks have no usable scale")
    return (centered / scale).reshape(NUM_FEATURES).astype(np.float32)


def validate_features(features: np.ndarray) -> np.ndarray:
    """Validate an already-normalized static feature vector."""
    array = np.asarray(features, dtype=np.float32)
    if array.shape != (NUM_FEATURES,) or not np.isfinite(array).all():
        raise ValueError("Expected 63 finite normalized landmark features")
    points = array.reshape(21, 3)
    if not np.allclose(points[0], 0.0, atol=1e-5):
        raise ValueError("Features do not match wrist_max_radius_v1 normalization")
    radius = float(np.linalg.norm(points, axis=1).max())
    if not np.isclose(radius, 1.0, atol=1e-4):
        raise ValueError("Features do not match wrist_max_radius_v1 normalization")
    return array


def normalize_landmark_sequence(
    frames: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize a raw landmark clip without deleting its hand trajectory.

    ``frames`` is ``(T, 21, 3)`` for one hand or ``(T, 2, 21, 3)`` for
    canonical ``Left, Right`` slots. A single anchor and scale are used for the
    entire clip, so wrist displacement remains visible to the sequence model.
    Missing frames/hands are represented as zero feature rows via ``valid``.
    """
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim == 3:
        array = array[:, None, :, :]
        squeeze_hand = True
    elif array.ndim == 4:
        squeeze_hand = False
    else:
        raise ValueError("Expected sequence shape (T, 21, 3) or (T, H, 21, 3)")
    if array.shape[0] == 0 or array.shape[2:] != (21, 3) or array.shape[1] not in (1, 2):
        raise ValueError("Expected a non-empty sequence with one or two 21x3 hands")

    if valid is None:
        mask = np.isfinite(array).all(axis=(2, 3))
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.ndim == 1:
            mask = mask[:, None]
        if mask.shape != array.shape[:2]:
            raise ValueError(f"valid mask must have shape {array.shape[:2]}")
        mask &= np.isfinite(array).all(axis=(2, 3))
    if not mask.any():
        raise ValueError("Sequence contains no detected hands")

    # Use the earliest visible wrist as a common clip origin. This retains
    # both whole-hand movement and the relative placement of two hands.
    first_time, first_hand = np.argwhere(mask)[0]
    origin = array[first_time, first_hand, 0].copy()
    radii: list[float] = []
    for time_index, hand_index in np.argwhere(mask):
        hand = array[time_index, hand_index]
        radii.append(float(np.linalg.norm(hand - hand[0], axis=1).max()))
    scale = float(np.median([radius for radius in radii if radius > 1e-8])) if any(
        radius > 1e-8 for radius in radii
    ) else 0.0
    if scale <= 1e-8:
        raise ValueError("Degenerate sequence: landmarks have no usable scale")

    normalized = np.zeros_like(array, dtype=np.float32)
    for time_index, hand_index in np.argwhere(mask):
        normalized[time_index, hand_index] = (
            array[time_index, hand_index] - origin
        ) / scale
    flattened = normalized.reshape(array.shape[0], array.shape[1] * NUM_FEATURES)
    if squeeze_hand:
        return flattened[:, :NUM_FEATURES]
    return flattened


def fill_landmark_gaps(
    frames: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate missed detections without shortening the gesture timeline.

    Interior gaps are linearly interpolated. Leading/trailing gaps hold the
    nearest detection. A hand that is never detected remains missing.
    """
    array = np.asarray(frames, dtype=np.float32).copy()
    mask = np.asarray(valid, dtype=bool).copy()
    if array.ndim != 4 or array.shape[2:] != (21, 3) or mask.shape != array.shape[:2]:
        raise ValueError("Expected frames (T, H, 21, 3) and valid mask (T, H)")
    timeline = np.arange(array.shape[0], dtype=np.float32)
    for hand_index in range(array.shape[1]):
        known = np.flatnonzero(mask[:, hand_index])
        if not len(known):
            continue
        values = array[known, hand_index].reshape(len(known), -1)
        filled = np.stack(
            [np.interp(timeline, known, values[:, column]) for column in range(values.shape[1])],
            axis=1,
        )
        array[:, hand_index] = filled.reshape(array.shape[0], 21, 3)
        mask[:, hand_index] = True
    return array, mask
