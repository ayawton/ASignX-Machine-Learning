"""Shared, versioned landmark feature contracts, kinematic enrichers, and 3D augmentations."""

from __future__ import annotations

import numpy as np

PREPROCESSING_VERSION = "wrist_max_radius_v1"
STATIC_HAND_LEGACY = "none"
STATIC_HAND_CANONICAL = "left_x_reflection_v1"
SEQUENCE_PREPROCESSING_VERSION = "sequence_anchor_trajectory_v2"
NUM_FEATURES = 63
DUAL_NUM_FEATURES = 126

# Kinematic feature dimensions per hand:
# 63 normalized coords + 15 joint flexion cosines + 10 fingertip pairwise distances + 3 palm normal
NUM_JOINT_ANGLES = 15
NUM_FINGERTIP_PAIRS = 10
NUM_PALM_NORMAL = 3
KINEMATIC_EXTRA_FEATURES = NUM_JOINT_ANGLES + NUM_FINGERTIP_PAIRS + NUM_PALM_NORMAL  # 28
KINEMATIC_NUM_FEATURES = NUM_FEATURES + KINEMATIC_EXTRA_FEATURES  # 91
DUAL_KINEMATIC_NUM_FEATURES = KINEMATIC_NUM_FEATURES * 2  # 182

# Upper-body reference keypoints (Nose, Mouth, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow)
UPPER_BODY_KEYPOINTS = 6
UPPER_BODY_FEATURES = UPPER_BODY_KEYPOINTS * 3  # 18

FEATURE_COLUMNS = [f"{axis}{index}" for index in range(21) for axis in "xyz"]

# 15 anatomical joint triplets (parent, joint, child) across all 5 fingers
JOINT_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    # Thumb
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # Index
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # Middle
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # Ring
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # Pinky
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
)

# All 10 unique pairs of fingertips (Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20)
FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
FINGERTIP_PAIRS: tuple[tuple[int, int], ...] = (
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (8, 16), (8, 20),
    (12, 16), (12, 20),
    (16, 20),
)


def _landmark_array(landmarks: np.ndarray | object) -> np.ndarray:
    if hasattr(landmarks, "to_numpy"):
        landmarks = landmarks.to_numpy()
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


def canonicalize_static_features(
    features: np.ndarray,
    handedness: str | None,
    mode: str = STATIC_HAND_CANONICAL,
) -> np.ndarray:
    """Reflect Left wrist-relative x coordinates into the Right representation.

    Labels must come from the same detector/image convention during extraction
    and inference. This preserves y/z and orientation; it is not palm alignment.
    CSV coordinates remain in their original, noncanonical representation.
    """
    result = validate_features(features).copy()
    if mode == STATIC_HAND_LEGACY:
        return result
    if mode != STATIC_HAND_CANONICAL:
        raise ValueError(f"Unsupported static hand preprocessing: {mode}")
    if handedness not in ("Left", "Right"):
        raise ValueError("Static hand canonicalization requires Left or Right handedness")
    if handedness == "Left":
        result[0::3] *= -1
    result[result == 0] = 0  # Stable fingerprints, including signed zero.
    return result


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


def compute_kinematic_features(normalized_features: np.ndarray) -> np.ndarray:
    """Enrich 63-dim normalized landmarks with 28 kinematic features -> 91 dims per hand.

    Accepts shapes:
    - ``(63,)`` -> ``(91,)``
    - ``(N, 63)`` -> ``(N, 91)``
    - ``(T, 126)`` -> ``(T, 182)`` (dual-hand sequence)
    """
    arr = np.asarray(normalized_features, dtype=np.float32)
    single_1d = arr.ndim == 1
    if single_1d:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] not in (NUM_FEATURES, DUAL_NUM_FEATURES):
        raise ValueError(
            f"Expected feature width {NUM_FEATURES} or {DUAL_NUM_FEATURES}, got {arr.shape}"
        )

    if arr.shape[1] == DUAL_NUM_FEATURES:
        left = compute_kinematic_features(arr[:, :NUM_FEATURES])
        right = compute_kinematic_features(arr[:, NUM_FEATURES:])
        out = np.concatenate([left, right], axis=1)
        return out[0] if single_1d else out

    n_samples = arr.shape[0]
    points = arr.reshape(n_samples, 21, 3)

    # 1. 15 Finger Joint Flexion Angles (cosines in [-1, 1])
    angles = np.zeros((n_samples, NUM_JOINT_ANGLES), dtype=np.float32)
    for idx, (a, b, c) in enumerate(JOINT_TRIPLETS):
        v1 = points[:, a] - points[:, b]
        v2 = points[:, c] - points[:, b]
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        denom = norm1 * norm2
        valid = denom > 1e-6
        cos_theta = np.zeros(n_samples, dtype=np.float32)
        cos_theta[valid] = np.clip(
            np.sum(v1[valid] * v2[valid], axis=1) / denom[valid], -1.0, 1.0
        )
        angles[:, idx] = cos_theta

    # 2. 10 Pairwise Fingertip Distances
    distances = np.zeros((n_samples, NUM_FINGERTIP_PAIRS), dtype=np.float32)
    for idx, (i, j) in enumerate(FINGERTIP_PAIRS):
        distances[:, idx] = np.linalg.norm(points[:, i] - points[:, j], axis=1)

    # 3. 3D Unit Palm Normal Vector: (Index_MCP - Wrist) x (Pinky_MCP - Wrist)
    v_index = points[:, 5] - points[:, 0]
    v_pinky = points[:, 17] - points[:, 0]
    cross = np.cross(v_index, v_pinky)
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = np.divide(
        cross,
        np.maximum(cross_norm, 1e-6),
        out=np.zeros_like(cross),
        where=cross_norm > 1e-6,
    ).astype(np.float32)

    enriched = np.concatenate([arr, angles, distances, normals], axis=1).astype(np.float32)
    return enriched[0] if single_1d else enriched


def augment_landmarks_3d(
    features: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    max_z_deg: float = 18.0,
    max_tilt_deg: float = 10.0,
    jitter_std: float = 0.004,
) -> np.ndarray:
    """Apply 3D geometric rotation and sensor jitter to a 63-dim normalized feature vector."""
    rng = rng or np.random.default_rng()
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape != (NUM_FEATURES,):
        raise ValueError(f"Expected ({NUM_FEATURES},) features, got {arr.shape}")
    points = arr.reshape(21, 3).copy()
    if np.allclose(points, 0.0):
        return arr.copy()

    # Random 3D Euler angles (pitch, yaw, roll/in-plane)
    rx = np.deg2rad(rng.uniform(-max_tilt_deg, max_tilt_deg))
    ry = np.deg2rad(rng.uniform(-max_tilt_deg, max_tilt_deg))
    rz = np.deg2rad(rng.uniform(-max_z_deg, max_z_deg))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    rot = rot_z @ rot_y @ rot_x

    rotated = points @ rot.T
    if jitter_std > 0:
        jitter = rng.normal(0.0, jitter_std, size=rotated.shape).astype(np.float32)
        jitter[0] = 0.0  # Keep wrist anchored at origin
        rotated += jitter

    return normalize_landmarks(rotated)


def augment_sequence_3d(
    sequence: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    max_z_deg: float = 15.0,
    max_tilt_deg: float = 8.0,
    jitter_std: float = 0.003,
    speed_range: tuple[float, float] = (0.85, 1.2),
) -> np.ndarray:
    """Apply consistent 3D rotation, jitter, and temporal speed warping to a sequence (T, 63|126)."""
    rng = rng or np.random.default_rng()
    seq = np.asarray(sequence, dtype=np.float32).copy()
    if seq.ndim != 2 or seq.shape[1] not in (NUM_FEATURES, DUAL_NUM_FEATURES):
        raise ValueError(f"Expected (T, 63) or (T, 126), got {seq.shape}")

    steps, width = seq.shape
    hands = width // NUM_FEATURES
    points = seq.reshape(steps, hands, 21, 3)

    # Consistent clip-wide 3D rotation
    rx = np.deg2rad(rng.uniform(-max_tilt_deg, max_tilt_deg))
    ry = np.deg2rad(rng.uniform(-max_tilt_deg, max_tilt_deg))
    rz = np.deg2rad(rng.uniform(-max_z_deg, max_z_deg))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    rot = rot_z @ rot_y @ rot_x

    valid_mask = np.any(np.abs(points) > 1e-7, axis=(2, 3))
    rotated = points @ rot.T
    if jitter_std > 0:
        noise = rng.normal(0.0, jitter_std, size=rotated.shape).astype(np.float32)
        rotated = np.where(valid_mask[:, :, None, None], rotated + noise, 0.0)

    out = rotated.reshape(steps, width)

    # Temporal speed warping via linear interpolation
    if steps >= 4 and speed_range[0] < speed_range[1]:
        speed = float(rng.uniform(speed_range[0], speed_range[1]))
        warped_len = max(2, int(round(steps / speed)))
        src_t = np.linspace(0.0, 1.0, steps)
        mid_t = np.linspace(0.0, 1.0, warped_len)
        warped = np.stack([np.interp(mid_t, src_t, out[:, col]) for col in range(width)], axis=1)
        dst_t = np.linspace(0.0, 1.0, steps)
        out = np.stack(
            [np.interp(dst_t, mid_t, warped[:, col]) for col in range(width)], axis=1
        ).astype(np.float32)

    return out


def normalize_holistic_landmarks(
    hand_landmarks: np.ndarray,
    upper_body_landmarks: np.ndarray | None = None,
) -> np.ndarray:
    """Compute body-anchored hand features using 6 upper-body reference points.

    ``upper_body_landmarks`` has shape ``(6, 3)`` ordered as:
    ``[NOSE, MOUTH_CENTER, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW]``.
    Returns ``(81,)`` features: 63 wrist-normalized hand features + 18 shoulder-normalized
    hand-to-body anchor vectors.
    """
    hand_norm = normalize_landmarks(hand_landmarks)
    if upper_body_landmarks is None:
        return np.concatenate([hand_norm, np.zeros(UPPER_BODY_FEATURES, dtype=np.float32)])

    body = np.asarray(upper_body_landmarks, dtype=np.float32)
    if body.shape != (UPPER_BODY_KEYPOINTS, 3) or not np.isfinite(body).all():
        raise ValueError(f"Expected ({UPPER_BODY_KEYPOINTS}, 3) finite upper-body landmarks")

    wrist = _landmark_array(hand_landmarks)[0]
    shoulder_dist = float(np.linalg.norm(body[2] - body[3]))
    scale = shoulder_dist if shoulder_dist > 1e-6 else 1.0
    relative_vectors = ((wrist[None, :] - body) / scale).reshape(UPPER_BODY_FEATURES).astype(np.float32)
    return np.concatenate([hand_norm, relative_vectors])


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
