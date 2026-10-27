"""Versioned two-hand shape + image-plane body context for dynamic signs.

Pose, face and hand depth have different origins; only x/y are compared across
trackers. Hand-local z is retained within each 63-feature hand block.
"""
from __future__ import annotations
import numpy as np
from ml.preprocessing import normalize_landmarks

BODY_CONTEXT_VERSION = "hands_upper_body_xy_v2"
BODY_NAMES = ("nose", "chin", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
TEMPLE_CONTEXT_VERSION = "hands_upper_body_temples_xy_v3"
TEMPLE_NUM_FEATURES = 180
BODY_FEATURE_WIDTHS = (150, TEMPLE_NUM_FEATURES)
DETECTED_BODY_NAMES = BODY_NAMES + ("temple A", "temple B")


def sequence_contract(width):
    from ml.preprocessing import SEQUENCE_PREPROCESSING_VERSION
    if width == TEMPLE_NUM_FEATURES:
        return TEMPLE_CONTEXT_VERSION
    return BODY_CONTEXT_VERSION if width == 150 else SEQUENCE_PREPROCESSING_VERSION


BODY_NUM_FEATURES = 150  # 126 hand-local XYZ + 8 (relative x, relative y, visibility)


def context_features(hands, body: np.ndarray | None, *, temple_context: bool = False) -> np.ndarray:
    """Return finite features, masking unavailable body points explicitly."""
    if temple_context:
        return temple_features(hands, body)
    out = np.zeros(BODY_NUM_FEATURES, dtype=np.float32)
    wrists = np.full((2, 2), np.nan, dtype=np.float32)
    for slot, label in enumerate(("Left", "Right")):
        hand = next((h for h in hands if h.handedness == label), None)
        if hand is not None:
            points = np.asarray(hand.to_numpy(), dtype=np.float32)
            if points.shape != (21, 3) or not np.isfinite(points).all():
                continue
            try:
                out[slot*63:(slot+1)*63] = normalize_landmarks(points)
                wrists[slot] = points[0, :2]
            except ValueError:
                continue
    if body is None:
        return out
    body = np.asarray(body, dtype=np.float32)
    if body.shape == (8, 2):
        body = body[:6]
    if body.shape != (6, 2):
        raise ValueError("Body context must contain six image-plane points (6, 2)")
    if not np.isfinite(body[2:4]).all():
        return out
    origin = body[2:4].mean(axis=0)
    scale = float(np.linalg.norm(body[2] - body[3]))
    if scale < 1e-5:
        return out
    points = np.concatenate((body, wrists))
    valid = np.isfinite(points).all(axis=1)
    context = out[126:].reshape(8, 3)
    context[valid, :2] = (points[valid] - origin) / scale
    context[valid, 2] = 1
    return out


def temple_features(hands, body):
    """Append temple-region positions and hand/head offsets; preserve v2 prefix.

    Face mesh vertices 127 and 356 are lateral-face temple-region proxies, not
    exact anatomical contact points. All cross-tracker geometry uses image x/y.
    Added layout: two shoulder-normalized anchor triples, then eight head-width
    normalized offset triples (Left/Right hand, anchor 127/356, wrist/index tip).
    """
    out = np.zeros(TEMPLE_NUM_FEATURES, dtype=np.float32)
    out[:150] = context_features(hands, body)
    if body is None:
        return out
    body = np.asarray(body, dtype=np.float32)
    if body.shape != (8, 2):
        raise ValueError("Temple mode requires eight detected reference points; re-record with temples enabled")
    shoulders, temples = body[2:4], body[6:8]
    if not np.isfinite(shoulders).all():
        return out
    shoulder_width = float(np.linalg.norm(shoulders[0] - shoulders[1]))
    if shoulder_width < 1e-5:
        return out
    anchors = out[150:156].reshape(2, 3)
    valid = np.isfinite(temples).all(axis=1)
    anchors[valid, :2] = (temples[valid] - shoulders.mean(axis=0)) / shoulder_width
    anchors[valid, 2] = 1
    if not valid.all():
        return out
    head_width = float(np.linalg.norm(temples[0] - temples[1]))
    if head_width < 1e-5:
        anchors[:] = 0
        return out
    offsets = out[156:].reshape(2, 2, 2, 3)
    for slot, label in enumerate(("Left", "Right")):
        hand = next((h for h in hands if h.handedness == label), None)
        if hand is None or not np.any(out[slot*63:(slot+1)*63]):
            continue
        points = np.asarray(hand.to_numpy(), dtype=np.float32)[[0,8], :2]
        offsets[slot, :, :, :2] = (points[None, :, :] - temples[:, None, :]) / head_width
        offsets[slot, :, :, 2] = 1
    return out


def resample_context(features, target_length):
    # Visibility stays binary; interpolate coordinates only when both endpoints
    # are visible, so tracking loss never becomes a fabricated point.
    features = np.asarray(features, dtype=np.float32)
    positions = np.linspace(0, len(features)-1, target_length)
    lo = np.floor(positions).astype(int)
    hi = np.ceil(positions).astype(int)
    alpha = (positions-lo)[:, None]
    out = (features[lo]*(1-alpha) + features[hi]*alpha).astype(np.float32)
    src = features[:, 126:].reshape(len(features), -1, 3)
    dst = out[:, 126:].reshape(target_length, -1, 3)
    valid = (src[lo, :, 2] == 1) & (src[hi, :, 2] == 1)
    dst[:, :, 2] = valid
    dst[:, :, :2] *= valid[:, :, None]
    for slot in range(2):
        valid_hand = np.any(features[lo, slot*63:(slot+1)*63], axis=1) & np.any(features[hi, slot*63:(slot+1)*63], axis=1)
        out[:, slot*63:(slot+1)*63] *= valid_hand[:, None]
    return out


def resample_context_timestamps(features, timestamps, target_fps, max_interval):
    """Sample a completed take on a uniform clock without inventing absences.

    Interpolate coordinates only between visible observations. Missing detections
    remain missing; a camera gap beyond the permitted interval rejects the take.
    """
    features=np.asarray(features,dtype=np.float32)
    timestamps=np.asarray(timestamps,dtype=np.float64)
    if (features.ndim!=2 or features.shape[1] not in BODY_FEATURE_WIDTHS
            or timestamps.ndim!=1 or len(features)!=len(timestamps) or not len(features)
            or not np.isfinite(features).all() or not np.isfinite(timestamps).all()):
        raise ValueError('Invalid body-context observations or timestamps')
    if not np.isfinite(target_fps) or target_fps<=0 or not np.isfinite(max_interval) or max_interval<=0:
        raise ValueError('Sampling rate and permitted camera interval must be positive')
    intervals=np.diff(timestamps)
    if np.any(intervals<=0):
        raise ValueError('Live sequence timestamps must increase and be finite')
    if np.any(intervals>max_interval+1e-8):
        raise ValueError('Camera gap is too long for a complete gesture')
    # Work relative to the first capture to avoid large-clock rounding errors.
    stamps=timestamps-timestamps[0]
    grid=np.arange(int(np.floor(stamps[-1]*target_fps+1e-8))+1,dtype=np.float64)/target_fps
    hi=np.searchsorted(stamps,grid,side='left').clip(0,len(stamps)-1)
    previous=np.maximum(hi-1,0)
    hi=np.where(np.isclose(stamps[previous],grid,rtol=0,atol=1e-8),previous,hi)
    lo=np.maximum(hi-1,0)
    exact=np.isclose(stamps[hi],grid,rtol=0,atol=1e-8)
    lo=np.where(exact,hi,lo)
    span=stamps[hi]-stamps[lo]
    alpha=np.divide(grid-stamps[lo],span,out=np.zeros_like(grid),where=span>0)[:,None]
    out=(features[lo]*(1-alpha)+features[hi]*alpha).astype(np.float32)
    src=features[:,126:].reshape(len(features),-1,3)
    dst=out[:,126:].reshape(len(out),-1,3)
    valid=(src[lo,:,2]==1)&(src[hi,:,2]==1)
    dst[:,:,2]=valid
    dst[:,:,:2]*=valid[:,:,None]
    for slot in range(2):
        present=np.any(features[lo,slot*63:(slot+1)*63]!=0,axis=1)&np.any(features[hi,slot*63:(slot+1)*63]!=0,axis=1)
        out[:,slot*63:(slot+1)*63]*=present[:,None]
    return out
