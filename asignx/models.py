"""
ASignX Hand Recognition Module - Data Models.

Defines structured data classes for 21 hand landmarks, hand collections,
and detection results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# MediaPipe standard 21 hand landmark names mapped by index (0-20)
HAND_LANDMARK_NAMES: Tuple[str, ...] = (
    "WRIST",                # 0
    "THUMB_CMC",            # 1
    "THUMB_MCP",            # 2
    "THUMB_IP",             # 3
    "THUMB_TIP",            # 4
    "INDEX_FINGER_MCP",     # 5
    "INDEX_FINGER_PIP",     # 6
    "INDEX_FINGER_DIP",     # 7
    "INDEX_FINGER_TIP",     # 8
    "MIDDLE_FINGER_MCP",    # 9
    "MIDDLE_FINGER_PIP",    # 10
    "MIDDLE_FINGER_DIP",    # 11
    "MIDDLE_FINGER_TIP",    # 12
    "RING_FINGER_MCP",      # 13
    "RING_FINGER_PIP",      # 14
    "RING_FINGER_DIP",      # 15
    "RING_FINGER_TIP",      # 16
    "PINKY_MCP",            # 17
    "PINKY_PIP",            # 18
    "PINKY_DIP",            # 19
    "PINKY_TIP",            # 20
)


@dataclass(frozen=True)
class Landmark:
    """Represents an individual hand landmark with normalized 3D coordinates.

    Attributes:
        id: Landmark index from 0 to 20.
        name: Anatomical name of the landmark.
        x: Normalized horizontal position [0.0, 1.0].
        y: Normalized vertical position [0.0, 1.0].
        z: Normalized depth coordinate relative to the wrist.
    """

    id: int
    name: str
    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return (x, y, z) tuple."""
        return (self.x, self.y, self.z)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize landmark to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "z": round(self.z, 6),
        }

    def __repr__(self) -> str:
        return f"Landmark {self.id} ({self.name}) → (x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


@dataclass
class HandLandmarks:
    """Represents a complete set of 21 landmarks for a single detected hand.

    Attributes:
        landmarks: List of exactly 21 Landmark instances, indexed 0 to 20.
        handedness: Predicted hand side ('Left', 'Right', or None).
        score: Confidence score of the handedness prediction.
    """

    landmarks: List[Landmark]
    handedness: Optional[str] = None
    score: float = 0.0

    def __post_init__(self) -> None:
        if len(self.landmarks) != 21:
            raise ValueError(
                f"HandLandmarks must contain exactly 21 landmarks, got {len(self.landmarks)}"
            )
        # Ensure landmarks are ordered by id 0..20
        self.landmarks = sorted(self.landmarks, key=lambda lm: lm.id)

    def __getitem__(self, idx: int) -> Landmark:
        """Access landmark by its integer index (0-20)."""
        return self.landmarks[idx]

    def __len__(self) -> int:
        return len(self.landmarks)

    def to_numpy(self) -> np.ndarray:
        """Convert landmarks to a (21, 3) numpy array [[x, y, z], ...].

        Useful for downstream preprocessing, feature extraction, and ML pipelines.
        """
        return np.array([[lm.x, lm.y, lm.z] for lm in self.landmarks], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hand data to dictionary structure."""
        return {
            "handedness": self.handedness,
            "score": round(self.score, 4),
            "landmarks": [lm.to_dict() for lm in self.landmarks],
        }

    def to_tree_string(self, ascii_only: Optional[bool] = None) -> str:
        """Render structured tree representation of hand landmarks.

        Args:
            ascii_only: If True, uses standard ASCII branch symbols (|--, \\--)
                and ASCII arrows (->). If None (default), auto-detects console encoding.
        """
        if ascii_only is None:
            encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
            try:
                "└── ├── →".encode(encoding)
                ascii_only = False
            except (UnicodeEncodeError, LookupError):
                ascii_only = True

        label = f"Hand ({self.handedness})" if self.handedness else "Hand"
        lines = [label]
        total = len(self.landmarks)
        arrow = "->" if ascii_only else "→"
        for i, lm in enumerate(self.landmarks):
            if ascii_only:
                prefix = "\\-- " if i == total - 1 else "|-- "
            else:
                prefix = "└── " if i == total - 1 else "├── "
            lines.append(
                f"{prefix}Landmark {lm.id:>2} ({lm.name:<17}) {arrow} ({lm.x:+.4f}, {lm.y:+.4f}, {lm.z:+.4f})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.to_tree_string()


@dataclass
class HandDetectionResult:
    """Encapsulates the complete result of processing a single video frame.

    Attributes:
        hand_detected: True if at least one hand was detected, False otherwise.
        hand: Primary HandLandmarks instance if hand_detected is True, else None.
        all_hands: List of all detected hands (up to max_num_hands, e.g. 2).
        frame_timestamp_ms: Timestamp of the processed frame in milliseconds.
    """

    hand_detected: bool
    hand: Optional[HandLandmarks] = None
    all_hands: List[HandLandmarks] = field(default_factory=list)
    frame_timestamp_ms: int = 0
    body_context: Optional[np.ndarray] = None

    @property
    def hands(self) -> List[HandLandmarks]:
        """Alias for all_hands."""
        return self.all_hands

    @property
    def hands_count(self) -> int:
        """Total number of detected hands."""
        return len(self.all_hands)

    @property
    def left_hand(self) -> Optional[HandLandmarks]:
        """Retrieve detected Left hand, or None if not detected."""
        for h in self.all_hands:
            if h.handedness == "Left":
                return h
        return None

    @property
    def right_hand(self) -> Optional[HandLandmarks]:
        """Retrieve detected Right hand, or None if not detected."""
        for h in self.all_hands:
            if h.handedness == "Right":
                return h
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detection result to dictionary.

        Guarantees that when no hand is detected, 'hand' is null (None).
        Provides access to 'hands' list as well as 'left_hand' and 'right_hand'.
        """
        return {
            "hand_detected": self.hand_detected,
            "hands_count": len(self.all_hands),
            "hand": self.hand.to_dict() if self.hand is not None else None,
            "hands": [h.to_dict() for h in self.all_hands],
            "left_hand": self.left_hand.to_dict() if self.left_hand is not None else None,
            "right_hand": self.right_hand.to_dict() if self.right_hand is not None else None,
            "frame_timestamp_ms": self.frame_timestamp_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export detection result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_tree_string(self, ascii_only: Optional[bool] = None) -> str:
        """Render human-readable tree string of the detection outcome.

        If multiple hands are detected, formats a nested tree with each hand.

        Args:
            ascii_only: If True, uses standard ASCII branch symbols (|--, \\--)
                and ASCII arrows (->). If None (default), auto-detects console encoding.
        """
        if ascii_only is None:
            encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
            try:
                "└── ├── │ →".encode(encoding)
                ascii_only = False
            except (UnicodeEncodeError, LookupError):
                ascii_only = True

        arrow = "->" if ascii_only else "→"
        if not self.hand_detected or not self.all_hands:
            return f"Hand {arrow} null"

        if len(self.all_hands) == 1:
            return self.all_hands[0].to_tree_string(ascii_only=ascii_only)

        # 2 or more hands detected
        tree_lines = [f"Hands ({len(self.all_hands)} detected)"]
        total_hands = len(self.all_hands)
        for h_idx, h in enumerate(self.all_hands):
            is_last_hand = (h_idx == total_hands - 1)
            hand_prefix = "\\-- " if (ascii_only and is_last_hand) else ("└── " if is_last_hand else ("|-- " if ascii_only else "├── "))
            continuation = "    " if is_last_hand else ("|   " if ascii_only else "│   ")

            hand_label = f"Hand ({h.handedness})" if h.handedness else f"Hand {h_idx+1}"
            tree_lines.append(f"{hand_prefix}{hand_label}")

            total_lm = len(h.landmarks)
            for lm_idx, lm in enumerate(h.landmarks):
                is_last_lm = (lm_idx == total_lm - 1)
                lm_prefix = "\\-- " if (ascii_only and is_last_lm) else ("└── " if is_last_lm else ("|-- " if ascii_only else "├── "))
                tree_lines.append(
                    f"{continuation}{lm_prefix}Landmark {lm.id:>2} ({lm.name:<17}) {arrow} ({lm.x:+.4f}, {lm.y:+.4f}, {lm.z:+.4f})"
                )

        return "\n".join(tree_lines)

    def __repr__(self) -> str:
        return self.to_tree_string()
