"""
ASignX Hand Recognition Module - Detector.

Uses MediaPipe Tasks to detect hand presence and extract 21 3D landmarks in
real time with duplicate-hand suppression and handedness stabilization.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any, Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from asignx.models import (
    HAND_LANDMARK_NAMES,
    HandDetectionResult,
    HandLandmarks,
    Landmark,
)


class HandDetector:
    """MediaPipe Tasks-based real-time hand detection engine."""

    _HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    )

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        duplicate_distance_threshold: float = 0.12,
        duplicate_iou_threshold: float = 0.30,
        left_hand_motion_distance_threshold: float = 0.45,
        min_presence_confidence: float = 0.45,
        recovery_enabled: bool = True,
        recovery_interval_ms: int = 150,
    ) -> None:
        """Initialize the MediaPipe Tasks hand detector.

        ``model_complexity`` is accepted for compatibility with the old CLI,
        but the Tasks API uses the bundled ``hand_landmarker.task`` model.
        """
        for value in (min_detection_confidence, min_tracking_confidence, min_presence_confidence):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Hand confidence thresholds must be between 0 and 1")
        if recovery_interval_ms < 1:
            raise ValueError("Recovery interval must be positive")
        self.min_presence_confidence = min_presence_confidence
        self.recovery_enabled = recovery_enabled
        self.recovery_interval_ms = recovery_interval_ms
        self._recovery_hands = None
        self._last_recovery_ms = -float("inf")
        self._last_full_hand_ms = -float("inf")
        self._recent_hand_count = 0
        self.recovery_attempts = self.recovery_successes = 0
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.duplicate_distance_threshold = duplicate_distance_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.left_hand_motion_distance_threshold = left_hand_motion_distance_threshold

        import sys

        model_path = Path(__file__).with_name("assets") / "hand_landmarker.task"
        if not model_path.exists() and hasattr(sys, "_MEIPASS"):
            alt_path = Path(getattr(sys, "_MEIPASS")) / "asignx" / "assets" / "hand_landmarker.task"
            if alt_path.exists():
                model_path = alt_path
        if not model_path.exists():
            raise FileNotFoundError(
                "Missing MediaPipe hand landmark model: "
                f"{model_path}. Download hand_landmarker.task into asignx/assets/."
            )

        self._last_timestamp_ms = -1
        if getattr(sys, "frozen", False):
            base_options = mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes())
        else:
            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        self._hand_base_options = base_options
        vision = mp.tasks.vision
        running_mode = (
            vision.RunningMode.IMAGE
            if self.static_image_mode
            else vision.RunningMode.VIDEO
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)

        self._history: Deque[List[Tuple[np.ndarray, str]]] = deque(maxlen=5)
        self._left_track: Deque[np.ndarray] = deque(maxlen=2)

    def _is_duplicate(self, hand_a: HandLandmarks, hand_b: HandLandmarks) -> bool:
        """Determine if two hand detections represent the same physical hand."""
        pts_a = hand_a.to_numpy()[:, :2]
        pts_b = hand_b.to_numpy()[:, :2]
        # Proximity or overlapping boxes alone do not imply a duplicate: two
        # genuine hands can touch or overlap during signs such as HELP.
        extent = max(float(np.linalg.norm(np.ptp(pts_a, axis=0))),
                     float(np.linalg.norm(np.ptp(pts_b, axis=0))), 1e-6)
        tolerance = min(self.duplicate_distance_threshold, extent * 0.25)
        distances = np.linalg.norm(pts_a - pts_b, axis=1)
        return bool(np.sqrt(np.mean(distances ** 2)) < tolerance
                    and np.quantile(distances, .9) < tolerance * 1.5)

    def _suppress_duplicates(
        self, candidates: List[Tuple[HandLandmarks, Any]]
    ) -> List[Tuple[HandLandmarks, Any]]:
        """Filter out duplicate ghost hands detected on the same physical hand."""
        if len(candidates) <= 1:
            return candidates

        sorted_candidates = sorted(candidates, key=lambda pair: pair[0].score, reverse=True)
        survivors: List[Tuple[HandLandmarks, Any]] = []

        for cand in sorted_candidates:
            if not any(self._is_duplicate(cand[0], kept[0]) for kept in survivors):
                survivors.append(cand)

        return survivors

    def _stabilize_handedness(
        self, candidates: List[Tuple[HandLandmarks, Any]]
    ) -> List[Tuple[HandLandmarks, Any]]:
        """Stabilize handedness predictions and resolve conflicting labels."""
        if not candidates:
            self._history.append([])
            self._left_track.clear()
            return candidates

        self._reacquire_fast_left_hand(candidates)

        if len(candidates) == 2:
            h0, h1 = candidates[0][0], candidates[1][0]
            if h0.handedness is not None and h0.handedness == h1.handedness:
                c0_x = float(np.mean([lm.x for lm in h0.landmarks]))
                c1_x = float(np.mean([lm.x for lm in h1.landmarks]))
                if c0_x < c1_x:
                    candidates[0][0].handedness = "Right"
                    candidates[1][0].handedness = "Left"
                else:
                    candidates[0][0].handedness = "Left"
                    candidates[1][0].handedness = "Right"

        current_records: List[Tuple[np.ndarray, str]] = []
        for hand_obj, _ in candidates:
            centroid = np.mean([(lm.x, lm.y) for lm in hand_obj.landmarks], axis=0)
            if hand_obj.handedness and self._history:
                prev_frame = self._history[-1]
                for prev_centroid, prev_label in prev_frame:
                    if float(np.linalg.norm(centroid - prev_centroid)) < 0.15:
                        if hand_obj.score < 0.85:
                            hand_obj.handedness = prev_label
                        break
            if hand_obj.handedness:
                current_records.append((centroid, hand_obj.handedness))

        self._history.append(current_records)
        self._update_left_track(candidates)
        return candidates

    @staticmethod
    def _centroid(hand: HandLandmarks) -> np.ndarray:
        """Return the normalized 2D landmark centroid for a hand."""
        return np.mean([(lm.x, lm.y) for lm in hand.landmarks], axis=0)

    def _reacquire_fast_left_hand(
        self, candidates: List[Tuple[HandLandmarks, Any]]
    ) -> None:
        """Recover the left label when a fast movement defeats normal matching."""
        if not self._left_track:
            return

        predicted = self._left_track[-1].copy()
        if len(self._left_track) == 2:
            predicted += self._left_track[-1] - self._left_track[-2]
        predicted = np.clip(predicted, 0.0, 1.0)

        nearest_hand, nearest_distance = min(
            ((hand, float(np.linalg.norm(self._centroid(hand) - predicted))) for hand, _ in candidates),
            key=lambda item: item[1],
        )
        labels = {hand.handedness for hand, _ in candidates}
        if (
            nearest_distance <= self.left_hand_motion_distance_threshold
            and nearest_hand.handedness != "Left"
            and "Left" not in labels
            # Motion history may repair uncertain labels, not override strong
            # evidence when a different physical hand enters the old location.
            and nearest_hand.score < 0.85
        ):
            nearest_hand.handedness = "Left"

    def _update_left_track(self, candidates: List[Tuple[HandLandmarks, Any]]) -> None:
        """Store only confirmed, current left-hand positions for next frame."""
        for hand, _ in candidates:
            if hand.handedness == "Left":
                self._left_track.append(self._centroid(hand))
                return
        self._left_track.clear()

    @staticmethod
    def _category_value(category: Any, attr: str, default: Any) -> Any:
        """Read both Tasks category objects and simple test doubles."""
        return getattr(category, attr, default)

    def _detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> Any:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self.static_image_mode:
            return self._hands.detect(mp_image)

        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return self._hands.detect_for_video(mp_image, timestamp_ms)

    def _recover_missing_hands(self, rgb_frame, results, timestamp_ms):
        """Retry current image on brief tracking loss; never reuse old landmarks."""
        primary = list(getattr(results, "hand_landmarks", []) or [])
        if (self.static_image_mode or not self.recovery_enabled
                or len(primary) >= self._recent_hand_count
                or timestamp_ms - self._last_full_hand_ms > 500
                or timestamp_ms - self._last_recovery_ms < self.recovery_interval_ms):
            return results
        self._last_recovery_ms = timestamp_ms
        if self._recovery_hands is None:
            vision = mp.tasks.vision
            self._recovery_hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=self._hand_base_options, running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_presence_confidence))
        self.recovery_attempts += 1
        retry = self._recovery_hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
        recovered = list(getattr(retry, "hand_landmarks", []) or [])
        if not recovered:
            return results
        self.recovery_successes += 1
        categories = list(getattr(results, "handedness", []) or [])
        categories += [[] for _ in range(len(primary) - len(categories))]
        retry_categories = list(getattr(retry, "handedness", []) or [])
        retry_categories += [[] for _ in range(len(recovered) - len(retry_categories))]
        return SimpleNamespace(hand_landmarks=primary + recovered,
                               handedness=categories + retry_categories)

    def _draw_hand(self, frame: np.ndarray, hand: HandLandmarks) -> None:
        height, width = frame.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand.landmarks]

        for start, end in self._HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        for point in points:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

    def process(
        self,
        frame: np.ndarray,
        draw: bool = True,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[HandDetectionResult, np.ndarray]:
        """Process a single BGR video frame and extract hand landmarks."""
        if frame is None or frame.size == 0:
            return (
                HandDetectionResult(
                    hand_detected=False,
                    hand=None,
                    all_hands=[],
                    frame_timestamp_ms=timestamp_ms or int(time.time() * 1000),
                ),
                frame,
            )

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        annotated_frame = frame.copy() if draw else frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._detect(rgb_frame, timestamp_ms)
        stamp = timestamp_ms if self.static_image_mode else self._last_timestamp_ms
        results = self._recover_missing_hands(rgb_frame, results, stamp)
        rgb_frame.flags.writeable = True

        raw_candidates: List[Tuple[HandLandmarks, Any]] = []
        hand_landmarks = getattr(results, "hand_landmarks", []) or []
        handedness = getattr(results, "handedness", []) or []

        for idx, raw_landmarks in enumerate(hand_landmarks):
            landmarks_list: List[Landmark] = []
            for lm_id, lm in enumerate(raw_landmarks):
                lm_name = (
                    HAND_LANDMARK_NAMES[lm_id]
                    if lm_id < len(HAND_LANDMARK_NAMES)
                    else f"LANDMARK_{lm_id}"
                )
                landmarks_list.append(
                    Landmark(
                        id=lm_id,
                        name=lm_name,
                        x=float(lm.x),
                        y=float(lm.y),
                        z=float(lm.z),
                    )
                )

            hand_label = None
            hand_score = 0.0
            if idx < len(handedness) and handedness[idx]:
                category = handedness[idx][0]
                hand_label = self._category_value(category, "category_name", None)
                hand_score = float(self._category_value(category, "score", 0.0))

            hand_obj = HandLandmarks(
                landmarks=landmarks_list,
                handedness=hand_label,
                score=hand_score,
            )
            raw_candidates.append((hand_obj, raw_landmarks))

        valid_candidates = self._suppress_duplicates(raw_candidates)[:self.max_num_hands]
        if not self.static_image_mode:
            valid_candidates = self._stabilize_handedness(valid_candidates)

        detected_hands: List[HandLandmarks] = [pair[0] for pair in valid_candidates]
        if len(detected_hands) >= self._recent_hand_count or stamp - self._last_full_hand_ms > 500:
            self._recent_hand_count = len(detected_hands)
            self._last_full_hand_ms = stamp

        if draw:
            for hand, _ in valid_candidates:
                self._draw_hand(annotated_frame, hand)

        detection_result = HandDetectionResult(
            hand_detected=bool(detected_hands),
            hand=detected_hands[0] if detected_hands else None,
            all_hands=detected_hands,
            frame_timestamp_ms=timestamp_ms,
        )
        return detection_result, annotated_frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if getattr(self, "_recovery_hands", None) is not None:
            self._recovery_hands.close()
            self._recovery_hands = None
        if hasattr(self, "_hands") and self._hands is not None:
            self._hands.close()
            self._hands = None

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
