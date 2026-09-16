"""
ASignX Hand Recognition Module - Detector.

Wraps Google MediaPipe Hands solution to detect hand presence and extract
21 3D landmarks in real time with duplicate-hand suppression and handedness
stabilization to prevent ghost double-detections and flickering.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Deque, List, Optional, Tuple
import cv2
import numpy as np

import mediapipe as mp

from asignx.models import (
    HAND_LANDMARK_NAMES,
    HandDetectionResult,
    HandLandmarks,
    Landmark,
)


class HandDetector:
    """MediaPipe Hands-based real-time hand detection and landmark extraction engine.

    Pipeline:
        Video Frame (BGR) -> RGB Conversion -> MediaPipe Hands ->
        Spatial Duplicate Suppression -> Handedness Stabilization ->
        21 Hand Landmarks -> Structured Coordinates
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        duplicate_distance_threshold: float = 0.12,
        duplicate_iou_threshold: float = 0.30,
        left_hand_motion_distance_threshold: float = 0.45,
    ) -> None:
        """Initialize MediaPipe Hands detector.

        Args:
            static_image_mode: Whether to treat the input images as a batch of
                static images or a continuous video stream (default False).
            max_num_hands: Maximum number of hands to track (default 2).
            min_detection_confidence: Confidence threshold for hand detection (default 0.6).
            min_tracking_confidence: Confidence threshold for tracking (default 0.5).
            model_complexity: Complexity of landmark model: 0 or 1 (default 1).
            duplicate_distance_threshold: Normalized distance threshold below which
                two hand proposals are considered duplicates of the same hand (default 0.12).
            duplicate_iou_threshold: Bounding box IoU threshold for duplicate
            suppression (default 0.30).
            left_hand_motion_distance_threshold: Largest normalized image-space
                jump accepted when reacquiring the left hand between consecutive
                frames (default 0.45). This keeps fast left-hand sweeps from
                being relabelled as Right after the tracker reacquires them.
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.duplicate_distance_threshold = duplicate_distance_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.left_hand_motion_distance_threshold = left_hand_motion_distance_threshold

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        # Temporal history for handedness stabilization
        # Stores recent (centroid, handedness) records across frames
        self._history: Deque[List[Tuple[np.ndarray, str]]] = deque(maxlen=5)
        # Recent left-hand positions are kept separately so a rapid sweep can
        # be associated by its predicted direction rather than a tiny static
        # proximity window.
        self._left_track: Deque[np.ndarray] = deque(maxlen=2)

    def _is_duplicate(self, hand_a: HandLandmarks, hand_b: HandLandmarks) -> bool:
        """Determine if two hand detections represent the same physical hand."""
        # 1. Wrist distance (Landmark 0)
        w_a = hand_a[0]
        w_b = hand_b[0]
        d_wrist = np.sqrt((w_a.x - w_b.x) ** 2 + (w_a.y - w_b.y) ** 2)

        # 2. Centroid distance
        pts_a = np.array([(lm.x, lm.y) for lm in hand_a.landmarks], dtype=np.float32)
        pts_b = np.array([(lm.x, lm.y) for lm in hand_b.landmarks], dtype=np.float32)
        c_a = np.mean(pts_a, axis=0)
        c_b = np.mean(pts_b, axis=0)
        d_centroid = float(np.linalg.norm(c_a - c_b))

        # 3. Bounding box IoU
        min_a = np.min(pts_a, axis=0)
        max_a = np.max(pts_a, axis=0)
        min_b = np.min(pts_b, axis=0)
        max_b = np.max(pts_b, axis=0)

        inter_min = np.maximum(min_a, min_b)
        inter_max = np.minimum(max_a, max_b)
        inter_dims = np.maximum(0.0, inter_max - inter_min)
        inter_area = float(inter_dims[0] * inter_dims[1])

        area_a = float(max(0.0, max_a[0] - min_a[0]) * max(0.0, max_a[1] - min_a[1]))
        area_b = float(max(0.0, max_b[0] - min_b[0]) * max(0.0, max_b[1] - min_b[1]))
        union_area = area_a + area_b - inter_area
        iou = (inter_area / union_area) if union_area > 1e-6 else 0.0

        return bool(
            d_wrist < self.duplicate_distance_threshold
            or d_centroid < self.duplicate_distance_threshold
            or iou > self.duplicate_iou_threshold
        )

    def _suppress_duplicates(
        self, candidates: List[Tuple[HandLandmarks, Any]]
    ) -> List[Tuple[HandLandmarks, Any]]:
        """Filter out duplicate ghost hands detected on the same physical hand."""
        if len(candidates) <= 1:
            return candidates

        # Sort by confidence score descending
        sorted_candidates = sorted(candidates, key=lambda pair: pair[0].score, reverse=True)
        survivors: List[Tuple[HandLandmarks, Any]] = []

        for cand in sorted_candidates:
            hand_obj = cand[0]
            is_dup = False
            for kept in survivors:
                if self._is_duplicate(hand_obj, kept[0]):
                    is_dup = True
                    break
            if not is_dup:
                survivors.append(cand)

        return survivors

    def _stabilize_handedness(
        self, candidates: List[Tuple[HandLandmarks, Any]]
    ) -> List[Tuple[HandLandmarks, Any]]:
        """Stabilize handedness predictions and resolve conflicting duplicate labels."""
        if not candidates:
            # Do not carry a left-hand identity through a detection gap. A
            # later hand must be observed again instead of inheriting a stale
            # label from several frames ago.
            self._history.append([])
            self._left_track.clear()
            return candidates

        self._reacquire_fast_left_hand(candidates)

        # If 2 hands are detected and MediaPipe gave both the same label, disambiguate by X coordinate
        if len(candidates) == 2:
            h0, h1 = candidates[0][0], candidates[1][0]
            if h0.handedness is not None and h0.handedness == h1.handedness:
                # Compare average X coordinates
                c0_x = float(np.mean([lm.x for lm in h0.landmarks]))
                c1_x = float(np.mean([lm.x for lm in h1.landmarks]))
                # The hand further to the left of the image (smaller X)
                # In standard webcam view: smaller X is user's physical Right hand (mirrored view) or Left
                if c0_x < c1_x:
                    candidates[0][0].handedness = "Right"
                    candidates[1][0].handedness = "Left"
                else:
                    candidates[0][0].handedness = "Left"
                    candidates[1][0].handedness = "Right"

        # Apply temporal smoothing against recent frame history
        current_records: List[Tuple[np.ndarray, str]] = []
        for hand_obj, _ in candidates:
            centroid = np.mean([(lm.x, lm.y) for lm in hand_obj.landmarks], axis=0)
            if hand_obj.handedness and self._history:
                # Find closest hand in previous frame
                prev_frame = self._history[-1]
                for prev_centroid, prev_label in prev_frame:
                    if float(np.linalg.norm(centroid - prev_centroid)) < 0.15:
                        # If the hand stayed at the same location, keep stable handedness unless score is low
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
        """Recover the left label when a fast movement defeats normal matching.

        MediaPipe normally tracks from the preceding frame. During a fast sweep
        it may redetect the same hand and briefly classify it as Right. Match
        the closest current proposal to the extrapolated left-hand position,
        using a bounded movement gate. This changes only the handedness label;
        it never returns cached landmarks as a fake detection.
        """
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
        ):
            nearest_hand.handedness = "Left"

    def _update_left_track(self, candidates: List[Tuple[HandLandmarks, Any]]) -> None:
        """Store only confirmed, current left-hand positions for next frame."""
        for hand, _ in candidates:
            if hand.handedness == "Left":
                self._left_track.append(self._centroid(hand))
                return
        self._left_track.clear()

    def process(
        self,
        frame: np.ndarray,
        draw: bool = True,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[HandDetectionResult, np.ndarray]:
        """Process a single BGR video frame and extract hand landmarks.

        Args:
            frame: Input video frame in OpenCV BGR format (H, W, 3).
            draw: Whether to render landmark points and skeleton connections
                on the returned annotated frame.
            timestamp_ms: Optional frame timestamp in milliseconds.

        Returns:
            Tuple of:
                - HandDetectionResult: Structured detection result with landmarks
                  or None when no hand is present.
                - np.ndarray: Output frame with landmark visualizations.
        """
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

        # Output frame for drawing overlays
        annotated_frame = frame.copy() if draw else frame

        # Convert BGR to RGB for MediaPipe inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Performance optimization: pass by reference
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        raw_candidates: List[Tuple[HandLandmarks, Any]] = []

        if results.multi_hand_landmarks:
            # Extract handedness if available
            handedness_labels = []
            if results.multi_handedness:
                for classification in results.multi_handedness:
                    label = classification.classification[0].label
                    score = classification.classification[0].score
                    handedness_labels.append((label, score))

            for idx, raw_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks_list: List[Landmark] = []
                for lm_id, lm in enumerate(raw_landmarks.landmark):
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
                if idx < len(handedness_labels):
                    hand_label, hand_score = handedness_labels[idx]

                hand_obj = HandLandmarks(
                    landmarks=landmarks_list,
                    handedness=hand_label,
                    score=hand_score,
                )
                raw_candidates.append((hand_obj, raw_landmarks))

        # 1. Spatial Non-Maximum Suppression (Eliminates ghost duplicate hands on same physical hand)
        valid_candidates = self._suppress_duplicates(raw_candidates)

        # 2. Handedness Stabilization (Eliminates handedness jitter & duplicate-label conflicts)
        if not self.static_image_mode:
            valid_candidates = self._stabilize_handedness(valid_candidates)

        detected_hands: List[HandLandmarks] = [pair[0] for pair in valid_candidates]

        # 3. Draw ONLY the surviving, valid hands (Prevents duplicate visual skeleton flickering)
        if draw:
            for _, raw_landmarks in valid_candidates:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    raw_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        if detected_hands:
            primary_hand = detected_hands[0]
            detection_result = HandDetectionResult(
                hand_detected=True,
                hand=primary_hand,
                all_hands=detected_hands,
                frame_timestamp_ms=timestamp_ms,
            )
        else:
            detection_result = HandDetectionResult(
                hand_detected=False,
                hand=None,
                all_hands=[],
                frame_timestamp_ms=timestamp_ms,
            )

        return detection_result, annotated_frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_hands") and self._hands is not None:
            self._hands.close()
            self._hands = None

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
