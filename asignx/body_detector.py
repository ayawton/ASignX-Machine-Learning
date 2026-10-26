"""Optional MediaPipe pose/face context alongside the existing hand detector."""
from pathlib import Path
import sys
import cv2
import mediapipe as mp
import numpy as np
from asignx.detector import HandDetector


class BodyHandDetector(HandDetector):
    def __init__(self, context_fps: float = 20.0, **kwargs):
        if not np.isfinite(context_fps) or context_fps <= 0:
            raise ValueError("Body context FPS must be positive")
        self._context_interval_ms = 1000.0 / context_fps
        self._body_cache = np.full((8, 2), np.nan, dtype=np.float32)
        super().__init__(**kwargs)
        self._pose = self._face = None
        self._body_timestamp = -1
        self._body_source_timestamp = -1
        try:
            vision = mp.tasks.vision
            mode = vision.RunningMode.IMAGE if self.static_image_mode else vision.RunningMode.VIDEO
            def options(name):
                path = Path(__file__).with_name("assets") / name
                if not path.exists() and hasattr(sys, "_MEIPASS"):
                    path = Path(sys._MEIPASS) / "asignx/assets" / name
                if not path.exists():
                    raise FileNotFoundError(f"Missing body-context model: {path}")
                return mp.tasks.BaseOptions(model_asset_buffer=path.read_bytes())
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=options("pose_landmarker_lite.task"), running_mode=mode, num_poses=1))
            self._face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=options("face_landmarker.task"), running_mode=mode, num_faces=1))
        except Exception:
            self.close()
            raise

    def process(self, frame, draw=True, timestamp_ms=None):
        result, annotated = super().process(frame, draw=draw, timestamp_ms=timestamp_ms)
        body = np.full((8, 2), np.nan, dtype=np.float32)
        if frame is not None and frame.size:
            stamp = result.frame_timestamp_ms
            refresh = (self.static_image_mode or self._body_source_timestamp < 0
                       or stamp < self._body_source_timestamp
                       or stamp - self._body_source_timestamp >= self._context_interval_ms)
            if refresh:
                self._body_source_timestamp = stamp
                stamp = max(stamp, self._body_timestamp + 1)
                self._body_timestamp = stamp
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pose = self._pose.detect(image) if self.static_image_mode else self._pose.detect_for_video(image, stamp)
                face = self._face.detect(image) if self.static_image_mode else self._face.detect_for_video(image, stamp)
                if pose.pose_landmarks:
                    for slot, index in ((0, 0), (2, 11), (3, 12), (4, 13), (5, 14)):
                        point = pose.pose_landmarks[0][index]
                        if point.visibility >= .5 and point.presence >= .5:
                            body[slot] = (point.x, point.y)
                if face.face_landmarks:
                    point = face.face_landmarks[0][152]
                    body[1] = (point.x, point.y)
                    # Lateral-face proxies from MediaPipe canonical face topology.
                    for slot, index in ((6, 127), (7, 356)):
                        point = face.face_landmarks[0][index]
                        body[slot] = (point.x, point.y)
                self._body_cache = body.copy()
            else:
                # Bounded 50 ms cache at the default rate; hands still run on
                # every frame. This does not extrapolate or hold hand landmarks.
                body = self._body_cache.copy()
            if draw:
                height, width = annotated.shape[:2]
                for index, point in enumerate(body):
                    if np.isfinite(point).all():
                        xy = tuple((point * (width, height)).astype(int))
                        cv2.circle(annotated, xy, 4, (255, 180, 0), -1)
                        from ml.body_context import DETECTED_BODY_NAMES
                        cv2.putText(annotated, DETECTED_BODY_NAMES[index], xy, cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 180, 0), 1)
        result.body_context = body
        return result, annotated

    def close(self):
        for name in ("_pose", "_face"):
            tracker = getattr(self, name, None)
            if tracker is not None:
                tracker.close()
                setattr(self, name, None)
        super().close()
