"""Record complete moving signs from a webcam as labeled landmark sequences."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import time
import uuid

import cv2

from asignx import HandDetector, WebcamStream
from ml.sequence_dataset import (
    SequenceSample,
    read_sequence_dir,
    save_sequence_file,
)
from ml.video import HAND_SELECTIONS, LandmarkSequenceCapture


def _label_folder(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip()).strip("_") or "sign"


def _save_take(
    capture: LandmarkSequenceCapture,
    *,
    label: str,
    output_dir: Path,
    sequence_length: int,
    target_fps: float,
    min_frames: int,
    min_detection_ratio: float,
    duration_seconds: float,
) -> Path:
    features, handedness, metadata = capture.finish(
        sequence_length,
        min_frames=min_frames,
        min_detection_ratio=min_detection_ratio,
        target_fps=target_fps,
        source="webcam",
    )
    metadata["duration_seconds"] = round(duration_seconds, 3)
    metadata["capture_method"] = "manual_webcam"
    folder = output_dir / _label_folder(label)
    session_id = uuid.uuid4().hex
    destination = folder / f"{_label_folder(label)}_{session_id}.npz"
    while destination.exists():
        session_id = uuid.uuid4().hex
        destination = folder / f"{_label_folder(label)}_{session_id}.npz"
    sample = SequenceSample(
        label=label,
        session_id=session_id,
        handedness=handedness,
        features=features,
        metadata=metadata,
    )
    return save_sequence_file(destination, sample)


def collect_sequence(
    label: str,
    *,
    output_dir: str | Path = "ml/data/sequences",
    hand_selection: str = "both",
    sequence_length: int = 30,
    target_fps: float = 15.0,
    min_frames: int = 10,
    min_detection_ratio: float = 0.5,
    max_seconds: float = 10.0,
    takes: int = 0,
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    exposure: float | None = -5.0,
) -> list[Path]:
    """Open the camera and save one session for each completed gesture take.

    Space starts/stops a take, R discards the active take, and Q/Esc exits.
    A maximum duration ends and saves a take automatically. Rejected takes
    remain unsaved and can be recorded again.
    """
    label = label.strip()
    if not label:
        raise ValueError("label must not be empty")
    if hand_selection not in HAND_SELECTIONS:
        raise ValueError(f"hand_selection must be one of {HAND_SELECTIONS}")
    if sequence_length < 2 or min_frames < 2:
        raise ValueError("sequence_length and min_frames must be at least 2")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive and finite")
    if not 0 < min_detection_ratio <= 1:
        raise ValueError("min_detection_ratio must be in (0, 1]")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    if takes < 0 or width < 1 or height < 1:
        raise ValueError("takes must be nonnegative and camera dimensions positive")
    if max_seconds * target_fps < min_frames - 1:
        raise ValueError("max_seconds is too short to collect min_frames at target_fps")

    output = Path(output_dir)
    if any(output.rglob("*.npz")):
        existing_samples = read_sequence_dir(output)
        expected_width = 126 if hand_selection == "both" else 63
        if existing_samples[0].features.shape != (sequence_length, expected_width):
            raise ValueError(
                "Existing sequences use a different hand mode or sequence length. "
                "Use matching settings or a different --output directory."
            )
    saved: list[Path] = []
    existing = len(list((output / _label_folder(label)).glob("*.npz")))
    interval = 1.0 / target_fps
    capture: LandmarkSequenceCapture | None = None
    started_at = 0.0
    next_sample_at = 0.0
    message = "SPACE: start a complete gesture"
    window = "ASignX - Capture Moving Signs"
    print(f"Label: {label} | hand selection: {hand_selection} | existing takes: {existing}")
    print("SPACE start/save | R discard current take | Q/Esc exit")

    try:
        with WebcamStream(
            camera_index=camera_index,
            width=width,
            height=height,
            target_fps=max(30, math.ceil(target_fps)),
            threaded=True,
            exposure=exposure,
        ) as camera:
            with HandDetector(max_num_hands=2) as detector:
                for frame in camera.stream():
                    now = time.monotonic()
                    result, annotated = detector.process(frame, draw=True)
                    if capture is not None and now >= next_sample_at:
                        capture.add_hands(result.all_hands)
                        next_sample_at = now + interval

                    if capture is not None and now - started_at >= max_seconds:
                        try:
                            path = _save_take(
                                capture,
                                label=label,
                                output_dir=output,
                                sequence_length=sequence_length,
                                target_fps=target_fps,
                                min_frames=min_frames,
                                min_detection_ratio=min_detection_ratio,
                                duration_seconds=now - started_at,
                            )
                            saved.append(path)
                            message = f"Saved {path.name}"
                            print(f"Saved {path} ({capture.detection_ratio:.0%} detection)")
                        except ValueError as error:
                            message = f"Take rejected: {error}"
                            print(message)
                        capture = None

                    state = "RECORDING" if capture is not None else "READY"
                    count = capture.frame_count if capture is not None else 0
                    quality = capture.detection_ratio if capture is not None else 0.0
                    lines = [
                        f"Label: {label} | {state} | saved: {existing + len(saved)}",
                        f"Frames: {count} | visible: {quality:.0%}",
                        "SPACE start/save | R discard | Q exit",
                        message,
                    ]
                    for row, line in enumerate(lines):
                        cv2.putText(
                            annotated,
                            line[:100],
                            (12, 28 + row * 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0) if capture is not None else (0, 255, 255),
                            2,
                        )
                    cv2.imshow(window, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        if capture is not None:
                            print("Discarded unfinished take")
                            capture = None
                        break
                    if key in (ord("r"), ord("R")):
                        if capture is not None:
                            capture = None
                            message = "Take discarded; press SPACE to start again"
                        else:
                            message = "No active take; press SPACE to start"
                        print(message)
                    elif key == 32:
                        if capture is None:
                            capture = LandmarkSequenceCapture(hand_selection)
                            started_at = time.monotonic()
                            next_sample_at = started_at
                            message = "Perform the complete gesture, then press SPACE"
                        else:
                            try:
                                path = _save_take(
                                    capture,
                                    label=label,
                                    output_dir=output,
                                    sequence_length=sequence_length,
                                    target_fps=target_fps,
                                    min_frames=min_frames,
                                    min_detection_ratio=min_detection_ratio,
                                    duration_seconds=time.monotonic() - started_at,
                                )
                                saved.append(path)
                                message = f"Saved {path.name}"
                                print(f"Saved {path} ({capture.detection_ratio:.0%} detection)")
                            except ValueError as error:
                                message = f"Take rejected: {error}"
                                print(message)
                            capture = None
                    if takes and len(saved) >= takes:
                        break
                if capture is not None:
                    print("Discarded unfinished take")
    except KeyboardInterrupt:
        print("Capture interrupted")
    finally:
        cv2.destroyAllWindows()
    print(f"Saved {len(saved)} new take(s) for {label}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Sign label, e.g. J, Z, HELLO")
    parser.add_argument("--output", default="ml/data/sequences")
    parser.add_argument("--hand", choices=HAND_SELECTIONS, default="both")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--min-detection-ratio", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--takes", type=int, default=0, help="Stop after this many saved takes (0: unlimited)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--exposure", default="-5.0", help="Camera exposure or 'auto'")
    args = parser.parse_args()
    exposure = None if args.exposure.lower() == "auto" else float(args.exposure)
    collect_sequence(
        args.label,
        output_dir=args.output,
        hand_selection=args.hand,
        sequence_length=args.sequence_length,
        target_fps=args.target_fps,
        min_frames=args.min_frames,
        min_detection_ratio=args.min_detection_ratio,
        max_seconds=args.max_seconds,
        takes=args.takes,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        exposure=exposure,
    )


if __name__ == "__main__":
    main()
