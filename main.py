"""
ASignX Hand Recognition Module - Main Application.

Launches the real-time webcam pipeline:
Webcam -> Video Frame -> MediaPipe Hands -> Hand Detection -> 21 Hand Landmarks
Supports dual-hand tracking and threaded capture for >=30 FPS performance.
"""

from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import datetime
from pathlib import Path
import sys
import time
import cv2

from asignx import CameraError, HandDetector, VisualDebugger, WebcamStream


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASignX Hand Recognition Module - Real-time Hand & Landmark Tracker"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Webcam frame width (default: 640 for >=30 FPS)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Webcam frame height (default: 480 for >=30 FPS)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to track (default: 2)",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1],
        default=1,
        help="MediaPipe Hands model complexity: 0=Lite (fastest), 1=Full (default: 1)",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.6,
        help="Minimum hand detection confidence [0.0 - 1.0] (default: 0.6; better rapid reacquisition)",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum hand tracking confidence [0.0 - 1.0] (default: 0.5)",
    )
    parser.add_argument(
        "--exposure",
        type=str,
        default="-5.0",
        help="Webcam exposure: '-5.0' (locks 30 FPS), '-6.0' (60 FPS), or 'auto' (default: -5.0)",
    )
    parser.add_argument(
        "--no-threads",
        action="store_true",
        help="Disable background threaded camera reader (falls back to synchronous)",
    )
    parser.add_argument(
        "--print-stream",
        action="store_true",
        help="Continuously print structured landmark tree to console",
    )
    parser.add_argument(
        "--sign-model",
        help="Optional static-sign .pth checkpoint for live classification",
    )
    parser.add_argument(
        "--sign-confidence",
        type=float,
        default=0.8,
        help="Minimum static-sign confidence (default: 0.8)",
    )
    parser.add_argument("--sign-stable-frames", type=int, default=5)
    parser.add_argument("--sign-release-frames", type=int, default=10)
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    print("=" * 60)
    print(" ASignX - Hand Recognition Module")
    print(f" Real-Time Hand Landmark Tracking (up to {args.max_hands} hands)")
    print("=" * 60)

    sign_recognizer = None
    if args.sign_model:
        from ml.predict import SignPredictor, SignRecognizer

        sign_recognizer = SignRecognizer(
            SignPredictor(args.sign_model, args.sign_confidence),
            stable_frames=args.sign_stable_frames,
            release_frames=args.sign_release_frames,
        )

    try:
        exposure_val = None if args.exposure.lower() == "auto" else float(args.exposure)
        camera = WebcamStream(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            threaded=not args.no_threads,
            exposure=exposure_val,
        )
        camera.start()
    except CameraError as e:
        print(f"\n[ERROR] Camera initialization failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    try:
        detector = HandDetector(
            static_image_mode=False,
            max_num_hands=args.max_hands,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
    except Exception:
        camera.stop()
        raise
    visualizer = VisualDebugger()

    window_name = "ASignX - Hand Recognition Module"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    res_w, res_h = camera.resolution
    mode_str = "Threaded (high-FPS)" if not args.no_threads else "Synchronous"
    exp_str = f"Exposure: {camera.exposure}" if camera.exposure is not None else "Auto-Exposure"
    print(f"[INFO] Camera initialized ({res_w}x{res_h} | {mode_str} | {exp_str})")
    print(f"[INFO] Tracking configured for up to {args.max_hands} hands (model complexity: {args.model_complexity})")
    print("[INFO] Controls:")
    print("       'q' / ESC : Exit application")
    print("       'd'       : Toggle landmark drawing")
    print("       'h'       : Toggle HUD overlay")
    print("       'e'       : Cycle exposure (30 FPS [-5.0] / 60 FPS [-6.0] / Auto)")
    print("       'p'       : Print structured landmark tree to console")
    print("       's'       : Save frame snapshot and JSON landmark coordinates")
    print("-" * 60)

    # FPS Calculation metrics
    prev_time = time.time()
    fps_smooth = 0.0
    alpha = 0.15  # Smoothing factor for FPS
    draw_landmarks = True

    try:
        for frame in camera.stream():
            # Process frame through MediaPipe Hands pipeline
            result, annotated_frame = detector.process(
                frame,
                draw=draw_landmarks,
            )

            if sign_recognizer is not None:
                for recognition in sign_recognizer.update(result.all_hands):
                    if recognition.emitted_label:
                        print(f"[SIGN] {recognition.hand}: {recognition.emitted_label}")

            # Compute smoothed FPS
            now = time.time()
            elapsed = now - prev_time
            prev_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps_smooth = (
                    current_fps if fps_smooth == 0.0 else (alpha * current_fps + (1 - alpha) * fps_smooth)
                )

            # Render visual debugging HUD overlay
            display_frame = visualizer.render_overlay(
                annotated_frame,
                result=result,
                fps=fps_smooth,
            )

            # If continuous printing is enabled
            if args.print_stream and result.hand_detected and result.all_hands:
                print(result.to_tree_string())
                print("-" * 40)

            # Display frame in OpenCV GUI window
            cv2.imshow(window_name, display_frame)

            # Handle keyboard input (1ms wait)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or Esc
                print("\n[INFO] Exiting...")
                break
            elif key in (ord("d"), ord("D")):
                draw_landmarks = not draw_landmarks
                print(f"[INFO] Landmark drawing toggled: {'ON' if draw_landmarks else 'OFF'}")
            elif key in (ord("h"), ord("H")):
                visualizer.show_hud = not visualizer.show_hud
                print(f"[INFO] HUD overlay toggled: {'ON' if visualizer.show_hud else 'OFF'}")
            elif key in (ord("e"), ord("E")):
                # Cycle exposure: -5.0 (30 FPS) -> -6.0 (60 FPS) -> None (Auto) -> -5.0
                if camera.exposure == -5.0:
                    new_exp = -6.0
                    label = "60 FPS mode (-6.0)"
                elif camera.exposure == -6.0:
                    new_exp = None
                    label = "Driver Auto-Exposure mode"
                else:
                    new_exp = -5.0
                    label = "30 FPS locked mode (-5.0)"
                camera.set_exposure(new_exp)
                print(f"[INFO] Camera exposure changed to: {label}")
            elif key in (ord("p"), ord("P")):
                print("\n--- Current Frame Structured Output ---")
                print(result.to_tree_string())
                print("---------------------------------------\n")
            elif key in (ord("s"), ord("S")):
                # Save snapshot and JSON landmark data
                output_dir = Path("snapshots")
                output_dir.mkdir(exist_ok=True)
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = output_dir / f"snapshot_{timestamp_str}.jpg"
                json_path = output_dir / f"snapshot_{timestamp_str}.json"

                cv2.imwrite(str(img_path), display_frame)
                json_path.write_text(result.to_json())
                print(f"[INFO] Saved snapshot to: {img_path}")
                print(f"[INFO] Saved landmarks JSON to: {json_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        # Graceful cleanup
        print("[INFO] Releasing camera and detector resources...")
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
