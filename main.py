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
        default=800,
        help="Webcam frame width (default: 800)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Webcam frame height (default: 600)",
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
        help="Legacy compatibility option; Tasks uses the bundled hand model for both values",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum hand detection confidence [0.0 - 1.0] (default: 0.5)",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum hand tracking confidence [0.0 - 1.0] (default: 0.5)",
    )
    parser.add_argument("--min-presence-confidence", type=float, default=0.45,
                        help="Minimum hand presence confidence (default: 0.45)")
    parser.add_argument("--no-hand-recovery", action="store_true",
                        help="Disable rate-limited full-image retry after hand tracking loss")
    parser.add_argument(
        "--exposure",
        type=str,
        default="auto",
        help="Webcam exposure value or 'auto' (default: auto; driver-dependent)",
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
    parser.add_argument("--sign-calibration", help="Optional validation-fitted threshold JSON matching the static checkpoint")
    parser.add_argument("--sign-release-frames", type=int, default=10)
    parser.add_argument(
        "--sequence-model",
        help="Optional dynamic sequence .pth checkpoint for live motion-sign classification",
    )
    parser.add_argument(
        "--sequence-confidence",
        type=float,
        default=0.8,
        help="Minimum dynamic sequence confidence (default: 0.8)",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.045,
        help="Normalized velocity threshold separating static and dynamic signs (default: 0.045)",
    )
    parser.add_argument('--gesture-min-excursion', type=float, default=.35,
                        help='Minimum completed-take movement in palm widths; requires validation tuning')
    parser.add_argument('--gesture-min-motion-seconds', type=float, default=.12,
                        help='Minimum sustained movement time before a dynamic sign may be accepted')
    parser.add_argument('--gesture-form-checks',action='store_true',
                        help='Pilot J/Z handshape checks and forehead-salute-only HELLO proximity check')
    parser.add_argument('--hold-static-after-gesture',action='store_true',
                        help='Prevent a completed dynamic signs ending pose from immediately emitting a static letter')
    parser.add_argument('--diagnostics-dir', type=Path,
                        help='Opt-in timing/tracking logs; keys 7=J, 8=Z, 9=HELLO, 0=rest')
    parser.add_argument('--diagnostic-person-id',
                        help='Optional signer identifier recorded in diagnostic metadata')
    parser.add_argument('--capture-gesture-inputs', action='store_true',
                        help='Also save completed landmark takes for exact diagnostic replay; requires --diagnostics-dir')
    args = parser.parse_args(argv)
    if args.capture_gesture_inputs and not args.diagnostics_dir:
        parser.error('--capture-gesture-inputs requires --diagnostics-dir')
    return args


def run_pipeline(args: argparse.Namespace) -> None:
    print("=" * 60)
    print(" ASignX - Hand Recognition Module")
    print(f" Real-Time Hand Landmark Tracking (up to {args.max_hands} hands)")
    print("=" * 60)

    sign_recognizer = None
    unified_recognizer = None
    sequence_model_path = getattr(args, "sequence_model", None)
    if sequence_model_path:
        from ml.predict import SignPredictor, UnifiedSignRecognizer
        from ml.predict_sequence import SequencePredictor
        from ml.gesture_acceptance import MotionAcceptance
        from ml.gesture_forms import GestureFormAcceptance

        static_pred = SignPredictor(args.sign_model, args.sign_confidence,
                                    calibration_path=getattr(args, "sign_calibration", None)) if args.sign_model else None
        seq_pred = SequencePredictor(sequence_model_path, getattr(args, "sequence_confidence", 0.8))
        print(f"[INFO] Dynamic model: {sequence_model_path}")
        print(f"[INFO] Dynamic vocabulary: {', '.join(seq_pred.labels)}")
        print(f"[INFO] Dynamic features: {seq_pred.input_size}; preprocessing: {seq_pred.metadata.get('preprocessing', 'unknown')}")
        unified_recognizer = UnifiedSignRecognizer(
            static_predictor=static_pred,
            sequence_predictor=seq_pred,
            velocity_threshold=getattr(args, "motion_threshold", 0.025),
            stable_frames=args.sign_stable_frames,
            release_frames=args.sign_release_frames,
            motion_acceptance=MotionAcceptance(
                min_excursion=getattr(args, 'gesture_min_excursion', .35),
                min_motion_seconds=getattr(args, 'gesture_min_motion_seconds', .12)),
            form_acceptance=GestureFormAcceptance() if args.gesture_form_checks else None,
            hold_static_after_gesture=args.hold_static_after_gesture,
        )
    elif args.sign_model:
        from ml.predict import SignPredictor, SignRecognizer

        sign_recognizer = SignRecognizer(
            SignPredictor(args.sign_model, args.sign_confidence,
                          calibration_path=getattr(args, "sign_calibration", None)),
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
        if exposure_val is None:
            camera.set_exposure(None)
    except CameraError as e:
        print(f"\n[ERROR] Camera initialization failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    try:
        from ml.body_context import BODY_FEATURE_WIDTHS
        from asignx.body_detector import BodyHandDetector
        detector_type = BodyHandDetector if unified_recognizer is not None and unified_recognizer.sequence_predictor.input_size in BODY_FEATURE_WIDTHS else HandDetector
        detector = detector_type(
            static_image_mode=False,
            max_num_hands=args.max_hands,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            min_presence_confidence=getattr(args, "min_presence_confidence", 0.45),
            recovery_enabled=not getattr(args, "no_hand_recovery", False),
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
    print(f"[INFO] Tracking configured for up to {args.max_hands} hands (bundled MediaPipe Tasks model)")
    print("[INFO] Controls:")
    print("       'q' / ESC : Exit application")
    print("       'd'       : Toggle landmark drawing")
    print("       'h'       : Toggle HUD overlay")
    print("       'p'       : Print structured landmark tree to console")
    print("       's'       : Save frame snapshot and JSON landmark coordinates")
    print("-" * 60)

    # FPS Calculation metrics
    prev_time = time.time()
    fps_smooth = 0.0
    alpha = 0.15  # Smoothing factor for FPS
    draw_landmarks = True
    diagnostics = None

    try:
        if getattr(args, 'diagnostics_dir', None):
            from asignx.live_diagnostics import LiveDiagnostics
            import hashlib
            diagnostics = LiveDiagnostics(args.diagnostics_dir, {
                'person_id': args.diagnostic_person_id,
                'sequence_model': sequence_model_path, 'static_model': args.sign_model,
                'static_model_sha256': (hashlib.sha256(Path(args.sign_model).read_bytes()).hexdigest()
                                        if args.sign_model else None),
                'static_confidence': args.sign_confidence,
                'static_stable_frames': args.sign_stable_frames,
                'static_release_frames': args.sign_release_frames,
                'sequence_model_sha256': (hashlib.sha256(Path(sequence_model_path).read_bytes()).hexdigest()
                                          if sequence_model_path else None),
                'sequence_confidence': args.sequence_confidence,
                'completed_take_sampling': 'timestamp_interpolation_v1',
                'sampling_fps': (unified_recognizer.sequence_predictor.target_fps if unified_recognizer else None),
                'motion_threshold': args.motion_threshold,
                'min_excursion': args.gesture_min_excursion,
                'min_motion_seconds': args.gesture_min_motion_seconds,
                'gesture_form_policy': ('j_z_salute_form_v1' if args.gesture_form_checks else None),
                'hold_static_after_gesture': args.hold_static_after_gesture,
            })
            print(f'[DIAGNOSTICS] Saving to {diagnostics.root.resolve()}')
            print('[DIAGNOSTICS] In camera window: 7=J, 8=Z, 9=HELLO, 0=REST; Q saves and exits.')
            if getattr(args, 'capture_gesture_inputs', False) and unified_recognizer is not None:
                unified_recognizer.trace_completed_takes = True
                print('[DIAGNOSTICS] Completed landmark inputs will also be saved locally for replay; no camera video.')
        for frame in camera.stream():
            capture_timestamp = getattr(camera, "frame_timestamp", None)
            if capture_timestamp is None:
                capture_timestamp = time.monotonic()
            # Process frame through MediaPipe Hands pipeline
            detector_started = time.perf_counter()
            result, annotated_frame = detector.process(
                frame,
                draw=draw_landmarks,
                timestamp_ms=int(capture_timestamp * 1000),
            )
            detector_ms = (time.perf_counter()-detector_started)*1000
            recognition_started = time.perf_counter()

            live_recognitions = None
            live_dynamic_info = None
            if unified_recognizer is not None:
                live_dynamic_info = unified_recognizer.update(result.all_hands, timestamp=capture_timestamp, body_context=getattr(result, "body_context", None))
                live_recognitions = live_dynamic_info["static"]
                for recognition in live_recognitions:
                    if recognition.emitted_label:
                        print(f"[STATIC SIGN] {recognition.hand}: {recognition.emitted_label}")
                if live_dynamic_info.get("dynamic_emitted"):
                    print(f"[DYNAMIC SIGN] {live_dynamic_info['dynamic_emitted']}")
                if live_dynamic_info.get("dynamic_rejected"):
                    print(f"[DYNAMIC REJECTED] {live_dynamic_info['dynamic_rejected']}")
            elif sign_recognizer is not None:
                live_recognitions = sign_recognizer.update(result.all_hands, timestamp=capture_timestamp)
                for recognition in live_recognitions:
                    if recognition.emitted_label:
                        print(f"[SIGN] {recognition.hand}: {recognition.emitted_label}")

            if diagnostics:
                diagnostics.record(result, live_dynamic_info, capture_time=capture_timestamp,
                    processed_time=time.monotonic(), detector_ms=detector_ms,
                    recognition_ms=(time.perf_counter()-recognition_started)*1000)

            # Compute smoothed FPS
            now = time.time()
            elapsed = now - prev_time
            prev_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps_smooth = (
                    current_fps if fps_smooth == 0.0 else (alpha * current_fps + (1 - alpha) * fps_smooth)
                )

            # Render visual debugging HUD overlay (including real-time sign recognition indicator)
            display_frame = visualizer.render_overlay(
                annotated_frame,
                result=result,
                fps=fps_smooth,
                recognitions=live_recognitions,
                dynamic_info=live_dynamic_info,
            )
            if diagnostics:
                cv2.putText(display_frame, f'TEST: {diagnostics.target} | 7=J 8=Z 9=HELLO 0=REST',
                            (16, max(20, display_frame.shape[0]-90)), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (255,255,0), 1, cv2.LINE_AA)

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
            elif diagnostics and diagnostics.mark(key):
                print(f'[DIAGNOSTICS] Test block: {diagnostics.target}')
            elif key in (ord("d"), ord("D")):
                draw_landmarks = not draw_landmarks
                print(f"[INFO] Landmark drawing toggled: {'ON' if draw_landmarks else 'OFF'}")
            elif key in (ord("h"), ord("H")):
                visualizer.show_hud = not visualizer.show_hud
                print(f"[INFO] HUD overlay toggled: {'ON' if visualizer.show_hud else 'OFF'}")
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
        if diagnostics:
            diagnostics.close()
            print(f'[DIAGNOSTICS] Saved report: {(diagnostics.root/"summary.json").resolve()}')
        print("[INFO] Done.")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
