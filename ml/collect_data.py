"""Record labeled static poses with the ASignX webcam detector."""

from __future__ import annotations

import argparse
import math
import time
import uuid

from ml.data_io import SampleWriter


def select_hand(hands: list, selection: str):
    if selection == "single":
        return hands[0] if len(hands) == 1 else None
    return next((hand for hand in hands if hand.handedness == selection), None)


def collect(args) -> int:
    if args.samples < 1 or not math.isfinite(args.interval) or args.interval <= 0:
        raise ValueError("Samples and sampling interval must be positive and finite")
    import cv2
    from asignx import HandDetector, WebcamStream

    session_id = args.session_id or uuid.uuid4().hex
    print(f"Label: {args.label}; session: {session_id}; selected hand: {args.hand}")
    print("SPACE starts/pauses recording. Q/Esc exits. Use independent recordings per sign.")
    recording = False
    saved = 0
    last_sample = -math.inf
    exposure = None if str(args.exposure).lower() == "auto" else float(args.exposure)
    try:
        with SampleWriter(args.output, args.label, session_id) as writer:
            try:
                camera_context = WebcamStream(
                    camera_index=args.camera,
                    width=args.width,
                    height=args.height,
                    exposure=exposure,
                )
            except TypeError:
                # Keeps lightweight camera test doubles usable.
                camera_context = WebcamStream()
            with camera_context as camera:
                try:
                    detector = HandDetector(max_num_hands=2)
                except TypeError:
                    detector = HandDetector()
                try:
                    for frame in camera.stream():
                        result, annotated = detector.process(frame, draw=True)
                        hand = select_hand(result.all_hands, args.hand)
                        now = time.monotonic()
                        if recording and hand is not None and now - last_sample >= args.interval:
                            try:
                                writer.add(hand.to_numpy(), hand.handedness or "")
                                saved += 1
                                last_sample = now
                            except ValueError:
                                print("Invalid landmarks - sample skipped")
                        status = "RECORDING" if recording else "PAUSED - SPACE to record"
                        cv2.putText(
                            annotated,
                            f"{status}: {saved}/{args.samples}",
                            (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        cv2.imshow("ASignX - Collect sign poses", annotated)
                        if saved >= args.samples:
                            break
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q"), ord("Q")):
                            break
                        if key == 32:
                            recording = not recording
                finally:
                    close = getattr(detector, "close", None)
                    if callable(close):
                        close()
    finally:
        cv2.destroyAllWindows()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", default="ml/data/landmarks.csv")
    parser.add_argument("--session-id")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument("--hand", choices=("single", "Left", "Right"), default="single")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--exposure", default="-5.0")
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
