"""Convert labeled image folders into ml/data/landmarks.csv for static pose training."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import cv2

from asignx import HandDetector
from ml.data_io import SampleWriter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_image_dataset(
    input_dir: str | Path,
    output_csv: str | Path = "ml/data/landmarks.csv",
    *,
    hand_selection: str = "single",
    chunk_size: int = 50,
) -> int:
    root = Path(input_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image dataset directory not found: {root}")

    label_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not label_dirs:
        raise ValueError(
            f"No subdirectories found in {root}. Structure must be: {root}/<LABEL>/*.jpg"
        )

    output_path = Path(output_csv).resolve()
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    total_samples = 0

    with HandDetector(static_image_mode=True, max_num_hands=2) as detector:
        for label_dir in label_dirs:
            label = label_dir.name
            image_files = sorted(
                [f for f in label_dir.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
            )
            if not image_files:
                continue

            print(f"Processing label '{label}' ({len(image_files)} images)...")
            label_samples = 0

            # Processing batches do not create independent recordings.
            session_id = "imported_" + hashlib.sha256(f"{root}:{label}".encode()).hexdigest()[:24]
            for chunk_idx, offset in enumerate(range(0, len(image_files), chunk_size)):
                batch_files = image_files[offset : offset + chunk_size]

                with SampleWriter(output_path, label=label, session_id=session_id) as writer:
                    for img_path in batch_files:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue

                        result, _ = detector.process(img, draw=False)
                        if not result.hand_detected:
                            continue

                        # Hand selection logic matching ASignX conventions
                        hand = None
                        if hand_selection == "single":
                            hand = result.all_hands[0] if len(result.all_hands) == 1 else None
                        elif hand_selection in ("Left", "Right"):
                            hand = next(
                                (h for h in result.all_hands if h.handedness == hand_selection),
                                None,
                            )

                        if hand is not None:
                            try:
                                writer.add(hand.to_numpy(), hand.handedness or "")
                                label_samples += 1
                                total_samples += 1
                            except ValueError:
                                # Skip invalid or degenerate landmarks
                                pass

            print(f"  -> Extracted {label_samples} valid landmark samples for '{label}'.")
            print("  Origin unknown: images stay in one group. Prefer ml.extract_dataset with a real signer label map.")

    print(f"\nFinished! Total {total_samples} samples saved to: {output_path}")
    return total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to root images directory structured as input/<LABEL>/*.jpg",
    )
    parser.add_argument(
        "--output",
        default="ml/data/landmarks.csv",
        help="Path to output CSV file (default: ml/data/landmarks.csv)",
    )
    parser.add_argument(
        "--hand",
        default="single",
        choices=["single", "Left", "Right"],
        help="Hand selection strategy (default: single)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of images per session ID for split stratification (default: 50)",
    )
    args = parser.parse_args()

    convert_image_dataset(
        args.input,
        args.output,
        hand_selection=args.hand,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

