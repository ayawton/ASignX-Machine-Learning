"""Process static images into the ASignX landmark dataset format."""

import argparse
import csv
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

from asignx import HandDetector
from ml.data_io import SampleWriter


def get_existing_labels(csv_path: str) -> set:
    """Read the CSV and return a set of all labels that have already been processed."""
    labels = set()
    path = Path(csv_path)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            try:
                reader = csv.DictReader(f)
                for row in reader:
                    if "label" in row:
                        labels.add(row["label"])
            except Exception as e:
                print(f"Warning: Could not read existing labels from {csv_path} - {e}")
    return labels


def process_images(args) -> None:
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} not found.")
        sys.exit(1)

    # Find all subdirectories which are treated as class labels
    labels = [d.name for d in input_dir.iterdir() if d.is_dir()]
    if not labels:
        print(f"Error: No label directories found in {input_dir}.")
        sys.exit(1)

    print(f"Found {len(labels)} labels: {', '.join(labels)}")
    print(f"Appending to {args.output}...")

    existing_labels = get_existing_labels(args.output)
    if existing_labels:
        print(f"Found existing data for {len(existing_labels)} labels. They will be skipped.")

    total_saved = 0
    detector = HandDetector(max_num_hands=2)
    
    try:
        for label in labels:
            if label in existing_labels:
                print(f"Skipping '{label}': already exists in dataset.")
                continue

            label_dir = input_dir / label
            image_paths = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpeg"))
            if not image_paths:
                continue
                
            print(f"Processing '{label}': found {len(image_paths)} images.")
            
            # Use one session ID per label for imported images
            session_id = uuid.uuid4().hex
            saved_for_label = 0
            
            with SampleWriter(args.output, label, session_id) as writer:
                for img_path in image_paths:
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        print(f"  Warning: could not read {img_path}")
                        continue
                        
                    result, _ = detector.process(frame, draw=False)
                    hands = result.all_hands
                    
                    if args.hand == "single":
                        selected = hands[0] if len(hands) == 1 else None
                    else:
                        selected = next((h for h in hands if h.handedness == args.hand), None)
                        
                    if selected is not None:
                        try:
                            writer.add(selected.to_numpy(), selected.handedness or "")
                            saved_for_label += 1
                            total_saved += 1
                        except ValueError:
                            # Invalid landmarks, skip
                            pass
                            
            print(f"  -> Extracted landmarks from {saved_for_label}/{len(image_paths)} images.")
            
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
            
    print(f"Done. Appended {total_saved} new landmark samples to {args.output}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory containing subfolders of images per label")
    parser.add_argument("--output", default="ml/data/landmarks.csv", help="Path to output CSV")
    parser.add_argument("--hand", choices=("single", "Left", "Right"), default="single", 
                        help="Which hand to extract if multiple are detected")
    process_images(parser.parse_args())


if __name__ == "__main__":
    main()
