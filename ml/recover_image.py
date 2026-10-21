"""Find an image that reproduces an existing static CSV row; never change its label."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from asignx.detector import HandDetector
from ml.data_io import import_session_id
from ml.data_quality import file_digest
from ml.extract_dataset import _detect_hand_with_retries, IMAGE_EXTENSIONS
from ml.preprocessing import FEATURE_COLUMNS, normalize_landmarks, validate_features


def same_sample(features, handedness, expected, expected_handedness, tolerance=1e-6):
    return handedness == expected_handedness and np.allclose(
        features, expected, rtol=0, atol=tolerance)


def recover(data, csv_row, archive, output, max_images=None):
    if csv_row < 2 or (max_images is not None and max_images < 1):
        raise ValueError("CSV row must include the header offset; max-images must be positive")
    with Path(data).open(newline="", encoding="utf-8") as handle:
        sample = next((r for i, r in enumerate(csv.DictReader(handle), 2) if i == csv_row), None)
    if sample is None:
        raise ValueError("CSV row not found")
    expected = validate_features(np.asarray([float(sample[k]) for k in FEATURE_COLUMNS], dtype=np.float32))
    archive = Path(archive).resolve()
    tags = {str(archive), archive.as_posix()}
    if not any(import_session_id(tag, sample["label"]) == sample["session_id"] for tag in tags):
        raise ValueError("Archive path does not reproduce this row's import-session hash")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    dataset_hash = file_digest(data)
    with zipfile.ZipFile(archive) as zipped, HandDetector(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=.5) as detector:
        members = sorted((m for m in zipped.infolist()
                          if sample["label"].lower() in [p.lower() for p in Path(m.filename).parts[:-1]]
                          and Path(m.filename).suffix.lower() in IMAGE_EXTENSIONS), key=lambda m: m.filename)
        for index, member in enumerate(members):
            if max_images is not None and index >= max_images:
                break
            if member.file_size > 64 * 1024 * 1024:
                continue
            raw = zipped.read(member)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            result = _detect_hand_with_retries(detector, image)
            if len(result.all_hands) != 1:
                continue
            hand = result.all_hands[0]
            try:
                features = normalize_landmarks(hand)
            except ValueError:
                continue
            if same_sample(features, hand.handedness, expected, sample["handedness"]):
                suffix = Path(member.filename).suffix.lower()
                image_name = f"row-{csv_row}-original{suffix}"
                (root / image_name).write_bytes(raw)
                overlay = image.copy()
                detector._draw_hand(overlay, hand)
                cv2.imwrite(str(root / f"row-{csv_row}-landmarks.jpg"), overlay)
                evidence = {"csv_row": csv_row, "dataset_sha256": dataset_hash,
                            "label": sample["label"], "session_id": sample["session_id"],
                            "archive": str(archive), "archive_sha256": file_digest(archive),
                            "archive_member": member.filename,
                            "image_sha256": hashlib.sha256(raw).hexdigest(),
                            "original_image": image_name,
                            "max_coordinate_difference": float(np.max(np.abs(features - expected))),
                            "csv_handedness": sample["handedness"], "detected_handedness": hand.handedness,
                            "exact_geometry_match": True, "label_verified": False,
                            "warning": "Found a geometrically matching source image, not proof of a unique image or a linguistically correct label."}
                (root / f"row-{csv_row}-recovery.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
                return evidence
            if index % 100 == 0:
                print(f"Checked {index}/{len(members)} images", flush=True)
    raise ValueError("No matching image within the search; CSV labels and source data were left unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml/data/landmarks.csv")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="output/data-repair")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    print(json.dumps(recover(args.data, args.row, args.archive, args.output, args.max_images), indent=2))


if __name__ == "__main__":
    main()
