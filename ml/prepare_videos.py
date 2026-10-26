"""Convert labeled video folders into leakage-safe landmark sequences."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import re
import sys
from typing import Callable

from asignx import HandDetector
from ml.sequence_io import SequenceSample, load_sequence_file, save_sequence_file
from ml.video import HAND_SELECTIONS, VIDEO_EXTENSIONS, extract_video_landmarks


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "sample"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_labeled_videos(input_dir: str | Path) -> list[tuple[str, Path]]:
    """Return ``(label, video)`` pairs from ``input/<label>/**/<video>``."""
    root = Path(input_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Video dataset directory not found: {root}")
    videos: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            relative = path.relative_to(root)
            if len(relative.parts) < 2:
                raise ValueError(
                    f"Video must be inside a label folder (input/<label>/video): {path}"
                )
            videos.append((relative.parts[0], path))
    if not videos:
        raise ValueError(f"No supported videos found under {root}")
    return videos


def prepare_video_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    sequence_length: int = 30,
    target_fps: float = 15.0,
    hand_selection: str = "both",
    body_context: bool = False,
    temple_context: bool = False,
    min_detection_ratio: float = 0.5,
    overwrite: bool = False,
    skip_invalid: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Path]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if not 0 < min_detection_ratio <= 1:
        raise ValueError("min_detection_ratio must be in (0, 1]")
    if hand_selection not in HAND_SELECTIONS:
        raise ValueError(f"hand_selection must be one of {HAND_SELECTIONS}")
    body_context = body_context or temple_context
    if body_context and hand_selection != "both":
        raise ValueError("Body context requires both hand slots")
    source_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    labeled_videos = discover_labeled_videos(source_root)
    counts = Counter(label for label, _ in labeled_videos)
    too_small = {label: count for label, count in counts.items() if count < 3}
    if len(counts) < 2:
        raise ValueError("Video training requires at least two label folders")
    if too_small:
        raise ValueError(
            "Each label needs at least 3 independent videos for train/validation/test; "
            f"found {too_small}"
        )

    safe_labels = [_safe_name(label) for label in counts]
    if len(set(safe_labels)) != len(safe_labels):
        raise ValueError("Two label folder names collapse to the same safe output name")

    output_root.mkdir(parents=True, exist_ok=True)
    expected: dict[Path, tuple[str, Path, str]] = {}
    for label, video_path in labeled_videos:
        relative = video_path.relative_to(source_root)
        session_hash = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:12]
        session_id = f"{_safe_name(label)}-{_safe_name(video_path.stem)}-{session_hash}"
        destination = output_root / _safe_name(label) / f"{session_id}.npz"
        expected[destination.resolve()] = (label, video_path, session_id)
    extras = sorted(
        path for path in output_root.rglob("*.npz") if path.resolve() not in expected
    )
    if extras:
        preview = ", ".join(str(path.relative_to(output_root)) for path in extras[:5])
        raise ValueError(
            "Output contains stale sequence files not present in this video dataset: "
            f"{preview}. Use a fresh output directory or remove the stale files."
        )

    written: list[Path] = []
    prepared_counts: Counter[str] = Counter()
    seen_hashes: dict[str, tuple[str, Path]] = {}
    failures: list[str] = []
    from asignx.body_detector import BodyHandDetector
    detector_type = BodyHandDetector if body_context else HandDetector
    with detector_type(max_num_hands=2) as detector:
        for number, (label, video_path) in enumerate(labeled_videos, start=1):
            if cancelled and cancelled():
                raise InterruptedError("Import stopped. Completed sequences were kept; rerun to resume.")
            if progress:
                progress(number - 1, len(labeled_videos), str(video_path))
            relative = video_path.relative_to(source_root)
            digest = _file_sha256(video_path)
            if digest in seen_hashes:
                prior_label, prior_path = seen_hashes[digest]
                message = f"duplicate video {relative} matches {prior_path.relative_to(source_root)}"
                if prior_label != label:
                    raise ValueError(f"{message} but uses conflicting labels {prior_label!r}/{label!r}")
                failures.append(message)
                continue
            seen_hashes[digest] = (label, video_path)
            destination = next(
                path for path, values in expected.items() if values[1] == video_path
            )
            session_id = expected[destination][2]
            if destination.exists() and not overwrite:
                existing = load_sequence_file(destination)
                expected_width = 180 if temple_context else 150 if body_context else 126 if hand_selection == "both" else 63
                if (
                    existing.label != label
                    or existing.session_id != session_id
                    or existing.features.shape != (sequence_length, expected_width)
                    or existing.metadata.get("source_sha256") != digest
                    or existing.metadata.get("target_fps") != target_fps
                    or existing.metadata.get("hand_selection") != hand_selection
                ):
                    raise ValueError(
                        f"Existing sequence does not match current source/settings: {destination}. "
                        "Run with --overwrite or use a fresh output directory."
                    )
                print(f"[{number}/{len(labeled_videos)}] keep {destination}")
                written.append(destination)
                prepared_counts[label] += 1
                continue
            try:
                features, handedness, metadata = extract_video_landmarks(
                    video_path,
                    detector=detector,
                    target_fps=target_fps,
                    hand_selection=hand_selection,
                    target_length=sequence_length,
                    interpolate=True,
                    return_metadata=True,
                    body_context=body_context,
                    temple_context=temple_context,
                )
                if metadata["detection_ratio"] < min_detection_ratio:
                    raise ValueError(
                        f"hand detected in only {metadata['detection_ratio']:.1%} of sampled frames"
                    )
                if temple_context and metadata.get("temple_detection_ratio", 0) < min_detection_ratio:
                    raise ValueError(
                        f"temples visible in only {metadata.get('temple_detection_ratio', 0):.1%} "
                        "of sampled frames; keep the face in view"
                    )
                metadata["relative_source"] = relative.as_posix()
                metadata["source_sha256"] = digest
                sample = SequenceSample(
                    label=label,
                    session_id=session_id,
                    handedness=handedness,
                    features=features,
                    metadata=metadata,
                )
                save_sequence_file(destination, sample)
                written.append(destination)
                prepared_counts[label] += 1
                print(
                    f"[{number}/{len(labeled_videos)}] wrote {destination} "
                    f"({metadata['detection_ratio']:.0%} detected)"
                )
            except Exception as error:
                message = f"{relative}: {error}"
                if not skip_invalid:
                    raise RuntimeError(message) from error
                failures.append(message)
                print(f"[SKIP] {message}", file=sys.stderr)
    if failures:
        print(f"Skipped {len(failures)} video(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
    if not written:
        raise ValueError("No sequence files were prepared")
    too_small_after = {
        label: prepared_counts[label] for label in counts if prepared_counts[label] < 3
    }
    if too_small_after:
        raise ValueError(
            "Too few valid, unique videos remain for session-disjoint splitting: "
            f"{too_small_after}"
        )
    print(f"Prepared {len(written)} sequence(s) in {output_root}")
    if progress:
        progress(len(labeled_videos), len(labeled_videos), "Import complete")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Folder laid out as <label>/*.mp4")
    parser.add_argument("--output", default="ml/data/sequences")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--temple-context", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--legacy-body-context", action="store_true", help="Use the older 150-feature body format without temple features")
    parser.add_argument("--body-context", action="store_true", help="Include upper-body and temple features; requires --hand both")
    parser.add_argument("--hand", choices=HAND_SELECTIONS, default="both")
    parser.add_argument("--min-detection-ratio", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args()
    if args.legacy_body_context and args.temple_context:
        parser.error("Choose standard body context or legacy body context, not both")
    body_enabled = args.body_context or args.temple_context or args.legacy_body_context
    temples_enabled = body_enabled and not args.legacy_body_context
    prepare_video_dataset(
        args.input,
        args.output,
        sequence_length=args.sequence_length,
        target_fps=args.target_fps,
        hand_selection=args.hand,
        body_context=body_enabled,
        temple_context=temples_enabled,
        min_detection_ratio=args.min_detection_ratio,
        overwrite=args.overwrite,
        skip_invalid=args.skip_invalid,
    )


if __name__ == "__main__":
    main()
