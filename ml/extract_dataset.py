"""Static image import with explicit label overrides, provenance groups and safe CSV append."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from uuid import uuid4

import cv2
import numpy as np

from asignx.detector import HandDetector
from ml.data_io import CSV_COLUMNS, import_session_id, read_samples
from ml.preprocessing import (
    FEATURE_COLUMNS,
    PREPROCESSING_VERSION,
    normalize_landmarks,
    validate_features,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif", ".m4v"}
# Outlier removal is opt-in: legitimate viewpoints and left hands can be rare.
CLEANING_PRESETS = {
    "Keep varied examples (recommended)": (0.0, 0.0),
    "Remove very similar examples": (0.0, 0.015),
    "Also remove unusual poses": (3.5, 0.015),
}
IGNORED_DIR_NAMES = {
    "train",
    "training",
    "val",
    "valid",
    "validation",
    "test",
    "testing",
    "images",
    "image",
    "photos",
    "dataset",
    "datasets",
    "data",
    "raw",
    "processed",
    "static",
    "hands",
    "asl",
    "fsl",
    "asl_alphabet_train",
    "asl_alphabet_test",
}
LABEL_PREFIX_PATTERN = re.compile(r"^(?:sign|class|label|letter|char|folder)[\s_\-]+", re.I)

BONE_SEGMENTS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class ExtractedCandidate:
    """Single candidate static hand landmark sample before multi-pass cleaning."""

    label: str
    hand: str
    features: np.ndarray
    file_hash: str
    source_path: str
    subgroup: str = "default"


@dataclass
class ExtractionReport:
    """Summary statistics from a dataset extraction and cleaning run."""

    total_files_scanned: int = 0
    exact_duplicates_skipped: int = 0
    label_conflicts_skipped: int = 0
    no_hand_detected: int = 0
    anatomical_outliers_rejected: int = 0
    statistical_outliers_rejected: int = 0
    near_duplicates_pruned: int = 0
    already_in_csv_skipped: int = 0
    samples_appended: int = 0
    per_label_counts: dict[str, int] = field(default_factory=dict)
    unreadable_images: int = 0
    multiple_hands_skipped: int = 0
    animated_images_skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    decisions: list[dict[str, str]] = field(default_factory=list)
    audit_path: str = ""


GENERIC_FILE_PREFIXES = {
    "img", "image", "images", "frame", "pic", "picture", "photo", "hand", "sample",
    "dsc", "dscn", "dscf", "vid", "shot", "snapshot", "capture", "take", "test",
    "train", "val", "valid", "person", "user", "subject", "signer", "color", "crop",
    "cropped", "aug", "augmented", "flip", "flipped", "rot", "rotated", "whatsapp",
    "screenshot", "screen", "webcam", "camera", "cam", "win", "pxl", "mvimg", "cimg",
    "raw", "untitled", "download", "clip",
}

SUBGROUP_DIR_PATTERN = re.compile(
    r"^(?:user|subject|person|signer|session|take|cam|camera|video|seq|clip|split|fold|group|batch|sample|rgb|color|depth)[\s_\-]*\d*$",
    re.I,
)


def canonicalize_label(raw_name: str) -> str:
    """Normalize folder names like 'a', 'A', 'sign_b', 'class-hello' to 'A', 'B', 'HELLO'."""
    cleaned = raw_name.strip()
    cleaned = LABEL_PREFIX_PATTERN.sub("", cleaned).strip()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned).strip("_")
    cleaned = cleaned.upper()
    if (not cleaned or not re.fullmatch(r"[\w]+", cleaned)
            or cleaned in {"CON", "PRN", "AUX", "NUL"}
            or re.fullmatch(r"(?:COM|LPT)[1-9]", cleaned)):
        raise ValueError(f"Unsafe or empty sign label: {raw_name!r}")
    return cleaned


def infer_label_from_path(
    image_path: Path,
    root_dir: Path,
    root_label_hint: str | None = None,
) -> tuple[str, str]:
    """Infer canonical sign label and optional subgroup from an image's relative path."""
    rel_parts = image_path.relative_to(root_dir).parts[:-1]
    meaningful = [
        p for p in rel_parts
        if p.lower() not in IGNORED_DIR_NAMES
        and not p.startswith(".")
        and not p.lower().startswith("asignx_zip_")
    ]
    if meaningful:
        label_parts = [p for p in meaningful if not SUBGROUP_DIR_PATTERN.match(p)]
        subgroup_parts = [p for p in meaningful if SUBGROUP_DIR_PATTERN.match(p)]
        if label_parts:
            label = canonicalize_label(label_parts[-1])
            subgroup = "/".join(subgroup_parts) or "default"
            return label, subgroup
        raise ValueError(f"Only person/session folders found; supply a label map: {image_path}")

    hint = (root_label_hint or root_dir.name).strip()
    hint_is_valid = (
        bool(hint)
        and hint.lower() not in IGNORED_DIR_NAMES
        and not hint.lower().startswith("asignx_zip_")
        and not SUBGROUP_DIR_PATTERN.match(hint)
    )
    hint_cand = canonicalize_label(hint) if hint_is_valid else ""

    # 1. If the selected root folder or zip is explicitly a single-char sign ('A'..'Z', '0'..'9')
    #    or explicitly prefixed ('sign_hello', 'class_B'), use it directly.
    if hint_cand and (len(hint_cand) == 1 or LABEL_PREFIX_PATTERN.match(hint)):
        return hint_cand, "default"

    # 2. Check filename stem for flat multi-class folders (e.g. "A_test.jpg", "B001.jpg")
    stem = re.sub(r"[_\-\s]*(?:test|train|val|valid)\d*$", "", image_path.stem, flags=re.I)
    stem_match = re.match(r"^([A-Za-z]+)", stem)
    if stem_match and stem_match.group(1).lower() not in GENERIC_FILE_PREFIXES:
        return canonicalize_label(stem_match.group(1)), "default"

    # 3. Fallback to the root folder / zip name for multi-letter sign folders ('HELLO/001.jpg')
    if hint_cand and hint_cand.lower() not in IGNORED_DIR_NAMES:
        return hint_cand, "default"

    raise ValueError(f"Could not infer sign label from path: {image_path}")


def discover_static_dataset(
    root_dir: Path,
    staging_dir: Path | None = None,
    root_label_hint: str | None = None,
    label_map: dict[str, tuple[str, str]] | None = None,
) -> list[tuple[Path, str, str, str]]:
    """Recursively discover static sign images across arbitrary folder structures.

    Returns a list of tuples: (image_path, canonical_label, subgroup, sha256_hex),
    performing Pass 1 (SHA-256 exact file deduplication) during discovery.
    """
    hash_labels: dict[str, set[str]] = defaultdict(set)
    discovered: list[tuple[Path, str, str, str]] = []

    all_files = sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") or part == "__MACOSX"
                    for part in p.relative_to(root_dir).parts)
    )
    if any(p.is_symlink() for p in root_dir.rglob("*")):
        raise ValueError("Dataset contains symbolic links; use ordinary image files.")
    if any(p.suffix.lower() in VIDEO_EXTENSIONS for p in root_dir.rglob("*") if p.is_file()):
        raise ValueError("This extractor accepts still images only. Remove videos and animated GIF files.")
    for img_path in all_files:
        relative = img_path.relative_to(root_dir).as_posix()
        if label_map is not None:
            if relative not in label_map:
                raise ValueError(f"Image is missing from the label map: {relative}")
            label, subgroup = label_map[relative]
        else:
            label, subgroup = infer_label_from_path(
                img_path, root_dir, root_label_hint=root_label_hint
            )
        if not label:
            continue

        digest = hashlib.sha256(img_path.read_bytes()).hexdigest()
        hash_labels[digest].add(label)

        target_path = img_path
        # Copy only accepted images after cleaning, never the rejected inputs.

        discovered.append((target_path, label, subgroup, digest))

    # Decide only after scanning every copy, so no arbitrarily chosen label wins.
    seen_hashes = set()
    marked = []
    for path, label, subgroup, digest in discovered:
        marker = digest
        if len(hash_labels[digest]) > 1:
            marker = "conflict:" + digest
        elif digest in seen_hashes:
            marker = ""
        seen_hashes.add(digest)
        marked.append((path, label, subgroup, marker))
    return marked


def load_label_map(path: str | Path) -> dict[str, tuple[str, str]]:
    """Explicit relative image paths override folder guessing; group is signer/origin."""
    mapping = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"path", "label"}.issubset(reader.fieldnames):
            raise ValueError("Label map needs path,label columns and optional group column.")
        for row in reader:
            relative = (row.get("path") or "").replace("\\", "/")
            if (not relative or relative.startswith("/") or ":" in relative
                    or any(p in ("", ".", "..") for p in relative.split("/"))):
                raise ValueError(f"Label map needs safe relative paths: {relative!r}")
            if relative in mapping:
                raise ValueError(f"Repeated path in label map: {relative}")
            mapping[relative] = (canonicalize_label(row["label"]),
                                 (row.get("group") or "default").strip() or "default")
    return mapping


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Reject unsafe names, links, encrypted entries and excessive expansion."""
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if sum(m.file_size for m in members) > 10 * 1024**3:
            raise ValueError("ZIP is too large (limit: 10 GiB expanded).")
        seen = set()
        for member in members:
            name = member.filename.replace("\\", "/").rstrip("/")
            parts = name.split("/")
            if (not name or name.startswith("/") or ":" in name
                    or any(p in ("", ".", "..") or p.endswith((" ", ".")) for p in parts)
                    or any(re.search(r'[<>:"|?*\x00-\x1f]', p)
                           or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", p, re.I)
                           for p in parts)
                    or (member.external_attr >> 16) & 0o170000 == 0o120000
                    or member.flag_bits & 1):
                raise ValueError(f"Unsafe or encrypted ZIP entry: {member.filename}")
            if name.casefold() in seen:
                raise ValueError(f"ZIP contains colliding paths: {member.filename}")
            seen.add(name.casefold())
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("ZIP entry escapes the extraction folder.")
        zf.extractall(destination)


def check_bone_plausibility(
    normalized_63d: np.ndarray,
    min_bone_len: float = 0.002,
    max_bone_len: float = 1.25,
) -> bool:
    """Pass 2: Verify normalized 21-point hand skeleton has plausible bone segment lengths.

    Note: In wrist_max_radius_v1 normalization, the farthest landmark from the wrist has
    distance 1.0. For closed-fist signs ('A', 'E', 'M', 'N', 'S', 'T'), an MCP knuckle (5, 9,
    13, 17) is often the farthest point from the wrist (length = 1.0), so max_bone_len must
    be >= 1.0 (default 1.25).
    """
    try:
        validate_features(normalized_63d)
    except ValueError:
        return False
    coords = np.asarray(normalized_63d, dtype=np.float32).reshape(21, 3)
    for start_idx, end_idx in BONE_SEGMENTS:
        length = float(np.linalg.norm(coords[end_idx] - coords[start_idx]))
        if length < min_bone_len or length > max_bone_len:
            return False
    return True


def filter_statistical_outliers_mad(
    candidates: list[ExtractedCandidate],
    mad_z_threshold: float = 3.5,
) -> tuple[list[ExtractedCandidate], int]:
    """Pass 3: Reject intra-class geometric outliers using Median Absolute Deviation (MAD)."""
    if len(candidates) < 5 or mad_z_threshold <= 0:
        return list(candidates), 0

    matrix = np.stack([c.features for c in candidates], axis=0)
    median_pose = np.median(matrix, axis=0)
    distances = np.linalg.norm(matrix - median_pose, axis=1)
    median_dist = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_dist)))

    if mad < 1e-6:
        return list(candidates), 0

    robust_sigma = 1.4826 * mad
    cutoff = median_dist + mad_z_threshold * robust_sigma

    kept: list[ExtractedCandidate] = []
    rejected = 0
    for candidate, dist in zip(candidates, distances):
        if float(dist) <= cutoff:
            kept.append(candidate)
        else:
            rejected += 1
    return kept, rejected


def prune_near_duplicates(
    candidates: list[ExtractedCandidate],
    epsilon: float = 0.015,
    existing_vectors: np.ndarray | None = None,
) -> tuple[list[ExtractedCandidate], int, int]:
    """Pass 4: Greedy epsilon-ball pruning against both existing CSV rows and new candidates."""
    kept: list[ExtractedCandidate] = []
    # Allocate once, avoiding a fresh growing stack for every input image.
    kept_vectors = np.empty((len(candidates), 63), dtype=np.float32)
    near_dup_count = 0
    already_in_csv_count = 0

    has_existing = existing_vectors is not None and len(existing_vectors) > 0
    threshold = max(epsilon, 1e-6)  # Exact landmark copies are always removed.

    for candidate in candidates:
        vec = candidate.features
        if has_existing:
            dists_existing = np.linalg.norm(existing_vectors - vec, axis=1)
            if float(np.min(dists_existing)) < threshold:
                already_in_csv_count += 1
                continue

        if kept:
            dists_kept = np.linalg.norm(kept_vectors[:len(kept)] - vec, axis=1)
            if float(np.min(dists_kept)) < threshold:
                near_dup_count += 1
                continue

        kept_vectors[len(kept)] = vec
        kept.append(candidate)

    return kept, near_dup_count, already_in_csv_count


def assign_balanced_sessions(
    candidates: list[ExtractedCandidate],
    source_tag: str,
    min_sessions: int = 5,
    target_chunk_size: int = 40,
) -> list[tuple[ExtractedCandidate, str]]:
    """Preserve origin groups; legacy arguments never manufacture independence.

    The trainer still requires three real groups per sign. Unknown provenance is
    deliberately a single group, even if there are thousands of images.
    """
    return [(candidate, import_session_id(source_tag, candidate.label, candidate.subgroup))
            for candidate in candidates]


def _load_existing_csv_vectors(csv_path: Path) -> dict[str, np.ndarray]:
    """Load existing 63D feature vectors grouped by label from landmarks.csv."""
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    if not csv_path.exists():
        return {}

    for sample in validate_existing_csv(csv_path):
        by_label[canonicalize_label(sample.label)].append(sample.features)

    return {lbl: np.stack(vecs, axis=0) for lbl, vecs in by_label.items() if vecs}


def validate_existing_csv(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != CSV_COLUMNS:
            raise ValueError("Existing CSV has an incompatible header. Choose a compatible landmarks.csv or a new file.")
        has_rows = False
        for row_number, row in enumerate(reader, 2):
            has_rows = True
            if len(row) != len(CSV_COLUMNS):
                raise ValueError(f"Existing CSV row {row_number} has the wrong number of columns.")
    return read_samples(csv_path) if has_rows else []


@contextmanager
def output_lock(path: Path):
    """Cooperative single-writer guard; never remove someone else's lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"Another extraction is writing this file: {lock}") from exc
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def append_to_landmarks_csv(
    rows_to_append: list[tuple[ExtractedCandidate, str]],
    csv_path: Path,
    expected_digest: str | None = None,
) -> int:
    """Non-destructively and atomically append cleaned rows to landmarks.csv."""
    if not rows_to_append:
        return 0

    validate_existing_csv(csv_path)
    for candidate, session_id in rows_to_append:
        validate_features(candidate.features)
        canonicalize_label(candidate.label)
        if not session_id.strip():
            raise ValueError("Missing source group")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{csv_path.name}.",
        suffix=".tmp",
        dir=str(csv_path.parent),
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)

    try:
        # Copy existing bytes intact, including their original numeric precision.
        if csv_path.exists() and csv_path.stat().st_size:
            shutil.copyfile(csv_path, temp_path)
            with temp_path.open("rb+") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) not in (b"\n", b"\r"):
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"\n")
        with temp_path.open("a", newline="", encoding="utf-8") as out_handle:
            writer = csv.writer(out_handle)
            if not temp_path.stat().st_size:
                writer.writerow(CSV_COLUMNS)

            for candidate, session_id in rows_to_append:
                writer.writerow(
                    [
                        candidate.label,
                        session_id,
                        candidate.hand,
                        PREPROCESSING_VERSION,
                        *[f"{val:.8f}" for val in candidate.features],
                    ]
                )
            out_handle.flush()
            os.fsync(out_handle.fileno())

        if expected_digest is not None and _csv_digest(csv_path) != expected_digest:
            raise ValueError("CSV changed during extraction. No extracted rows were saved; retry when other writers stop.")
        os.replace(temp_path, csv_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return len(rows_to_append)


def _csv_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def _is_animated_image(path: Path) -> bool:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        offset = 8
        while offset + 8 <= len(data):
            length = int.from_bytes(data[offset:offset + 4], "big")
            kind = data[offset + 4:offset + 8]
            if kind == b"acTL":
                return True
            offset += 12 + length
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        offset = 12
        while offset + 8 <= len(data):
            kind = data[offset:offset + 4]
            length = int.from_bytes(data[offset + 4:offset + 8], "little")
            if kind in (b"ANIM", b"ANMF"):
                return True
            offset += 8 + length + length % 2
    return False


def _detect_hand_with_retries(detector: object, bgr: np.ndarray) -> object:
    """Run MediaPipe hand detection with proportional border-padding & contrast retries.

    Tightly cropped static dataset images (e.g., 200x200 ASL Alphabet crops) often clip
    the palm/wrist margin needed by MediaPipe's SSD palm detector. Proportional border
    padding restores spatial context without altering wrist-centered normalized features.
    """
    result, _ = detector.process(bgr, draw=False)  # type: ignore[attr-defined]
    if result.hand_detected and result.all_hands:
        return result

    h, w = bgr.shape[:2]
    work = bgr
    if min(h, w) < 256:
        scale = 256.0 / float(min(h, w))
        work = cv2.resize(
            bgr,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        h, w = work.shape[:2]

    # Retry 1: 30% proportional border padding (replicated edge pixels)
    pad_y, pad_x = int(round(h * 0.30)), int(round(w * 0.30))
    padded = cv2.copyMakeBorder(
        work, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
    )
    result, _ = detector.process(padded, draw=False)  # type: ignore[attr-defined]
    if result.hand_detected and result.all_hands:
        _undo_padding(result, w, h, pad_x, pad_y)
        return result

    # Retry 2: CLAHE contrast enhancement + 45% proportional border padding
    try:
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(
            cv2.merge((clahe.apply(l_chan), a_chan, b_chan)),
            cv2.COLOR_LAB2BGR,
        )
        pad_y2, pad_x2 = int(round(h * 0.45)), int(round(w * 0.45))
        padded2 = cv2.copyMakeBorder(
            enhanced, pad_y2, pad_y2, pad_x2, pad_x2, cv2.BORDER_REPLICATE
        )
        result, _ = detector.process(padded2, draw=False)  # type: ignore[attr-defined]
        if result.hand_detected and result.all_hands:
            _undo_padding(result, w, h, pad_x2, pad_y2)
    except Exception:
        pass
    return result


def _undo_padding(result, width: int, height: int, pad_x: int, pad_y: int) -> None:
    """Restore image-normalized x/y/z before applying the training feature contract."""
    for hand in result.all_hands:
        from dataclasses import replace
        hand.landmarks = [replace(point,
            x=(point.x * (width + 2 * pad_x) - pad_x) / width,
            y=(point.y * (height + 2 * pad_y) - pad_y) / height,
            z=point.z * (width + 2 * pad_x) / width,
        ) for point in hand.landmarks]


def _extract_and_clean_dataset(
    input_path: str | Path,
    output_csv: str | Path = Path("ml/data/landmarks.csv"),
    staging_dir: str | Path | None = None,
    mad_z_threshold: float = 0.0,
    duplicate_epsilon: float = 0.0,
    min_sessions: int = 5,
    min_detection_confidence: float = 0.5,
    detector: object | None = None,
    progress_callback: object | None = None,
    manifest_path: str | Path | None = None,
    source_id: str | None = None,
) -> ExtractionReport:
    """End-to-end static dataset extraction, 4-pass cleaning, and non-destructive CSV append."""
    input_path = Path(input_path)
    output_csv = Path(output_csv)
    staging_path = Path(staging_dir) if staging_dir else None

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {input_path}")
    if (not np.isfinite([mad_z_threshold, duplicate_epsilon, min_detection_confidence]).all()
            or mad_z_threshold < 0 or duplicate_epsilon < 0
            or not 0 <= min_detection_confidence <= 1):
        raise ValueError("Cleaning values must be finite and nonnegative; detection confidence must be 0 to 1.")
    if staging_path is not None and input_path.is_dir():
        if (staging_path.resolve().is_relative_to(input_path.resolve())
                or input_path.resolve().is_relative_to(staging_path.resolve())):
            raise ValueError("Accepted-image folder must be separate from the input folder.")
    existing_by_label = _load_existing_csv_vectors(output_csv)
    original_digest = _csv_digest(output_csv)
    label_map = load_label_map(manifest_path) if manifest_path else None

    report = ExtractionReport()
    source_tag = source_id.strip() if source_id and source_id.strip() else str(input_path.resolve())

    temp_extract_dir: tempfile.TemporaryDirectory[str] | None = None
    owns_detector = detector is None

    def _notify(curr: int, total: int, msg: str) -> None:
        if callable(progress_callback):
            progress_callback(curr, total, msg)

    try:
        _notify(0, 100, f"Scanning dataset: {input_path.name}...")
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            temp_extract_dir = tempfile.TemporaryDirectory(prefix="asignx_zip_")
            safe_extract_zip(input_path, Path(temp_extract_dir.name))
            scan_root = Path(temp_extract_dir.name)
        elif input_path.is_dir():
            scan_root = input_path
        else:
            raise ValueError(f"Unsupported input path (must be directory or .zip): {input_path}")

        discovered = discover_static_dataset(
            scan_root,
            root_label_hint=input_path.stem,
            label_map=label_map,
        )
        if not discovered:
            raise ValueError("No supported still images found in the dataset.")
        report.total_files_scanned = len(discovered)
        total_files = max(1, len(discovered))

        if owns_detector:
            detector = HandDetector(
                static_image_mode=True,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
            )

        candidates_by_label: dict[str, list[ExtractedCandidate]] = defaultdict(list)
        def record(path, label, reason, digest="", group=""):
            report.decisions.append({"path": str(Path(path).relative_to(scan_root)),
                                     "label": label, "reason": reason,
                                     "sha256": digest, "group": group})

        for idx, (img_path, label, subgroup, file_hash) in enumerate(discovered, start=1):
            if idx == 1 or idx % 10 == 0 or idx == len(discovered):
                _notify(idx, total_files, f"[{idx}/{len(discovered)}] Extracting label '{label}'...")

            if file_hash.startswith("conflict:"):
                report.label_conflicts_skipped += 1
                record(img_path, label, "Identical image assigned to different signs; review labels",
                       file_hash.removeprefix("conflict:"), subgroup)
                continue
            if not file_hash:
                report.exact_duplicates_skipped += 1
                record(img_path, label, "Repeated image file")
                continue
            if _is_animated_image(img_path):
                report.animated_images_skipped += 1
                record(img_path, label, "Animated image; only still images accepted", file_hash, subgroup)
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                try:
                    raw_bytes = np.fromfile(str(img_path), dtype=np.uint8)
                    bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                except Exception:
                    bgr = None
            if bgr is None:
                report.unreadable_images += 1
                record(img_path, label, "Image cannot be read", file_hash, subgroup)
                continue

            result = _detect_hand_with_retries(detector, bgr)
            if not result.hand_detected or not result.all_hands:
                report.no_hand_detected += 1
                record(img_path, label, "No hand found", file_hash, subgroup)
                continue
            if len(result.all_hands) != 1:
                report.multiple_hands_skipped += 1
                record(img_path, label, "More than one hand; needs a two-hand model", file_hash, subgroup)
                continue

            hand = result.all_hands[0]
            try:
                features = normalize_landmarks(hand)
            except ValueError:
                report.anatomical_outliers_rejected += 1
                record(img_path, label, "Invalid hand points", file_hash, subgroup)
                continue

            # Pass 2: Anatomical bone-ratio plausibility check
            if not check_bone_plausibility(features):
                report.anatomical_outliers_rejected += 1
                record(img_path, label, "Unusable hand shape", file_hash, subgroup)
                continue

            candidates_by_label[label].append(
                ExtractedCandidate(
                    label=label,
                    hand=hand.handedness,
                    features=features,
                    file_hash=file_hash,
                    source_path=str(img_path),
                    subgroup=subgroup,
                )
            )

        _notify(total_files, total_files, "Checking repeated examples and unusual poses...")
        final_rows: list[tuple[ExtractedCandidate, str]] = []
        feature_labels = defaultdict(set)
        for lbl, vectors in existing_by_label.items():
            for vector in vectors:
                feature_labels[tuple(np.round(vector, 6))].add(lbl)
        for lbl, candidates in candidates_by_label.items():
            for candidate in candidates:
                feature_labels[tuple(np.round(candidate.features, 6))].add(lbl)

        for label in sorted(candidates_by_label):
            raw_list = candidates_by_label[label]
            unambiguous = []
            for candidate in raw_list:
                if len(feature_labels[tuple(np.round(candidate.features, 6))]) > 1:
                    report.label_conflicts_skipped += 1
                    record(candidate.source_path, label,
                           "Identical hand points assigned to different signs; review labels",
                           candidate.file_hash, candidate.subgroup)
                else:
                    unambiguous.append(candidate)
            raw_list = unambiguous

            # Pass 3: Statistical MAD outlier rejection
            mad_cleaned, mad_rejected = filter_statistical_outliers_mad(
                raw_list, mad_z_threshold=mad_z_threshold
            )
            report.statistical_outliers_rejected += mad_rejected
            mad_ids = {id(c) for c in mad_cleaned}
            for c in raw_list:
                if id(c) not in mad_ids:
                    record(c.source_path, label, "Unusual pose (optional filter)", c.file_hash, c.subgroup)

            # Pass 4: Epsilon-ball near-duplicate pruning
            pruned, near_dups, csv_dups = prune_near_duplicates(
                mad_cleaned,
                epsilon=duplicate_epsilon,
                existing_vectors=existing_by_label.get(label),
            )
            report.near_duplicates_pruned += near_dups
            report.already_in_csv_skipped += csv_dups
            kept_ids = {id(c) for c in pruned}
            for c in mad_cleaned:
                reason = "Accepted" if id(c) in kept_ids else "Repeated or very similar hand points"
                record(c.source_path, label, reason, c.file_hash, c.subgroup)

            if not pruned:
                continue
            session_rows = assign_balanced_sessions(
                pruned, source_tag=source_tag, min_sessions=min_sessions
            )
            final_rows.extend(session_rows)
            report.per_label_counts[label] = len(session_rows)
            if len({session for _, session in session_rows}) < 3:
                report.warnings.append(f"{label}: fewer than 3 source groups. Extraction succeeds, but honest train/validation/test splitting needs more independent recordings or signers.")
            if any(c.subgroup == "default" for c in pruned):
                report.warnings.append(f"{label}: origin unknown; all ungrouped images stay together. Supply a label map with real signer groups when available.")
            if label in {"J", "Z"}:
                report.warnings.append(f"{label}: still images cannot verify movement. Check the target sign language's definition before using this as a static class.")

        if report.label_conflicts_skipped:
            report.warnings.append(
                f"Skipped {report.label_conflicts_skipped} examples assigned to conflicting sign labels. "
                "Other examples were processed. Review the paths in the detailed results file; original images and existing rows were kept."
            )
        # Stage accepted images before committing the CSV. Original images are untouched.
        if staging_path:
            for candidate, _ in final_rows:
                target = staging_path / candidate.label / (candidate.file_hash + Path(candidate.source_path).suffix.lower())
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(candidate.source_path, target)
        audit_path = output_csv.with_name(f"{output_csv.stem}.extraction-{uuid4().hex}.json")
        report.audit_path = str(audit_path.resolve())
        from dataclasses import asdict
        first_csv_row = sum(len(v) for v in existing_by_label.values()) + 2
        output_rows = []
        for offset, (candidate, session) in enumerate(final_rows):
            relative = Path(candidate.source_path).relative_to(scan_root).as_posix()
            # Bind the fingerprint to exactly the numeric precision written by the CSV writer.
            stored = np.asarray([float(f"{v:.8f}") for v in candidate.features], dtype=np.float32)
            output_rows.append({
                "csv_row": first_csv_row + offset, "label": candidate.label,
                "session_id": session, "handedness": candidate.hand,
                "source_relative_path": relative, "image_sha256": candidate.file_hash,
                "stored_feature_sha256": hashlib.sha256(stored.tobytes()).hexdigest(),
                "original_image_path": str((input_path / relative).resolve()) if input_path.is_dir() else None,
                "archive_path": str(input_path.resolve()) if input_path.is_file() else None,
                "archive_member": relative if input_path.is_file() else None,
            })
        payload = {"source": source_tag, "source_input_path": str(input_path.resolve()),
                   "dataset_before_sha256": original_digest,
                   "output_rows": output_rows, "preprocessing": PREPROCESSING_VERSION,
                   "status": "prepared", "settings": {"unusual_pose_threshold": mad_z_threshold,
                   "similarity_distance": duplicate_epsilon}, "report": asdict(report)}
        audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _notify(total_files, total_files, f"Atomically appending {len(final_rows)} clean samples to {output_csv.name}...")
        report.samples_appended = append_to_landmarks_csv(final_rows, output_csv, original_digest)
        payload.update(status="committed", report=asdict(report),
                       dataset_after_sha256=_csv_digest(output_csv))
        try:
            audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            report.warnings.append("CSV saved, but the audit status could not be updated. The prepared audit remains available.")
        _notify(total_files, total_files, f"Completed! Appended {report.samples_appended} clean samples.")
        return report

    finally:
        if owns_detector and detector is not None and hasattr(detector, "close"):
            detector.close()
        if temp_extract_dir is not None:
            temp_extract_dir.cleanup()


def extract_and_clean_dataset(input_path, output_csv=Path("ml/data/landmarks.csv"), **kwargs) -> ExtractionReport:
    """Import static samples with a cooperative single-writer lock around the transaction."""
    with output_lock(Path(output_csv)):
        return _extract_and_clean_dataset(input_path, output_csv, **kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Dataset folder or ZIP; use --manifest to override folder label guesses",
    )
    parser.add_argument(
        "--output",
        default="ml/data/landmarks.csv",
        help="Target landmarks.csv to non-destructively append to (default: ml/data/landmarks.csv)",
    )
    parser.add_argument(
        "--staging-dir",
        default=None,
        help="Optional folder for accepted image copies, grouped by sign name",
    )
    parser.add_argument(
        "--mad-threshold",
        type=float,
        default=None,
        help="Advanced: unusual-pose cutoff (0 disables); overrides the cleaning preset",
    )
    parser.add_argument(
        "--dedup-epsilon",
        type=float,
        default=None,
        help="Advanced: similarity distance (0 removes exact repeats only)",
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cleaning", choices=("gentle", "similar", "unusual"), default="gentle",
                        help="Keep varied examples, remove similar examples, or also remove unusual poses")
    parser.add_argument("--manifest", help="CSV with path,label and optional group (real signer/origin)")
    parser.add_argument("--source-id", help="Stable dataset origin name; reuse for copies of the same source")
    args = parser.parse_args(argv)
    threshold, epsilon = list(CLEANING_PRESETS.values())[("gentle", "similar", "unusual").index(args.cleaning)]

    report = extract_and_clean_dataset(
        input_path=args.input,
        output_csv=args.output,
        staging_dir=args.staging_dir,
        mad_z_threshold=threshold if args.mad_threshold is None else args.mad_threshold,
        duplicate_epsilon=epsilon if args.dedup_epsilon is None else args.dedup_epsilon,
        min_sessions=args.min_sessions,
        manifest_path=args.manifest,
        source_id=args.source_id,
    )

    print("=" * 60)
    print(" ASignX Static Dataset Extraction & Cleaning Report")
    print("=" * 60)
    print(f" Total images scanned          : {report.total_files_scanned}")
    print(f" Exact file duplicates skipped : {report.exact_duplicates_skipped}")
    print(f" Conflicting labels skipped    : {report.label_conflicts_skipped}")
    print(f" No hand detected              : {report.no_hand_detected}")
    print(f" Anatomical outliers rejected  : {report.anatomical_outliers_rejected}")
    print(f" Statistical (MAD) outliers    : {report.statistical_outliers_rejected}")
    print(f" Near-duplicates pruned        : {report.near_duplicates_pruned}")
    print(f" Already in CSV skipped        : {report.already_in_csv_skipped}")
    print(f" Clean samples appended        : {report.samples_appended}")
    print(f" Unreadable images             : {report.unreadable_images}")
    print(f" Multiple hands skipped        : {report.multiple_hands_skipped}")
    print(f" Detailed results              : {report.audit_path}")
    for warning in report.warnings:
        print(f" Note: {warning}")
    print("-" * 60)
    for label, count in sorted(report.per_label_counts.items()):
        print(f"   [{label}] : +{count} clean samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
