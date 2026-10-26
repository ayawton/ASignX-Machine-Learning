"""Report and fix mixed one-hand (63) / two-hand (126) sequence .npz files.

Two-hand layout used by the project: Left hand = columns 0-62, Right hand = 63-125.
One-hand files are padded into the slot matching their stored ``handedness``;
the empty slot is zero-filled (same as recordings where that hand was absent).

Usage:
    python -m ml.unify_sequence_width ml/data/sequences            # report only
    python -m ml.unify_sequence_width ml/data/sequences --fix      # convert in place (.bak backups)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np

ONE = 63
TWO = 126


def _pad(features: np.ndarray, handedness: str) -> np.ndarray:
    out = np.zeros((features.shape[0], TWO), dtype=np.float32)
    if handedness == "Left":
        out[:, :ONE] = features
    elif handedness == "Right":
        out[:, ONE:] = features
    else:
        raise ValueError(f"handedness must be Left or Right, got {handedness!r}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--fix", action="store_true", help="Pad 63-wide files to 126 in place")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)

    paths = sorted(Path(args.directory).rglob("*.npz"))
    if not paths:
        raise SystemExit(f"No .npz files under {args.directory}")

    widths: Counter[int] = Counter()
    lengths: Counter[int] = Counter()
    one_hand: list[Path] = []
    bad: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            shape = data["features"].shape
        widths[shape[1]] += 1
        lengths[shape[0]] += 1
        if shape[1] == ONE:
            one_hand.append(path)
        elif shape[1] != TWO:
            bad.append(f"{path} width={shape[1]}")

    print(f"Files: {len(paths)}  widths: {dict(widths)}  frame lengths: {dict(lengths)}")
    if len(lengths) > 1:
        print("WARNING: mixed sequence lengths - training also requires one length (re-extract to the same T).")
    for line in bad:
        print("UNSUPPORTED:", line)
    if not one_hand:
        print("No 63-wide files; nothing to convert.")
        return
    print(f"{len(one_hand)} one-hand (63) file(s) need padding to 126:")
    for p in one_hand[:10]:
        print("  ", p)
    if len(one_hand) > 10:
        print(f"   ... +{len(one_hand) - 10} more")
    if not args.fix:
        print("Run again with --fix to convert them.")
        return

    for path in one_hand:
        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}
        payload["features"] = _pad(payload["features"].astype(np.float32), str(payload["handedness"]))
        if not args.no_backup:
            shutil.copy2(path, path.with_suffix(".npz.bak"))
        np.savez_compressed(path, **payload)
    print(f"Converted {len(one_hand)} file(s) to width 126.")


if __name__ == "__main__":
    main()

