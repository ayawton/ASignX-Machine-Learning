"""Explicit, cross-label identity groups for static and sequence benchmarks.

CSV columns: session_id,group_id,signer_id,recording_id,source_id,device,split.
group_id must be a real global identity (usually signer); never infer it from labels.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from ml.dataset import SPLITS

GROUP_COLUMNS = ["session_id", "group_id", "signer_id", "recording_id", "source_id", "device", "split"]


def load_group_plan(path, samples):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"session_id", "group_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Group CSV requires session_id,group_id,split")
        rows = list(reader)
    sessions = {s.session_id for s in samples}
    mapping, owners, identity_owners = {}, {}, {}
    for row in rows:
        session, group, split = (row[k].strip() for k in ("session_id", "group_id", "split"))
        if session not in sessions or session in mapping or not group or split not in SPLITS:
            raise ValueError(f"Unknown/duplicate session, missing identity or invalid split: {session}")
        if owners.setdefault(group, split) != split:
            raise ValueError(f"Global group leaks across labels/splits: {group}")
        for key in ("signer_id", "recording_id", "source_id"):
            value = (row.get(key) or "").strip()
            if value and identity_owners.setdefault((key, value), split) != split:
                raise ValueError(f"{key} leaks across splits: {value}")
        mapping[session] = (group, split)
    if set(mapping) != sessions:
        raise ValueError("Provide explicit global groups and splits for every session; unknown identities cannot be fabricated")
    splits = {name: [] for name in SPLITS}
    for index, sample in enumerate(samples):
        splits[mapping[sample.session_id][1]].append(index)
    labels = {s.label for s in samples}
    for name, indices in splits.items():
        if {samples[i].label for i in indices} != labels:
            raise ValueError(f"{name} must contain every label; collect more independent groups")
    return splits, {"group_plan_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "group_plan": str(Path(path).resolve()), "global_identity_groups": True}


def write_group_template(path, samples):
    """Leave identity fields blank rather than manufacture signer independence."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_COLUMNS)
        writer.writeheader()
        writer.writerows({"session_id": session} for session in sorted({s.session_id for s in samples}))


def class_weights(counts, mode):
    if mode not in ("none", "sqrt", "inverse"):
        raise ValueError("class_weighting must be none, sqrt or inverse")
    if mode == "none":
        return None
    weights = counts.sum() / (len(counts) * counts.clamp_min(1))
    return weights.sqrt() if mode == "sqrt" else weights


def selection_score(metrics, selection_metric):
    return metrics["macro_f1"] if selection_metric == "macro_f1" else -metrics["loss"]
