"""Recover import-source evidence and diagnose a frozen static benchmark.

Does not relabel rows, delete difficult samples, or regenerate benchmark splits.
Source-tag recovery identifies an import, not a signer or a particular image.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from ml.data_io import import_session_id, read_samples
from ml.dataset import dataset_digest, validate_splits
from ml.preprocessing import STATIC_HAND_CANONICAL, canonicalize_static_features


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_import_sources(samples, paths):
    """Recover only exact source-tag hash matches to existing supplied files.

    File content is hashed as present-day evidence, not proof it has never changed.
    Non-default groups and manually assigned source IDs remain unresolved.
    """
    sessions = {s.session_id for s in samples}
    labels = {s.label for s in samples}
    resolved = {}
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise ValueError(f"Source archive is missing: {path}")
        tags = {str(path), path.as_posix()}
        matches = [(import_session_id(tag, label), tag) for tag in tags for label in labels
                   if import_session_id(tag, label) in sessions]
        if not matches:
            continue
        digest = file_digest(path)
        for session, tag in matches:
            evidence = {"source_path": str(path), "source_tag": tag,
                        "source_sha256": digest, "source_id": "sha256:" + digest,
                        "evidence": "exact_import_session_hash", "signer_known": False}
            if session in resolved and resolved[session]["source_id"] != evidence["source_id"]:
                raise ValueError(f"Ambiguous source evidence: {session}")
            resolved[session] = evidence
    return resolved


def source_coverage(samples, splits, resolved):
    counts = defaultdict(lambda: {name: Counter() for name in splits})
    unresolved = Counter()
    for split, indices in splits.items():
        for index in indices:
            sample = samples[index]
            evidence = resolved.get(sample.session_id)
            if evidence:
                counts[evidence["source_id"]][split][sample.label] += 1
            else:
                unresolved[split] += 1
    missing = []
    for source, parts in counts.items():
        for split in ("validation", "test"):
            for label, support in parts[split].items():
                if not parts["train"][label]:
                    missing.append({"source_id": source, "split": split,
                                    "label": label, "support": support})
    return {"sources": {source: {split: dict(sorted(c.items())) for split, c in parts.items()}
                        for source, parts in counts.items()},
            "unresolved_rows_by_split": dict(unresolved),
            "source_label_conditions_absent_from_training": missing,
            "source_overlap_across_splits": [source for source, parts in counts.items()
                                             if sum(bool(c) for c in parts.values()) > 1],
            "signer_independence_established": False}


def audit_quality(data, manifest, sources, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(manifest).read_text(encoding="utf-8"))
    digest = dataset_digest(data)
    if plan.get("dataset_sha256") != digest:
        raise ValueError("Manifest does not match dataset; refusing a misleading row/source report")
    samples = read_samples(data)
    validation_status = "passed"
    try:
        validate_splits(samples, plan["splits"], static_hand_preprocessing=STATIC_HAND_CANONICAL)
    except ValueError as error:
        if "conflicting labels" not in str(error):
            raise
        validation_status = f"failed: {error}"
    resolved = resolve_import_sources(samples, sources)
    coverage = source_coverage(samples, plan["splits"], resolved)
    owners = defaultdict(list)
    for index, sample in enumerate(samples):
        fingerprint = canonicalize_static_features(sample.features, sample.handedness).tobytes()
        owners[fingerprint].append((index + 2, sample.label))
    conflicts = [rows for rows in owners.values() if len({label for _, label in rows}) > 1]
    session_samples = {s.session_id: s for s in samples}
    with (root / "session-provenance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_sha256", "session_id", "label", "source_path", "source_sha256", "evidence", "signer_id"])
        for session, sample in sorted(session_samples.items()):
            e = resolved.get(session, {})
            writer.writerow([digest, session, sample.label, e.get("source_path", ""),
                             e.get("source_sha256", ""), e.get("evidence", "unresolved"), ""])
    with (root / "benchmark-plan-template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session_id", "group_id", "signer_id", "recording_id", "source_id", "device", "split"])
        for session in sorted(session_samples):
            source = resolved.get(session, {}).get("source_id", "")
            # Known archive groups must stay together across ALL labels. Leave
            # unknown identities and split choices blank, rather than inventing them.
            writer.writerow([session, source, "", "", source, "", ""])
    report = {"dataset_sha256": digest, "rows": len(samples),
              "resolved_sessions": len(resolved),
              "resolved_rows": sum(s.session_id in resolved for s in samples),
              "source_evidence": list({e["source_id"]: e for e in resolved.values()}.values()),
              "coverage": coverage, "exact_conflicting_label_groups": conflicts,
              "numerical_and_canonical_split_validation": validation_status,
              "labels_changed": 0, "rows_removed": 0, "splits_changed": False,
              "benchmark_ready": False,
              "identity_warning": "Archive identity is not signer identity. Unresolved source/signer identities remain blank.",
              "next_action": "Inspect original images and verify source conventions. Keep historical scores. Build a new explicitly source/signer-grouped benchmark after provenance review."}
    (root / "quality-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = "".join(f"<tr><td>{html.escape(r['split'])}</td><td>{html.escape(r['label'])}</td><td>{r['support']}</td></tr>"
                   for r in coverage["source_label_conditions_absent_from_training"])
    links = ""
    recovery_file = root / "row-110182-recovery.json"
    if recovery_file.exists():
        recovery = json.loads(recovery_file.read_text())
        if (recovery.get("dataset_sha256") == digest and recovery.get("exact_geometry_match")
                and (root / "row-110182-original.jpg").exists()):
            links = ('<h2>Recovered row 110182</h2><p>Original image and detected landmarks. '
                     'The original M label is unchanged and still needs linguistic review.</p>'
                     '<img src="row-110182-original.jpg" width="300" alt="Original image"> '
                     '<img src="row-110182-landmarks.jpg" width="300" alt="Detected landmarks">')
    page = f'''<!doctype html><meta charset="utf-8"><title>ASignX data quality repair</title>
<style>body{{font:16px system-ui;max-width:960px;margin:40px auto;padding:20px;color:#15304b}}td,th{{padding:8px;border:1px solid #ccd}}table{{border-collapse:collapse}}img{{max-width:45%}}</style>
<h1>ASignX source and preprocessing investigation</h1>
<p>{report['resolved_rows']:,} of {len(samples):,} rows now have recovered import-source tags across {len(resolved)} sessions. These are source archives, not verified people.</p>
<p>Numerical and canonical split validation: {html.escape(validation_status)}. No labels, rows or original splits were changed. No F1 increase is claimed.</p>
<h2>Evaluation conditions absent from training</h2><p>These labels have evaluation rows from a recovered source but no training rows of that label from the same source. This is evidence of source-domain coverage differences, not proof of bad test labels. Other unresolved sources may have additional gaps.</p>
<table><tr><th>Partition</th><th>Label</th><th>Rows</th></tr>{rows}</table>
{links}<h2>Artifacts</h2><p><a href="quality-report.json">Full evidence</a> | <a href="session-provenance.csv">Session provenance</a></p>'''
    (root / "review.html").write_text(page, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml/data/landmarks.csv")
    parser.add_argument("--manifest", default="ml/models/sign_classifier_canonical.splits.json")
    parser.add_argument("--source", action="append", required=True, help="Existing source archive; repeat as needed")
    parser.add_argument("--output", default="output/data-repair")
    args = parser.parse_args()
    report = audit_quality(args.data, args.manifest, args.source, args.output)
    print(json.dumps({k: report[k] for k in ("rows", "resolved_sessions", "resolved_rows", "exact_conflicting_label_groups")}, indent=2))


if __name__ == "__main__":
    main()
