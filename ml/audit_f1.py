"""Reproduce static predictions and build a source/hand/session error review.

No labels are corrected automatically. Landmark sketches are evidence for review,
not substitutes for original images. Use --split validation for tuning diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
import torch

from ml.benchmark import write_group_template
from ml.data_io import read_samples
from ml.dataset import LandmarkDataset, load_manifest, validate_splits
from ml.evaluate import classification_metrics
from ml.model import load_checkpoint


def source_hint(session):
    match = re.match(r"ext_(.+)_[A-Z0-9_]+_s\d+$", session)
    return f"legacy:{match[1]}" if match else "hashed_import:source_identity_unavailable"


def rejection_metrics(targets, predictions, confidence, threshold, labels):
    """Rejected known signs remain false negatives; no unknown-class claims."""
    targets, predictions, confidence = map(np.asarray, (targets, predictions, confidence))
    accepted = confidence >= threshold
    precision, recall, f1 = [], [], []
    for index in range(len(labels)):
        tp = np.sum((targets == index) & (predictions == index) & accepted)
        fp = np.sum((targets != index) & (predictions == index) & accepted)
        fn = np.sum((targets == index) & ((predictions != index) | ~accepted))
        precision.append(float(tp / max(1, tp + fp)))
        recall.append(float(tp / max(1, tp + fn)))
        f1.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
    return {"threshold": float(threshold), "coverage": float(accepted.mean()),
            "accepted_accuracy": float((targets[accepted] == predictions[accepted]).mean()) if accepted.any() else None,
            "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1))}


def sketch(features):
    points = np.asarray(features).reshape(21, 3)[:, :2]
    low, high = points.min(0), points.max(0)
    points = 12 + 136 * (points - low) / max(float((high - low).max()), 1e-6)
    edges = [(0, k) for k in (1, 5, 9, 13, 17)]
    edges += [(k, k + 1) for start in (1, 5, 9, 13, 17) for k in range(start, start + 3)]
    lines = ''.join(f'<line x1="{points[a,0]:.1f}" y1="{points[a,1]:.1f}" x2="{points[b,0]:.1f}" y2="{points[b,1]:.1f}"/>' for a, b in edges)
    return f'<svg viewBox="0 0 160 160" width="160" height="160" aria-label="Normalized landmark projection"><g stroke="#126e82" stroke-width="2">{lines}</g></svg>'


@torch.inference_mode()
def audit(data, model_path, output, split="test"):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    samples = read_samples(data)
    model, labels, checkpoint = load_checkpoint(model_path)
    manifest = load_manifest(Path(model_path).with_suffix(".splits.json"), data)
    if checkpoint["dataset_sha256"] != manifest["dataset_sha256"] or checkpoint["splits"] != manifest["splits"]:
        raise ValueError("Checkpoint and manifest do not match")
    validate_splits(samples, manifest["splits"], static_hand_preprocessing=model.static_hand_preprocessing)
    indices = manifest["splits"][split]
    dataset = LandmarkDataset(samples, indices, labels, static_hand_preprocessing=model.static_hand_preprocessing)
    logits = torch.cat([model(batch) for batch in dataset.features.split(1024)]).numpy()
    shifted = logits - logits.max(1, keepdims=True)
    probs = np.exp(shifted); probs /= probs.sum(1, keepdims=True)
    predictions, confidence = probs.argmax(1), probs.max(1)
    targets = dataset.targets.numpy()
    metrics = classification_metrics(targets.tolist(), predictions.tolist(), labels)
    np.savez_compressed(root / "predictions.npz", logits=logits, targets=targets, indices=indices, labels=labels,
                        dataset_sha256=manifest["dataset_sha256"], model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(), split=split)
    buckets = defaultdict(list)
    errors, pairs = [], Counter()
    with (root / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["csv_row", "label", "prediction", "confidence", "correct", "handedness", "session_id", "source_hint"])
        for pos, index in enumerate(indices):
            sample = samples[index]
            source = source_hint(sample.session_id)
            correct = int(targets[pos] == predictions[pos])
            writer.writerow([index + 2, sample.label, labels[predictions[pos]], float(confidence[pos]), correct, sample.handedness, sample.session_id, source])
            for category, key in (("session", sample.session_id), ("source_hint", source), ("handedness", sample.handedness), ("source_label", source + ":" + sample.label)):
                buckets[(category, key)].append(pos)
            if not correct:
                errors.append(pos); pairs[(sample.label, labels[predictions[pos]])] += 1
    summaries = []
    for (category, key), positions in buckets.items():
        subset = classification_metrics(targets[positions].tolist(), predictions[positions].tolist(), labels)
        summaries.append({"category": category, "group": key, "support": len(positions),
                          "accuracy": subset["accuracy"], "macro_f1_all_vocabulary": subset["macro_f1"]})
    thresholds = [rejection_metrics(targets, predictions, confidence, t, labels) for t in np.linspace(0, .99, 100)]
    summary = {"split": split, "dataset_sha256": manifest["dataset_sha256"], "model": str(model_path),
               "metrics": metrics, "groups": summaries,
               "top_confusions": [{"true": a, "predicted": b, "count": n} for (a, b), n in pairs.most_common(30)],
               "threshold_sweep": thresholds, "unknown_false_activations_measured": False,
               "identity_warning": "Source hints are unverified diagnostics, not signer IDs or safe grouping claims"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_group_template(root / "identity-plan-template.csv", samples)
    with (root / "review-decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["csv_row", "original_label", "prediction", "session_id", "reviewed_label", "decision", "reason", "original_image_path"])
        for pos in sorted(errors, key=lambda p: -confidence[p])[:500]:
            sample = samples[indices[pos]]
            writer.writerow([indices[pos]+2, sample.label, labels[predictions[pos]], sample.session_id, "", "pending", "", ""])
    cards = []
    # Balanced review: show each confusion pair, rather than only one large source.
    for (true, predicted), count in pairs.most_common(24):
        candidates = [p for p in errors if labels[targets[p]] == true and labels[predictions[p]] == predicted]
        for pos in sorted(candidates, key=lambda p: -confidence[p])[:3]:
            sample = samples[indices[pos]]
            cards.append(f'<article><h3>{html.escape(true)} → {html.escape(predicted)}</h3>{sketch(sample.features)}<p>CSV row {indices[pos]+2}; {confidence[pos]:.1%}<br>{html.escape(sample.handedness)}<br>{html.escape(sample.session_id)}</p></article>')
    rows = ''.join(f'<tr><td>{html.escape(r["category"])}</td><td>{html.escape(r["group"])}</td><td>{r["support"]}</td><td>{r["accuracy"]:.1%}</td></tr>' for r in sorted(summaries,key=lambda r:r['accuracy']) if r['category'] in ('source_hint','handedness','source_label'))
    page = f'''<!doctype html><meta charset="utf-8"><title>ASignX error review</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#15304b}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccd}}.cards{{display:flex;flex-wrap:wrap;gap:16px}}article{{border:1px solid #ccd;padding:14px;width:250px;overflow-wrap:anywhere}}input{{padding:10px;width:90%}}</style>
<h1>ASignX {html.escape(split)} error review</h1><p>Accuracy {metrics['accuracy']:.2%}; macro F1 {metrics['macro_f1']:.2%}. Labels have not been changed.</p>
<p>Source hints are unverified. Sketches show stored x/y geometry, not original images. Complete predictions and pending review decisions are adjacent CSV files. Missing source images require manual recovery.</p>
<input id="filter" placeholder="Filter cards by label, session or CSV row" aria-label="Filter review cards">
<h2>Confusion examples</h2><div class="cards">{''.join(cards)}</div><h2>Source and handedness breakdown</h2>
<table><tr><th>Type</th><th>Group</th><th>Samples</th><th>Accuracy</th></tr>{rows}</table>
<script>document.querySelector('#filter').oninput=e=>document.querySelectorAll('article').forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>'''
    (root / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"accuracy":metrics['accuracy'],"macro_f1":metrics['macro_f1'],"top_confusions":summary['top_confusions'][:8]}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml/data/landmarks.csv")
    parser.add_argument("--model", default="ml/models/sign_classifier_canonical.pth")
    parser.add_argument("--output", default="output/f1-review/baseline")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args(); audit(args.data, args.model, args.output, args.split)


if __name__ == "__main__":
    main()
