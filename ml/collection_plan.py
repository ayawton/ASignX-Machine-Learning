"""Create a recording checklist, never synthetic training examples or identities."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from ml.benchmark import GROUP_COLUMNS, load_group_plan
from ml.sequence_io import load_sequence_file


def prepare(output="output/f1-review/collection", sequence_dir=None):
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    if sequence_dir is None:
        sequence_dir = next((path for path in ("ml/data/sequences", "dist/data/sequences", "ml/data/recovered_sequence_baseline")
                             if any(Path(path).rglob("*.npz"))), "ml/data/sequences")
    columns = ["planned_person", "split", "visit", "kind", "label", "take", "condition", "status", "actual_signer_id", "sequence_path"]
    conditions = ["normal pace, front view", "slow pace, slight viewpoint change", "fast pace, different distance", "different lighting", "other supported handedness"]
    with (root / "recording-checklist.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns); writer.writeheader()
        for person in range(1, 11):
            split = "train" if person <= 6 else "validation" if person <= 8 else "test"
            for visit in (1, 2):
                for label in ("J", "Z"):
                    for take, condition in enumerate(conditions, 1):
                        writer.writerow(dict(planned_person=f"planned-{person:02}", split=split, visit=visit,
                                             kind="positive", label=label, take=take, condition=condition, status="not recorded"))
                for label in ("IDLE", "TRANSITION", "PARTIAL_SIGN", "UNRELATED_MOTION",
                              "J_HANDSHAPE_HOLD", "Z_HANDSHAPE_HOLD", "INCORRECT_J_PATH", "INCORRECT_Z_PATH"):
                    writer.writerow(dict(planned_person=f"planned-{person:02}", split=split, visit=visit,
                                         kind="negative_keep_separate", label=label, take=1,
                                         condition="continuous realistic movement", status="not recorded"))
    paths = sorted(Path(sequence_dir).rglob("*.npz"))
    samples, errors, rows = [], [], []
    for path in paths:
        try:
            sample = load_sequence_file(path); samples.append(sample)
            m = sample.metadata
            rows.append(dict(session_id=sample.session_id, group_id=m.get("signer_id", ""),
                             signer_id=m.get("signer_id", ""), recording_id=m.get("recording_id", ""),
                             source_id=m.get("source_id", ""), device=m.get("device", ""), split=m.get("split", "")))
        except (ValueError, KeyError, OSError) as error:
            errors.append({"path":str(path), "error":str(error)})
    plan = root / "sequence-identity-plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, GROUP_COLUMNS); writer.writeheader(); writer.writerows(rows)
    ready = False
    try:
        if samples and not errors:
            load_group_plan(plan, samples); ready = True
    except ValueError as error:
        errors.append({"identity_plan":str(error)})
    status = {"data_dir": str(sequence_dir), "recorded_sequences":len(samples), "labels":dict(Counter(s.label for s in samples)),
              "benchmark_ready":ready, "problems":errors,
              "missing_data":not bool(samples), "planned_examples_are_not_training_data":True}
    (root / "readiness.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status)); return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/f1-review/collection")
    parser.add_argument("--sequence-dir")
    args=parser.parse_args(); prepare(args.output, args.sequence_dir)
