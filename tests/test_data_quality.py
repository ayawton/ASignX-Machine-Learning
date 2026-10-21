"""Evidence-backed source recovery and canonical dataset safety."""
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from ml.data_io import Sample, import_session_id
from ml.data_quality import resolve_import_sources, source_coverage
from ml.dataset import save_manifest, load_manifest, validate_splits
from ml.preprocessing import normalize_landmarks, STATIC_HAND_CANONICAL
from ml.recover_image import same_sample


def pose(seed):
    return normalize_landmarks(np.random.default_rng(seed).normal(size=(21, 3)))


def test_import_hash_matches_historical_row_and_recovers_only_existing_sources(tmp_path):
    assert import_session_id(r"C:\Users\jibbi\Downloads\archive (4).zip", "M") == "ext_c72cf97fc7fa5995335a73a1"
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"original source content")
    sample = SimpleNamespace(session_id=import_session_id(str(archive.resolve()), "M"), label="M")
    other = SimpleNamespace(session_id="unverified-session", label="N")
    found = resolve_import_sources([sample, other], [archive])
    assert set(found) == {sample.session_id}
    assert found[sample.session_id]["source_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert found[sample.session_id]["signer_known"] is False
    with pytest.raises(ValueError, match="missing"):
        resolve_import_sources([sample], [tmp_path / "missing.zip"])


def test_source_gaps_are_reported_without_fabricating_identity_or_changing_splits():
    samples = [SimpleNamespace(session_id="a", label="A"), SimpleNamespace(session_id="b", label="B"),
               SimpleNamespace(session_id="c", label="C")]
    splits = {"train": [0], "validation": [2], "test": [1]}
    resolved = {s: {"source_id": "real-archive"} for s in ("a", "b")}
    report = source_coverage(samples, splits, resolved)
    assert report["source_label_conditions_absent_from_training"] == [
        {"source_id": "real-archive", "split": "test", "label": "B", "support": 1}]
    assert report["unresolved_rows_by_split"] == {"validation": 1}
    assert not report["signer_independence_established"]
    assert splits == {"train": [0], "validation": [2], "test": [1]}


def test_conflicting_exact_features_within_training_are_rejected():
    samples = [Sample("A", "a", "Right", pose(0)), Sample("B", "b", "Right", pose(0))]
    samples += [Sample(label, str(i), "Right", pose(i)) for i, label in enumerate("ABAB", 2)]
    with pytest.raises(ValueError, match="conflicting labels"):
        validate_splits(samples, {"train": [0, 1], "validation": [2, 3], "test": [4, 5]},
                        static_hand_preprocessing=STATIC_HAND_CANONICAL)


def test_saved_manifest_retains_and_enforces_canonical_contract(tmp_path):
    from ml.data_io import CSV_COLUMNS
    import csv
    samples = [Sample(label, str(i), "Right", pose(i)) for i, label in enumerate("ABABAB")]
    path = tmp_path / "samples.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(CSV_COLUMNS)
        for s in samples:
            writer.writerow([s.label, s.session_id, s.handedness, "wrist_max_radius_v1", *s.features.tolist()])
    manifest = tmp_path / "splits.json"
    splits = {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}
    save_manifest(manifest, path, samples, splits, 42, static_hand_preprocessing=STATIC_HAND_CANONICAL)
    assert load_manifest(manifest, path)["static_hand_preprocessing"] == STATIC_HAND_CANONICAL
    payload = json.loads(manifest.read_text()); payload["static_hand_preprocessing"] = "invalid"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unsupported static hand preprocessing"):
        load_manifest(manifest, path)


def test_image_match_requires_all_coordinates_and_matching_handedness():
    expected = pose(1)
    assert same_sample(expected.copy(), "Right", expected, "Right")
    assert not same_sample(expected.copy(), "Left", expected, "Right")
    changed = expected.copy(); changed[12] += .001
    assert not same_sample(changed, "Right", expected, "Right")
