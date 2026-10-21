"""Regression checks for imports that could corrupt data or inflate evaluation."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest

from ml.data_io import CSV_COLUMNS, read_samples
from ml.extract_dataset import (
    ExtractedCandidate, append_to_landmarks_csv, assign_balanced_sessions,
    canonicalize_label, discover_static_dataset, extract_and_clean_dataset,
    load_label_map, output_lock, prune_near_duplicates, safe_extract_zip,
    _undo_padding,
)
from ml.preprocessing import normalize_landmarks
from tests.test_upgraded_architecture import _make_plausible_hand


def candidate(label="A", group="default"):
    return ExtractedCandidate(label, "Right", normalize_landmarks(_make_plausible_hand()),
                              "hash", "sample.png", group)


def images(root):
    (root / "A").mkdir(parents=True)
    for i in range(2):
        cv2.imwrite(str(root / "A" / f"{i}.png"), np.full((40, 70, 3), i + 20, np.uint8))


class Detector:
    def process(self, image, draw=False):
        hand = _make_plausible_hand(spread=float(image[0, 0, 0]) / 10)
        return SimpleNamespace(hand_detected=True, all_hands=[hand]), image


def test_existing_bytes_preserved_and_bad_schema_refused(tmp_path):
    path = tmp_path / "landmarks.csv"
    append_to_landmarks_csv([(candidate(), "source")], path)
    original = path.read_bytes()
    append_to_landmarks_csv([(candidate("B"), "other")], path)
    assert path.read_bytes().startswith(original)
    assert len(read_samples(path)) == 2
    path.write_text("label,x0\nA,0\n")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="header"):
        append_to_landmarks_csv([(candidate(), "source")], path)
    assert path.read_bytes() == original


def test_wrong_preprocessing_and_changed_target_refused(tmp_path):
    path = tmp_path / "landmarks.csv"
    append_to_landmarks_csv([(candidate(), "source")], path)
    with pytest.raises(ValueError, match="changed"):
        append_to_landmarks_csv([(candidate(), "source")], path, "old-hash")
    path.write_text(path.read_text().replace("wrist_max_radius_v1", "wrong"))
    with pytest.raises(ValueError, match="preprocessing"):
        append_to_landmarks_csv([(candidate(), "source")], path)


def test_groups_preserve_real_origin_and_never_invent_sessions():
    rows = assign_balanced_sessions([candidate(), candidate(), candidate(group="signer2")], "dataset")
    assert rows[0][1] == rows[1][1]
    assert rows[0][1] != rows[2][1]
    assert rows[0][1] == assign_balanced_sessions([candidate()], "dataset")[0][1]


def test_exact_landmarks_removed_even_with_optional_similarity_off():
    first, duplicate = candidate(), candidate()
    kept, count, _ = prune_near_duplicates([first, duplicate], epsilon=0)
    assert len(kept) == count == 1
    kept, _, old = prune_near_duplicates([first], epsilon=0, existing_vectors=first.features[None])
    assert not kept and old == 1


@pytest.mark.parametrize("name", ["../escape.png", "C:/escape.png", "..\\escape.png", "a./x.png"])
def test_unsafe_zip_refused(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(name, b"image")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract_zip(archive, tmp_path / "target")


def test_manifest_overrides_layout_and_groups(tmp_path):
    root = tmp_path / "input"
    images(root)
    mapping = tmp_path / "names.csv"
    mapping.write_text("path,label,group\nA/0.png,b,signer1\nA/1.png,B,signer2\n")
    found = discover_static_dataset(root, label_map=load_label_map(mapping))
    assert [(label, group) for _, label, group, _ in found] == [("B", "signer1"), ("B", "signer2")]
    mapping.write_text("path,label\n../file.png,A\n")
    with pytest.raises(ValueError, match="relative"):
        load_label_map(mapping)


def test_static_only_and_conflicting_file_labels(tmp_path):
    images(tmp_path)
    (tmp_path / "movie.mp4").write_bytes(b"video")
    with pytest.raises(ValueError, match="still images"):
        discover_static_dataset(tmp_path)
    (tmp_path / "movie.mp4").unlink()
    (tmp_path / "B").mkdir()
    (tmp_path / "B" / "copy.png").write_bytes((tmp_path / "A" / "0.png").read_bytes())
    found = discover_static_dataset(tmp_path)
    conflicts = [(path, label, digest) for path, label, _, digest in found
                 if digest.startswith("conflict:")]
    assert {label for _, label, _ in conflicts} == {"A", "B"}
    assert len(conflicts) == 2


def test_reimport_idempotence_audit_and_manifests_retained(tmp_path):
    root = tmp_path / "input"
    images(root)
    path = tmp_path / "data" / "landmarks.csv"
    manifest = tmp_path / "models" / "saved.splits.json"
    manifest.parent.mkdir()
    manifest.write_text("keep this evidence")
    report = extract_and_clean_dataset(root, path, detector=Detector(), source_id="origin")
    assert report.samples_appended == 2
    original = path.read_bytes()
    audit = json.loads(Path(report.audit_path).read_text())
    assert audit["status"] == "committed"
    assert audit["source_input_path"] == str(root.resolve())
    assert [r["csv_row"] for r in audit["output_rows"]] == [2, 3]
    import hashlib
    assert audit["dataset_after_sha256"] == hashlib.sha256(original).hexdigest()
    for record, sample in zip(audit["output_rows"], read_samples(path)):
        assert Path(record["original_image_path"]).is_file()
        assert record["image_sha256"] == hashlib.sha256(Path(record["original_image_path"]).read_bytes()).hexdigest()
        assert record["stored_feature_sha256"] == hashlib.sha256(sample.features.tobytes()).hexdigest()
        assert record["archive_path"] is None
    assert all(d["reason"] == "Accepted" for d in audit["report"]["decisions"])
    second = extract_and_clean_dataset(root, path, detector=Detector(), source_id="origin")
    assert second.samples_appended == 0 and second.already_in_csv_skipped == 2
    assert path.read_bytes() == original
    assert manifest.read_text() == "keep this evidence"


def test_conflicting_copies_all_skipped_but_other_images_import(tmp_path):
    root = tmp_path / "input"
    images(root)
    (root / "B").mkdir()
    original = (root / "A" / "0.png").read_bytes()
    (root / "A" / "duplicate.png").write_bytes(original)
    (root / "B" / "conflicting.png").write_bytes(original)
    output = tmp_path / "landmarks.csv"
    report = extract_and_clean_dataset(root, output, detector=Detector())
    assert report.label_conflicts_skipped == 3
    assert report.samples_appended == 1
    assert report.exact_duplicates_skipped == 0
    assert [sample.label for sample in read_samples(output)] == ["A"]
    decisions = json.loads(Path(report.audit_path).read_text())["report"]["decisions"]
    assert sum("different signs" in decision["reason"] for decision in decisions) == 3
    assert all((root / decision["path"]).exists() for decision in decisions)


def test_zip_provenance_survives_temporary_extraction_cleanup(tmp_path):
    root = tmp_path / "input"
    images(root)
    archive = tmp_path / "original.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for image in sorted(root.rglob("*.png")):
            zipped.write(image, "dataset/" + image.relative_to(root).as_posix())
    report = extract_and_clean_dataset(archive, tmp_path / "samples.csv", detector=Detector())
    audit = json.loads(Path(report.audit_path).read_text())
    assert report.samples_appended == len(audit["output_rows"]) == 2
    import hashlib
    with zipfile.ZipFile(archive) as zipped:
        for record in audit["output_rows"]:
            assert record["archive_path"] == str(archive.resolve())
            assert record["original_image_path"] is None
            assert record["image_sha256"] == hashlib.sha256(zipped.read(record["archive_member"])).hexdigest()


def test_conflicting_landmarks_do_not_assign_an_arbitrary_label(tmp_path):
    root = tmp_path / "input"
    images(root)
    (root / "B").mkdir()
    cv2.imwrite(str(root / "B" / "other.png"), np.full((40, 70, 3), 80, np.uint8))
    class SamePoseDetector:
        def process(self, image, draw=False):
            return SimpleNamespace(hand_detected=True, all_hands=[_make_plausible_hand()]), image
    report = extract_and_clean_dataset(root, tmp_path / "landmarks.csv", detector=SamePoseDetector())
    assert report.samples_appended == 0
    assert report.label_conflicts_skipped == 3
    assert any("conflicting sign labels" in note for note in report.warnings)


def test_lock_and_nested_staging_refused(tmp_path):
    root = tmp_path / "input"
    images(root)
    path = tmp_path / "landmarks.csv"
    with output_lock(path):
        with pytest.raises(ValueError, match="Another extraction"):
            extract_and_clean_dataset(root, path, detector=Detector())
    with pytest.raises(ValueError, match="separate"):
        extract_and_clean_dataset(root, path, staging_dir=root / "accepted", detector=Detector())
    assert not path.exists()


def test_padding_coordinates_restored_for_non_square_images():
    hand = _make_plausible_hand()
    original = hand.to_numpy().copy()
    width, height, px, py = 500, 200, 150, 60
    from dataclasses import replace
    hand.landmarks = [replace(p,
        x=(p.x * width + px) / (width + 2 * px),
        y=(p.y * height + py) / (height + 2 * py),
        z=p.z * width / (width + 2 * px)) for p in hand.landmarks]
    _undo_padding(SimpleNamespace(all_hands=[hand]), width, height, px, py)
    np.testing.assert_allclose(hand.to_numpy(), original, atol=1e-6)


def test_only_accepted_images_staged_and_animation_skipped(tmp_path):
    root = tmp_path / "input"
    images(root)
    # Minimal APNG header: animation is identified before attempting image decode.
    (root / "A" / "animated.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + (8).to_bytes(4, "big") + b"acTL" + b"\0" * 12)
    (root / "A" / "broken.png").write_bytes(b"not a picture")
    staged = tmp_path / "accepted"
    report = extract_and_clean_dataset(root, tmp_path / "landmarks.csv",
                                      staging_dir=staged, detector=Detector())
    assert report.samples_appended == 2
    assert report.unreadable_images == report.animated_images_skipped == 1
    assert len(list(staged.rglob("*.png"))) == 2


def test_legacy_session_repair_leaves_csv_unchanged(tmp_path):
    from ml.fix_sessions import fix_sessions
    path = tmp_path / "landmarks.csv"
    path.write_text("do not rewrite provenance")
    with pytest.raises(ValueError, match="independent"):
        fix_sessions(str(path))
    assert path.read_text() == "do not rewrite provenance"


@pytest.mark.parametrize("label", ["../A", "C:A", "CON", "", "A/B"])
def test_unsafe_label_refused(label):
    with pytest.raises(ValueError):
        canonicalize_label(label)
