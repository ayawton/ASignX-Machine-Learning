"""Reevaluate a sequence checkpoint only against its matching dataset/manifest."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.sequence_dataset import SequenceDataset, load_sequence_manifest, read_sequence_dir
from ml.sequence_model import load_sequence_checkpoint
from ml.train_sequence import evaluate_sequence_model
from ml.evaluate import classification_metrics


def audit(data_dir, model_path, output):
    samples = read_sequence_dir(data_dir)
    manifest = load_sequence_manifest(Path(model_path).with_suffix(".splits.json"), samples)
    model, labels, checkpoint = load_sequence_checkpoint(model_path)
    if checkpoint["dataset_sha256"] != manifest["dataset_sha256"] or checkpoint["splits"] != manifest["splits"]:
        raise ValueError("Sequence checkpoint and dataset manifest do not match")
    result = {"dataset_sha256": manifest["dataset_sha256"], "data_dir": str(data_dir),
              "identity_known": all(bool(s.metadata.get("signer_id")) for s in samples),
              "sample_count": len(samples), "splits": {},
              "clips": [{"label":s.label, "session_id":s.session_id, "metadata":s.metadata} for s in samples]}
    for split in ("validation", "test"):
        dataset = SequenceDataset(samples, manifest["splits"][split], labels)
        result["splits"][split] = evaluate_sequence_model(model, DataLoader(dataset, batch_size=16), labels)
    # Diagnostic only: a model that passes after removing motion may use handshape shortcuts.
    test_data = SequenceDataset(samples, manifest["splits"]["test"], labels)
    features = test_data.features.clone()
    variants = {"time_reversed": features.flip(1),
                "last_frame_repeated": features[:, -1:].expand_as(features),
                "time_shuffled": features[:, torch.randperm(features.shape[1], generator=torch.Generator().manual_seed(42))]}
    without_travel = features.reshape(len(features), features.shape[1], -1, 21, 3).clone()
    without_travel -= without_travel[:, :, :, :1].clone()
    variants["wrist_travel_removed"] = without_travel.reshape_as(features)
    with torch.inference_mode():
        result["temporal_diagnostics"] = {name: classification_metrics(test_data.targets.tolist(), model(value).argmax(1).tolist(), labels)
                                          for name, value in variants.items()}
    result["temporal_diagnostic_warning"] = "Two test clips cannot establish motion understanding; unchanged scores after motion removal suggest a shortcut worth testing on broader data"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"samples":len(samples), "test_macro_f1":result["splits"]["test"]["macro_f1"],
                      "identity_known":result["identity_known"]})); return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="ml/data/recovered_sequence_baseline")
    parser.add_argument("--model", default="ml/models/sequence_classifier.pth")
    parser.add_argument("--output", default="output/f1-review/dynamic/reproduced.json")
    args=parser.parse_args(); torch.set_num_threads(1); audit(args.data_dir,args.model,args.output)
