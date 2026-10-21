"""Export ASignX static and dynamic PyTorch checkpoints for client-side Web deployment.

Produces:
  1. <output>.onnx         : ONNX Runtime Web graph (when onnx/torch.onnx is available)
  2. <output>.meta.json    : Labels, architecture config, and 63D class_centroids for
                             ASignLearn per-finger coaching & ASignLive stabilization
  3. <output>.weights.json : Portable JSON tensor bundle for zero-dependency WebAssembly/JS
                             inference and client-side zero-retrain prototype addition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ml.model import load_checkpoint
from ml.sequence_model import load_sequence_checkpoint


def _tensors_to_serializable(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Convert a PyTorch state_dict into compact nested float lists with shape metadata."""
    serialized: dict[str, Any] = {}
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().to(torch.float32).numpy()
        serialized[name] = {
            "shape": list(arr.shape),
            "data": [round(float(x), 6) for x in arr.ravel()],
        }
    return serialized


def export_static_checkpoint(
    checkpoint_path: str | Path,
    output_onnx_path: str | Path,
    export_weights_json: bool = True,
) -> dict[str, Path]:
    """Export a static SignMLP/SignResMLP checkpoint to ONNX + .meta.json + .weights.json."""
    checkpoint_path = Path(checkpoint_path)
    output_onnx_path = Path(output_onnx_path)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model, labels, raw_ckpt = load_checkpoint(checkpoint_path)
    model.eval()

    meta_path = output_onnx_path.with_suffix(".meta.json")
    weights_path = output_onnx_path.with_suffix(".weights.json")

    cfg = model.config
    input_dim = int(raw_ckpt.get("num_features", 63))
    hidden_sizes = list(cfg.get("hidden_sizes", (128, 64)))

    metadata: dict[str, Any] = {
        "model_type": "static_sign_resmlp",
        "labels": list(labels),
        "input_dim": input_dim,
        "hidden_sizes": hidden_sizes,
        "use_kinematic_features": bool(cfg.get("use_kinematics", model.use_kinematics)),
        "use_cosine_head": bool(cfg.get("cosine_head", model.cosine_head)),
        "residual": bool(cfg.get("residual", model.residual)),
        "class_centroids": raw_ckpt.get("class_centroids", {}),
        "preprocessing": raw_ckpt["preprocessing"],
        "static_hand_preprocessing": model.static_hand_preprocessing,
        "input_contract": "Externally normalize and apply static_hand_preprocessing before graph inference",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    artifacts: dict[str, Path] = {"meta": meta_path}

    if export_weights_json:
        weights_payload = {
            "metadata": metadata,
            "state_dict": _tensors_to_serializable(model.state_dict()),
        }
        weights_path.write_text(json.dumps(weights_payload), encoding="utf-8")
        artifacts["weights"] = weights_path

    dummy_input = torch.zeros(1, metadata["input_dim"], dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_onnx_path),
            input_names=["landmarks"],
            output_names=["logits"],
            dynamic_axes={"landmarks": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
        )
        artifacts["onnx"] = output_onnx_path
    except Exception:
        # Fallback to TorchScript trace if the optional `onnx` python package is not installed
        pt_trace_path = output_onnx_path.with_suffix(".traced.pt")
        traced = torch.jit.trace(model, dummy_input, check_trace=False)
        traced.save(str(pt_trace_path))
        artifacts["torchscript"] = pt_trace_path

    return artifacts


def export_sequence_checkpoint(
    checkpoint_path: str | Path,
    output_onnx_path: str | Path,
    export_weights_json: bool = True,
) -> dict[str, Path]:
    """Export a dynamic SignLSTM/SignTCNAttention checkpoint to ONNX + .meta.json + .weights.json."""
    checkpoint_path = Path(checkpoint_path)
    output_onnx_path = Path(output_onnx_path)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model, labels, raw_ckpt = load_sequence_checkpoint(checkpoint_path)
    model.eval()

    meta_path = output_onnx_path.with_suffix(".meta.json")
    weights_path = output_onnx_path.with_suffix(".weights.json")

    cfg = model.config
    seq_len = int(raw_ckpt.get("sequence_length", 30))
    input_dim = int(raw_ckpt.get("num_features", cfg.get("input_size", 63)))

    metadata: dict[str, Any] = {
        "model_type": "dynamic_sequence",
        "backbone": str(cfg.get("backbone", model.backbone)),
        "labels": list(labels),
        "sequence_length": seq_len,
        "input_dim": input_dim,
        "hidden_size": int(cfg.get("hidden_size", 128)),
        "num_layers": int(cfg.get("num_layers", 2)),
        "use_kinematic_features": bool(cfg.get("use_kinematics", model.use_kinematics)),
        "use_cosine_head": bool(cfg.get("cosine_head", model.cosine_head)),
        "preprocessing": str(raw_ckpt.get("preprocessing", "sequence_anchor_trajectory_v2")),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    artifacts: dict[str, Path] = {"meta": meta_path}

    if export_weights_json:
        weights_payload = {
            "metadata": metadata,
            "state_dict": _tensors_to_serializable(model.state_dict()),
        }
        weights_path.write_text(json.dumps(weights_payload), encoding="utf-8")
        artifacts["weights"] = weights_path

    dummy_input = torch.zeros(1, seq_len, input_dim, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_onnx_path),
            input_names=["sequence"],
            output_names=["logits"],
            dynamic_axes={"sequence": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
        )
        artifacts["onnx"] = output_onnx_path
    except Exception:
        pt_trace_path = output_onnx_path.with_suffix(".traced.pt")
        traced = torch.jit.trace(model, dummy_input, check_trace=False)
        traced.save(str(pt_trace_path))
        artifacts["torchscript"] = pt_trace_path

    return artifacts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export ASignX static or dynamic checkpoints for Web client deployment"
    )
    parser.add_argument("--checkpoint", required=True, help="Input .pth checkpoint path")
    parser.add_argument("--output", required=True, help="Output .onnx file path")
    parser.add_argument(
        "--type",
        choices=("static", "sequence"),
        default="static",
        help="Checkpoint type: 'static' (SignResMLP) or 'sequence' (SignTCNAttention/LSTM)",
    )
    args = parser.parse_args(argv)

    if args.type == "static":
        artifacts = export_static_checkpoint(args.checkpoint, args.output)
    else:
        artifacts = export_sequence_checkpoint(args.checkpoint, args.output)

    print("Exported client-side web deployment bundle:")
    for key, path in artifacts.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
