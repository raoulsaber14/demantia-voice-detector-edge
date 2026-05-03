#!/usr/bin/env python3
"""Inspect and smoke-test the two-stage wav2vec2 inference pipeline.

Prints every input/output tensor name, shape, and dtype for both models,
then runs a synthetic forward pass end-to-end.  Reveals any mismatch between
what Wav2Vec2TwoStageBackend expects and what the ONNX files actually contain.

Usage:
    python -m scripts.verify_two_stage --backbone <path/to/backbone/model.onnx> \
                                       --head     <path/to/head.onnx>

    # Or use the project-default paths:
    python -m scripts.verify_two_stage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Defaults (relative to this file; adjust if your layout differs)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODELS_ROOT = HERE.parent / "models" / "phaseD_frozen_ensemble_deploy"

DEFAULT_BACKBONE = (
    MODELS_ROOT
    / "wav2vec2_base"
    / "wav2vec2_base_chunk8_accumulator_fp32_external"
    / "model.onnx"
)
DEFAULT_HEAD = MODELS_ROOT / "wav2vec2_component_head_fp32.onnx"

EXPECTED = {
    "backbone_inputs":  {"input_audio": ((1, 128000), "float32"),
                         "valid_samples": ((1,), "int64")},
    "backbone_outputs": {"sum_vec": (1, 768), "sumsq_vec": (1, 768), "frame_count": None},
    "head_inputs":      {"input_features": ((1, 1536), "float32")},
    "head_outputs":     {"logits": (1, 2)},
}

SEP = "-" * 64


def _dtype_name(dtype) -> str:
    """Normalise onnxruntime dtype string (e.g. 'tensor(float)' → 'float32')."""
    s = str(dtype)
    mapping = {
        "tensor(float)": "float32",
        "tensor(double)": "float64",
        "tensor(int64)": "int64",
        "tensor(int32)": "int32",
        "tensor(int8)": "int8",
        "tensor(uint8)": "uint8",
    }
    return mapping.get(s, s)


def inspect_session(label: str, session) -> dict[str, dict]:
    """Pretty-print and return {name: {shape, dtype}} for all I/O."""
    print(f"\n  {label}")
    info = {}
    for role, get_fn in (("INPUTS", session.get_inputs), ("OUTPUTS", session.get_outputs)):
        print(f"    {role}:")
        for t in get_fn():
            dtype = _dtype_name(t.type)
            shape = tuple(t.shape)
            print(f"      {t.name!r:40s}  shape={str(shape):<20s}  dtype={dtype}")
            info[t.name] = {"shape": shape, "dtype": dtype}
    return info


def check_name(label: str, actual: dict, expected_name: str) -> bool:
    if expected_name not in actual:
        print(f"  [MISMATCH] {label}: expected tensor {expected_name!r} "
              f"— not found. Actual names: {list(actual)}")
        return False
    return True


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def run_smoke_test(backbone_session, head_session,
                   backbone_info: dict, head_info: dict) -> None:
    print(f"\n{SEP}")
    print("SMOKE TEST — synthetic 8 s white-noise chunk")
    print(SEP)

    # Determine actual input names from the model (don't hard-code)
    b_input_names = [t.name for t in backbone_session.get_inputs()]
    b_output_names = [t.name for t in backbone_session.get_outputs()]
    h_input_names  = [t.name for t in head_session.get_inputs()]
    h_output_names = [t.name for t in head_session.get_outputs()]

    # Build backbone inputs
    rng = np.random.default_rng(42)
    audio_chunk = rng.standard_normal(128_000).astype(np.float32) * 0.05

    # Map by position (first input = audio, second = valid_samples) if names differ
    b_feeds: dict[str, np.ndarray] = {}
    for name in b_input_names:
        info = backbone_info.get(name, {})
        dtype = info.get("dtype", "float32")
        if "int" in dtype:
            b_feeds[name] = np.array([128_000], dtype=np.int64)
        else:
            b_feeds[name] = audio_chunk.reshape(1, 128_000)

    print(f"  Backbone feeds: { {k: v.shape for k, v in b_feeds.items()} }")

    try:
        b_outs = backbone_session.run(b_output_names, b_feeds)
    except Exception as exc:
        print(f"  [ERROR] backbone.run() failed: {exc}")
        return

    b_dict = dict(zip(b_output_names, b_outs))
    print("  Backbone outputs:")
    for name, arr in b_dict.items():
        print(f"    {name!r:30s}  shape={arr.shape}  dtype={arr.dtype}"
              f"  min={arr.min():.4f}  max={arr.max():.4f}")

    # Reconstruct embedding — try expected names, fall back to positional
    sum_key   = "sum_vec"   if "sum_vec"   in b_dict else b_output_names[0]
    sumsq_key = "sumsq_vec" if "sumsq_vec" in b_dict else b_output_names[1]
    fc_key    = "frame_count" if "frame_count" in b_dict else b_output_names[2]

    fc = int(np.squeeze(b_dict[fc_key]))
    print(f"\n  frame_count = {fc}")

    if fc == 0:
        print("  [WARNING] frame_count=0 — the accumulator produced no frames "
              "(silent / too-short input or valid_samples=0). "
              "Try valid_samples=128000.")
        return

    mean = b_dict[sum_key] / float(fc)
    var  = b_dict[sumsq_key] / float(fc) - mean ** 2
    std  = np.sqrt(np.maximum(var, 0.0) + 1e-8)
    embedding = np.concatenate([mean, std], axis=-1).reshape(1, -1).astype(np.float32)
    print(f"  Embedding  shape={embedding.shape}  "
          f"mean_of_mean={mean.mean():.4f}  mean_of_std={std.mean():.4f}")

    # Head feed
    h_feeds: dict[str, np.ndarray] = {}
    for name in h_input_names:
        h_feeds[name] = embedding

    print(f"  Head feeds: { {k: v.shape for k, v in h_feeds.items()} }")

    try:
        h_outs = head_session.run(h_output_names, h_feeds)
    except Exception as exc:
        print(f"  [ERROR] head.run() failed: {exc}")
        return

    h_dict = dict(zip(h_output_names, h_outs))
    logits_key = "logits" if "logits" in h_dict else h_output_names[0]
    logits = h_dict[logits_key]
    probs  = softmax(logits.ravel())

    print("  Head outputs:")
    for name, arr in h_dict.items():
        print(f"    {name!r:30s}  shape={arr.shape}  values={arr.ravel().tolist()}")

    print(f"\n  Softmax probabilities: non_dementia={probs[0]:.4f}  dementia={probs[1]:.4f}")
    print(f"  risk_score (P*100):    {int(round(probs[1] * 100))}")


def check_name_alignment(backbone_info: dict, head_info: dict) -> None:
    print(f"\n{SEP}")
    print("NAME ALIGNMENT CHECK (expected vs actual)")
    print(SEP)
    ok = True
    for expected_name, (expected_shape, expected_dtype) in EXPECTED["backbone_inputs"].items():
        if expected_name not in backbone_info:
            print(f"  [MISMATCH] backbone input {expected_name!r} not found  "
                  f"(actual: {list(backbone_info)})")
            ok = False
        else:
            print(f"  [OK] backbone input  {expected_name!r}")
    for expected_name in EXPECTED["backbone_outputs"]:
        if expected_name not in backbone_info:
            print(f"  [MISMATCH] backbone output {expected_name!r} not found  "
                  f"(actual: {[k for k in backbone_info if k not in EXPECTED['backbone_inputs']]})")
            ok = False
        else:
            print(f"  [OK] backbone output {expected_name!r}")
    for expected_name in EXPECTED["head_inputs"]:
        if expected_name not in head_info:
            print(f"  [MISMATCH] head input {expected_name!r} not found  "
                  f"(actual: {list(head_info)})")
            ok = False
        else:
            print(f"  [OK] head input      {expected_name!r}")
    for expected_name in EXPECTED["head_outputs"]:
        if expected_name not in head_info:
            print(f"  [MISMATCH] head output {expected_name!r} not found  "
                  f"(actual: {list(head_info)})")
            ok = False
        else:
            print(f"  [OK] head output     {expected_name!r}")

    if ok:
        print("\n  All expected tensor names present — backend should work as coded.")
    else:
        print(
            "\n  ACTION: update the hard-coded tensor names in "
            "Wav2Vec2TwoStageBackend.run() to match the actual names above."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE,
                    help="Path to backbone model.onnx")
    ap.add_argument("--head",     type=Path, default=DEFAULT_HEAD,
                    help="Path to head .onnx")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — pip install onnxruntime")
        return 1

    print(SEP)
    print("MODEL FILE CHECK")
    print(SEP)
    for label, p in (("backbone", args.backbone), ("head", args.head)):
        exists = p.exists()
        size   = f"{p.stat().st_size / 1024:.0f} KB" if exists else "---"
        status = "OK" if exists else "NOT FOUND"
        print(f"  [{status}] {label}: {p}  ({size})")
    if not args.backbone.exists() or not args.head.exists():
        print("\nFix the missing paths and re-run.")
        return 1

    print(f"\n{SEP}")
    print("TENSOR INSPECTION")
    print(SEP)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]

    backbone_session = ort.InferenceSession(str(args.backbone), opts, providers=providers)
    head_session     = ort.InferenceSession(str(args.head),     opts, providers=providers)

    backbone_info = inspect_session("BACKBONE", backbone_session)
    head_info     = inspect_session("HEAD",     head_session)

    check_name_alignment(backbone_info, head_info)
    run_smoke_test(backbone_session, head_session, backbone_info, head_info)

    print(f"\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
