#!/usr/bin/env python3
"""Package deployable assets for the official Phase D max-probability ensemble.

This script does not retrain a new architecture. It:
1. exports the frozen HuBERT and wav2vec2-base backbones used by Phase C/D to
   ONNX INT8 external-data packages;
2. fits the same simple calibrated logistic heads on the trusted cleaned
   subset using the locked Phase D recipe so the hardware team has concrete
   deployable component artifacts;
3. writes a manifest describing the official max-probability ensemble.

Important: the deployment heads created here are full-data packaging artifacts.
They are not new evaluation results and should not be presented as unbiased
held-out metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx.external_data_helper import convert_model_to_external_data
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "models" / "phaseD_frozen_ensemble_deploy"
DEFAULT_SEED = 42
CHUNK_SECONDS = 8.0
MIN_CHUNK_SECONDS = 0.5
MAX_COMPONENT_THRESHOLD = 0.5


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    slug: str
    model_id: str


BACKBONES = [
    BackboneSpec(name="HuBERT", slug="hubert_base_ls960", model_id="facebook/hubert-base-ls960"),
    BackboneSpec(name="wav2vec2-base", slug="wav2vec2_base", model_id="facebook/wav2vec2-base"),
]


class ChunkAccumulator(torch.nn.Module):
    """Return framewise sums so the host can reproduce Phase C mean+std pooling."""

    def __init__(self, model: Any, chunk_samples: int):
        super().__init__()
        self.model = model
        self.chunk_samples = int(chunk_samples)

    def forward(self, input_audio: torch.Tensor, valid_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input_audio.shape[0]
        clipped_lengths = valid_samples.to(dtype=torch.long).clamp(min=1, max=self.chunk_samples)
        position_ids = torch.arange(self.chunk_samples, device=input_audio.device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = (position_ids < clipped_lengths.unsqueeze(1)).to(dtype=torch.long)
        outputs = self.model(input_values=input_audio, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        frame_lengths = self.model._get_feat_extract_output_lengths(clipped_lengths).to(dtype=torch.long)
        frame_positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand(batch_size, -1)
        frame_mask = (frame_positions < frame_lengths.unsqueeze(1)).to(dtype=hidden.dtype).unsqueeze(-1)
        masked_hidden = hidden * frame_mask
        sum_vec = masked_hidden.sum(dim=1)
        sumsq_vec = (masked_hidden * masked_hidden).sum(dim=1)
        frame_count = frame_lengths.to(dtype=hidden.dtype).unsqueeze(1)
        return sum_vec, sumsq_vec, frame_count


class CalibratedLogisticHead(torch.nn.Module):
    """Tiny ONNX-exportable head matching the packaged JSON scorer."""

    def __init__(self, pipeline: dict[str, Any]):
        super().__init__()
        self.register_buffer("imputer_statistics", torch.tensor(pipeline["imputer_statistics"], dtype=torch.float32))
        self.register_buffer("scaler_mean", torch.tensor(pipeline["scaler_mean"], dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.tensor(pipeline["scaler_scale"], dtype=torch.float32))
        self.register_buffer("linear_model_coef", torch.tensor(pipeline["linear_model_coef"], dtype=torch.float32))
        self.register_buffer("linear_model_intercept", torch.tensor([pipeline["linear_model_intercept"]], dtype=torch.float32))
        self.register_buffer("calibrator_coef", torch.tensor([pipeline["calibrator_coef"]], dtype=torch.float32))
        self.register_buffer("calibrator_intercept", torch.tensor([pipeline["calibrator_intercept"]], dtype=torch.float32))

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        stats = self.imputer_statistics.unsqueeze(0).expand_as(input_features)
        mean = self.scaler_mean.unsqueeze(0).expand_as(input_features)
        scale = torch.where(self.scaler_scale == 0, torch.ones_like(self.scaler_scale), self.scaler_scale)
        scale = scale.unsqueeze(0).expand_as(input_features)
        x = torch.where(torch.isfinite(input_features), input_features, stats)
        x = (x - mean) / scale
        raw_score = torch.matmul(x, self.linear_model_coef.unsqueeze(1)).squeeze(1) + self.linear_model_intercept
        calibrated_score = (raw_score * self.calibrator_coef) + self.calibrator_intercept
        zeros = torch.zeros_like(calibrated_score)
        return torch.stack([zeros, calibrated_score], dim=1)


def file_size_bytes(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def dir_total_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def dir_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def dir_max_file_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    sizes = [p.stat().st_size for p in path.rglob("*") if p.is_file()]
    return max(sizes) if sizes else None


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_phase_d_module() -> Any:
    module_path = ROOT / "scripts" / "phaseD_hubert_recall_recovery_benchmark.py"
    spec = importlib.util.spec_from_file_location("phase_d_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Phase D module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def export_backbone(backbone: BackboneSpec) -> dict[str, Any]:
    model = AutoModel.from_pretrained(backbone.model_id, local_files_only=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model = model.to(torch.device("cpu"))
    sampling_rate = 16000
    chunk_samples = int(round(CHUNK_SECONDS * sampling_rate))
    wrapper = ChunkAccumulator(model=model, chunk_samples=chunk_samples).to(torch.device("cpu"))
    wrapper.eval()

    backbone_dir = OUTPUT_DIR / backbone.slug
    fp32_path = backbone_dir / f"{backbone.slug}_chunk8_accumulator_fp32.onnx"
    int8_path = backbone_dir / f"{backbone.slug}_chunk8_accumulator_int8.onnx"
    fp32_external_dir = backbone_dir / f"{backbone.slug}_chunk8_accumulator_fp32_external"
    fp32_external_model_path = fp32_external_dir / "model.onnx"
    external_dir = backbone_dir / f"{backbone.slug}_chunk8_accumulator_int8_external"
    external_model_path = external_dir / "model.onnx"
    contract_path = backbone_dir / f"{backbone.slug}_chunk8_accumulator_contract.json"

    if backbone_dir.exists():
        shutil.rmtree(backbone_dir)
    backbone_dir.mkdir(parents=True, exist_ok=True)

    dummy_audio = torch.zeros(1, chunk_samples, dtype=torch.float32)
    dummy_valid = torch.tensor([chunk_samples], dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_audio, dummy_valid),
            str(fp32_path),
            input_names=["input_audio", "valid_samples"],
            output_names=["sum_vec", "sumsq_vec", "frame_count"],
            opset_version=17,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(str(fp32_path))
    onnx.checker.check_model(onnx_model)

    rng = np.random.default_rng(42)
    sample_valid = int(round(5.25 * sampling_rate))
    sample_audio = rng.normal(0.0, 0.05, size=(1, chunk_samples)).astype(np.float32)
    sample_audio[:, sample_valid:] = 0.0
    sample_valid_np = np.asarray([sample_valid], dtype=np.int64)

    with torch.no_grad():
        torch_sum, torch_sumsq, torch_count = wrapper(
            torch.from_numpy(sample_audio), torch.from_numpy(sample_valid_np)
        )
        torch_outputs = [torch_sum.numpy(), torch_sumsq.numpy(), torch_count.numpy()]

    ort_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    ort_outputs = ort_session.run(
        ["sum_vec", "sumsq_vec", "frame_count"],
        {"input_audio": sample_audio, "valid_samples": sample_valid_np},
    )
    fp32_max_abs_diff = float(max(np.max(np.abs(a - b)) for a, b in zip(torch_outputs, ort_outputs)))

    fp32_external_dir.mkdir(parents=True, exist_ok=True)
    fp32_external_model = onnx.load(str(fp32_path))
    convert_model_to_external_data(
        fp32_external_model,
        all_tensors_to_one_file=False,
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.save_model(fp32_external_model, str(fp32_external_model_path))
    fp32_external_session = ort.InferenceSession(str(fp32_external_model_path), providers=["CPUExecutionProvider"])
    fp32_external_outputs = fp32_external_session.run(
        ["sum_vec", "sumsq_vec", "frame_count"],
        {"input_audio": sample_audio, "valid_samples": sample_valid_np},
    )
    fp32_external_max_abs_diff = float(max(np.max(np.abs(a - b)) for a, b in zip(torch_outputs, fp32_external_outputs)))

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    int8_outputs = int8_session.run(
        ["sum_vec", "sumsq_vec", "frame_count"],
        {"input_audio": sample_audio, "valid_samples": sample_valid_np},
    )
    int8_max_abs_diff = float(max(np.max(np.abs(a - b)) for a, b in zip(torch_outputs, int8_outputs)))

    external_dir.mkdir(parents=True, exist_ok=True)
    q_model = onnx.load(str(int8_path))
    convert_model_to_external_data(
        q_model,
        all_tensors_to_one_file=False,
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.save_model(q_model, str(external_model_path))
    external_session = ort.InferenceSession(str(external_model_path), providers=["CPUExecutionProvider"])
    external_outputs = external_session.run(
        ["sum_vec", "sumsq_vec", "frame_count"],
        {"input_audio": sample_audio, "valid_samples": sample_valid_np},
    )
    external_max_abs_diff = float(max(np.max(np.abs(a - b)) for a, b in zip(torch_outputs, external_outputs)))

    contract = {
        "export_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backbone": backbone.model_id,
        "source_model_type": "frozen_backbone_chunk_accumulator",
        "input_contract": {
            "input_audio_tensor": {"name": "input_audio", "shape": [1, chunk_samples], "dtype": "float32"},
            "valid_samples_tensor": {"name": "valid_samples", "shape": [1], "dtype": "int64"},
            "sampling_rate_hz": sampling_rate,
            "chunk_duration_seconds": CHUNK_SECONDS,
            "minimum_retained_chunk_seconds": MIN_CHUNK_SECONDS,
        },
        "output_contract": {
            "sum_vec": {"shape": [1, 768], "dtype": "float32"},
            "sumsq_vec": {"shape": [1, 768], "dtype": "float32"},
            "frame_count": {"shape": [1, 1], "dtype": "float32"},
            "embedding_reconstruction": "mean = sum_vec / frame_count; std = sqrt(max(sumsq_vec / frame_count - mean^2, 0)); embedding = concat(mean, std)",
        },
        "clip_level_rule": {
            "chunk_generation": "contiguous non-overlapping 8-second windows",
            "drop_final_chunk_if_shorter_than_seconds_when_multiple_chunks": MIN_CHUNK_SECONDS,
            "if_single_chunk_shorter_than_minimum_seconds": "treat valid_samples as 0.5 seconds before padding to full chunk tensor",
            "pooling": "global mean+std across all valid backbone frames across all retained chunks",
        },
        "artifacts": {
            "preferred_runtime_variant": "fp32_external",
            "fp32_external_model": rel(fp32_external_model_path),
            "fp32_external_total_size_bytes": dir_total_size_bytes(fp32_external_dir),
            "fp32_external_file_count": dir_file_count(fp32_external_dir),
            "fp32_external_max_file_size_bytes": dir_max_file_size_bytes(fp32_external_dir),
            "external_model": rel(external_model_path),
            "external_total_size_bytes": dir_total_size_bytes(external_dir),
            "external_file_count": dir_file_count(external_dir),
            "external_max_file_size_bytes": dir_max_file_size_bytes(external_dir),
        },
        "validation": {
            "sample_valid_samples": sample_valid,
            "fp32_max_abs_diff_vs_torch": fp32_max_abs_diff,
            "fp32_external_max_abs_diff_vs_torch": fp32_external_max_abs_diff,
            "int8_max_abs_diff_vs_torch": int8_max_abs_diff,
            "int8_external_max_abs_diff_vs_torch": external_max_abs_diff,
        },
    }
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    fp32_path.unlink(missing_ok=True)
    int8_path.unlink(missing_ok=True)

    return {
        "backbone": backbone.model_id,
        "slug": backbone.slug,
        "contract_json": rel(contract_path),
        "preferred_external_model": rel(fp32_external_model_path),
        "preferred_runtime_variant": "fp32_external",
        "fp32_external_model": rel(fp32_external_model_path),
        "fp32_external_total_size_bytes": dir_total_size_bytes(fp32_external_dir),
        "fp32_external_max_file_size_bytes": dir_max_file_size_bytes(fp32_external_dir),
        "external_model": rel(external_model_path),
        "external_total_size_bytes": dir_total_size_bytes(external_dir),
        "external_file_count": dir_file_count(external_dir),
        "external_max_file_size_bytes": dir_max_file_size_bytes(external_dir),
        "fp32_max_abs_diff_vs_torch": fp32_max_abs_diff,
        "fp32_external_max_abs_diff_vs_torch": fp32_external_max_abs_diff,
        "int8_external_max_abs_diff_vs_torch": external_max_abs_diff,
        "chunk_samples": chunk_samples,
    }


def fit_calibrated_logistic_package(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[Any, Any]:
    estimator = phase_d.build_estimator(phase_d.LOGISTIC, seed=seed, c_value=1.0)
    estimator.fit(x, y)
    raw = phase_d.raw_scores(estimator, x)
    calibrator = LogisticRegression(max_iter=2000, random_state=seed)
    calibrator.fit(raw.reshape(-1, 1), y)
    return estimator, calibrator


def score_pipeline_vector(x: np.ndarray, pipeline: dict[str, Any]) -> float:
    stats = np.asarray(pipeline["imputer_statistics"], dtype=np.float32)
    mean = np.asarray(pipeline["scaler_mean"], dtype=np.float32)
    scale = np.asarray(pipeline["scaler_scale"], dtype=np.float32)
    coef = np.asarray(pipeline["linear_model_coef"], dtype=np.float32)
    intercept = float(pipeline["linear_model_intercept"])
    calibrator_coef = float(pipeline["calibrator_coef"])
    calibrator_intercept = float(pipeline["calibrator_intercept"])

    x = np.asarray(x, dtype=np.float32).copy()
    missing_mask = ~np.isfinite(x)
    if missing_mask.any():
        x[missing_mask] = stats[missing_mask]
    safe_scale = np.where(scale == 0, 1.0, scale)
    x = (x - mean) / safe_scale
    raw_score = float(np.dot(x, coef) + intercept)
    calibrated_score = (calibrator_coef * raw_score) + calibrator_intercept
    return float(1.0 / (1.0 + np.exp(-calibrated_score)))


def softmax_positive_probability(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs[:, 1].astype(np.float64)


def pipeline_json(estimator: Any, calibrator: Any, feature_names: list[str]) -> dict[str, Any]:
    imputer = estimator.named_steps["imputer"]
    scaler = estimator.named_steps["scaler"]
    model = estimator.named_steps["model"]
    return {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "imputer_strategy": "median",
        "imputer_statistics": [float(v) for v in imputer.statistics_.tolist()],
        "scaler_mean": [float(v) for v in scaler.mean_.tolist()],
        "scaler_scale": [float(v) for v in scaler.scale_.tolist()],
        "linear_model_type": type(model).__name__,
        "linear_model_coef": [float(v) for v in model.coef_[0].tolist()],
        "linear_model_intercept": float(model.intercept_[0]),
        "calibrator_type": type(calibrator).__name__,
        "calibrator_coef": float(calibrator.coef_[0][0]),
        "calibrator_intercept": float(calibrator.intercept_[0]),
        "positive_class_index": 1,
        "score_semantics": "raw linear decision score -> Platt calibrator sigmoid probability",
    }


def feature_input_description(feature_mode: str, feature_names: list[str]) -> str:
    if feature_mode == "hubert_only":
        return "Input must be a 1536-dim Phase D HuBERT mean+std embedding in exact pipeline.feature_names order."
    if feature_mode == "wav2vec2_only":
        return "Input must be a 1536-dim Phase D wav2vec2 mean+std embedding in exact pipeline.feature_names order."
    if feature_mode == "hubert_plus_handcrafted":
        handcrafted_count = len([name for name in feature_names if not name.startswith("emb_")])
        return (
            "Input must be a 1792-dim feature vector in exact pipeline.feature_names order: "
            f"1536 HuBERT embedding values followed by {handcrafted_count} handcrafted features."
        )
    return "Input must follow pipeline.feature_names order."


def export_component_head_onnx(
    *,
    component_name: str,
    feature_mode: str,
    feature_names: list[str],
    pipeline: dict[str, Any],
    output_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    head = CalibratedLogisticHead(pipeline).to(torch.device("cpu"))
    head.eval()
    feature_count = len(feature_names)
    dummy_input = torch.zeros(1, feature_count, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            head,
            dummy_input,
            str(output_path),
            input_names=["input_features"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={"input_features": {0: "batch"}, "logits": {0: "batch"}},
        )

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    sample_input = rng.normal(0.0, 0.5, size=(8, feature_count)).astype(np.float32)
    if feature_count >= 2:
        sample_input[0, 0] = np.nan
        sample_input[1, -1] = np.nan
    logits = session.run(["logits"], {"input_features": sample_input})[0]
    onnx_probs = softmax_positive_probability(logits)
    reference_probs = np.asarray([score_pipeline_vector(row, pipeline) for row in sample_input], dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(onnx_probs - reference_probs)))

    contract = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "component_name": component_name,
        "feature_mode": feature_mode,
        "deployment_note": "Tiny FP32 ONNX head for use after Phase D embedding reconstruction.",
        "input_contract": {
            "tensor": {"name": "input_features", "shape": [1, feature_count], "dtype": "float32"},
            "feature_order": feature_names,
            "description": feature_input_description(feature_mode, feature_names),
        },
        "output_contract": {
            "tensor": {"name": "logits", "shape": [1, 2], "dtype": "float32"},
            "positive_class_index": 1,
            "logit_semantics": "softmax(logits)[1] equals the calibrated dementia probability",
            "construction": "logits[:, 0] = 0; logits[:, 1] = Platt-calibrated scalar score",
        },
        "artifacts": {
            "onnx_model": rel(output_path),
            "size_bytes": file_size_bytes(output_path),
        },
        "validation": {
            "sample_batch_size": int(sample_input.shape[0]),
            "max_abs_diff_probability_vs_json_pipeline": max_abs_diff,
        },
    }
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return {
        "head_onnx": rel(output_path),
        "head_contract_json": rel(contract_path),
        "head_validation_max_abs_diff": max_abs_diff,
    }


def write_component_package(
    *,
    path: Path,
    component_name: str,
    feature_mode: str,
    backbone: str,
    aggregation: str,
    threshold: float,
    feature_names: list[str],
    estimator: Any,
    calibrator: Any,
) -> dict[str, Any]:
    pipeline = pipeline_json(estimator, calibrator, feature_names)
    head_export = export_component_head_onnx(
        component_name=component_name,
        feature_mode=feature_mode,
        feature_names=feature_names,
        pipeline=pipeline,
        output_path=path.with_name(f"{path.stem}_head_fp32.onnx"),
        contract_path=path.with_name(f"{path.stem}_head_contract.json"),
    )
    package = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "component_name": component_name,
        "feature_mode": feature_mode,
        "backbone": backbone,
        "classifier": phase_d.LOGISTIC,
        "aggregation": aggregation,
        "threshold_policy": "default_0.5",
        "threshold": threshold,
        "deployment_note": (
            "Full-data deployment fit using the locked Phase D recipe on the trusted cleaned subset. "
            "This artifact is for runtime packaging, not new held-out evaluation."
        ),
        "pipeline": pipeline,
        **head_export,
    }
    path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    return {
        "path": rel(path),
        "component_name": component_name,
        "feature_count": len(feature_names),
        "head_onnx": head_export["head_onnx"],
        "head_contract_json": head_export["head_contract_json"],
    }


def load_existing_backbone_exports() -> list[dict[str, Any]]:
    manifest_path = OUTPUT_DIR / "max_probability_ensemble_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist, so --heads-only cannot reuse the existing backbone exports."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backbones = manifest.get("backbones", [])
    if not backbones:
        raise ValueError("Existing ensemble manifest has no backbone entries.")
    return backbones


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heads-only",
        action="store_true",
        help="Regenerate only the small classifier-head ONNX files and refresh component/manifest metadata.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global phase_d  # noqa: PLW0603 - script-local singleton for helper reuse.
    phase_d = load_phase_d_module()

    backbone_exports = load_existing_backbone_exports() if args.heads_only else [export_backbone(spec) for spec in BACKBONES]

    manifest = phase_d.load_manifest()
    y = manifest["label"].astype(int).to_numpy()

    hubert_embeddings, hubert_meta = phase_d.load_cached_embeddings(phase_d.HUBERT_BACKBONE, manifest)
    wav2vec_embeddings, wav2vec_meta = phase_d.load_cached_embeddings(phase_d.WAV2VEC2_BACKBONE, manifest)
    x_hubert, hubert_feature_names = phase_d.embedding_matrix(hubert_embeddings)
    x_wav2vec, wav2vec_feature_names = phase_d.embedding_matrix(wav2vec_embeddings)
    handcrafted_df, handcrafted_feature_names, handcrafted_join = phase_d.load_aligned_handcrafted_features(manifest)
    x_handcrafted = handcrafted_df.to_numpy(dtype=np.float32)
    x_hubert_plus_handcrafted = np.hstack([x_hubert, x_handcrafted]).astype(np.float32)
    fusion_feature_names = [*hubert_feature_names, *handcrafted_feature_names]

    hubert_estimator, hubert_calibrator = fit_calibrated_logistic_package(x_hubert, y, DEFAULT_SEED)
    fusion_estimator, fusion_calibrator = fit_calibrated_logistic_package(x_hubert_plus_handcrafted, y, DEFAULT_SEED)
    wav2vec_estimator, wav2vec_calibrator = fit_calibrated_logistic_package(x_wav2vec, y, DEFAULT_SEED)

    hubert_component = write_component_package(
        path=OUTPUT_DIR / "hubert_only_component.json",
        component_name="HuBERT-only logistic majority_vote default_0.5",
        feature_mode="hubert_only",
        backbone=phase_d.HUBERT_BACKBONE,
        aggregation="majority_vote",
        threshold=MAX_COMPONENT_THRESHOLD,
        feature_names=hubert_feature_names,
        estimator=hubert_estimator,
        calibrator=hubert_calibrator,
    )
    fusion_component = write_component_package(
        path=OUTPUT_DIR / "hubert_plus_handcrafted_component.json",
        component_name="HuBERT+handcrafted logistic_regression_platt majority_vote default_0.5",
        feature_mode="hubert_plus_handcrafted",
        backbone=phase_d.HUBERT_BACKBONE,
        aggregation="majority_vote",
        threshold=MAX_COMPONENT_THRESHOLD,
        feature_names=fusion_feature_names,
        estimator=fusion_estimator,
        calibrator=fusion_calibrator,
    )
    wav2vec_component = write_component_package(
        path=OUTPUT_DIR / "wav2vec2_component.json",
        component_name="facebook/wav2vec2-base logistic mean_score default_0.5",
        feature_mode="wav2vec2_only",
        backbone=phase_d.WAV2VEC2_BACKBONE,
        aggregation="mean_score",
        threshold=MAX_COMPONENT_THRESHOLD,
        feature_names=wav2vec_feature_names,
        estimator=wav2vec_estimator,
        calibrator=wav2vec_calibrator,
    )

    manifest_json = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "official_candidate_name": "max_probability_ensemble default_0.5",
        "ensemble_method": "max_probability_ensemble",
        "ensemble_threshold": MAX_COMPONENT_THRESHOLD,
        "deployment_note": (
            "This manifest packages the official Phase D ensemble components for runtime use. "
            "The backbone exports are frozen pretrained models; the small linear heads are full-data "
            "deployment fits using the same locked Phase D recipe."
        ),
        "trusted_protocol_reference": {
            "dataset_manifest": rel(phase_d.TRUSTED_MANIFEST),
            "hubert_cache": hubert_meta.get("cache_csv"),
            "wav2vec2_cache": wav2vec_meta.get("cache_csv"),
            "handcrafted_feature_table": rel(phase_d.FEATURE_TABLE),
            "handcrafted_join": handcrafted_join,
        },
        "backbones": backbone_exports,
        "components": [hubert_component, fusion_component, wav2vec_component],
        "ensemble_rule": {
            "speaker_level_score": "max(component_probability_1, component_probability_2, component_probability_3)",
            "speaker_level_prediction": "positive if max score >= 0.5 else negative",
            "component_aggregations": {
                "HuBERT-only logistic majority_vote default_0.5": "majority_vote over clip probabilities at threshold 0.5",
                "HuBERT+handcrafted logistic_regression_platt majority_vote default_0.5": "majority_vote over clip probabilities at threshold 0.5",
                "facebook/wav2vec2-base logistic mean_score default_0.5": "mean of clip probabilities",
            },
        },
    }
    manifest_path = OUTPUT_DIR / "max_probability_ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest_json, indent=2) + "\n", encoding="utf-8")

    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    for export in backbone_exports:
        print(
            f"BACKBONE={export['backbone']} EXTERNAL_MODEL={export['external_model']} "
            f"TOTAL_SIZE_BYTES={export['external_total_size_bytes']} MAX_FILE_BYTES={export['external_max_file_size_bytes']}"
        )
    print(f"COMPONENT_JSON={hubert_component['path']}")
    print(f"COMPONENT_HEAD_ONNX={hubert_component['head_onnx']}")
    print(f"COMPONENT_JSON={fusion_component['path']}")
    print(f"COMPONENT_HEAD_ONNX={fusion_component['head_onnx']}")
    print(f"COMPONENT_JSON={wav2vec_component['path']}")
    print(f"COMPONENT_HEAD_ONNX={wav2vec_component['head_onnx']}")
    print(f"ENSEMBLE_MANIFEST={rel(manifest_path)}")


if __name__ == "__main__":
    main()
