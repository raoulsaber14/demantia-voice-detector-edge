"""Runtime for the official Phase D max-probability ensemble."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from src.audio_utils import load_audio_safe
from src.features import extract_features_from_audio

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASED_ENSEMBLE_MANIFEST = ROOT / "models" / "phaseD_frozen_ensemble_deploy" / "max_probability_ensemble_manifest.json"


def _load_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - runtime dependency error.
        raise RuntimeError("onnxruntime is required for Phase D ensemble inference.") from exc
    return ort


def _repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=None)
def _load_json(path_str: str) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _load_session(model_path_str: str):
    ort = _load_onnxruntime()
    return ort.InferenceSession(str(model_path_str), providers=["CPUExecutionProvider"])


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax_positive_probability(logits: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return float(probs[0, 1])


def _segment_audio(audio: np.ndarray, sr: int, chunk_seconds: float, min_chunk_seconds: float) -> list[tuple[np.ndarray, int]]:
    chunk_samples = int(round(float(chunk_seconds) * sr))
    min_samples = int(round(float(min_chunk_seconds) * sr))
    if chunk_samples <= 0:
        raise ValueError("chunk_seconds must be positive.")
    if audio.size == 0:
        padded = np.zeros(chunk_samples, dtype=np.float32)
        return [(padded, min_samples)]

    raw_chunks = [audio[start : start + chunk_samples].astype(np.float32) for start in range(0, len(audio), chunk_samples)]
    raw_chunks = [chunk for chunk in raw_chunks if len(chunk) > 0]
    if len(raw_chunks) > 1 and len(raw_chunks[-1]) < min_samples:
        raw_chunks = raw_chunks[:-1]
    if not raw_chunks:
        padded = np.zeros(chunk_samples, dtype=np.float32)
        return [(padded, min_samples)]

    chunk_specs: list[tuple[np.ndarray, int]] = []
    for idx, chunk in enumerate(raw_chunks):
        valid_samples = len(chunk)
        if len(raw_chunks) == 1 and valid_samples < min_samples:
            valid_samples = min_samples
        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[: len(chunk)] = chunk[:chunk_samples]
        chunk_specs.append((padded, int(valid_samples)))
    return chunk_specs


def _embedding_from_accumulator(audio_path: Path, backbone_info: dict[str, Any]) -> np.ndarray:
    contract = _load_json(str(_repo_path(backbone_info["contract_json"]).resolve()))
    model_path = _repo_path(backbone_info["preferred_external_model"])
    session = _load_session(str(model_path.resolve()))

    input_contract = contract["input_contract"]
    chunk_duration = float(input_contract["chunk_duration_seconds"])
    min_chunk_duration = float(input_contract["minimum_retained_chunk_seconds"])
    sampling_rate = int(input_contract["sampling_rate_hz"])

    audio, sr, error = load_audio_safe(audio_path, target_sr=sampling_rate)
    if error or audio is None or sr is None:
        raise RuntimeError(f"Audio load failed for {audio_path}: {error or 'unknown error'}")
    audio = np.asarray(audio, dtype=np.float32)
    chunk_specs = _segment_audio(audio, sr, chunk_duration, min_chunk_duration)

    sum_vec: np.ndarray | None = None
    sumsq_vec: np.ndarray | None = None
    frame_count_total = 0.0
    for padded_chunk, valid_samples in chunk_specs:
        ort_outputs = session.run(
            ["sum_vec", "sumsq_vec", "frame_count"],
            {
                "input_audio": padded_chunk.reshape(1, -1).astype(np.float32),
                "valid_samples": np.asarray([valid_samples], dtype=np.int64),
            },
        )
        chunk_sum = np.asarray(ort_outputs[0], dtype=np.float32)[0]
        chunk_sumsq = np.asarray(ort_outputs[1], dtype=np.float32)[0]
        chunk_frame_count = float(np.asarray(ort_outputs[2], dtype=np.float32)[0, 0])
        sum_vec = chunk_sum if sum_vec is None else sum_vec + chunk_sum
        sumsq_vec = chunk_sumsq if sumsq_vec is None else sumsq_vec + chunk_sumsq
        frame_count_total += chunk_frame_count

    if sum_vec is None or sumsq_vec is None or frame_count_total <= 0:
        raise RuntimeError(f"No valid backbone frames were produced for {audio_path}.")

    mean = sum_vec / frame_count_total
    variance = np.maximum((sumsq_vec / frame_count_total) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return np.concatenate([mean, std]).astype(np.float32)


def _score_component(feature_vector: np.ndarray, pipeline: dict[str, Any]) -> float:
    stats = np.asarray(pipeline["imputer_statistics"], dtype=np.float32)
    mean = np.asarray(pipeline["scaler_mean"], dtype=np.float32)
    scale = np.asarray(pipeline["scaler_scale"], dtype=np.float32)
    coef = np.asarray(pipeline["linear_model_coef"], dtype=np.float32)
    intercept = float(pipeline["linear_model_intercept"])
    calibrator_coef = float(pipeline["calibrator_coef"])
    calibrator_intercept = float(pipeline["calibrator_intercept"])

    x = np.asarray(feature_vector, dtype=np.float32).copy()
    if x.shape[0] != stats.shape[0]:
        raise ValueError(f"Feature length mismatch: got {x.shape[0]}, expected {stats.shape[0]}.")
    missing_mask = ~np.isfinite(x)
    if missing_mask.any():
        x[missing_mask] = stats[missing_mask]
    safe_scale = np.where(scale == 0, 1.0, scale)
    x = (x - mean) / safe_scale
    raw_score = float(np.dot(x, coef) + intercept)
    calibrated_score = calibrator_coef * raw_score + calibrator_intercept
    return float(_sigmoid(calibrated_score))


def _score_component_onnx(feature_vector: np.ndarray, component: dict[str, Any]) -> float:
    model_path = _repo_path(component["head_onnx"])
    session = _load_session(str(model_path.resolve()))
    logits = session.run(
        ["logits"],
        {"input_features": np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)},
    )[0]
    return _softmax_positive_probability(logits)


def _build_component_vector(
    component: dict[str, Any],
    hubert_embedding: np.ndarray,
    wav2vec_embedding: np.ndarray,
    handcrafted_features: dict[str, float],
) -> np.ndarray:
    pipeline = component["pipeline"]
    backbone = str(component["backbone"])
    embedding = hubert_embedding if "hubert" in backbone.lower() else wav2vec_embedding
    values: list[float] = []
    for name in pipeline["feature_names"]:
        if name.startswith("emb_"):
            idx = int(name.split("_", 1)[1])
            values.append(float(embedding[idx]))
        else:
            values.append(float(handcrafted_features.get(name, np.nan)))
    return np.asarray(values, dtype=np.float32)


def _aggregate_component(probabilities: Iterable[float], aggregation: str, threshold: float) -> tuple[float, int]:
    probs = np.asarray(list(probabilities), dtype=float)
    if probs.size == 0:
        raise ValueError("Cannot aggregate zero clip probabilities.")
    if aggregation == "majority_vote":
        clip_votes = (probs >= threshold).astype(float)
        score = float(np.mean(clip_votes))
        pred = int(score >= 0.5)
        return score, pred
    if aggregation == "mean_score":
        score = float(np.mean(probs))
        pred = int(score >= threshold)
        return score, pred
    raise ValueError(f"Unsupported Phase D runtime aggregation: {aggregation}")


def predict_with_phaseD_ensemble(
    audio_paths: Path | list[Path],
    manifest_path: Path = DEFAULT_PHASED_ENSEMBLE_MANIFEST,
    threshold_override: float | None = None,
) -> Dict[str, object]:
    """Run the official Phase D max-probability ensemble on one or more cleaned clips."""
    if isinstance(audio_paths, Path):
        clip_paths = [audio_paths]
    else:
        clip_paths = [Path(p) for p in audio_paths]
    if not clip_paths:
        raise ValueError("At least one audio path is required for Phase D ensemble inference.")

    manifest = _load_json(str(manifest_path.resolve()))
    threshold = float(manifest["ensemble_threshold"] if threshold_override is None else threshold_override)
    backbone_map = {item["backbone"]: item for item in manifest["backbones"]}
    components = [
        _load_json(str(_repo_path(item["path"]).resolve())) if "pipeline" not in item else item
        for item in manifest["components"]
    ]

    component_probabilities: dict[str, list[float]] = {
        item["component_name"]: [] for item in components
    }

    for clip_path in clip_paths:
        hubert_embedding = _embedding_from_accumulator(clip_path, backbone_map["facebook/hubert-base-ls960"])
        wav2vec_embedding = _embedding_from_accumulator(clip_path, backbone_map["facebook/wav2vec2-base"])
        handcrafted = extract_features_from_audio(clip_path)
        for component in components:
            vector = _build_component_vector(component, hubert_embedding, wav2vec_embedding, handcrafted)
            if component.get("head_onnx"):
                probability = _score_component_onnx(vector, component)
            else:
                probability = _score_component(vector, component["pipeline"])
            component_probabilities[component["component_name"]].append(probability)

    component_scores: dict[str, dict[str, object]] = {}
    for component in components:
        name = component["component_name"]
        score, pred = _aggregate_component(component_probabilities[name], component["aggregation"], float(component["threshold"]))
        component_scores[name] = {
            "clip_probabilities": [float(v) for v in component_probabilities[name]],
            "speaker_level_score": float(score),
            "speaker_level_prediction": int(pred),
            "aggregation": component["aggregation"],
            "threshold": float(component["threshold"]),
        }

    max_component_name, max_component = max(
        component_scores.items(),
        key=lambda item: (float(item[1]["speaker_level_score"]), item[0]),
    )
    final_score = float(max_component["speaker_level_score"])
    final_pred = "dementia" if final_score >= threshold else "non_dementia"

    return {
        "predicted_label": final_pred,
        "confidence_score": final_score if final_pred == "dementia" else 1.0 - final_score,
        "probability_dementia": final_score,
        "threshold": threshold,
        "ensemble_method": manifest["ensemble_method"],
        "official_candidate_name": manifest["official_candidate_name"],
        "max_component_name": max_component_name,
        "component_scores": component_scores,
        "n_clips": int(len(clip_paths)),
        "explanation": (
            "Prediction from the official Phase D max-probability ensemble using HuBERT-only, "
            "HuBERT + handcrafted, and wav2vec2-base components."
        ),
    }
