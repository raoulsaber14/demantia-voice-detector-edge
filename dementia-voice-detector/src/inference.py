"""Inference utilities for baseline, wav2vec, and the final Phase D ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Config, Wav2Vec2Model

from src.audio_utils import load_audio_safe, normalize_amplitude, trim_silence
from src.features import extract_features_from_audio
from src.phaseD_ensemble_runtime import DEFAULT_PHASED_ENSEMBLE_MANIFEST, predict_with_phaseD_ensemble

DISCLAIMER = "This tool is for screening support only and is not a medical diagnosis."


def predict_with_baseline(audio_path: Path, model_path: Path, threshold: float = 0.5) -> Dict[str, object]:
    """Run prediction using a saved baseline sklearn model."""
    feats = extract_features_from_audio(audio_path)
    model = joblib.load(model_path)
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        # Fallback for pipelines that do not preserve names.
        feature_names = [k for k, v in feats.items() if isinstance(v, (int, float))]
    x = np.array([[feats.get(c, np.nan) for c in feature_names]])
    prob = float(model.predict_proba(x)[0, 1])
    pred = "dementia" if prob >= threshold else "non_dementia"
    return {
        "predicted_label": pred,
        "confidence_score": prob if pred == "dementia" else 1.0 - prob,
        "probability_dementia": prob,
        "threshold": float(threshold),
        "explanation": "Prediction from acoustic baseline features (MFCC, energy, pitch, pauses).",
        "disclaimer": DISCLAIMER,
    }


def _wav2vec_segments(
    y: np.ndarray,
    sr: int,
    segment_duration_sec: float,
    min_segment_duration_sec: float,
    hop_length_sec: float | None = None,
) -> list[np.ndarray]:
    """Split one clip into fixed-length segments matching Phase 6 training."""
    target_len = int(round(float(segment_duration_sec) * sr))
    min_len = int(round(float(min_segment_duration_sec) * sr))
    if target_len <= 0:
        raise ValueError("segment_duration_sec must be positive.")
    hop_len = int(round((hop_length_sec or segment_duration_sec) * sr))
    hop_len = max(1, hop_len)
    if len(y) == 0:
        return [np.zeros(target_len, dtype=np.float32)]

    segments: list[np.ndarray] = []
    start = 0
    while start < len(y):
        end = min(start + target_len, len(y))
        chunk = np.asarray(y[start:end], dtype=np.float32)
        if len(chunk) >= min_len or not segments:
            if len(chunk) < target_len:
                chunk = np.pad(chunk, (0, target_len - len(chunk)))
            else:
                chunk = chunk[:target_len]
            segments.append(chunk.astype(np.float32))
        start += hop_len
    return segments or [np.zeros(target_len, dtype=np.float32)]


class Phase6Wav2VecClassifier(torch.nn.Module):
    """Lightweight runtime copy of the Phase 6 wav2vec classifier head."""

    def __init__(self, backbone_config: Wav2Vec2Config, num_classes: int = 2, dropout: float = 0.35, pooling_strategy: str = "attention"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(backbone_config)
        self.pooling_strategy = pooling_strategy
        hidden_size = self.wav2vec2.config.hidden_size
        if pooling_strategy == "attention":
            self.attention_query = torch.nn.Parameter(torch.empty(hidden_size))
            torch.nn.init.normal_(self.attention_query, std=0.02)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if attention_mask is not None and hasattr(self.wav2vec2, "_get_feature_vector_attention_mask"):
            feature_mask = self.wav2vec2._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            if self.pooling_strategy == "attention":
                scores = (hidden @ self.attention_query) / np.sqrt(hidden.shape[-1])
                scores = scores.masked_fill(~feature_mask, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                pooled = (hidden * weights).sum(dim=1)
            else:
                mask = feature_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            if self.pooling_strategy == "attention":
                scores = (hidden @ self.attention_query) / np.sqrt(hidden.shape[-1])
                weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                pooled = (hidden * weights).sum(dim=1)
            else:
                pooled = hidden.mean(dim=1)
        return self.classifier(self.dropout(pooled))


def _load_phase6_wav2vec_checkpoint(model_path: Path) -> Tuple[AutoFeatureExtractor, torch.nn.Module, int, float, float]:
    """Load the custom Phase 6 wav2vec checkpoint and its feature extractor."""
    checkpoint_path = model_path if model_path.is_file() else model_path / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    cfg = checkpoint.get("config", {}) or {}
    state_dict = checkpoint["model_state_dict"]
    model_name = str(cfg.get("model_name", "facebook/wav2vec2-base"))
    dropout = float(cfg.get("dropout", 0.35))
    inferred_pooling = "attention" if "attention_query" in state_dict else "mean"
    pooling_strategy = str(cfg.get("pooling_strategy", inferred_pooling))
    sampling_rate = int(cfg.get("target_sampling_rate", 16000))
    segment_duration_sec = float(cfg.get("segment_duration_seconds", 3.0))
    min_segment_duration_sec = float(cfg.get("min_segment_duration", 1.5))

    feature_extractor_dir = checkpoint_path.parent / "feature_extractor"
    extractor_source = feature_extractor_dir if feature_extractor_dir.exists() else model_name
    extractor = AutoFeatureExtractor.from_pretrained(extractor_source)

    model_config_dir = checkpoint_path.parent if (checkpoint_path.parent / "config.json").exists() else model_name
    backbone_config = Wav2Vec2Config.from_pretrained(model_config_dir)
    model = Phase6Wav2VecClassifier(
        backbone_config,
        num_classes=2,
        dropout=dropout,
        pooling_strategy=pooling_strategy,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"attention_query"} if pooling_strategy == "attention" else set()
    unexpected_set = set(unexpected)
    missing_set = set(missing)
    if unexpected_set or (missing_set - allowed_missing):
        raise RuntimeError(
            "Checkpoint load mismatch: "
            f"missing={sorted(missing_set)}, unexpected={sorted(unexpected_set)}"
        )
    model.eval()
    return extractor, model, sampling_rate, segment_duration_sec, min_segment_duration_sec


def predict_with_wav2vec(audio_path: Path, model_path: Path, threshold: float = 0.5) -> Dict[str, object]:
    """Run prediction using either a HF-exported wav2vec model or the saved Phase 6 checkpoint."""
    checkpoint_path = model_path if model_path.is_file() else model_path / "best_model.pt"
    if checkpoint_path.exists():
        extractor, model, sampling_rate, segment_duration_sec, min_segment_duration_sec = _load_phase6_wav2vec_checkpoint(
            model_path
        )
        y, sr, error = load_audio_safe(audio_path, target_sr=sampling_rate)
        if error or y is None or sr is None:
            raise RuntimeError(f"Audio load failed: {error}")
        y = trim_silence(y)
        y = normalize_amplitude(y)
        segments = _wav2vec_segments(y, sr, segment_duration_sec, min_segment_duration_sec)
        inputs = extractor(
            segments,
            sampling_rate=sr,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = model(
                inputs["input_values"].to(device),
                attention_mask=inputs.get("attention_mask").to(device) if "attention_mask" in inputs else None,
            )
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[:, 1]
        prob_dementia = float(np.mean(probs))
        explanation = "Prediction from Phase 6 wav2vec2 checkpoint using mean segment aggregation."
        extra = {"n_segments": int(len(segments))}
    else:
        extractor = AutoFeatureExtractor.from_pretrained(model_path)
        model = AutoModelForAudioClassification.from_pretrained(model_path)
        y, sr = librosa.load(audio_path, sr=extractor.sampling_rate, mono=True)
        y = trim_silence(y)
        y = normalize_amplitude(y)
        inputs = extractor(y, sampling_rate=sr, return_tensors="pt", truncation=True, max_length=sr * 20)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        prob_dementia = float(probs[1])
        explanation = "Prediction from fine-tuned wav2vec2 acoustic representation."
        extra = {}

    pred = "dementia" if prob_dementia >= threshold else "non_dementia"
    return {
        "predicted_label": pred,
        "confidence_score": prob_dementia if pred == "dementia" else 1.0 - prob_dementia,
        "probability_dementia": prob_dementia,
        "threshold": float(threshold),
        "explanation": explanation,
        "disclaimer": DISCLAIMER,
        **extra,
    }


def parse_args() -> argparse.Namespace:
    """CLI for one-file inference."""
    parser = argparse.ArgumentParser(description="Run dementia screening inference on an audio file.")
    parser.add_argument("--audio-path", nargs="+", type=str, required=True)
    parser.add_argument("--model-type", type=str, choices=["baseline", "wav2vec", "phaseD_ensemble"], default="baseline")
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Path to .joblib, wav2vec checkpoint/directory, or Phase D ensemble manifest",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for dementia-positive classification")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audio_paths = [Path(p) for p in args.audio_path]
    model = Path(args.model_path) if args.model_path else DEFAULT_PHASED_ENSEMBLE_MANIFEST
    if args.model_type == "baseline":
        if len(audio_paths) != 1:
            raise ValueError("Baseline inference expects exactly one --audio-path.")
        result = predict_with_baseline(audio_paths[0], model, threshold=args.threshold)
    elif args.model_type == "wav2vec":
        if len(audio_paths) != 1:
            raise ValueError("wav2vec inference expects exactly one --audio-path.")
        result = predict_with_wav2vec(audio_paths[0], model, threshold=args.threshold)
    else:
        result = predict_with_phaseD_ensemble(audio_paths, model, threshold_override=args.threshold)
        result["disclaimer"] = DISCLAIMER
    print(result)
