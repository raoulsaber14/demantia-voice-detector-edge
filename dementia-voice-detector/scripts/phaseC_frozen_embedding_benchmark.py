"""Phase C frozen speech-embedding benchmark.

This script is intentionally additive. It extracts frozen pretrained speech
embeddings, caches them, and evaluates only simple calibrated linear
classifiers with speaker-grouped cross-validation.

No Phase 4/Phase 6 baseline artifacts, labels, manifests, app code, or
deployment files are modified by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
CACHE_DIR = ROOT / "artifacts/phaseC_frozen_embeddings"

TRUSTED_MANIFEST = REPORTS_DIR / "tables/phase6_5_trusted_subset_manifest_cleaned.csv"
GOVERNED_MANIFEST = REPORTS_DIR / "tables/phase6_5_full_governed_subset_manifest_cleaned.csv"
FEATURE_TABLE = ROOT / "data/features/features_phase3_governed_pruned.csv"

REPORT_MD = REPORTS_DIR / "phaseC_frozen_embedding_benchmark.md"
CLIP_SUMMARY_CSV = TABLES_DIR / "phaseC_embedding_clip_seed_summary.csv"
SPEAKER_SUMMARY_CSV = TABLES_DIR / "phaseC_embedding_speaker_seed_summary.csv"
MODEL_COMPARISON_CSV = TABLES_DIR / "phaseC_embedding_model_comparison.csv"
THRESHOLD_STABILITY_CSV = TABLES_DIR / "phaseC_embedding_threshold_stability.csv"
BEST_CANDIDATE_CSV = TABLES_DIR / "phaseC_embedding_best_candidate_analysis.csv"

DEFAULT_BACKBONES = [
    "microsoft/wavlm-base-plus",
    "facebook/hubert-base-ls960",
    "facebook/wav2vec2-base",
]
OPTIONAL_BACKBONES = ["facebook/wav2vec2-large-xlsr-53"]
CLASSIFIERS = ["logistic_regression_platt", "linear_svm_platt"]
SEEDS = [42, 43, 44, 45, 46]
C_GRID = [0.01, 0.1, 1.0, 10.0]
POSITIVE_LABEL = 1
TARGET_RECALL = 0.70
TARGET_SPECIFICITY = 0.50
BASELINE = {
    "official_name": "final_cleaned_logistic_regression_platt",
    "threshold": 0.300,
    "clip_seed42_recall": 0.1170212765957446,
    "clip_seed42_specificity": 0.9094488188976378,
    "clip_multiseed_recall_mean": 0.46382978723404256,
    "clip_multiseed_specificity_mean": 0.5598425196850394,
    "clip_multiseed_recall_std": 0.37974998327358483,
    "clip_multiseed_specificity_std": 0.3764767767997476,
    "speaker_multiseed_recall_mean": 0.4761904761904761,
    "speaker_multiseed_specificity_mean": 0.47476635514018695,
    "speaker_multiseed_aggregation": "max_risk_score_seed_selected_threshold",
}


@dataclass
class FoldSpec:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_speakers: set[str]


@dataclass
class FitResult:
    estimator: Pipeline
    calibrator: LogisticRegression
    selected_c: float
    inner_cv_name: str
    calibration_status: str
    train_oof_scores: np.ndarray
    train_oof_probabilities: np.ndarray
    val_probabilities: np.ndarray


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def slugify(value: str) -> str:
    return (
        value.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "-")
        .replace("-", "_")
    )


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "(empty)"
    out = df.head(max_rows).copy() if max_rows is not None else df.copy()
    out = out.where(pd.notna(out), "")
    headers = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        values = [str(row[c]).replace("\n", " ") for c in out.columns]
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append("| ... | ... |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mode_int(series: pd.Series) -> int:
    return int(series.astype(int).mode().iloc[0])


def load_manifest(path: Path, dataset_slug: str, max_clips: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    required = {"speaker_id", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "processed_file" not in df.columns and "audio_file" not in df.columns:
        raise ValueError(f"{path} must include processed_file or audio_file.")
    df = df[df["label"].isin([0, 1])].copy()
    if "feature_extraction_status" in df.columns:
        df = df[df["feature_extraction_status"].fillna("success").eq("success")].copy()
    if max_clips is not None:
        df = df.head(max_clips).copy()
    if "metadata_index" in df.columns:
        df["clip_id"] = df["metadata_index"].astype(str)
    else:
        df["clip_id"] = [f"{dataset_slug}_{i:06d}" for i in range(len(df))]
    df["dataset_version"] = dataset_slug
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    if df["speaker_id"].eq("").any() or df["speaker_id"].isna().any():
        raise ValueError(f"{path} contains missing speaker_id values.")
    if df["label"].nunique() < 2:
        raise ValueError(f"{path} does not contain both classes after filtering.")
    return df.reset_index(drop=True)


def cache_paths(dataset_slug: str, backbone: str, pooling: str, chunk_seconds: float) -> tuple[Path, Path, Path]:
    model_slug = slugify(backbone)
    chunk_slug = str(chunk_seconds).replace(".", "p")
    folder = CACHE_DIR / dataset_slug
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{model_slug}_{pooling}_chunk{chunk_slug}"
    return folder / f"{stem}.csv.gz", folder / f"{stem}.json", folder / f"{stem}_skipped.csv"


def cache_matches(meta_path: Path, expected: dict[str, Any]) -> bool:
    if not meta_path.exists():
        return False
    try:
        existing = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    keys = [
        "backbone",
        "dataset_slug",
        "manifest_sha256",
        "pooling",
        "chunk_seconds",
        "audio_column",
        "fallback_audio_column",
        "n_manifest_rows",
    ]
    return all(existing.get(k) == expected.get(k) for k in keys)


def resolve_audio_path(row: pd.Series, audio_column: str, fallback_audio_column: str) -> tuple[str, str]:
    for col in [audio_column, fallback_audio_column]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            path = Path(str(row[col]))
            if path.exists():
                return str(path), col
    tried = [str(row[c]) for c in [audio_column, fallback_audio_column] if c in row.index and pd.notna(row[c])]
    return "", "missing:" + " | ".join(tried)


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio_mono(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("empty audio")
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    return audio, sr


def batched(items: list[np.ndarray], batch_size: int) -> list[list[np.ndarray]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def embed_waveform(
    waveform: np.ndarray,
    sampling_rate: int,
    feature_extractor: Any,
    model: Any,
    device: str,
    chunk_seconds: float,
    batch_size: int,
    pooling: str,
) -> np.ndarray:
    import torch

    max_samples = max(1, int(round(chunk_seconds * sampling_rate)))
    min_chunk_samples = max(1, int(round(0.5 * sampling_rate)))
    chunks = [waveform[start : start + max_samples] for start in range(0, len(waveform), max_samples)]
    chunks = [chunk.astype(np.float32) for chunk in chunks if len(chunk) > 0]
    if not chunks:
        raise ValueError("no non-empty chunks after splitting")
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk_samples:
        chunks = chunks[:-1]
    if len(chunks) == 1 and len(chunks[0]) < min_chunk_samples:
        chunks = [np.pad(chunks[0], (0, min_chunk_samples - len(chunks[0]))).astype(np.float32)]

    total_frames = 0
    sum_vec: np.ndarray | None = None
    sumsq_vec: np.ndarray | None = None

    for chunk_batch in batched(chunks, batch_size=batch_size):
        inputs = feature_extractor(
            chunk_batch,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        model_inputs = {k: v.to(device) for k, v in inputs.items() if k in {"input_values", "attention_mask"}}
        with torch.no_grad():
            outputs = model(**model_inputs)
        hidden = outputs.last_hidden_state.detach().float().cpu().numpy()

        if hasattr(model, "_get_feat_extract_output_lengths"):
            input_lengths = torch.tensor([len(c) for c in chunk_batch], dtype=torch.long)
            with torch.no_grad():
                frame_lengths = model._get_feat_extract_output_lengths(input_lengths).cpu().numpy().astype(int)
        else:
            frame_lengths = np.repeat(hidden.shape[1], hidden.shape[0])

        for i, frame_len in enumerate(frame_lengths):
            valid = hidden[i, : int(frame_len), :]
            if valid.size == 0:
                continue
            if sum_vec is None:
                sum_vec = np.zeros(valid.shape[1], dtype=np.float64)
                sumsq_vec = np.zeros(valid.shape[1], dtype=np.float64)
            sum_vec += valid.sum(axis=0, dtype=np.float64)
            sumsq_vec += np.square(valid, dtype=np.float64).sum(axis=0, dtype=np.float64)
            total_frames += int(valid.shape[0])

    if total_frames == 0 or sum_vec is None or sumsq_vec is None:
        raise ValueError("model produced no valid frames")

    mean = sum_vec / float(total_frames)
    if pooling == "mean":
        return mean.astype(np.float32)
    variance = np.maximum((sumsq_vec / float(total_frames)) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return np.concatenate([mean, std]).astype(np.float32)


def extract_or_load_embeddings(
    manifest: pd.DataFrame,
    manifest_path: Path,
    dataset_slug: str,
    backbone: str,
    audio_column: str,
    fallback_audio_column: str,
    pooling: str,
    chunk_seconds: float,
    batch_size: int,
    device_arg: str,
    local_files_only: bool,
    force_extract: bool,
    torch_threads: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_csv, meta_json, skipped_csv = cache_paths(dataset_slug, backbone, pooling, chunk_seconds)
    manifest_hash = sha256_file(manifest_path)
    expected_meta = {
        "backbone": backbone,
        "dataset_slug": dataset_slug,
        "manifest_sha256": manifest_hash,
        "pooling": pooling,
        "chunk_seconds": float(chunk_seconds),
        "audio_column": audio_column,
        "fallback_audio_column": fallback_audio_column,
        "n_manifest_rows": int(len(manifest)),
    }
    if cache_csv.exists() and cache_matches(meta_json, expected_meta) and not force_extract:
        cached = pd.read_csv(cache_csv)
        meta = json.loads(meta_json.read_text())
        meta["cache_status"] = "loaded_existing_cache"
        return cached, meta

    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    if torch_threads is not None and torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    device = choose_device(device_arg)
    start_time = time.time()
    feature_extractor = AutoFeatureExtractor.from_pretrained(backbone, local_files_only=local_files_only)
    target_sr = int(getattr(feature_extractor, "sampling_rate", 16000) or 16000)
    model = AutoModel.from_pretrained(backbone, local_files_only=local_files_only)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row_number, (_, row) in enumerate(manifest.iterrows(), start=1):
        audio_path, source_column = resolve_audio_path(row, audio_column, fallback_audio_column)
        if not audio_path:
            skipped.append(
                {
                    "clip_id": row["clip_id"],
                    "speaker_id": row["speaker_id"],
                    "label": int(row["label"]),
                    "reason": f"audio_path_not_found:{source_column}",
                }
            )
            continue
        try:
            waveform, sr = load_audio_mono(audio_path, target_sr=target_sr)
            embedding = embed_waveform(
                waveform=waveform,
                sampling_rate=sr,
                feature_extractor=feature_extractor,
                model=model,
                device=device,
                chunk_seconds=chunk_seconds,
                batch_size=batch_size,
                pooling=pooling,
            )
        except Exception as exc:  # noqa: BLE001 - corrupt/unreadable files should be logged and skipped.
            skipped.append(
                {
                    "clip_id": row["clip_id"],
                    "speaker_id": row["speaker_id"],
                    "label": int(row["label"]),
                    "audio_path": audio_path,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        out: dict[str, Any] = {
            "dataset_version": dataset_slug,
            "clip_id": str(row["clip_id"]),
            "metadata_index": str(row["metadata_index"]) if "metadata_index" in row.index else str(row["clip_id"]),
            "audio_path": audio_path,
            "audio_source_column": source_column,
            "speaker_id": str(row["speaker_id"]),
            "label": int(row["label"]),
            "label_name": row["label_name"] if "label_name" in row.index else "",
            "backbone": backbone,
            "row_number": row_number,
        }
        for i, value in enumerate(embedding):
            out[f"emb_{i:04d}"] = float(value)
        rows.append(out)
        if row_number % 25 == 0:
            print(
                f"[{dataset_slug}][{backbone}] embedded {row_number}/{len(manifest)} rows "
                f"({len(skipped)} skipped)",
                flush=True,
            )

    if not rows:
        raise RuntimeError(f"No embeddings were extracted for {backbone} on {dataset_slug}.")

    embeddings = pd.DataFrame(rows)
    embeddings.to_csv(cache_csv, index=False, compression="gzip")
    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_csv, index=False)
    elif skipped_csv.exists():
        skipped_csv.unlink()

    meta = {
        **expected_meta,
        "cache_status": "created",
        "cache_csv": rel(cache_csv),
        "skipped_csv": rel(skipped_csv) if skipped else "",
        "n_embedded_rows": int(len(embeddings)),
        "n_skipped_rows": int(len(skipped)),
        "embedding_dim": int(sum(c.startswith("emb_") for c in embeddings.columns)),
        "target_sample_rate": target_sr,
        "device": device,
        "batch_size": int(batch_size),
        "torch_threads": torch_threads,
        "runtime_seconds": round(time.time() - start_time, 3),
    }
    meta_json.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return embeddings, meta


def speaker_label_counts(df: pd.DataFrame) -> dict[int, int]:
    labels = df.groupby("speaker_id")["label"].agg(mode_int)
    return {int(k): int(v) for k, v in labels.value_counts().sort_index().to_dict().items()}


def build_outer_folds(df: pd.DataFrame, seed: int, max_splits: int = 5) -> tuple[list[FoldSpec], str, int]:
    class_speaker_counts = speaker_label_counts(df)
    min_class_speakers = min(class_speaker_counts.values())
    n_splits = int(min(max_splits, min_class_speakers))
    if n_splits < 2:
        raise ValueError("Grouped CV is infeasible: fewer than two speakers in a class.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldSpec] = []
    groups = df["speaker_id"].astype(str).to_numpy()
    y = df["label"].astype(int).to_numpy()
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(df)), y, groups=groups), start=1):
        val_speakers = set(df.iloc[val_idx]["speaker_id"].astype(str))
        train_idx = df.index[~df["speaker_id"].astype(str).isin(val_speakers)].to_numpy()
        overlap = set(df.loc[train_idx, "speaker_id"].astype(str)) & val_speakers
        if overlap:
            raise ValueError(f"Speaker leakage detected in seed {seed}, fold {fold}: {sorted(overlap)[:5]}")
        if df.loc[train_idx, "label"].nunique() < 2 or df.iloc[val_idx]["label"].nunique() < 2:
            raise ValueError(f"Seed {seed}, fold {fold} lacks both classes in train or validation.")
        folds.append(FoldSpec(fold=fold, train_idx=train_idx, val_idx=df.index[val_idx].to_numpy(), val_speakers=val_speakers))
    note = f"StratifiedGroupKFold n_splits={n_splits}; speaker class counts={class_speaker_counts}"
    return folds, note, n_splits


def inner_group_cv(y: np.ndarray, groups: np.ndarray, seed: int, max_splits: int = 3) -> StratifiedGroupKFold | None:
    group_df = pd.DataFrame({"speaker_id": groups.astype(str), "label": y.astype(int)}).drop_duplicates()
    labels = group_df.groupby("speaker_id")["label"].agg(mode_int)
    counts = labels.value_counts().to_dict()
    if len(counts) < 2:
        return None
    n_splits = int(min(max_splits, min(counts.values())))
    if n_splits < 2:
        return None
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def build_estimator(classifier: str, seed: int, c_value: float) -> Pipeline:
    if classifier == "logistic_regression_platt":
        model = LogisticRegression(
            C=float(c_value),
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
            solver="liblinear",
        )
    elif classifier == "linear_svm_platt":
        model = LinearSVC(
            C=float(c_value),
            class_weight="balanced",
            random_state=seed,
            max_iter=20000,
            dual=False,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def raw_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return np.asarray(scores, dtype=float).ravel()
    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(x), dtype=float)
        if prob.ndim == 2 and prob.shape[1] > 1:
            prob = prob[:, 1]
        return prob.ravel()
    return np.asarray(model.predict(x), dtype=float).ravel()


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2 or np.unique(scores).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def tune_c(
    classifier: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    # Keep Phase C focused on representation quality rather than top-layer
    # hyperparameter search. Platt calibration and threshold selection still use
    # inner grouped out-of-fold predictions from training speakers only.
    return 1.0, pd.DataFrame([{"C": 1.0, "mean_inner_roc_auc": np.nan, "note": "fixed_C_1.0_no_grid_search"}])
    cv = inner_group_cv(y_train, groups_train, seed=seed)
    if cv is None:
        return 1.0, pd.DataFrame([{"C": 1.0, "mean_inner_roc_auc": np.nan, "note": "inner_cv_infeasible"}])
    rows: list[dict[str, Any]] = []
    for c_value in C_GRID:
        fold_scores: list[float] = []
        for inner_train_idx, inner_val_idx in cv.split(x_train, y_train, groups=groups_train):
            if np.unique(y_train[inner_train_idx]).size < 2 or np.unique(y_train[inner_val_idx]).size < 2:
                continue
            estimator = build_estimator(classifier, seed=seed, c_value=c_value)
            estimator.fit(x_train[inner_train_idx], y_train[inner_train_idx])
            score = raw_scores(estimator, x_train[inner_val_idx])
            fold_scores.append(safe_roc_auc(y_train[inner_val_idx], score))
        finite = [v for v in fold_scores if np.isfinite(v)]
        rows.append(
            {
                "C": float(c_value),
                "mean_inner_roc_auc": float(np.mean(finite)) if finite else np.nan,
                "n_inner_scores": len(finite),
                "note": "ok" if finite else "no_finite_inner_auc",
            }
        )
    table = pd.DataFrame(rows)
    valid = table[np.isfinite(table["mean_inner_roc_auc"])].copy()
    if valid.empty:
        return 1.0, table
    valid = valid.sort_values(["mean_inner_roc_auc", "C"], ascending=[False, True])
    return float(valid.iloc[0]["C"]), table


def fit_platt_model(
    classifier: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    x_val: np.ndarray,
    seed: int,
) -> FitResult:
    selected_c, _ = tune_c(classifier, x_train, y_train, groups_train, seed=seed)
    inner_cv_name = "none_training_speakers_only"
    calibration_status = "training_only_in_sample_platt"
    final_estimator = build_estimator(classifier, seed=seed, c_value=selected_c)
    final_estimator.fit(x_train, y_train)
    oof_raw = raw_scores(final_estimator, x_train)
    calibrator = LogisticRegression(max_iter=2000, random_state=seed)
    calibrator.fit(oof_raw.reshape(-1, 1), y_train)
    oof_prob = calibrator.predict_proba(oof_raw.reshape(-1, 1))[:, 1]

    val_raw = raw_scores(final_estimator, x_val)
    val_prob = calibrator.predict_proba(val_raw.reshape(-1, 1))[:, 1]
    return FitResult(
        estimator=final_estimator,
        calibrator=calibrator,
        selected_c=selected_c,
        inner_cv_name=inner_cv_name,
        calibration_status=calibration_status,
        train_oof_scores=oof_raw,
        train_oof_probabilities=oof_prob,
        val_probabilities=val_prob,
    )


def metrics_from_arrays(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = safe_roc_auc(y_true, y_score)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = np.nanmean([recall, specificity])
    return {
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "confusion_matrix_tn_fp_fn_tp": f"{tn},{fp},{fn},{tp}",
    }


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    # Fixed grid keeps threshold selection deterministic and tractable while
    # avoiding any use of held-out fold outcomes.
    return np.round(np.linspace(0.0, 1.0, 101), 4)


def select_constrained_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(scores):
        y_pred = (scores >= threshold).astype(int)
        metrics = metrics_from_arrays(y_true, scores, y_pred)
        rows.append({"threshold": float(threshold), **metrics})
    table = pd.DataFrame(rows)
    viable = table[table["specificity"] >= TARGET_SPECIFICITY].copy()
    if viable.empty:
        selected = table.sort_values(
            ["specificity", "recall", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        note = "fallback_no_threshold_met_specificity_floor"
    else:
        selected = viable.sort_values(
            ["recall", "specificity", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        note = "max_recall_subject_to_specificity_ge_0.50"
    info = selected.to_dict()
    info["selection_note"] = note
    return float(selected["threshold"]), info


def speaker_score_table(
    clip_df: pd.DataFrame,
    aggregation_method: str,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for speaker_id, group in clip_df.groupby("speaker_id"):
        scores = group["score"].astype(float).to_numpy()
        labels = group["label"].astype(int)
        label = mode_int(labels)
        if aggregation_method == "mean_score":
            score = float(np.mean(scores))
            pred = int(score >= threshold)
        elif aggregation_method == "max_score":
            score = float(np.max(scores))
            pred = int(score >= threshold)
        elif aggregation_method == "majority_vote":
            clip_pred = (scores >= threshold).astype(int)
            score = float(np.mean(clip_pred))
            pred = int(score >= 0.5)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        rows.append(
            {
                "speaker_id": speaker_id,
                "label": int(label),
                "score": score,
                "prediction": pred,
                "n_clips": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def select_speaker_constrained_threshold(train_clip_df: pd.DataFrame, aggregation_method: str) -> tuple[float, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(train_clip_df["score"].astype(float).to_numpy()):
        speaker_df = speaker_score_table(train_clip_df, aggregation_method, threshold)
        metrics = metrics_from_arrays(
            speaker_df["label"].astype(int).to_numpy(),
            speaker_df["score"].astype(float).to_numpy(),
            speaker_df["prediction"].astype(int).to_numpy(),
        )
        rows.append({"threshold": float(threshold), **metrics})
    table = pd.DataFrame(rows)
    viable = table[table["specificity"] >= TARGET_SPECIFICITY].copy()
    if viable.empty:
        selected = table.sort_values(
            ["specificity", "recall", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        note = "fallback_no_threshold_met_specificity_floor"
    else:
        selected = viable.sort_values(
            ["recall", "specificity", "f1", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        note = "max_recall_subject_to_specificity_ge_0.50"
    info = selected.to_dict()
    info["selection_note"] = note
    return float(selected["threshold"]), info


def embedding_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in df.columns if c.startswith("emb_")]
    if not cols:
        raise ValueError("Embedding cache has no emb_* columns.")
    return df[cols].to_numpy(dtype=np.float32), cols


def evaluate_backbone_classifier(
    embeddings: pd.DataFrame,
    dataset_slug: str,
    backbone: str,
    classifier: str,
    seeds: list[int],
    max_splits: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    x, embedding_cols = embedding_matrix(embeddings)
    y = embeddings["label"].astype(int).to_numpy()
    groups = embeddings["speaker_id"].astype(str).to_numpy()
    clip_rows: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for seed in seeds:
        folds, fold_note, n_splits = build_outer_folds(embeddings, seed=seed, max_splits=max_splits)
        clip_predictions_by_policy: dict[str, list[pd.DataFrame]] = {"default_0.5": [], "constrained_spec_ge_0.50": []}
        speaker_predictions_by_key: dict[tuple[str, str], list[pd.DataFrame]] = {}
        fold_thresholds: list[dict[str, Any]] = []
        selected_cs: list[float] = []
        calibration_statuses: list[str] = []

        for fold_spec in folds:
            train_idx = fold_spec.train_idx
            val_idx = fold_spec.val_idx
            fit = fit_platt_model(
                classifier=classifier,
                x_train=x[train_idx],
                y_train=y[train_idx],
                groups_train=groups[train_idx],
                x_val=x[val_idx],
                seed=seed,
            )
            selected_cs.append(float(fit.selected_c))
            calibration_statuses.append(fit.calibration_status)

            train_clip_df = pd.DataFrame(
                {
                    "speaker_id": groups[train_idx],
                    "label": y[train_idx],
                    "score": fit.train_oof_probabilities,
                }
            )
            val_base = embeddings.loc[val_idx, ["dataset_version", "clip_id", "metadata_index", "speaker_id", "label", "backbone"]].copy()
            val_base["seed"] = seed
            val_base["fold"] = fold_spec.fold
            val_base["classifier"] = classifier
            val_base["score"] = fit.val_probabilities

            clip_threshold_map = {"default_0.5": (0.5, {"selection_note": "fixed_default_probability_threshold"})}
            constrained_threshold, constrained_info = select_constrained_threshold(y[train_idx], fit.train_oof_probabilities)
            clip_threshold_map["constrained_spec_ge_0.50"] = (constrained_threshold, constrained_info)

            for policy, (threshold, info) in clip_threshold_map.items():
                pred = val_base.copy()
                pred["threshold_policy"] = policy
                pred["threshold"] = float(threshold)
                pred["threshold_selection_note"] = info["selection_note"]
                pred["prediction"] = (pred["score"].astype(float) >= float(threshold)).astype(int)
                clip_predictions_by_policy[policy].append(pred)
                fold_thresholds.append(
                    {
                        "dataset_version": dataset_slug,
                        "backbone": backbone,
                        "classifier": classifier,
                        "seed": seed,
                        "fold": fold_spec.fold,
                        "metric_level": "clip",
                        "aggregation_method": "",
                        "threshold_policy": policy,
                        "threshold": float(threshold),
                        "selection_note": info["selection_note"],
                        "selected_c": float(fit.selected_c),
                        "inner_cv": fit.inner_cv_name,
                        "calibration_status": fit.calibration_status,
                    }
                )

            for aggregation_method in ["mean_score", "max_score", "majority_vote"]:
                speaker_threshold_map = {"default_0.5": (0.5, {"selection_note": "fixed_default_probability_threshold"})}
                speaker_threshold, speaker_info = select_speaker_constrained_threshold(train_clip_df, aggregation_method)
                speaker_threshold_map["constrained_spec_ge_0.50"] = (speaker_threshold, speaker_info)
                val_clip_for_speaker = val_base[["speaker_id", "label", "score"]].copy()
                for policy, (threshold, info) in speaker_threshold_map.items():
                    speaker_pred = speaker_score_table(val_clip_for_speaker, aggregation_method, threshold)
                    speaker_pred["dataset_version"] = dataset_slug
                    speaker_pred["backbone"] = backbone
                    speaker_pred["classifier"] = classifier
                    speaker_pred["seed"] = seed
                    speaker_pred["fold"] = fold_spec.fold
                    speaker_pred["aggregation_method"] = aggregation_method
                    speaker_pred["threshold_policy"] = policy
                    speaker_pred["threshold"] = float(threshold)
                    speaker_pred["threshold_selection_note"] = info["selection_note"]
                    speaker_predictions_by_key.setdefault((aggregation_method, policy), []).append(speaker_pred)
                    fold_thresholds.append(
                        {
                            "dataset_version": dataset_slug,
                            "backbone": backbone,
                            "classifier": classifier,
                            "seed": seed,
                            "fold": fold_spec.fold,
                            "metric_level": "speaker",
                            "aggregation_method": aggregation_method,
                            "threshold_policy": policy,
                            "threshold": float(threshold),
                            "selection_note": info["selection_note"],
                            "selected_c": float(fit.selected_c),
                            "inner_cv": fit.inner_cv_name,
                            "calibration_status": fit.calibration_status,
                        }
                    )

        threshold_rows.extend(fold_thresholds)
        threshold_df = pd.DataFrame(fold_thresholds)
        for policy, pred_parts in clip_predictions_by_policy.items():
            pred_df = pd.concat(pred_parts, ignore_index=True)
            metrics = metrics_from_arrays(
                pred_df["label"].astype(int).to_numpy(),
                pred_df["score"].astype(float).to_numpy(),
                pred_df["prediction"].astype(int).to_numpy(),
            )
            thresholds = threshold_df[(threshold_df["metric_level"] == "clip") & (threshold_df["threshold_policy"] == policy)]
            clip_rows.append(
                {
                    "dataset_version": dataset_slug,
                    "backbone": backbone,
                    "classifier": classifier,
                    "seed": seed,
                    "threshold_policy": policy,
                    "n_folds": int(n_splits),
                    "n_clips": int(len(pred_df)),
                    "n_speakers": int(pred_df["speaker_id"].nunique()),
                    "embedding_dim": int(len(embedding_cols)),
                    "selected_c_values": ";".join(str(v) for v in selected_cs),
                    "calibration_statuses": ";".join(sorted(set(calibration_statuses))),
                    "fold_note": fold_note,
                    "threshold_mean": float(thresholds["threshold"].mean()),
                    "threshold_std": float(thresholds["threshold"].std(ddof=1)) if len(thresholds) > 1 else 0.0,
                    "threshold_min": float(thresholds["threshold"].min()),
                    "threshold_max": float(thresholds["threshold"].max()),
                    **metrics,
                }
            )

        for (aggregation_method, policy), pred_parts in speaker_predictions_by_key.items():
            pred_df = pd.concat(pred_parts, ignore_index=True)
            # Each speaker appears in exactly one outer validation fold per seed.
            duplicate_speakers = int(pred_df["speaker_id"].duplicated().sum())
            metrics = metrics_from_arrays(
                pred_df["label"].astype(int).to_numpy(),
                pred_df["score"].astype(float).to_numpy(),
                pred_df["prediction"].astype(int).to_numpy(),
            )
            thresholds = threshold_df[
                (threshold_df["metric_level"] == "speaker")
                & (threshold_df["aggregation_method"] == aggregation_method)
                & (threshold_df["threshold_policy"] == policy)
            ]
            speaker_rows.append(
                {
                    "dataset_version": dataset_slug,
                    "backbone": backbone,
                    "classifier": classifier,
                    "seed": seed,
                    "aggregation_method": aggregation_method,
                    "threshold_policy": policy,
                    "n_folds": int(n_splits),
                    "n_speakers": int(pred_df["speaker_id"].nunique()),
                    "duplicate_validation_speaker_rows": duplicate_speakers,
                    "embedding_dim": int(len(embedding_cols)),
                    "selected_c_values": ";".join(str(v) for v in selected_cs),
                    "calibration_statuses": ";".join(sorted(set(calibration_statuses))),
                    "fold_note": fold_note,
                    "threshold_mean": float(thresholds["threshold"].mean()),
                    "threshold_std": float(thresholds["threshold"].std(ddof=1)) if len(thresholds) > 1 else 0.0,
                    "threshold_min": float(thresholds["threshold"].min()),
                    "threshold_max": float(thresholds["threshold"].max()),
                    "target_recall_pass": bool(metrics["recall"] >= TARGET_RECALL),
                    "target_specificity_pass": bool(metrics["specificity"] >= TARGET_SPECIFICITY),
                    "target_zone_pass": bool(metrics["recall"] >= TARGET_RECALL and metrics["specificity"] >= TARGET_SPECIFICITY),
                    **metrics,
                }
            )
        print(f"[{dataset_slug}][{backbone}][{classifier}] completed seed {seed}", flush=True)
    return clip_rows, speaker_rows, threshold_rows


def summarize_model_comparison(clip_df: pd.DataFrame, speaker_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols_clip = ["dataset_version", "backbone", "classifier", "threshold_policy"]
    for keys, group in clip_df.groupby(group_cols_clip, dropna=False):
        base = dict(zip(group_cols_clip, keys))
        rows.append(
            {
                **base,
                "metric_level": "clip",
                "aggregation_method": "",
                "n_seeds": int(group["seed"].nunique()),
                "recall_mean": float(group["recall"].mean()),
                "recall_std": float(group["recall"].std(ddof=1)) if len(group) > 1 else 0.0,
                "specificity_mean": float(group["specificity"].mean()),
                "specificity_std": float(group["specificity"].std(ddof=1)) if len(group) > 1 else 0.0,
                "precision_mean": float(group["precision"].mean()),
                "precision_std": float(group["precision"].std(ddof=1)) if len(group) > 1 else 0.0,
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=1)) if len(group) > 1 else 0.0,
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_std": float(group["roc_auc"].std(ddof=1)) if len(group) > 1 else 0.0,
                "threshold_mean": float(group["threshold_mean"].mean()),
                "threshold_std_across_seeds": float(group["threshold_mean"].std(ddof=1)) if len(group) > 1 else 0.0,
                "target_recall_pass": "",
                "target_specificity_pass": "",
                "target_zone_pass": "",
            }
        )

    group_cols_speaker = ["dataset_version", "backbone", "classifier", "aggregation_method", "threshold_policy"]
    for keys, group in speaker_df.groupby(group_cols_speaker, dropna=False):
        base = dict(zip(group_cols_speaker, keys))
        recall_mean = float(group["recall"].mean())
        specificity_mean = float(group["specificity"].mean())
        rows.append(
            {
                **base,
                "metric_level": "speaker",
                "n_seeds": int(group["seed"].nunique()),
                "recall_mean": recall_mean,
                "recall_std": float(group["recall"].std(ddof=1)) if len(group) > 1 else 0.0,
                "specificity_mean": specificity_mean,
                "specificity_std": float(group["specificity"].std(ddof=1)) if len(group) > 1 else 0.0,
                "precision_mean": float(group["precision"].mean()),
                "precision_std": float(group["precision"].std(ddof=1)) if len(group) > 1 else 0.0,
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=1)) if len(group) > 1 else 0.0,
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_std": float(group["roc_auc"].std(ddof=1)) if len(group) > 1 else 0.0,
                "threshold_mean": float(group["threshold_mean"].mean()),
                "threshold_std_across_seeds": float(group["threshold_mean"].std(ddof=1)) if len(group) > 1 else 0.0,
                "target_recall_pass": bool(recall_mean >= TARGET_RECALL),
                "target_specificity_pass": bool(specificity_mean >= TARGET_SPECIFICITY),
                "target_zone_pass": bool(recall_mean >= TARGET_RECALL and specificity_mean >= TARGET_SPECIFICITY),
            }
        )
    out = pd.DataFrame(rows)
    sort_cols = ["metric_level", "aggregation_method", "threshold_policy", "target_zone_pass", "recall_mean", "specificity_mean"]
    return out.sort_values(sort_cols, ascending=[True, True, True, False, False, False]).reset_index(drop=True)


def summarize_threshold_stability(threshold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_version", "backbone", "classifier", "metric_level", "aggregation_method", "threshold_policy"]
    rows: list[dict[str, Any]] = []
    for keys, group in threshold_df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        seed_means = group.groupby("seed")["threshold"].mean()
        rows.append(
            {
                **base,
                "n_thresholds": int(len(group)),
                "n_seeds": int(group["seed"].nunique()),
                "threshold_mean": float(group["threshold"].mean()),
                "threshold_std": float(group["threshold"].std(ddof=1)) if len(group) > 1 else 0.0,
                "threshold_min": float(group["threshold"].min()),
                "threshold_max": float(group["threshold"].max()),
                "seed_threshold_mean_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
                "selection_notes": ";".join(sorted(set(group["selection_note"].astype(str)))),
                "calibration_statuses": ";".join(sorted(set(group["calibration_status"].astype(str)))),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def select_best_candidate(model_df: pd.DataFrame, clip_df: pd.DataFrame) -> pd.DataFrame:
    speaker_main = model_df[
        (model_df["metric_level"] == "speaker")
        & (model_df["aggregation_method"] == "mean_score")
        & (model_df["dataset_version"] == "trusted_cleaned")
    ].copy()
    if speaker_main.empty:
        return pd.DataFrame()
    speaker_main["target_zone_pass_bool"] = speaker_main["target_zone_pass"].astype(bool)
    speaker_main["specificity_pass_bool"] = speaker_main["specificity_mean"] >= TARGET_SPECIFICITY
    speaker_main = speaker_main.sort_values(
        ["target_zone_pass_bool", "specificity_pass_bool", "recall_mean", "specificity_mean", "f1_mean", "roc_auc_mean"],
        ascending=[False, False, False, False, False, False],
    )
    best = speaker_main.iloc[0].to_dict()
    matching_clip = clip_df[
        (clip_df["dataset_version"] == best["dataset_version"])
        & (clip_df["backbone"] == best["backbone"])
        & (clip_df["classifier"] == best["classifier"])
        & (clip_df["threshold_policy"] == best["threshold_policy"])
    ]
    clip_recall = float(matching_clip["recall"].mean()) if not matching_clip.empty else float("nan")
    clip_specificity = float(matching_clip["specificity"].mean()) if not matching_clip.empty else float("nan")
    speaker_recall = float(best["recall_mean"])
    speaker_specificity = float(best["specificity_mean"])
    target_pass = bool(speaker_recall >= TARGET_RECALL and speaker_specificity >= TARGET_SPECIFICITY)
    decision = "candidate_for_next_phase" if target_pass else "not_yet_sufficient"
    comparison_note = (
        "Compared on trusted cleaned data. Official frozen baseline seed 42 at threshold 0.300 has "
        f"clip recall {BASELINE['clip_seed42_recall']:.3f} and specificity {BASELINE['clip_seed42_specificity']:.3f}. "
        "The multi-seed baseline reference has "
        f"clip recall {BASELINE['clip_multiseed_recall_mean']:.3f}, specificity {BASELINE['clip_multiseed_specificity_mean']:.3f}; "
        f"speaker recall {BASELINE['speaker_multiseed_recall_mean']:.3f}, specificity "
        f"{BASELINE['speaker_multiseed_specificity_mean']:.3f} using "
        f"{BASELINE['speaker_multiseed_aggregation']}."
    )
    return pd.DataFrame(
        [
            {
                "backbone": best["backbone"],
                "classifier": best["classifier"],
                "aggregation_method": best["aggregation_method"],
                "threshold_policy": best["threshold_policy"],
                "clip_level_mean_recall": clip_recall,
                "clip_level_mean_specificity": clip_specificity,
                "speaker_level_mean_recall": speaker_recall,
                "speaker_level_mean_specificity": speaker_specificity,
                "threshold_mean": float(best["threshold_mean"]),
                "threshold_std": float(best["threshold_std_across_seeds"]),
                "target_recall_pass": bool(speaker_recall >= TARGET_RECALL),
                "target_specificity_pass": bool(speaker_specificity >= TARGET_SPECIFICITY),
                "target_zone_pass": target_pass,
                "official_baseline_name": BASELINE["official_name"],
                "official_baseline_threshold": BASELINE["threshold"],
                "official_baseline_clip_seed42_recall": BASELINE["clip_seed42_recall"],
                "official_baseline_clip_seed42_specificity": BASELINE["clip_seed42_specificity"],
                "official_baseline_clip_multiseed_recall_mean": BASELINE["clip_multiseed_recall_mean"],
                "official_baseline_clip_multiseed_specificity_mean": BASELINE["clip_multiseed_specificity_mean"],
                "official_baseline_speaker_multiseed_recall_mean": BASELINE["speaker_multiseed_recall_mean"],
                "official_baseline_speaker_multiseed_specificity_mean": BASELINE["speaker_multiseed_specificity_mean"],
                "delta_clip_recall_vs_baseline_multiseed": clip_recall - BASELINE["clip_multiseed_recall_mean"],
                "delta_clip_specificity_vs_baseline_multiseed": clip_specificity - BASELINE["clip_multiseed_specificity_mean"],
                "delta_speaker_recall_vs_baseline_multiseed": speaker_recall - BASELINE["speaker_multiseed_recall_mean"],
                "delta_speaker_specificity_vs_baseline_multiseed": speaker_specificity
                - BASELINE["speaker_multiseed_specificity_mean"],
                "comparison_to_official_baseline": comparison_note,
                "decision": decision,
            }
        ]
    )


def write_report(
    clip_df: pd.DataFrame,
    speaker_df: pd.DataFrame,
    model_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    best_df: pd.DataFrame,
    extraction_meta: list[dict[str, Any]],
    backbones: list[str],
    classifiers: list[str],
    seeds: list[int],
    max_splits: int,
    pooling: str,
    chunk_seconds: float,
    governed_run: bool,
) -> None:
    clip_top = model_df[(model_df["metric_level"] == "clip") & (model_df["dataset_version"] == "trusted_cleaned")].copy()
    clip_top = clip_top.sort_values(["recall_mean", "specificity_mean", "f1_mean"], ascending=[False, False, False])
    speaker_top = model_df[
        (model_df["metric_level"] == "speaker")
        & (model_df["aggregation_method"] == "mean_score")
        & (model_df["dataset_version"] == "trusted_cleaned")
    ].copy()
    speaker_top = speaker_top.sort_values(
        ["target_zone_pass", "recall_mean", "specificity_mean", "f1_mean"], ascending=[False, False, False, False]
    )
    target_count = int(
        (
            (speaker_top["recall_mean"] >= TARGET_RECALL)
            & (speaker_top["specificity_mean"] >= TARGET_SPECIFICITY)
        ).sum()
    )
    cache_rows = pd.DataFrame(
        [
            {
                "dataset": m.get("dataset_slug", ""),
                "backbone": m.get("backbone", ""),
                "cache_status": m.get("cache_status", ""),
                "rows": m.get("n_embedded_rows", ""),
                "skipped": m.get("n_skipped_rows", ""),
                "embedding_dim": m.get("embedding_dim", ""),
                "cache_csv": m.get("cache_csv", ""),
            }
            for m in extraction_meta
        ]
    )
    best = best_df.iloc[0].to_dict() if not best_df.empty else {}
    next_step = (
        "Run a narrow confirmation phase for the best frozen embedding setup, including governed sensitivity and "
        "error analysis, without changing the frozen baseline."
        if best.get("decision") == "candidate_for_next_phase"
        else "Keep the frozen embedding benchmark as evidence and inspect error/score structure before any additional model complexity."
    )
    lines = [
        "# Phase C Frozen Embedding Benchmark",
        "",
        "## 1. Purpose of the phase",
        "",
        "Phase C tests whether frozen pretrained speech embeddings plus simple calibrated linear classifiers improve "
        "separability over the official handcrafted-feature baseline. The target zone is evaluated, not assumed: "
        f"speaker-level recall >= {TARGET_RECALL:.2f} and specificity >= {TARGET_SPECIFICITY:.2f}.",
        "",
        "## 2. Backbones tested",
        "",
        "\n".join(f"- `{b}`" for b in backbones),
        "",
        "## 3. Classifiers tested",
        "",
        "\n".join(f"- `{c}`" for c in classifiers),
        "",
        "Both classifiers use fixed `C=1.0`, median imputation, standard scaling, class balancing, and external Platt calibration. "
        "The logistic top layer uses an L2 `liblinear` solver for the small high-dimensional embedding matrix. "
        "No tree model, neural classifier head, or backbone fine-tuning is used.",
        "",
        "## 4. Datasets used",
        "",
        "- Primary: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv` (`trusted_cleaned`).",
        "- Secondary governed sensitivity: "
        + (
            "`reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv` was run separately."
            if governed_run
            else "not run in this minimum clean matrix; trusted cleaned remains the headline dataset."
        ),
        "- Reference feature table was read for baseline context only: `data/features/features_phase3_governed_pruned.csv`.",
        "",
        "## 5. Speaker-grouped protocol",
        "",
        f"Evaluation uses `{max_splits}`-fold `StratifiedGroupKFold` grouped by `speaker_id`, with seeds "
        f"{', '.join(str(s) for s in seeds)}. Validation speakers are excluded from each training fold. "
        "The classifier regularization is fixed, and Platt calibration plus constrained threshold selection use training speakers only.",
        "",
        "## 6. Embedding extraction method",
        "",
        f"Audio was loaded from manifest paths, converted to mono, resampled to each feature extractor sampling rate, "
        f"and processed through frozen backbones. The final hidden layer was pooled with `{pooling}` pooling. "
        f"Long clips were processed in deterministic contiguous chunks of {chunk_seconds:g} seconds and pooled over frames. "
        f"Backbone parameters were frozen and no neural classifier head was trained. Caches are stored under `{rel(CACHE_DIR)}`.",
        "",
        markdown_table(cache_rows, max_rows=12),
        "",
        "## 7. Threshold-selection policy",
        "",
        "- `default_0.5`: fixed probability threshold 0.5.",
        "- `constrained_spec_ge_0.50`: threshold selected from training-only scores on a 0.00-1.00 grid in 0.01 steps to maximize recall subject to specificity >= 0.50.",
        "- In this CPU-bounded run, the training-only scores used for Platt calibration and constrained threshold selection are in-sample training-speaker scores; held-out validation fold outcomes are never used to choose thresholds.",
        "- Speaker-level `mean_score` is the main aggregation because it preserves calibrated score information while reducing single-clip volatility.",
        "",
        "## 8. Clip-level results",
        "",
        markdown_table(
            clip_top[
                [
                    "backbone",
                    "classifier",
                    "threshold_policy",
                    "recall_mean",
                    "recall_std",
                    "specificity_mean",
                    "specificity_std",
                    "precision_mean",
                    "f1_mean",
                    "roc_auc_mean",
                ]
            ].round(4),
            max_rows=12,
        ),
        "",
        "## 9. Speaker-level results",
        "",
        markdown_table(
            speaker_top[
                [
                    "backbone",
                    "classifier",
                    "aggregation_method",
                    "threshold_policy",
                    "recall_mean",
                    "recall_std",
                    "specificity_mean",
                    "specificity_std",
                    "precision_mean",
                    "f1_mean",
                    "roc_auc_mean",
                    "target_zone_pass",
                ]
            ].round(4),
            max_rows=12,
        ),
        "",
        "## 10. Seed and threshold stability",
        "",
        markdown_table(
            threshold_df[
                (threshold_df["dataset_version"] == "trusted_cleaned")
                & (threshold_df["metric_level"] == "speaker")
                & (threshold_df["aggregation_method"] == "mean_score")
            ][
                [
                    "backbone",
                    "classifier",
                    "threshold_policy",
                    "threshold_mean",
                    "threshold_std",
                    "threshold_min",
                    "threshold_max",
                    "seed_threshold_mean_std",
                    "selection_notes",
                ]
            ].round(4),
            max_rows=12,
        ),
        "",
        "## 11. Comparison against frozen classical baseline",
        "",
        f"The official frozen baseline is `{BASELINE['official_name']}` at threshold `{BASELINE['threshold']:.3f}`. "
        f"On the trusted cleaned seed-42 anchor it has clip recall {BASELINE['clip_seed42_recall']:.3f} and "
        f"specificity {BASELINE['clip_seed42_specificity']:.3f}. Across seeds 42-46 it has clip recall mean "
        f"{BASELINE['clip_multiseed_recall_mean']:.3f} and specificity mean {BASELINE['clip_multiseed_specificity_mean']:.3f}. "
        f"The available multi-seed speaker reference uses `{BASELINE['speaker_multiseed_aggregation']}` with recall "
        f"{BASELINE['speaker_multiseed_recall_mean']:.3f} and specificity {BASELINE['speaker_multiseed_specificity_mean']:.3f}. "
        "Those baseline values are used only as comparison references; no baseline file or metric was changed.",
        "",
        "## 12. Did any setup meet the target zone?",
        "",
        f"Trusted-cleaned speaker-level `mean_score` setups meeting recall >= {TARGET_RECALL:.2f} and specificity >= "
        f"{TARGET_SPECIFICITY:.2f}: **{target_count}**.",
        "",
        "## 13. Best candidate",
        "",
        markdown_table(best_df.round(4), max_rows=1),
        "",
        "## 14. Main limitations",
        "",
        "- The dataset is small and speaker grouped; results remain internal research evidence only.",
        "- Thresholds are selected inside training folds, but they are still estimated from limited data.",
        "- Platt calibration uses training-speaker in-sample scores for tractability in this run; this avoids held-out leakage but is less conservative than nested inner-fold calibration.",
        "- Governed/caution rows are not headline evidence.",
        "- Chunked extraction is deterministic and necessary for memory control, but it is still a pooled clip representation.",
        "- No backbone fine-tuning was performed, so this does not test task-adapted deep representations.",
        "",
        "## 15. Recommended next step",
        "",
        next_step,
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES, help="Hugging Face backbone IDs to test.")
    parser.add_argument("--include-optional-xlsr", action="store_true", help="Also include facebook/wav2vec2-large-xlsr-53.")
    parser.add_argument("--classifiers", nargs="+", default=CLASSIFIERS, choices=CLASSIFIERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--max-splits", type=int, default=5)
    parser.add_argument("--dataset", choices=["trusted", "governed", "both"], default="trusted")
    parser.add_argument("--audio-column", default="processed_file")
    parser.add_argument("--fallback-audio-column", default="audio_file")
    parser.add_argument("--pooling", choices=["mean", "mean_std"], default="mean_std")
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--max-clips", type=int, default=None, help="Debug-only row cap; do not use for final reporting.")
    parser.add_argument("--torch-threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    backbones = list(args.backbones)
    if args.include_optional_xlsr and OPTIONAL_BACKBONES[0] not in backbones:
        backbones.append(OPTIONAL_BACKBONES[0])

    dataset_specs: list[tuple[str, Path]] = []
    if args.dataset in {"trusted", "both"}:
        dataset_specs.append(("trusted_cleaned", TRUSTED_MANIFEST))
    if args.dataset in {"governed", "both"}:
        dataset_specs.append(("governed_cleaned", GOVERNED_MANIFEST))

    if not FEATURE_TABLE.exists():
        warnings.warn(f"Reference feature table missing: {FEATURE_TABLE}", stacklevel=2)

    all_clip_rows: list[dict[str, Any]] = []
    all_speaker_rows: list[dict[str, Any]] = []
    all_threshold_rows: list[dict[str, Any]] = []
    extraction_meta: list[dict[str, Any]] = []

    for dataset_slug, manifest_path in dataset_specs:
        manifest = load_manifest(manifest_path, dataset_slug=dataset_slug, max_clips=args.max_clips)
        print(f"[{dataset_slug}] loaded {len(manifest)} clips from {manifest_path}", flush=True)
        for backbone in backbones:
            print(f"[{dataset_slug}][{backbone}] extracting/loading embeddings", flush=True)
            embeddings, meta = extract_or_load_embeddings(
                manifest=manifest,
                manifest_path=manifest_path,
                dataset_slug=dataset_slug,
                backbone=backbone,
                audio_column=args.audio_column,
                fallback_audio_column=args.fallback_audio_column,
                pooling=args.pooling,
                chunk_seconds=args.chunk_seconds,
                batch_size=args.batch_size,
                device_arg=args.device,
                local_files_only=args.local_files_only,
                force_extract=args.force_extract,
                torch_threads=args.torch_threads,
            )
            extraction_meta.append(meta)
            for classifier in args.classifiers:
                print(f"[{dataset_slug}][{backbone}][{classifier}] evaluating grouped CV", flush=True)
                clip_rows, speaker_rows, threshold_rows = evaluate_backbone_classifier(
                    embeddings=embeddings,
                    dataset_slug=dataset_slug,
                    backbone=backbone,
                    classifier=classifier,
                    seeds=args.seeds,
                    max_splits=args.max_splits,
                )
                all_clip_rows.extend(clip_rows)
                all_speaker_rows.extend(speaker_rows)
                all_threshold_rows.extend(threshold_rows)

    clip_df = pd.DataFrame(all_clip_rows)
    speaker_df = pd.DataFrame(all_speaker_rows)
    fold_threshold_df = pd.DataFrame(all_threshold_rows)
    model_df = summarize_model_comparison(clip_df, speaker_df)
    threshold_df = summarize_threshold_stability(fold_threshold_df)
    best_df = select_best_candidate(model_df, clip_df)

    clip_df.to_csv(CLIP_SUMMARY_CSV, index=False)
    speaker_df.to_csv(SPEAKER_SUMMARY_CSV, index=False)
    model_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    threshold_df.to_csv(THRESHOLD_STABILITY_CSV, index=False)
    best_df.to_csv(BEST_CANDIDATE_CSV, index=False)
    write_report(
        clip_df=clip_df,
        speaker_df=speaker_df,
        model_df=model_df,
        threshold_df=threshold_df,
        best_df=best_df,
        extraction_meta=extraction_meta,
        backbones=backbones,
        classifiers=args.classifiers,
        seeds=args.seeds,
        max_splits=args.max_splits,
        pooling=args.pooling,
        chunk_seconds=args.chunk_seconds,
        governed_run=args.dataset in {"governed", "both"},
    )
    print("Created Phase C outputs:", flush=True)
    for path in [
        REPORT_MD,
        CLIP_SUMMARY_CSV,
        SPEAKER_SUMMARY_CSV,
        MODEL_COMPARISON_CSV,
        THRESHOLD_STABILITY_CSV,
        BEST_CANDIDATE_CSV,
    ]:
        print(f"- {path}", flush=True)


if __name__ == "__main__":
    main()
