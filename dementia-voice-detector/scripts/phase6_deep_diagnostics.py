"""Phase 6 deep-learning fixes and diagnostics.

This runner is intentionally additive. It does not overwrite the frozen Phase 4
baseline, and it does not modify the original Phase 6 pilot/full outputs. The
goal is to make the Priority 1 diagnostics reproducible from governed records:

- speaker-grouped CV with explicit no-overlap checks;
- same-split baseline, wav2vec, ensemble, fixed-embedding fusion, and attention
  pooling comparisons;
- threshold, calibration, segmentation aggregation, embedding visualization, and
  targeted error analysis tables/figures.

Full wav2vec fine-tuning remains in ``src/train_wav2vec.py``. This script uses
the saved fine-tuned checkpoint as a fixed embedding extractor so diagnostics
can run locally without retraining wav2vec for every ablation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import classification_metrics  # noqa: E402
from src.train_baseline import select_feature_columns  # noqa: E402
from src.train_wav2vec import (  # noqa: E402
    CLIP_MANIFEST_CSV,
    CONFIG_PATH,
    MODEL_ROOT,
    SEGMENT_MANIFEST_CSV,
    SegmentDataset,
    Wav2VecPhase6Config,
    aggregate_segments_to_clips,
    choose_device,
    collate_factory,
    ensure_manifest,
    load_config,
    set_reproducible_seed,
)


FEATURE_CSV = ROOT / "data/features/features_phase3_governed_pruned.csv"
REPORT_DIR = ROOT / "reports/phase6_deep_diagnostics"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
EMBEDDING_NPZ = TABLE_DIR / "phase6_wav2vec_segment_embeddings.npz"
SUMMARY_MD = REPORT_DIR / "phase6_deep_diagnostics_summary.md"

PHASE4_OFFICIAL = {
    "model": "phase4_logistic_regression_platt",
    "threshold": 0.300,
    "recall": 0.8617021276595744,
    "precision": 0.2892857142857143,
    "specificity": 0.21653543307086615,
    "balanced_accuracy": 0.5391187803652203,
    "f1": 2 * 0.2892857142857143 * 0.8617021276595744 / (0.2892857142857143 + 0.8617021276595744),
    "pr_auc": 0.3322394307030126,
    "brier_score": 0.1980628044332236,
}

METRIC_ORDER = [
    "balanced_accuracy",
    "recall",
    "precision",
    "f1",
    "specificity",
    "roc_auc",
    "pr_auc",
    "brier_score",
]

HANDCRAFTED_PRIORITY_COLUMNS = [
    "pause_ratio",
    "speaking_rate_proxy",
    "pause_rate_per_min",
    "speech_segment_rate_per_min",
    "speech_duration_sec",
    "duration_sec",
    "f0_semitone_std",
    "f0_std",
    "f0_range",
    "voiced_ratio",
    "jitter_local_proxy",
    "jitter_rap_proxy",
    "shimmer_local_proxy",
    "shimmer_db_proxy",
    "energy_modulation_std",
    "log_energy_modulation_std",
    "rms_mean",
    "rms_std",
    "spectral_centroid_mean",
    "spectral_flatness_mean",
    "mfcc_1_mean",
    "mfcc_1_std",
    "mfcc_2_mean",
    "mfcc_2_std",
]


@dataclass
class ModelResult:
    name: str
    split: str
    threshold: float
    threshold_note: str
    metrics: dict[str, float]
    calibration_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    extra: dict[str, Any]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "(empty)"
    out = df.head(max_rows).copy() if max_rows else df.copy()
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


def as_id(series: pd.Series) -> pd.Series:
    return series.astype(str)


def load_feature_table() -> pd.DataFrame:
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURE_CSV}")
    df = pd.read_csv(FEATURE_CSV)
    df = df[df["label"].isin([0, 1])].copy()
    if "feature_extraction_status" in df.columns:
        df = df[df["feature_extraction_status"].fillna("success").eq("success")].copy()
    df["metadata_index"] = as_id(df["metadata_index"])
    df["speaker_id"] = as_id(df["speaker_id"]).str.strip()
    return df


def merge_phase6_split(feature_df: pd.DataFrame) -> pd.DataFrame:
    if not CLIP_MANIFEST_CSV.exists():
        return feature_df.copy()
    clips = pd.read_csv(CLIP_MANIFEST_CSV)
    clips["metadata_index"] = as_id(clips["metadata_index"])
    split_cols = ["metadata_index", "phase6_split"]
    return feature_df.merge(clips[split_cols], on="metadata_index", how="left")


def available_handcrafted_columns(df: pd.DataFrame, max_columns: int = 24) -> list[str]:
    cols = [c for c in HANDCRAFTED_PRIORITY_COLUMNS if c in df.columns]
    if len(cols) < 10:
        fallback = [c for c in select_feature_columns(df) if c not in cols]
        cols.extend(fallback[: 10 - len(cols)])
    return cols[:max_columns]


def predict_at_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_prob, dtype=float) >= float(threshold)).astype(int)


def threshold_candidates(y_prob: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(y_prob, dtype=float))
    return np.unique(np.concatenate(([0.0, 1.0], values)))


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.50,
    target_recall: float = 0.70,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """Select a screening threshold from validation/calibration probabilities.

    Primary rule: maximize recall among thresholds with precision >= min_precision.
    Secondary rule: among those, prefer higher F1, balanced accuracy, specificity,
    and threshold. If no threshold reaches min_precision, fall back to max F1.
    """
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(y_prob):
        pred = predict_at_threshold(y_prob, threshold)
        metrics = classification_metrics(y_true, pred, y_prob)
        rows.append({"threshold": float(threshold), **metrics})
    table = pd.DataFrame(rows)
    viable = table[table["precision"] >= min_precision].copy()
    if not viable.empty:
        viable["target_recall_met"] = viable["recall"] >= target_recall
        selected = viable.sort_values(
            ["recall", "target_recall_met", "f1", "balanced_accuracy", "specificity", "threshold"],
            ascending=[False, False, False, False, False, False],
        ).iloc[0]
        note = "max_recall_with_precision_floor"
    else:
        selected = table.sort_values(
            ["f1", "balanced_accuracy", "recall", "precision", "specificity", "threshold"],
            ascending=[False, False, False, False, False, False],
        ).iloc[0]
        note = "precision_floor_not_reached_max_f1"
    info = selected.to_dict()
    info["selection_note"] = note
    info["min_precision"] = min_precision
    info["target_recall"] = target_recall
    return float(selected["threshold"]), info, table


def metrics_for_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    split: str = "test",
) -> dict[str, Any]:
    y_pred = predict_at_threshold(y_prob, threshold)
    metrics = classification_metrics(y_true, y_pred, y_prob)
    return {"model": model_name, "split": split, "threshold": threshold, **metrics}


def bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    speaker_ids: np.ndarray,
    seed: int,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = predict_at_threshold(y_prob, threshold)
    speakers = np.asarray(speaker_ids).astype(str)
    unique_speakers = np.unique(speakers)
    speaker_to_idx = {speaker: np.where(speakers == speaker)[0] for speaker in unique_speakers}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {m: [] for m in METRIC_ORDER}
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_speakers, size=len(unique_speakers), replace=True)
        idx = np.concatenate([speaker_to_idx[s] for s in sampled])
        if np.unique(y_true[idx]).size < 2:
            continue
        metrics = classification_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for metric in METRIC_ORDER:
            val = metrics.get(metric, np.nan)
            if not np.isnan(val):
                values[metric].append(float(val))
    rows = []
    point_metrics = classification_metrics(y_true, y_pred, y_prob)
    for metric in METRIC_ORDER:
        arr = np.asarray(values[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "point": point_metrics.get(metric, np.nan),
                "ci_low": float(np.percentile(arr, 2.5)) if arr.size else np.nan,
                "ci_high": float(np.percentile(arr, 97.5)) if arr.size else np.nan,
                "bootstrap_n": int(arr.size),
            }
        )
    return pd.DataFrame(rows)


def run_data_quality_audit(feature_df: pd.DataFrame) -> pd.DataFrame:
    speech_duration = feature_df.get("speech_duration_sec", feature_df.get("duration_sec", pd.Series(np.nan, index=feature_df.index)))
    silence_ratio = feature_df.get("silence_ratio", feature_df.get("pause_ratio", pd.Series(np.nan, index=feature_df.index)))
    audit = feature_df[["metadata_index", "speaker_id", "label", "datasplit", "duration_sec"]].copy()
    audit["speech_duration_sec"] = speech_duration
    audit["silence_ratio_or_pause_proxy"] = silence_ratio
    audit["flag_short_speech_lt_1p5s"] = speech_duration.astype(float) < 1.5
    audit["flag_silence_gt_80pct"] = silence_ratio.astype(float) > 0.80
    speaker_counts = feature_df.groupby("speaker_id")["metadata_index"].count()
    audit["speaker_clip_count"] = audit["speaker_id"].map(speaker_counts).astype(int)
    audit["flag_single_clip_speaker"] = audit["speaker_clip_count"].eq(1)
    audit["quality_pass"] = ~(audit["flag_short_speech_lt_1p5s"] | audit["flag_silence_gt_80pct"])
    audit.to_csv(TABLE_DIR / "data_quality_audit.csv", index=False)
    summary = pd.DataFrame(
        [
            {"check": "clips_total", "count": len(audit)},
            {"check": "short_speech_lt_1p5s", "count": int(audit["flag_short_speech_lt_1p5s"].sum())},
            {"check": "silence_gt_80pct", "count": int(audit["flag_silence_gt_80pct"].sum())},
            {"check": "single_clip_speakers", "count": int(audit["flag_single_clip_speaker"].sum())},
            {"check": "quality_pass_clips", "count": int(audit["quality_pass"].sum())},
        ]
    )
    summary.to_csv(TABLE_DIR / "data_quality_audit_summary.csv", index=False)
    return audit


def grouped_cv_splitter(y: np.ndarray, groups: np.ndarray, max_splits: int = 5) -> StratifiedGroupKFold:
    speaker_labels = pd.DataFrame({"speaker_id": groups.astype(str), "label": y.astype(int)}).drop_duplicates("speaker_id")
    min_class_speakers = int(speaker_labels["label"].value_counts().min())
    n_splits = max(2, min(max_splits, min_class_speakers))
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)


def summarize_cv_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in fold_df.groupby("model"):
        row: dict[str, Any] = {"model": model, "n_folds": int(sub["fold"].nunique())}
        for metric in METRIC_ORDER:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("balanced_accuracy_mean", ascending=False)


def run_speaker_grouped_cv(feature_df: pd.DataFrame) -> pd.DataFrame:
    cv_df = feature_df[feature_df["datasplit"].isin(["train", "valid"])].copy()
    feature_cols = select_feature_columns(cv_df)
    x = cv_df[feature_cols]
    y = cv_df["label"].astype(int).to_numpy()
    groups = cv_df["speaker_id"].astype(str).to_numpy()
    splitter = grouped_cv_splitter(y, groups)
    models: dict[str, Pipeline] = {
        "grouped_cv_logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=42)),
            ]
        ),
        "grouped_cv_random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        random_state=42,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
        train_speakers = set(groups[train_idx])
        val_speakers = set(groups[val_idx])
        overlap = sorted(train_speakers & val_speakers)
        overlap_rows.append(
            {
                "fold": fold,
                "train_speakers": len(train_speakers),
                "validation_speakers": len(val_speakers),
                "overlap_speaker_count": len(overlap),
                "status": "pass" if not overlap else "fail",
            }
        )
        for model_name, estimator in models.items():
            model = clone(estimator)
            model.fit(x.iloc[train_idx], y[train_idx])
            y_prob = model.predict_proba(x.iloc[val_idx])[:, 1]
            row = metrics_for_predictions(model_name, y[val_idx], y_prob, threshold=0.5, split=f"fold_{fold}")
            row["fold"] = fold
            rows.append(row)
    fold_metrics = pd.DataFrame(rows)
    overlap_df = pd.DataFrame(overlap_rows)
    summary = summarize_cv_metrics(fold_metrics)
    fold_metrics.to_csv(TABLE_DIR / "speaker_grouped_cv_fold_metrics.csv", index=False)
    overlap_df.to_csv(TABLE_DIR / "speaker_grouped_cv_overlap_check.csv", index=False)
    summary.to_csv(TABLE_DIR / "speaker_grouped_cv_summary.csv", index=False)
    return summary


def fit_same_split_baseline(feature_df: pd.DataFrame, cfg: Wav2VecPhase6Config) -> tuple[pd.DataFrame, pd.DataFrame, ModelResult]:
    df = merge_phase6_split(feature_df)
    feature_cols = select_feature_columns(df)
    train_df = df[df["phase6_split"].eq("train")].copy()
    cal_df = df[df["phase6_split"].eq("calibration")].copy()
    test_df = df[df["phase6_split"].eq("test")].copy()
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=cfg.seed)),
        ]
    )
    model.fit(train_df[feature_cols], train_df["label"].astype(int))
    cal_prob = model.predict_proba(cal_df[feature_cols])[:, 1]
    threshold, threshold_info, threshold_table = select_threshold(
        cal_df["label"].astype(int).to_numpy(),
        cal_prob,
        min_precision=0.50,
        target_recall=0.70,
    )
    threshold_table.to_csv(TABLE_DIR / "baseline_same_split_threshold_selection_precision_floor.csv", index=False)
    test_prob = model.predict_proba(test_df[feature_cols])[:, 1]
    cal_pred = cal_df[["metadata_index", "speaker_id", "label", "phase6_split"]].copy()
    cal_pred["probability"] = cal_prob
    cal_pred["prediction"] = predict_at_threshold(cal_prob, threshold)
    test_pred = test_df[["metadata_index", "speaker_id", "processed_file", "label", "phase6_split"]].copy()
    test_pred["probability"] = test_prob
    test_pred["prediction"] = predict_at_threshold(test_prob, threshold)
    test_pred.to_csv(TABLE_DIR / "baseline_same_split_test_predictions_precision_floor.csv", index=False)
    metrics = classification_metrics(test_pred["label"].to_numpy(), test_pred["prediction"].to_numpy(), test_pred["probability"].to_numpy())
    result = ModelResult(
        name="baseline_logistic_same_split",
        split="test",
        threshold=threshold,
        threshold_note=str(threshold_info["selection_note"]),
        metrics=metrics,
        calibration_predictions=cal_pred,
        test_predictions=test_pred,
        extra={"threshold_info": threshold_info},
    )
    return cal_pred, test_pred, result


def load_existing_wav2vec_predictions(cfg: Wav2VecPhase6Config) -> ModelResult | None:
    cal_path = ROOT / "reports/tables/phase6_wav2vec_calibration_segment_predictions.csv"
    test_path = ROOT / "reports/tables/phase6_wav2vec_test_segment_predictions.csv"
    if not cal_path.exists() or not test_path.exists():
        return None
    cal_segments = pd.read_csv(cal_path)
    test_segments = pd.read_csv(test_path)
    cal_segments["metadata_index"] = as_id(cal_segments["metadata_index"])
    test_segments["metadata_index"] = as_id(test_segments["metadata_index"])
    cal_clip = aggregate_segments_to_clips(cal_segments, threshold=0.5)
    test_clip = aggregate_segments_to_clips(test_segments, threshold=0.5)
    cal_clip["metadata_index"] = as_id(cal_clip["metadata_index"])
    test_clip["metadata_index"] = as_id(test_clip["metadata_index"])
    threshold, threshold_info, threshold_table = select_threshold(
        cal_clip["label"].to_numpy(),
        cal_clip["probability"].to_numpy(),
        min_precision=0.50,
        target_recall=0.70,
    )
    threshold_table.to_csv(TABLE_DIR / "wav2vec_threshold_selection_precision_floor.csv", index=False)
    cal_clip["prediction"] = predict_at_threshold(cal_clip["probability"], threshold)
    test_clip["prediction"] = predict_at_threshold(test_clip["probability"], threshold)
    test_clip.to_csv(TABLE_DIR / "wav2vec_test_clip_predictions_precision_floor.csv", index=False)
    metrics = classification_metrics(test_clip["label"].to_numpy(), test_clip["prediction"].to_numpy(), test_clip["probability"].to_numpy())
    return ModelResult(
        name="wav2vec_mean_pool_existing",
        split="test",
        threshold=threshold,
        threshold_note=str(threshold_info["selection_note"]),
        metrics=metrics,
        calibration_predictions=cal_clip,
        test_predictions=test_clip,
        extra={"threshold_info": threshold_info},
    )


def safe_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def run_ensemble_variants(
    baseline_result: ModelResult,
    wav_result: ModelResult | None,
    cfg: Wav2VecPhase6Config,
) -> list[ModelResult]:
    if wav_result is None:
        return []
    cal = baseline_result.calibration_predictions[["metadata_index", "label", "speaker_id", "probability"]].merge(
        wav_result.calibration_predictions[["metadata_index", "probability"]],
        on="metadata_index",
        how="inner",
        suffixes=("_ml", "_w2v"),
    )
    test = baseline_result.test_predictions[["metadata_index", "label", "speaker_id", "probability"]].merge(
        wav_result.test_predictions[["metadata_index", "probability"]],
        on="metadata_index",
        how="inner",
        suffixes=("_ml", "_w2v"),
    )
    results: list[ModelResult] = []

    def make_result(name: str, cal_prob: np.ndarray, test_prob: np.ndarray, extra: dict[str, Any]) -> ModelResult:
        threshold, threshold_info, threshold_table = select_threshold(cal["label"].to_numpy(), cal_prob)
        threshold_table.to_csv(TABLE_DIR / f"{name}_threshold_selection.csv", index=False)
        cal_pred = cal[["metadata_index", "speaker_id", "label"]].copy()
        cal_pred["probability"] = cal_prob
        cal_pred["prediction"] = predict_at_threshold(cal_prob, threshold)
        test_pred = test[["metadata_index", "speaker_id", "label"]].copy()
        test_pred["probability"] = test_prob
        test_pred["prediction"] = predict_at_threshold(test_prob, threshold)
        test_pred.to_csv(TABLE_DIR / f"{name}_test_predictions.csv", index=False)
        metrics = classification_metrics(test_pred["label"].to_numpy(), test_pred["prediction"].to_numpy(), test_pred["probability"].to_numpy())
        return ModelResult(name, "test", threshold, str(threshold_info["selection_note"]), metrics, cal_pred, test_pred, {**extra, "threshold_info": threshold_info})

    simple_cal = 0.5 * cal["probability_ml"].to_numpy() + 0.5 * cal["probability_w2v"].to_numpy()
    simple_test = 0.5 * test["probability_ml"].to_numpy() + 0.5 * test["probability_w2v"].to_numpy()
    results.append(make_result("ensemble_simple_average", simple_cal, simple_test, {"alpha_ml": 0.5}))

    weight_rows = []
    best: tuple[tuple[float, float, float, float, float], float, float, np.ndarray, np.ndarray, dict[str, Any]] | None = None
    for alpha in np.linspace(0.0, 1.0, 21):
        cal_prob = alpha * cal["probability_ml"].to_numpy() + (1.0 - alpha) * cal["probability_w2v"].to_numpy()
        test_prob = alpha * test["probability_ml"].to_numpy() + (1.0 - alpha) * test["probability_w2v"].to_numpy()
        threshold, threshold_info, _ = select_threshold(cal["label"].to_numpy(), cal_prob)
        cal_metrics = classification_metrics(cal["label"].to_numpy(), predict_at_threshold(cal_prob, threshold), cal_prob)
        key = (
            float(cal_metrics["precision"] >= 0.50),
            float(cal_metrics["recall"]),
            float(cal_metrics["f1"]),
            float(cal_metrics["balanced_accuracy"]),
            float(alpha),
        )
        weight_rows.append({"alpha_ml": alpha, "threshold": threshold, **cal_metrics})
        if best is None or key > best[0]:
            best = (key, float(alpha), threshold, cal_prob, test_prob, threshold_info)
    pd.DataFrame(weight_rows).to_csv(TABLE_DIR / "ensemble_weight_search.csv", index=False)
    assert best is not None
    _, alpha, _, best_cal_prob, best_test_prob, _ = best
    results.append(make_result("ensemble_weighted_average", best_cal_prob, best_test_prob, {"alpha_ml": alpha}))

    stack = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", solver="lbfgs", random_state=cfg.seed)),
        ]
    )
    x_cal = cal[["probability_ml", "probability_w2v"]].to_numpy()
    x_test = test[["probability_ml", "probability_w2v"]].to_numpy()
    stack.fit(x_cal, cal["label"].astype(int).to_numpy())
    results.append(make_result("ensemble_stacking_logistic", stack.predict_proba(x_cal)[:, 1], stack.predict_proba(x_test)[:, 1], {}))
    return results


def checkpoint_path_for_config(cfg: Wav2VecPhase6Config) -> Path:
    default = MODEL_ROOT / cfg.run_id / "full/best_model.pt"
    if default.exists():
        return default
    candidates = sorted(MODEL_ROOT.glob("*/full/best_model.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("No wav2vec full checkpoint found under models/wav2vec/*/full/best_model.pt")


def load_embedding_cache() -> dict[str, Any] | None:
    if not EMBEDDING_NPZ.exists():
        return None
    data = np.load(EMBEDDING_NPZ, allow_pickle=True)
    return {key: data[key] for key in data.files}


def extract_wav2vec_embeddings(cfg: Wav2VecPhase6Config, batch_size: int = 16, force: bool = False) -> dict[str, Any] | None:
    if EMBEDDING_NPZ.exists() and not force:
        return load_embedding_cache()
    try:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2Model
    except Exception as exc:
        print(f"Embedding extraction skipped: torch/transformers unavailable ({exc})", flush=True)
        return None

    _, segments = ensure_manifest(cfg)
    segments = segments.copy()
    segments["metadata_index"] = as_id(segments["metadata_index"])
    checkpoint_path = checkpoint_path_for_config(cfg)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_cfg = checkpoint.get("config", {})
    dropout = float(checkpoint_cfg.get("dropout", cfg.dropout))

    class EmbeddingClassifier(torch.nn.Module):
        def __init__(self, model_name: str, dropout_rate: float):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, local_files_only=True)
            self.dropout = torch.nn.Dropout(dropout_rate)
            self.classifier = torch.nn.Linear(self.wav2vec2.config.hidden_size, 2)

        def pool(self, hidden: Any, attention_mask: Any | None) -> Any:
            if attention_mask is not None and hasattr(self.wav2vec2, "_get_feature_vector_attention_mask"):
                feature_mask = self.wav2vec2._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
                mask = feature_mask.unsqueeze(-1).to(hidden.dtype)
                return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return hidden.mean(dim=1)

        def forward(self, input_values: Any, attention_mask: Any | None = None) -> tuple[Any, Any]:
            outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
            pooled = self.pool(outputs.last_hidden_state, attention_mask)
            logits = self.classifier(self.dropout(pooled))
            return logits, pooled

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name, local_files_only=True)
    device = choose_device(torch)
    model = EmbeddingClassifier(cfg.model_name, dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    dataset = SegmentDataset(
        segments,
        sampling_rate=cfg.target_sampling_rate,
        segment_duration_seconds=cfg.segment_duration_seconds,
        augment=False,
        seed=cfg.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_factory(feature_extractor, cfg.target_sampling_rate),
    )

    embeddings: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            inputs = batch["input_values"].to(device)
            mask = batch.get("attention_mask")
            mask = mask.to(device) if mask is not None else None
            logits, pooled = model(inputs, attention_mask=mask)
            prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
            probabilities.append(prob.astype(np.float32))
            for i, meta in enumerate(batch["metadata"]):
                metadata_rows.append(
                    {
                        **meta,
                        "label": int(batch["labels"][i].detach().cpu()),
                        "phase6_split": str(segments.iloc[len(metadata_rows)]["phase6_split"]),
                    }
                )
            if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == len(loader):
                print(f"Embedding extraction: batch {batch_idx}/{len(loader)}", flush=True)

    meta = pd.DataFrame(metadata_rows)
    meta["metadata_index"] = as_id(meta["metadata_index"])
    all_embeddings = np.vstack(embeddings)
    all_prob = np.concatenate(probabilities)
    np.savez_compressed(
        EMBEDDING_NPZ,
        embeddings=all_embeddings,
        probabilities=all_prob,
        segment_id=meta["segment_id"].astype(str).to_numpy(),
        metadata_index=meta["metadata_index"].astype(str).to_numpy(),
        speaker_id=meta["speaker_id"].astype(str).to_numpy(),
        label=meta["label"].astype(int).to_numpy(),
        phase6_split=meta["phase6_split"].astype(str).to_numpy(),
        processed_file=meta["processed_file"].astype(str).to_numpy(),
        start_sec=meta["start_sec"].astype(float).to_numpy(),
        end_sec=meta["end_sec"].astype(float).to_numpy(),
    )
    segment_pred = meta.copy()
    segment_pred["probability"] = all_prob
    segment_pred["prediction"] = predict_at_threshold(all_prob, 0.5)
    segment_pred.to_csv(TABLE_DIR / "wav2vec_all_segment_predictions_from_checkpoint.csv", index=False)
    return load_embedding_cache()


def clip_embedding_frame(cache: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    emb = np.asarray(cache["embeddings"], dtype=np.float32)
    meta = pd.DataFrame(
        {
            "metadata_index": cache["metadata_index"].astype(str),
            "speaker_id": cache["speaker_id"].astype(str),
            "label": cache["label"].astype(int),
            "phase6_split": cache["phase6_split"].astype(str),
            "probability": np.asarray(cache["probabilities"], dtype=float),
        }
    )
    rows = []
    clip_to_embedding: dict[str, np.ndarray] = {}
    for clip_id, idx in meta.groupby("metadata_index").groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        pooled = emb[idx_arr].mean(axis=0)
        clip_to_embedding[str(clip_id)] = pooled
        sub = meta.iloc[idx_arr]
        rows.append(
            {
                "metadata_index": str(clip_id),
                "speaker_id": sub["speaker_id"].iloc[0],
                "label": int(sub["label"].iloc[0]),
                "phase6_split": sub["phase6_split"].iloc[0],
                "probability": float(sub["probability"].mean()),
                "n_segments": int(len(idx_arr)),
            }
        )
    return pd.DataFrame(rows), clip_to_embedding


def matrix_from_clip_embeddings(df: pd.DataFrame, clip_to_embedding: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([clip_to_embedding[str(mid)] for mid in df["metadata_index"].astype(str)])


def run_fixed_embedding_fusion(
    feature_df: pd.DataFrame,
    cache: dict[str, Any] | None,
    seeds: list[int],
) -> list[ModelResult]:
    if cache is None:
        return []
    clip_emb_df, clip_to_embedding = clip_embedding_frame(cache)
    features = merge_phase6_split(feature_df)
    features["metadata_index"] = as_id(features["metadata_index"])
    handcrafted_cols = available_handcrafted_columns(features)
    merged = clip_emb_df.merge(
        features[["metadata_index", "processed_file", *handcrafted_cols]],
        on="metadata_index",
        how="inner",
    )
    split_frames = {split: merged[merged["phase6_split"].eq(split)].copy() for split in ["train", "calibration", "test"]}
    if any(split_frames[s].empty for s in split_frames):
        return []

    model_specs: dict[str, Callable[[int], Pipeline]] = {
        "wav2vec_embedding_mlp": lambda seed: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=5e-4,
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.20,
                        n_iter_no_change=25,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hybrid_fusion_mlp": lambda seed: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=2e-3,
                        learning_rate_init=5e-4,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.20,
                        n_iter_no_change=30,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    result_rows = []
    best_by_model: dict[str, ModelResult] = {}
    for model_name, factory in model_specs.items():
        for seed in seeds:
            set_reproducible_seed(seed)
            x_train_emb = matrix_from_clip_embeddings(split_frames["train"], clip_to_embedding)
            x_cal_emb = matrix_from_clip_embeddings(split_frames["calibration"], clip_to_embedding)
            x_test_emb = matrix_from_clip_embeddings(split_frames["test"], clip_to_embedding)
            if model_name == "hybrid_fusion_mlp":
                x_train = np.hstack([x_train_emb, split_frames["train"][handcrafted_cols].to_numpy(dtype=float)])
                x_cal = np.hstack([x_cal_emb, split_frames["calibration"][handcrafted_cols].to_numpy(dtype=float)])
                x_test = np.hstack([x_test_emb, split_frames["test"][handcrafted_cols].to_numpy(dtype=float)])
            else:
                x_train, x_cal, x_test = x_train_emb, x_cal_emb, x_test_emb

            y_train = split_frames["train"]["label"].astype(int).to_numpy()
            y_cal = split_frames["calibration"]["label"].astype(int).to_numpy()
            y_test = split_frames["test"]["label"].astype(int).to_numpy()
            model = factory(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)
            cal_prob = model.predict_proba(x_cal)[:, 1]
            threshold, threshold_info, threshold_table = select_threshold(y_cal, cal_prob)
            test_prob = model.predict_proba(x_test)[:, 1]
            test_pred = split_frames["test"][["metadata_index", "speaker_id", "processed_file", "label"]].copy()
            test_pred["probability"] = test_prob
            test_pred["prediction"] = predict_at_threshold(test_prob, threshold)
            metrics = classification_metrics(y_test, test_pred["prediction"].to_numpy(), test_prob)
            result_rows.append({"model": model_name, "seed": seed, "threshold": threshold, "threshold_note": threshold_info["selection_note"], **metrics})
            threshold_table.to_csv(TABLE_DIR / f"{model_name}_seed{seed}_threshold_selection.csv", index=False)

            cal_pred = split_frames["calibration"][["metadata_index", "speaker_id", "label"]].copy()
            cal_pred["probability"] = cal_prob
            cal_pred["prediction"] = predict_at_threshold(cal_prob, threshold)
            result = ModelResult(
                name=f"{model_name}_seed{seed}",
                split="test",
                threshold=threshold,
                threshold_note=str(threshold_info["selection_note"]),
                metrics=metrics,
                calibration_predictions=cal_pred,
                test_predictions=test_pred,
                extra={"seed": seed, "handcrafted_columns": handcrafted_cols if model_name == "hybrid_fusion_mlp" else []},
            )
            current_best = best_by_model.get(model_name)
            if current_best is None or (metrics["f1"], metrics["precision"], metrics["recall"]) > (
                current_best.metrics["f1"],
                current_best.metrics["precision"],
                current_best.metrics["recall"],
            ):
                best_by_model[model_name] = result

    seed_metrics = pd.DataFrame(result_rows)
    seed_metrics.to_csv(TABLE_DIR / "fusion_seed_metrics.csv", index=False)
    seed_summary = summarize_seed_metrics(seed_metrics)
    seed_summary.to_csv(TABLE_DIR / "fusion_seed_summary.csv", index=False)
    for result in best_by_model.values():
        result.test_predictions.to_csv(TABLE_DIR / f"{result.name}_test_predictions.csv", index=False)
    return list(best_by_model.values())


def summarize_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in seed_metrics.groupby("model"):
        row: dict[str, Any] = {"model": model, "n_seeds": int(sub["seed"].nunique())}
        for metric in METRIC_ORDER:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False)


def run_fusion_grouped_cv(feature_df: pd.DataFrame, cache: dict[str, Any] | None) -> pd.DataFrame:
    if cache is None:
        return pd.DataFrame()
    clip_emb_df, clip_to_embedding = clip_embedding_frame(cache)
    features = feature_df[feature_df["datasplit"].isin(["train", "valid"])].copy()
    handcrafted_cols = available_handcrafted_columns(features)
    merged = features[["metadata_index", "speaker_id", "label", *handcrafted_cols]].merge(
        clip_emb_df[["metadata_index"]],
        on="metadata_index",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    x_emb = matrix_from_clip_embeddings(merged, clip_to_embedding)
    x_fusion = np.hstack([x_emb, merged[handcrafted_cols].to_numpy(dtype=float)])
    y = merged["label"].astype(int).to_numpy()
    groups = merged["speaker_id"].astype(str).to_numpy()
    splitter = grouped_cv_splitter(y, groups)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=42),
            ),
        ]
    )
    rows = []
    overlap_rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_fusion, y, groups), start=1):
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        overlap_rows.append({"fold": fold, "overlap_speaker_count": len(overlap), "status": "pass" if not overlap else "fail"})
        estimator = clone(model)
        estimator.fit(x_fusion[train_idx], y[train_idx])
        y_prob = estimator.predict_proba(x_fusion[val_idx])[:, 1]
        row = metrics_for_predictions("grouped_cv_hybrid_fixed_embedding_logistic", y[val_idx], y_prob, 0.5, split=f"fold_{fold}")
        row["fold"] = fold
        rows.append(row)
    fold_df = pd.DataFrame(rows)
    summary = summarize_cv_metrics(fold_df)
    fold_df.to_csv(TABLE_DIR / "fusion_grouped_cv_fold_metrics.csv", index=False)
    summary.to_csv(TABLE_DIR / "fusion_grouped_cv_summary.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(TABLE_DIR / "fusion_grouped_cv_overlap_check.csv", index=False)
    return summary


def train_attention_pooling(
    feature_df: pd.DataFrame,
    cache: dict[str, Any] | None,
    seed: int,
    epochs: int = 120,
) -> ModelResult | None:
    if cache is None:
        return None
    try:
        import torch
    except Exception:
        return None

    emb = np.asarray(cache["embeddings"], dtype=np.float32)
    meta = pd.DataFrame(
        {
            "metadata_index": cache["metadata_index"].astype(str),
            "speaker_id": cache["speaker_id"].astype(str),
            "label": cache["label"].astype(int),
            "phase6_split": cache["phase6_split"].astype(str),
        }
    )
    features = merge_phase6_split(feature_df)
    features["metadata_index"] = as_id(features["metadata_index"])
    handcrafted_cols = available_handcrafted_columns(features)
    feature_lookup = features.set_index("metadata_index")[handcrafted_cols]
    clip_ids = sorted(meta["metadata_index"].unique())
    train_segment_idx = meta.index[meta["phase6_split"].eq("train")].to_numpy()
    emb_mean = emb[train_segment_idx].mean(axis=0)
    emb_std = emb[train_segment_idx].std(axis=0) + 1e-6
    feature_imputer = SimpleImputer(strategy="median")
    feature_scaler = StandardScaler()
    train_clip_ids = [c for c in clip_ids if meta.loc[meta["metadata_index"].eq(c), "phase6_split"].iloc[0] == "train"]
    train_features = feature_lookup.loc[train_clip_ids].to_numpy(dtype=float)
    feature_scaler.fit(feature_imputer.fit_transform(train_features))

    def build_clip_records(split: str) -> list[dict[str, Any]]:
        records = []
        for clip_id in clip_ids:
            idx = meta.index[meta["metadata_index"].eq(clip_id)].to_numpy()
            if len(idx) == 0:
                continue
            sub = meta.iloc[idx]
            if sub["phase6_split"].iloc[0] != split:
                continue
            feature_vec = feature_lookup.loc[[clip_id]].to_numpy(dtype=float)
            feature_vec = feature_scaler.transform(feature_imputer.transform(feature_vec))[0].astype(np.float32)
            records.append(
                {
                    "metadata_index": clip_id,
                    "speaker_id": sub["speaker_id"].iloc[0],
                    "label": int(sub["label"].iloc[0]),
                    "embeddings": ((emb[idx] - emb_mean) / emb_std).astype(np.float32),
                    "features": feature_vec,
                }
            )
        return records

    train_records = build_clip_records("train")
    cal_records = build_clip_records("calibration")
    test_records = build_clip_records("test")
    if not train_records or not cal_records or not test_records:
        return None

    class AttentionClassifier(torch.nn.Module):
        def __init__(self, embedding_dim: int, feature_dim: int):
            super().__init__()
            self.query = torch.nn.Parameter(torch.zeros(embedding_dim))
            torch.nn.init.normal_(self.query, std=0.02)
            self.net = torch.nn.Sequential(
                torch.nn.Dropout(0.35),
                torch.nn.Linear(embedding_dim + feature_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.35),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )

        def forward(self, seq: Any, features: Any) -> Any:
            scores = (seq @ self.query) / math.sqrt(seq.shape[-1])
            weights = torch.softmax(scores, dim=0)
            pooled = (seq * weights.unsqueeze(1)).sum(dim=0)
            return self.net(torch.cat([pooled, features], dim=0)).squeeze(0)

    set_reproducible_seed(seed)
    device = torch.device("cpu")
    model = AttentionClassifier(emb.shape[1], len(handcrafted_cols)).to(device)
    pos = sum(r["label"] for r in train_records)
    neg = len(train_records) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    rng = np.random.default_rng(seed)
    best_state = None
    best_score = -np.inf
    patience = 25
    patience_left = patience
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_records))
        for idx in order:
            rec = train_records[int(idx)]
            seq = torch.tensor(rec["embeddings"], dtype=torch.float32, device=device)
            feats = torch.tensor(rec["features"], dtype=torch.float32, device=device)
            label = torch.tensor(float(rec["label"]), dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(seq, feats).view(1), label.view(1))
            loss.backward()
            optimizer.step()
        cal_prob = predict_attention_records(model, cal_records, device)
        cal_y = np.asarray([r["label"] for r in cal_records], dtype=int)
        try:
            score = average_precision_score(cal_y, cal_prob)
        except Exception:
            score = classification_metrics(cal_y, predict_at_threshold(cal_prob, 0.5), cal_prob)["f1"]
        if score > best_score:
            best_score = float(score)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    cal_prob = predict_attention_records(model, cal_records, device)
    cal_y = np.asarray([r["label"] for r in cal_records], dtype=int)
    threshold, threshold_info, threshold_table = select_threshold(cal_y, cal_prob)
    threshold_table.to_csv(TABLE_DIR / "attention_pooling_threshold_selection.csv", index=False)
    test_prob = predict_attention_records(model, test_records, device)
    test_pred = records_to_prediction_frame(test_records, test_prob, threshold)
    test_pred.to_csv(TABLE_DIR / "attention_pooling_test_predictions.csv", index=False)
    cal_pred = records_to_prediction_frame(cal_records, cal_prob, threshold)
    metrics = classification_metrics(test_pred["label"].to_numpy(), test_pred["prediction"].to_numpy(), test_pred["probability"].to_numpy())
    return ModelResult(
        "attention_pooling_hybrid",
        "test",
        threshold,
        str(threshold_info["selection_note"]),
        metrics,
        cal_pred,
        test_pred,
        {"seed": seed, "validation_pr_auc": best_score, "handcrafted_columns": handcrafted_cols},
    )


def predict_attention_records(model: Any, records: list[dict[str, Any]], device: Any) -> np.ndarray:
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for rec in records:
            seq = torch.tensor(rec["embeddings"], dtype=torch.float32, device=device)
            feats = torch.tensor(rec["features"], dtype=torch.float32, device=device)
            prob = torch.sigmoid(model(seq, feats)).detach().cpu().item()
            probs.append(prob)
    return np.asarray(probs, dtype=float)


def records_to_prediction_frame(records: list[dict[str, Any]], probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "metadata_index": [r["metadata_index"] for r in records],
            "speaker_id": [r["speaker_id"] for r in records],
            "label": [r["label"] for r in records],
            "probability": probabilities,
        }
    )
    out["prediction"] = predict_at_threshold(out["probability"].to_numpy(), threshold)
    return out


def run_segmentation_aggregation_comparison(
    wav_result: ModelResult | None,
    cfg: Wav2VecPhase6Config,
) -> pd.DataFrame:
    rows = []
    cal_path = ROOT / "reports/tables/phase6_wav2vec_calibration_segment_predictions.csv"
    test_path = ROOT / "reports/tables/phase6_wav2vec_test_segment_predictions.csv"
    if wav_result is not None and cal_path.exists() and test_path.exists():
        cal_segments = pd.read_csv(cal_path)
        test_segments = pd.read_csv(test_path)
        aggregators: dict[str, Callable[[pd.Series], float]] = {
            "mean": lambda s: float(s.mean()),
            "median": lambda s: float(s.median()),
            "max": lambda s: float(s.max()),
            "p75": lambda s: float(np.percentile(s, 75)),
            "top2_mean": lambda s: float(np.mean(np.sort(s.to_numpy())[-min(2, len(s)) :])),
        }
        for agg_name, func in aggregators.items():
            cal_clip = aggregate_with_function(cal_segments, func)
            test_clip = aggregate_with_function(test_segments, func)
            threshold, threshold_info, _ = select_threshold(cal_clip["label"].to_numpy(), cal_clip["probability"].to_numpy())
            pred = predict_at_threshold(test_clip["probability"].to_numpy(), threshold)
            metrics = classification_metrics(test_clip["label"].to_numpy(), pred, test_clip["probability"].to_numpy())
            rows.append(
                {
                    "strategy": "3s_window_0s_overlap",
                    "aggregation": agg_name,
                    "threshold": threshold,
                    "threshold_note": threshold_info["selection_note"],
                    "mean_segment_probability_std": float(test_segments.groupby("metadata_index")["probability"].std().mean()),
                    **metrics,
                }
            )

    # Manifest-only counts for requested alternatives. Inference is intentionally
    # separate from this fast diagnostic because every variant requires another
    # wav2vec pass over calibration/test audio.
    if CLIP_MANIFEST_CSV.exists():
        clips = pd.read_csv(CLIP_MANIFEST_CSV)
        for window, overlap in [(3.0, 0.0), (3.0, 1.5), (5.0, 0.0), (5.0, 2.5)]:
            hop = window - overlap
            counts = []
            for _, row in clips.iterrows():
                duration = float(row.get("duration_sec", np.nan))
                if np.isnan(duration) and Path(str(row["processed_file"])).exists():
                    import soundfile as sf

                    info = sf.info(str(row["processed_file"]))
                    duration = float(info.frames / info.samplerate)
                starts = np.arange(0.0, max(duration, 0.0), hop)
                n_segments = sum(1 for start in starts if min(start + window, duration) - start >= cfg.min_segment_duration)
                counts.append(n_segments)
            rows.append(
                {
                    "strategy": f"{window:g}s_window_{overlap:g}s_overlap",
                    "aggregation": "manifest_count_only",
                    "threshold": np.nan,
                    "threshold_note": "probability_inference_not_rerun_for_variant",
                    "mean_segments_per_clip": float(np.mean(counts)) if counts else np.nan,
                    "median_segments_per_clip": float(np.median(counts)) if counts else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "segmentation_aggregation_comparison.csv", index=False)
    return table


def aggregate_with_function(segment_df: pd.DataFrame, func: Callable[[pd.Series], float]) -> pd.DataFrame:
    rows = []
    segment_df = segment_df.copy()
    segment_df["metadata_index"] = as_id(segment_df["metadata_index"])
    for clip_id, sub in segment_df.groupby("metadata_index"):
        prob = func(sub["probability"])
        rows.append(
            {
                "metadata_index": clip_id,
                "speaker_id": sub["speaker_id"].iloc[0],
                "label": int(sub["label"].iloc[0]),
                "probability": prob,
                "n_segments": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def choose_best_result(results: list[ModelResult]) -> ModelResult:
    def key(result: ModelResult) -> tuple[float, float, float, float, float]:
        metrics = result.metrics
        success_precision = metrics.get("precision", 0.0) >= 0.50
        success_recall = metrics.get("recall", 0.0) >= 0.70
        return (
            float(success_precision and success_recall),
            float(metrics.get("f1", 0.0)),
            float(metrics.get("pr_auc", 0.0)),
            float(metrics.get("balanced_accuracy", 0.0)),
            float(success_precision),
            float(success_recall),
        )

    return sorted(results, key=key, reverse=True)[0]


def apply_platt_scaling(cal_pred: pd.DataFrame, test_pred: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    x_cal = safe_logit(cal_pred["probability"].to_numpy()).reshape(-1, 1)
    y_cal = cal_pred["label"].astype(int).to_numpy()
    x_test = safe_logit(test_pred["probability"].to_numpy()).reshape(-1, 1)
    scaler = LogisticRegression(solver="lbfgs", class_weight="balanced", random_state=42)
    scaler.fit(x_cal, y_cal)
    calibrated_prob = scaler.predict_proba(x_test)[:, 1]
    before = classification_metrics(
        test_pred["label"].to_numpy(),
        test_pred["prediction"].to_numpy(),
        test_pred["probability"].to_numpy(),
    )
    after = classification_metrics(
        test_pred["label"].to_numpy(),
        predict_at_threshold(calibrated_prob, 0.5),
        calibrated_prob,
    )
    return calibrated_prob, {"brier_before": before["brier_score"], "brier_after": after["brier_score"]}


def save_pr_operating_point(result: ModelResult) -> None:
    y_true = result.test_predictions["label"].astype(int).to_numpy()
    y_prob = result.test_predictions["probability"].to_numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob) if np.unique(y_true).size == 2 else np.nan
    operating = classification_metrics(y_true, result.test_predictions["prediction"].to_numpy(), y_prob)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
    ax.scatter([operating["recall"]], [operating["precision"]], color="red", label=f"threshold={result.threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall: {result.name}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_precision_recall_operating_point.png", dpi=200)
    plt.close(fig)


def save_score_distribution(result: ModelResult) -> None:
    pred = result.test_predictions.copy()
    pred["error_group"] = np.select(
        [
            pred["label"].eq(1) & pred["prediction"].eq(1),
            pred["label"].eq(1) & pred["prediction"].eq(0),
            pred["label"].eq(0) & pred["prediction"].eq(0),
            pred["label"].eq(0) & pred["prediction"].eq(1),
        ],
        ["TP", "FN", "TN", "FP"],
        default="other",
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group, sub in pred.groupby("error_group"):
        ax.hist(sub["probability"], bins=12, alpha=0.55, label=f"{group} (n={len(sub)})")
    ax.axvline(result.threshold, color="black", linestyle="--", linewidth=1, label=f"threshold={result.threshold:.3f}")
    ax.set_xlabel("Positive-class score")
    ax.set_ylabel("Clip count")
    ax.set_title(f"Score Distribution: {result.name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_score_distribution_tp_fn_tn_fp.png", dpi=200)
    plt.close(fig)


def save_calibration_plot(result: ModelResult) -> pd.DataFrame:
    calibrated_prob, brier = apply_platt_scaling(result.calibration_predictions, result.test_predictions)
    y_true = result.test_predictions["label"].astype(int).to_numpy()
    raw_prob = result.test_predictions["probability"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if np.unique(y_true).size == 2:
        for label, prob in [("before", raw_prob), ("after_platt", calibrated_prob)]:
            observed, predicted = calibration_curve(y_true, prob, n_bins=5, strategy="quantile")
            ax.plot(predicted, observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="ideal")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"Calibration: {result.name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_calibration_before_after.png", dpi=200)
    plt.close(fig)
    table = pd.DataFrame([{"model": result.name, **brier}])
    table.to_csv(TABLE_DIR / "final_calibration_before_after.csv", index=False)
    return table


def speaker_level_metrics(result: ModelResult) -> pd.DataFrame:
    pred = result.test_predictions.copy()
    rows = []
    for speaker, sub in pred.groupby("speaker_id"):
        prob = float(sub["probability"].mean())
        label = int(sub["label"].mode().iloc[0])
        rows.append(
            {
                "speaker_id": speaker,
                "label": label,
                "probability": prob,
                "prediction": int(prob >= result.threshold),
                "n_clips": int(len(sub)),
            }
        )
    speaker_pred = pd.DataFrame(rows)
    metrics = classification_metrics(
        speaker_pred["label"].to_numpy(),
        speaker_pred["prediction"].to_numpy(),
        speaker_pred["probability"].to_numpy(),
    )
    out = pd.DataFrame([{"model": result.name, "n_speakers": len(speaker_pred), **metrics}])
    speaker_pred.to_csv(TABLE_DIR / "final_speaker_level_predictions.csv", index=False)
    out.to_csv(TABLE_DIR / "final_speaker_level_metrics.csv", index=False)
    return out


def noise_bucket(feature_df: pd.DataFrame) -> pd.Series:
    if "spectral_flatness_mean" in feature_df.columns:
        score = feature_df["spectral_flatness_mean"].astype(float)
    elif "zcr_mean" in feature_df.columns:
        score = feature_df["zcr_mean"].astype(float)
    else:
        return pd.Series("unknown", index=feature_df.index)
    q1, q2 = score.quantile([0.50, 0.75])
    return pd.Series(
        np.select(
            [score <= q1, score <= q2, score > q2],
            ["clean_proxy", "moderate_proxy", "noisy_proxy"],
            default="unknown",
        ),
        index=feature_df.index,
    )


def run_error_analysis(result: ModelResult, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = feature_df.copy()
    features["metadata_index"] = as_id(features["metadata_index"])
    features["noise_bucket"] = noise_bucket(features)
    handcrafted_cols = available_handcrafted_columns(features, max_columns=12)
    pred = result.test_predictions.copy()
    pred["metadata_index"] = as_id(pred["metadata_index"])
    detail_cols = [
        "metadata_index",
        "duration_sec",
        "speech_duration_sec",
        "pause_ratio",
        "f0_semitone_std",
        "speaker_id",
        "noise_bucket",
        *[c for c in handcrafted_cols if c not in {"duration_sec", "speech_duration_sec", "pause_ratio", "f0_semitone_std"}],
    ]
    detail_cols = [c for c in detail_cols if c in features.columns]
    detail = pred.merge(features[detail_cols], on=["metadata_index", "speaker_id"], how="left", suffixes=("", "_feature"))
    detail["duration_bucket"] = pd.cut(
        detail["duration_sec"].astype(float),
        bins=[-np.inf, 2.0, 4.0, np.inf],
        labels=["short_lt_2s", "medium_2_4s", "long_gt_4s"],
    )
    prosody = detail["f0_semitone_std"].astype(float)
    stable_cutoff = prosody.quantile(0.25)
    detail["stable_prosody_proxy"] = prosody <= stable_cutoff
    detail["is_fn"] = detail["label"].eq(1) & detail["prediction"].eq(0)
    detail["is_fp"] = detail["label"].eq(0) & detail["prediction"].eq(1)
    detail.to_csv(TABLE_DIR / "final_error_case_details.csv", index=False)

    fn = detail[detail["is_fn"]].copy()
    fp = detail[detail["is_fp"]].copy()
    fn.to_csv(TABLE_DIR / "final_false_negative_cases.csv", index=False)
    fp.to_csv(TABLE_DIR / "final_false_positive_cases.csv", index=False)

    def patterns(sub: pd.DataFrame, label: str) -> pd.DataFrame:
        total = len(sub)
        if total == 0:
            return pd.DataFrame(
                [
                    {"error_type": label, "pattern": "no_cases", "count": 0, "pct_of_errors": 0.0},
                ]
            )
        speaker_counts = sub["speaker_id"].value_counts()
        dominant_speaker_cases = int(speaker_counts.iloc[0]) if not speaker_counts.empty else 0
        rows = [
            {"error_type": label, "pattern": "Short clips (<2s)", "count": int(sub["duration_bucket"].eq("short_lt_2s").sum())},
            {"error_type": label, "pattern": "High noise proxy", "count": int(sub["noise_bucket"].eq("noisy_proxy").sum())},
            {"error_type": label, "pattern": "Dominant speaker(s)", "count": dominant_speaker_cases},
            {"error_type": label, "pattern": "Stable prosody proxy", "count": int(sub["stable_prosody_proxy"].sum())},
        ]
        out = pd.DataFrame(rows)
        out["pct_of_errors"] = (100.0 * out["count"] / total).round(1)
        return out

    fn_patterns = patterns(fn, "false_negative")
    fp_patterns = patterns(fp, "false_positive")
    pattern_table = pd.concat([fn_patterns, fp_patterns], ignore_index=True)
    pattern_table.to_csv(TABLE_DIR / "final_error_pattern_summary.csv", index=False)
    return pattern_table, detail


def save_embedding_tsne(cache: dict[str, Any] | None, feature_df: pd.DataFrame) -> None:
    if cache is None:
        return
    clip_emb_df, clip_to_embedding = clip_embedding_frame(cache)
    validation = clip_emb_df[clip_emb_df["phase6_split"].eq("validation")].copy()
    if len(validation) < 5:
        return
    features = feature_df[["metadata_index", "duration_sec"]].copy()
    features["metadata_index"] = as_id(features["metadata_index"])
    validation = validation.merge(features, on="metadata_index", how="left")
    x = matrix_from_clip_embeddings(validation, clip_to_embedding)
    perplexity = min(30, max(2, len(validation) // 3))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(x)
    validation["tsne_x"] = coords[:, 0]
    validation["tsne_y"] = coords[:, 1]
    validation["predicted_label"] = predict_at_threshold(validation["probability"].to_numpy(), 0.5)
    validation.to_csv(TABLE_DIR / "validation_embedding_tsne_coordinates.csv", index=False)
    color_specs = {
        "true_label": validation["label"],
        "predicted_label": validation["predicted_label"],
        "speaker_id": pd.factorize(validation["speaker_id"])[0],
        "duration": validation["duration_sec"].astype(float),
    }
    for name, colors in color_specs.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(validation["tsne_x"], validation["tsne_y"], c=colors, cmap="viridis", s=38, alpha=0.85)
        ax.set_title(f"Validation wav2vec t-SNE by {name}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"validation_embedding_tsne_by_{name}.png", dpi=200)
        plt.close(fig)


def result_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "model": result.name,
            "threshold": result.threshold,
            "threshold_note": result.threshold_note,
            **{metric: result.metrics.get(metric, np.nan) for metric in METRIC_ORDER},
        }
        row["delta_f1_vs_phase4"] = row["f1"] - PHASE4_OFFICIAL["f1"]
        row["delta_pr_auc_vs_phase4"] = row["pr_auc"] - PHASE4_OFFICIAL["pr_auc"]
        row["meets_precision_target"] = row["precision"] >= 0.50
        row["meets_recall_target"] = row["recall"] >= 0.70
        row["meets_f1_target"] = row["f1"] >= 0.55
        row["meets_pr_auc_target"] = row["pr_auc"] >= 0.60
        rows.append(row)
    table = pd.DataFrame(rows)
    table["meets_precision_and_recall_target"] = table["meets_precision_target"] & table["meets_recall_target"]
    return table.sort_values(
        ["meets_precision_and_recall_target", "f1", "pr_auc", "balanced_accuracy"],
        ascending=[False, False, False, False],
    )


def write_summary(
    cv_summary: pd.DataFrame,
    fusion_cv_summary: pd.DataFrame,
    model_table: pd.DataFrame,
    best: ModelResult,
    ci_table: pd.DataFrame,
    segmentation_table: pd.DataFrame,
    error_patterns: pd.DataFrame,
    calibration_table: pd.DataFrame,
    embedding_cache_available: bool,
) -> None:
    lines = [
        "# Phase 6 Deep Learning Fixes And Diagnostics",
        "",
        "## Priority 1: Speaker-Grouped CV",
        "",
        markdown_table(cv_summary.round(4)),
        "",
    ]
    if not fusion_cv_summary.empty:
        lines.extend(
            [
                "### Fixed-Embedding Fusion Grouped CV",
                "",
                markdown_table(fusion_cv_summary.round(4)),
                "",
            ]
        )
    elif not embedding_cache_available:
        lines.extend(
            [
                "### Fixed-Embedding Fusion Status",
                "",
                "Fixed-embedding fusion, attention pooling, and t-SNE were implemented in the runner but were not executed in this completed pass because no embedding cache was available. Run with `--extract-embeddings` to populate `tables/phase6_wav2vec_segment_embeddings.npz`; in this local session, full wav2vec embedding extraction was stopped after repeated slow checkpoint inference.",
                "",
            ]
        )
    lines.extend(
        [
            "## Priority 1: Fusion And Ensemble Holdout Results",
            "",
            markdown_table(model_table.round(4)),
            "",
            "## Final Selected Diagnostic Model",
            "",
            f"- Model: `{best.name}`",
            f"- Threshold: `{best.threshold:.6f}` ({best.threshold_note})",
            f"- Recall: `{best.metrics.get('recall', np.nan):.3f}`",
            f"- Precision: `{best.metrics.get('precision', np.nan):.3f}`",
            f"- F1: `{best.metrics.get('f1', np.nan):.3f}`",
            f"- PR-AUC: `{best.metrics.get('pr_auc', np.nan):.3f}`",
            "",
            "### Bootstrap CIs",
            "",
            markdown_table(ci_table.round(4)),
            "",
            "## Segmentation And Aggregation",
            "",
            markdown_table(segmentation_table.round(4), max_rows=12),
            "",
            "## Calibration",
            "",
            markdown_table(calibration_table.round(4)),
            "",
            "## Error Pattern Summary",
            "",
            markdown_table(error_patterns),
            "",
            "## Generated Figures",
            "",
            "- `figures/final_precision_recall_operating_point.png`",
            "- `figures/final_score_distribution_tp_fn_tn_fp.png`",
            "- `figures/final_calibration_before_after.png`",
        ]
    )
    if embedding_cache_available:
        lines.extend(
            [
                "- `figures/validation_embedding_tsne_by_true_label.png`",
                "- `figures/validation_embedding_tsne_by_predicted_label.png`",
                "- `figures/validation_embedding_tsne_by_speaker_id.png`",
                "- `figures/validation_embedding_tsne_by_duration.png`",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use this diagnostic winner only as a Phase 6 experiment until it is rerun with true multi-seed grouped folds and a freshly fine-tuned gentle wav2vec checkpoint. The frozen Phase 4 baseline remains the official model unless the post-fix model beats it without a specificity collapse.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 deep-learning fixes and diagnostics.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--extract-embeddings", action="store_true", help="Extract/cache wav2vec segment embeddings from the saved full checkpoint.")
    parser.add_argument("--force-embeddings", action="store_true", help="Recompute the embedding cache even if it exists.")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--seeds", type=str, default="42,7,13,21,100")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    cfg = load_config(Path(args.config))
    cfg = Wav2VecPhase6Config(**{**cfg.__dict__, "bootstrap_iterations": int(args.bootstrap_iterations)})
    ensure_manifest(cfg)
    feature_df = load_feature_table()
    run_data_quality_audit(feature_df)
    cv_summary = run_speaker_grouped_cv(feature_df)

    cache = load_embedding_cache()
    if args.extract_embeddings or args.force_embeddings:
        cache = extract_wav2vec_embeddings(cfg, batch_size=args.embedding_batch_size, force=args.force_embeddings)

    fusion_cv_summary = run_fusion_grouped_cv(feature_df, cache)
    baseline_cal, baseline_test, baseline_result = fit_same_split_baseline(feature_df, cfg)
    wav_result = load_existing_wav2vec_predictions(cfg)
    results = [baseline_result]
    if wav_result is not None:
        results.append(wav_result)
    results.extend(run_ensemble_variants(baseline_result, wav_result, cfg))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    results.extend(run_fixed_embedding_fusion(feature_df, cache, seeds))
    if not args.skip_attention:
        attention_result = train_attention_pooling(feature_df, cache, seed=cfg.seed)
        if attention_result is not None:
            results.append(attention_result)

    model_table = result_table(results)
    model_table.to_csv(TABLE_DIR / "ablation_table.csv", index=False)
    best_name = model_table.iloc[0]["model"]
    best = next(result for result in results if result.name == best_name)
    best.test_predictions.to_csv(TABLE_DIR / "final_best_model_test_predictions.csv", index=False)
    best.calibration_predictions.to_csv(TABLE_DIR / "final_best_model_calibration_predictions.csv", index=False)

    ci_table = bootstrap_cis(
        best.test_predictions["label"].to_numpy(),
        best.test_predictions["probability"].to_numpy(),
        best.threshold,
        best.test_predictions["speaker_id"].to_numpy(),
        seed=cfg.seed,
        n_bootstrap=args.bootstrap_iterations,
    )
    ci_table.to_csv(TABLE_DIR / "final_best_model_bootstrap_ci.csv", index=False)
    save_pr_operating_point(best)
    save_score_distribution(best)
    calibration_table = save_calibration_plot(best)
    speaker_level_metrics(best)
    error_patterns, _ = run_error_analysis(best, feature_df)
    segmentation_table = run_segmentation_aggregation_comparison(wav_result, cfg)
    save_embedding_tsne(cache, feature_df)
    write_summary(
        cv_summary,
        fusion_cv_summary,
        model_table,
        best,
        ci_table,
        segmentation_table,
        error_patterns,
        calibration_table,
        embedding_cache_available=cache is not None,
    )
    print(f"Phase 6 diagnostics complete: {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
