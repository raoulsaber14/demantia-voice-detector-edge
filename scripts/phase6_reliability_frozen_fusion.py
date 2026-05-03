"""Reliability-first frozen wav2vec + handcrafted fusion experiment.

This is intentionally narrow:
- frozen wav2vec2-base embeddings from the existing cache;
- selected handcrafted acoustic features;
- single-head clip attention pooling;
- one linear classifier;
- BCE loss, AdamW(lr=1e-4, wd=0.01);
- 5-fold speaker-grouped CV, seeds 3/42/123;
- training-only speaker calibration split;
- no tuning and no second modeling cycle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase6_focused_experiments import (  # noqa: E402
    PHASE4_BASELINE,
    build_clip_records,
    load_embedding_cache,
    load_features,
    metric_dict,
)
from src.train_baseline import select_feature_columns  # noqa: E402

OUT_DIR = ROOT / "final_results/reliability_frozen_fusion"
TABLE_DIR = OUT_DIR / "tables"
CM_DIR = OUT_DIR / "confusion_matrices"

SEEDS = [3, 42, 123]
N_SPLITS = 5
CALIBRATION_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 0.01
PRECISION_FLOOR = 0.30
TARGET_RECALL = 0.65
TARGET_SPECIFICITY_FALLBACK = 0.50

ACOUSTIC_FEATURE_PATTERNS = (
    "mfcc_",
    "delta_mfcc_",
    "delta2_mfcc_",
    "f0_",
    "voiced_ratio",
    "jitter_",
    "pitch_",
    "shimmer_",
    "rms_",
    "energy_",
    "log_energy_",
    "zcr_",
    "spectral_centroid_",
)

METRIC_COLUMNS = [
    "balanced_accuracy_0p5",
    "balanced_accuracy_calibrated",
    "recall_calibrated",
    "specificity_calibrated",
    "precision_calibrated",
    "f1_calibrated",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "recall_minus_0p05",
    "specificity_minus_0p05",
    "recall_plus_0p05",
    "specificity_plus_0p05",
]


@dataclass
class SplitBundle:
    train_idx: np.ndarray
    val_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray


class ScorePassthroughEstimator(ClassifierMixin, BaseEstimator):
    """Estimator wrapper whose predict_proba returns the supplied score column."""

    _estimator_type = "classifier"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ScorePassthroughEstimator":
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        score = np.clip(np.asarray(x, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - score, score])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class ReliabilityFusionLinear(torch.nn.Module):
    def __init__(self, embedding_dim: int, handcrafted_dim: int):
        super().__init__()
        self.query = torch.nn.Parameter(torch.empty(embedding_dim))
        torch.nn.init.normal_(self.query, std=0.02)
        self.classifier = torch.nn.Linear(embedding_dim + handcrafted_dim, 1)

    def attention_pool(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = (seq @ self.query) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(scores, dim=0)
        pooled = (seq * weights.unsqueeze(1)).sum(dim=0)
        return pooled, weights

    def forward(self, seq: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.attention_pool(seq)
        fused = torch.cat([pooled, handcrafted], dim=0)
        return self.classifier(fused).squeeze()


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_reliability_feature_columns(df: pd.DataFrame) -> list[str]:
    official_numeric = select_feature_columns(df)
    selected = [
        col
        for col in official_numeric
        if any(col.startswith(pattern) or col == pattern for pattern in ACOUSTIC_FEATURE_PATTERNS)
    ]
    return selected


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metadata_index": r["metadata_index"],
                "speaker_id": r["speaker_id"],
                "label": int(r["label"]),
                "duration_sec": float(r.get("duration_sec", np.nan)),
            }
            for r in records
        ]
    )


def split_train_val_cal(frame: pd.DataFrame, outer_train_idx: np.ndarray, seed: int, fold: int) -> SplitBundle:
    train_frame = frame.iloc[outer_train_idx].copy()
    speaker_labels = train_frame.groupby("speaker_id")["label"].agg(lambda s: int(s.mode().iloc[0])).reset_index()
    labels = speaker_labels["label"].to_numpy()
    speakers = speaker_labels["speaker_id"].astype(str).to_numpy()
    rng_seed = seed * 1000 + fold

    if len(np.unique(labels)) == 2 and pd.Series(labels).value_counts().min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=CALIBRATION_FRACTION, random_state=rng_seed)
        remain_spk_idx, cal_spk_idx = next(splitter.split(np.arange(len(speakers)), labels))
    else:
        rng = np.random.default_rng(rng_seed)
        cal_size = max(1, int(round(CALIBRATION_FRACTION * len(speakers))))
        cal_spk_idx = rng.choice(np.arange(len(speakers)), size=cal_size, replace=False)
        remain_spk_idx = np.setdiff1d(np.arange(len(speakers)), cal_spk_idx)

    remain_speakers = speakers[remain_spk_idx]
    remain_labels = labels[remain_spk_idx]
    cal_speakers = set(speakers[cal_spk_idx])

    if len(np.unique(remain_labels)) == 2 and pd.Series(remain_labels).value_counts().min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=rng_seed + 17)
        train_spk_idx, val_spk_idx = next(splitter.split(np.arange(len(remain_speakers)), remain_labels))
    else:
        rng = np.random.default_rng(rng_seed + 17)
        val_size = max(1, int(round(VALIDATION_FRACTION * len(remain_speakers))))
        val_spk_idx = rng.choice(np.arange(len(remain_speakers)), size=val_size, replace=False)
        train_spk_idx = np.setdiff1d(np.arange(len(remain_speakers)), val_spk_idx)

    train_speakers = set(remain_speakers[train_spk_idx])
    val_speakers = set(remain_speakers[val_spk_idx])
    outer = frame.iloc[outer_train_idx]
    train_idx = outer_train_idx[outer["speaker_id"].astype(str).isin(train_speakers).to_numpy()]
    val_idx = outer_train_idx[outer["speaker_id"].astype(str).isin(val_speakers).to_numpy()]
    cal_idx = outer_train_idx[outer["speaker_id"].astype(str).isin(cal_speakers).to_numpy()]
    return SplitBundle(train_idx=train_idx, val_idx=val_idx, cal_idx=cal_idx, test_idx=np.array([], dtype=int))


def standardize_embeddings(records: list[dict[str, Any]], train_idx: np.ndarray, *other_indices: np.ndarray) -> tuple[list[list[dict[str, Any]]], dict[str, np.ndarray]]:
    train_segments = np.vstack([records[i]["embeddings"] for i in train_idx])
    mean = train_segments.mean(axis=0)
    std = train_segments.std(axis=0) + 1e-6

    def transform(indices: np.ndarray) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx in indices:
            rec = dict(records[int(idx)])
            rec["embeddings"] = ((rec["embeddings"] - mean) / std).astype(np.float32)
            out.append(rec)
        return out

    return [transform(train_idx), *[transform(idx) for idx in other_indices]], {"embedding_mean": mean, "embedding_std": std}


def preprocess_handcrafted(feature_frame: pd.DataFrame, train_idx: np.ndarray, *other_indices: np.ndarray, columns: list[str]) -> tuple[list[np.ndarray], dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(feature_frame.iloc[train_idx][columns].to_numpy(dtype=float)))
    outputs = [train_x.astype(np.float32)]
    for idx in other_indices:
        outputs.append(scaler.transform(imputer.transform(feature_frame.iloc[idx][columns].to_numpy(dtype=float))).astype(np.float32))
    return outputs, {"imputer": imputer, "scaler": scaler}


def pos_weight(labels: np.ndarray) -> torch.Tensor:
    labels = np.asarray(labels).astype(int)
    pos = max(int(labels.sum()), 1)
    neg = max(int(len(labels) - labels.sum()), 1)
    return torch.tensor(float(neg / pos), dtype=torch.float32)


def iterate_batches(n_items: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n_items)
    return [order[start : start + batch_size] for start in range(0, n_items, batch_size)]


def predict_model(model: ReliabilityFusionLinear, records: list[dict[str, Any]], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[float] = []
    logits: list[float] = []
    with torch.no_grad():
        for rec, feature in zip(records, features):
            seq = torch.tensor(rec["embeddings"], dtype=torch.float32)
            feat = torch.tensor(feature, dtype=torch.float32)
            logit = model(seq, feat)
            logits.append(float(logit.detach().cpu()))
            probs.append(float(torch.sigmoid(logit).detach().cpu()))
    return np.asarray(probs, dtype=float), np.asarray(logits, dtype=float)


def evaluate_at_threshold(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    return metric_dict(y_true, prob, threshold)


def train_reliability_model(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    x_train: np.ndarray,
    x_val: np.ndarray,
    seed: int,
) -> tuple[ReliabilityFusionLinear, pd.DataFrame, dict[str, float]]:
    set_seed(seed)
    model = ReliabilityFusionLinear(train_records[0]["embeddings"].shape[1], x_train.shape[1])
    y_train = np.asarray([r["label"] for r in train_records], dtype=int)
    y_val = np.asarray([r["label"] for r in val_records], dtype=int)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight(y_train))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    best_state = None
    best_val_loss = np.inf
    patience_left = PATIENCE
    rows: list[dict[str, Any]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for batch_idx in iterate_batches(len(train_records), BATCH_SIZE, rng):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for idx in batch_idx:
                rec = train_records[int(idx)]
                seq = torch.tensor(rec["embeddings"], dtype=torch.float32)
                feat = torch.tensor(x_train[int(idx)], dtype=torch.float32)
                label = torch.tensor(float(rec["label"]), dtype=torch.float32)
                losses.append(loss_fn(model(seq, feat).view(1), label.view(1)))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_probs, val_logits = predict_model(model, val_records, x_val)
        with torch.no_grad():
            val_losses = []
            for logit, rec in zip(val_logits, val_records):
                val_losses.append(float(loss_fn(torch.tensor([logit], dtype=torch.float32), torch.tensor([float(rec["label"])], dtype=torch.float32))))
        val_loss = float(np.mean(val_losses)) if val_losses else np.inf
        val_metrics = evaluate_at_threshold(y_val, val_probs, 0.5)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
                "val_loss": val_loss,
                "val_balanced_accuracy_0p5": val_metrics["balanced_accuracy"],
                "val_recall_0p5": val_metrics["recall"],
                "val_specificity_0p5": val_metrics["specificity"],
            }
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_probs, _ = predict_model(model, val_records, x_val)
    final_val_metrics = evaluate_at_threshold(y_val, final_val_probs, 0.5)
    return model, pd.DataFrame(rows), final_val_metrics


def fit_calibrator(cal_prob: np.ndarray, y_cal: np.ndarray) -> CalibratedClassifierCV:
    base = ScorePassthroughEstimator().fit(cal_prob.reshape(-1, 1), y_cal)
    calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(base), method="sigmoid")
    calibrator.fit(cal_prob.reshape(-1, 1), y_cal)
    return calibrator


def select_operating_threshold(y_cal: np.ndarray, cal_prob: np.ndarray) -> tuple[float, str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 4):
        rows.append({"threshold": float(threshold), **evaluate_at_threshold(y_cal, cal_prob, float(threshold))})
    table = pd.DataFrame(rows)
    candidates = table[table["recall"] >= TARGET_RECALL].copy()
    if not candidates.empty:
        selected = candidates.sort_values(["specificity", "precision", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
        return float(selected["threshold"]), "recall_ge_0p65_highest_specificity", table
    candidates = table[table["specificity"] >= TARGET_SPECIFICITY_FALLBACK].copy()
    if not candidates.empty:
        selected = candidates.sort_values(["recall", "precision", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
        return float(selected["threshold"]), "recall_unattainable_highest_recall_specificity_ge_0p50", table
    selected = table.sort_values(["balanced_accuracy", "recall", "specificity"], ascending=[False, False, False]).iloc[0]
    return float(selected["threshold"]), "fallback_max_balanced_accuracy", table


def confusion_to_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def run_phase4_same_protocol(
    feature_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    columns: list[str],
    split_records: dict[str, Any],
) -> dict[str, float]:
    train_idx = split_records["train_idx"]
    cal_idx = split_records["cal_idx"]
    test_idx = split_records["test_idx"]
    x_train = feature_frame.iloc[train_idx][columns]
    y_train = feature_frame.iloc[train_idx]["label"].astype(int)
    x_cal = feature_frame.iloc[cal_idx][columns]
    y_cal = feature_frame.iloc[cal_idx]["label"].astype(int).to_numpy()
    x_test = feature_frame.iloc[test_idx][columns]
    y_test = feature_frame.iloc[test_idx]["label"].astype(int).to_numpy()

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    cal_raw = model.predict_proba(x_cal)[:, 1]
    calibrator = fit_calibrator(cal_raw, y_cal)
    cal_prob = calibrator.predict_proba(cal_raw.reshape(-1, 1))[:, 1]
    test_raw = model.predict_proba(x_test)[:, 1]
    test_prob = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    threshold, _, _ = select_operating_threshold(y_cal, cal_prob)
    return evaluate_at_threshold(y_test, test_prob, threshold)


def speaker_recall_std(pred_df: pd.DataFrame) -> float:
    positives = pred_df[pred_df["label"].astype(int) == 1].copy()
    if positives.empty:
        return np.nan
    speaker_hits = positives.groupby("speaker_id")["prediction_calibrated"].max().astype(float)
    return float(speaker_hits.std(ddof=0)) if len(speaker_hits) else np.nan


def run_experiment() -> None:
    ensure_dirs()
    feature_df = load_features()
    cache = load_embedding_cache()
    records = build_clip_records(feature_df, cache)
    frame = records_frame(records)
    feature_cols = select_reliability_feature_columns(feature_df)
    feature_frame = frame[["metadata_index", "speaker_id", "label"]].merge(
        feature_df[["metadata_index", *feature_cols]],
        on="metadata_index",
        how="left",
    )
    full_feature_frame = frame[["metadata_index"]].merge(feature_df, on="metadata_index", how="left")
    full_feature_frame["label"] = frame["label"].astype(int).to_numpy()
    phase4_feature_cols = select_feature_columns(full_feature_frame)
    y = frame["label"].astype(int).to_numpy()
    groups = frame["speaker_id"].astype(str).to_numpy()

    fold_assignment_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    training_logs: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    phase4_rows: list[dict[str, Any]] = []
    seed_validation_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (outer_train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups=groups), start=1):
            split = split_train_val_cal(frame, outer_train_idx, seed, fold)
            split.test_idx = test_idx

            for split_name, indices in [
                ("train", split.train_idx),
                ("validation", split.val_idx),
                ("calibration", split.cal_idx),
                ("test", split.test_idx),
            ]:
                speakers = sorted(frame.iloc[indices]["speaker_id"].astype(str).unique())
                for speaker_id in speakers:
                    fold_assignment_rows.append({"seed": seed, "fold": fold, "split": split_name, "speaker_id": speaker_id})

            [train_records, val_records, cal_records, test_records], _ = standardize_embeddings(
                records,
                split.train_idx,
                split.val_idx,
                split.cal_idx,
                split.test_idx,
            )
            feature_arrays, _ = preprocess_handcrafted(
                feature_frame,
                split.train_idx,
                split.val_idx,
                split.cal_idx,
                split.test_idx,
                columns=feature_cols,
            )
            x_train, x_val, x_cal, x_test = feature_arrays
            model, train_log, val_metrics = train_reliability_model(train_records, val_records, x_train, x_val, seed=seed)
            train_log["seed"] = seed
            train_log["fold"] = fold
            training_logs.append(train_log)
            seed_validation_rows.append({"seed": seed, "fold": fold, **{f"val_{k}": v for k, v in val_metrics.items()}})

            cal_raw_prob, _ = predict_model(model, cal_records, x_cal)
            test_raw_prob, _ = predict_model(model, test_records, x_test)
            y_cal = np.asarray([r["label"] for r in cal_records], dtype=int)
            y_test = np.asarray([r["label"] for r in test_records], dtype=int)
            calibrator = fit_calibrator(cal_raw_prob, y_cal)
            cal_prob = calibrator.predict_proba(cal_raw_prob.reshape(-1, 1))[:, 1]
            test_prob = calibrator.predict_proba(test_raw_prob.reshape(-1, 1))[:, 1]
            threshold, threshold_note, threshold_table = select_operating_threshold(y_cal, cal_prob)
            threshold_table.to_csv(TABLE_DIR / f"threshold_sweep_seed{seed}_fold{fold}.csv", index=False)

            pred_0p5 = (test_raw_prob >= 0.5).astype(int)
            pred_cal = (test_prob >= threshold).astype(int)
            metrics_0p5 = evaluate_at_threshold(y_test, test_raw_prob, 0.5)
            metrics_cal = evaluate_at_threshold(y_test, test_prob, threshold)
            metrics_minus = evaluate_at_threshold(y_test, test_prob, max(0.0, threshold - 0.05))
            metrics_plus = evaluate_at_threshold(y_test, test_prob, min(1.0, threshold + 0.05))
            fragile = (
                abs(metrics_minus["recall"] - metrics_plus["recall"]) > 0.15
                or abs(metrics_minus["specificity"] - metrics_plus["specificity"]) > 0.15
            )

            cm_row = {"seed": seed, "fold": fold, **confusion_to_row(y_test, pred_cal)}
            pd.DataFrame([cm_row]).to_csv(CM_DIR / f"confusion_seed{seed}_fold{fold}.csv", index=False)

            test_frame = frame.iloc[split.test_idx][["metadata_index", "speaker_id", "label", "duration_sec"]].copy()
            test_frame["seed"] = seed
            test_frame["fold"] = fold
            test_frame["raw_probability"] = test_raw_prob
            test_frame["calibrated_probability"] = test_prob
            test_frame["threshold"] = threshold
            test_frame["prediction_0p5"] = pred_0p5
            test_frame["prediction_calibrated"] = pred_cal
            test_frame["snr_estimate"] = np.nan
            prediction_rows.append(test_frame)

            fold_metric_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "threshold": threshold,
                    "threshold_note": threshold_note,
                    "balanced_accuracy_0p5": metrics_0p5["balanced_accuracy"],
                    "recall_0p5": metrics_0p5["recall"],
                    "specificity_0p5": metrics_0p5["specificity"],
                    "precision_0p5": metrics_0p5["precision"],
                    "balanced_accuracy_calibrated": metrics_cal["balanced_accuracy"],
                    "recall_calibrated": metrics_cal["recall"],
                    "specificity_calibrated": metrics_cal["specificity"],
                    "precision_calibrated": metrics_cal["precision"],
                    "f1_calibrated": metrics_cal["f1"],
                    "roc_auc": metrics_cal["roc_auc"],
                    "pr_auc": metrics_cal["pr_auc"],
                    "brier_score": metrics_cal["brier_score"],
                    "recall_minus_0p05": metrics_minus["recall"],
                    "specificity_minus_0p05": metrics_minus["specificity"],
                    "recall_plus_0p05": metrics_plus["recall"],
                    "specificity_plus_0p05": metrics_plus["specificity"],
                    "threshold_fragile": fragile,
                }
            )
            threshold_rows.append({"seed": seed, "fold": fold, "threshold": threshold, "threshold_note": threshold_note})
            robustness_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "threshold": threshold,
                    "recall_minus_0p05": metrics_minus["recall"],
                    "specificity_minus_0p05": metrics_minus["specificity"],
                    "recall_plus_0p05": metrics_plus["recall"],
                    "specificity_plus_0p05": metrics_plus["specificity"],
                    "threshold_fragile": fragile,
                }
            )

            phase4_metrics = run_phase4_same_protocol(
                feature_df,
                full_feature_frame,
                phase4_feature_cols,
                {"train_idx": split.train_idx, "cal_idx": split.cal_idx, "test_idx": split.test_idx},
            )
            phase4_rows.append({"seed": seed, "fold": fold, **phase4_metrics})

    fold_assignments = pd.DataFrame(fold_assignment_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    training_log = pd.concat(training_logs, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    robustness = pd.DataFrame(robustness_rows)
    phase4_fold_metrics = pd.DataFrame(phase4_rows)
    validation_metrics = pd.DataFrame(seed_validation_rows)

    fold_assignments.to_csv(OUT_DIR / "fold_assignments.csv", index=False)
    predictions.to_csv(OUT_DIR / "seed_wise_test_predictions.csv", index=False)
    predictions.to_csv(OUT_DIR / "calibrated_probabilities_per_fold_seed.csv", index=False)
    thresholds.to_csv(OUT_DIR / "selected_thresholds.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    training_log.to_csv(OUT_DIR / "training_logs.csv", index=False)
    robustness.to_csv(OUT_DIR / "threshold_robustness.csv", index=False)
    validation_metrics.to_csv(OUT_DIR / "validation_model_selection_metrics.csv", index=False)
    phase4_fold_metrics.to_csv(OUT_DIR / "phase4_same_protocol_fold_metrics.csv", index=False)

    aggregate_and_decide(fold_metrics, predictions, phase4_fold_metrics, validation_metrics)


def aggregate_and_decide(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    phase4_fold_metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
) -> None:
    seed_rows: list[dict[str, Any]] = []
    seed_rejection_rows: list[dict[str, Any]] = []
    speaker_std_rows: list[dict[str, Any]] = []
    for seed, sub in fold_metrics.groupby("seed"):
        pred_sub = predictions[predictions["seed"].eq(seed)].copy()
        speaker_std = speaker_recall_std(pred_sub)
        speaker_std_rows.append({"seed": seed, "speaker_level_recall_std": speaker_std})
        row: dict[str, Any] = {"seed": seed}
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=0))
        row["per_fold_recall_min"] = float(sub["recall_calibrated"].min())
        row["per_fold_recall_max"] = float(sub["recall_calibrated"].max())
        row["per_seed_variance_balanced_accuracy"] = float(sub["balanced_accuracy_calibrated"].var(ddof=0))
        seed_rows.append(row)

        reject_recall_std = float(sub["recall_calibrated"].std(ddof=0)) > 0.15
        reject_min_recall = float(sub["recall_calibrated"].min()) < 0.30
        reject_specificity = float(sub["specificity_0p5"].mean()) < 0.40
        reject_precision = float(sub["precision_calibrated"].mean()) < PRECISION_FLOOR
        rejected = reject_recall_std or reject_min_recall or reject_specificity or reject_precision
        seed_rejection_rows.append(
            {
                "seed": seed,
                "rejected": rejected,
                "recall_std_gt_0p15": reject_recall_std,
                "min_fold_recall_lt_0p30": reject_min_recall,
                "mean_specificity_0p5_lt_0p40": reject_specificity,
                "precision_calibrated_lt_0p30": reject_precision,
                "recall_std": float(sub["recall_calibrated"].std(ddof=0)),
                "min_fold_recall": float(sub["recall_calibrated"].min()),
                "mean_specificity_0p5": float(sub["specificity_0p5"].mean()),
                "mean_precision_calibrated": float(sub["precision_calibrated"].mean()),
            }
        )

    seed_summary = pd.DataFrame(seed_rows)
    seed_rejections = pd.DataFrame(seed_rejection_rows)
    speaker_std_df = pd.DataFrame(speaker_std_rows)
    seed_summary = seed_summary.merge(speaker_std_df, on="seed", how="left")
    seed_summary.to_csv(OUT_DIR / "seed_summary_metrics.csv", index=False)
    seed_rejections.to_csv(OUT_DIR / "seed_rejection_summary.csv", index=False)

    metric_rows = []
    for col in METRIC_COLUMNS:
        seed_col = f"{col}_mean"
        metric_rows.append(
            {
                "metric": col,
                "value_mean": float(seed_summary[seed_col].mean()),
                "value_std": float(seed_summary[seed_col].std(ddof=0)),
            }
        )
    metric_rows.extend(
        [
            {
                "metric": "per_fold_recall_range_min",
                "value_mean": float(seed_summary["per_fold_recall_min"].mean()),
                "value_std": float(seed_summary["per_fold_recall_min"].std(ddof=0)),
            },
            {
                "metric": "per_fold_recall_range_max",
                "value_mean": float(seed_summary["per_fold_recall_max"].mean()),
                "value_std": float(seed_summary["per_fold_recall_max"].std(ddof=0)),
            },
            {
                "metric": "per_seed_variance_balanced_accuracy",
                "value_mean": float(seed_summary["per_seed_variance_balanced_accuracy"].mean()),
                "value_std": float(seed_summary["per_seed_variance_balanced_accuracy"].std(ddof=0)),
            },
            {
                "metric": "speaker_level_recall_std",
                "value_mean": float(seed_summary["speaker_level_recall_std"].mean()),
                "value_std": float(seed_summary["speaker_level_recall_std"].std(ddof=0)),
            },
            {
                "metric": "number_of_seeds_rejected",
                "value_mean": int(seed_rejections["rejected"].sum()),
                "value_std": 0.0,
            },
        ]
    )
    aggregate_metrics = pd.DataFrame(metric_rows)
    aggregate_metrics.to_csv(OUT_DIR / "final_metrics_aggregated.csv", index=False)

    phase4_summary = pd.DataFrame(
        [
            {
                "model": "phase4_logistic_regression_platt_same_protocol",
                "recall": float(phase4_fold_metrics["recall"].mean()),
                "specificity": float(phase4_fold_metrics["specificity"].mean()),
                "balanced_accuracy": float(phase4_fold_metrics["balanced_accuracy"].mean()),
                "precision": float(phase4_fold_metrics["precision"].mean()),
            },
            {
                "model": "phase4_official_anchor",
                "recall": PHASE4_BASELINE["recall"],
                "specificity": PHASE4_BASELINE["specificity"],
                "balanced_accuracy": PHASE4_BASELINE["balanced_accuracy"],
                "precision": PHASE4_BASELINE["precision"],
            },
            {
                "model": "reliability_frozen_fusion",
                "recall": float(seed_summary["recall_calibrated_mean"].mean()),
                "specificity": float(seed_summary["specificity_calibrated_mean"].mean()),
                "balanced_accuracy": float(seed_summary["balanced_accuracy_calibrated_mean"].mean()),
                "precision": float(seed_summary["precision_calibrated_mean"].mean()),
            },
        ]
    )
    phase4_summary.to_csv(OUT_DIR / "phase4_comparison.csv", index=False)

    chosen_seed_row = seed_summary.sort_values(
        ["balanced_accuracy_calibrated_mean", "per_seed_variance_balanced_accuracy"],
        ascending=[False, True],
    ).iloc[0]
    chosen_seed = int(chosen_seed_row["seed"])
    chosen_predictions = predictions[predictions["seed"].eq(chosen_seed)].copy()
    chosen_predictions.to_csv(OUT_DIR / "final_chosen_seed_predictions.csv", index=False)
    top_fn = chosen_predictions[(chosen_predictions["label"].eq(1)) & (chosen_predictions["prediction_calibrated"].eq(0))].sort_values(
        "calibrated_probability", ascending=True
    ).head(5)
    top_fp = chosen_predictions[(chosen_predictions["label"].eq(0)) & (chosen_predictions["prediction_calibrated"].eq(1))].sort_values(
        "calibrated_probability", ascending=False
    ).head(5)
    error_cols = ["speaker_id", "metadata_index", "label", "calibrated_probability", "duration_sec", "snr_estimate"]
    top_fn[error_cols].rename(columns={"metadata_index": "clip_id", "calibrated_probability": "predicted_probability"}).to_csv(
        OUT_DIR / "top5_false_negatives.csv", index=False
    )
    top_fp[error_cols].rename(columns={"metadata_index": "clip_id", "calibrated_probability": "predicted_probability"}).to_csv(
        OUT_DIR / "top5_false_positives.csv", index=False
    )

    recall_mean = float(seed_summary["recall_calibrated_mean"].mean())
    recall_std_mean = float(seed_summary["recall_calibrated_std"].mean())
    specificity_mean = float(seed_summary["specificity_calibrated_mean"].mean())
    precision_mean = float(seed_summary["precision_calibrated_mean"].mean())
    balanced_accuracy_mean = float(seed_summary["balanced_accuracy_calibrated_mean"].mean())
    threshold_fragile = bool(fold_metrics["threshold_fragile"].any())
    no_seed_rejected = int(seed_rejections["rejected"].sum()) == 0
    phase4_bal_acc = float(phase4_summary[phase4_summary["model"].eq("phase4_official_anchor")]["balanced_accuracy"].iloc[0])
    phase4_recall_std_proxy = 0.0
    usability = {
        "mean_recall_ge_0p65": recall_mean >= 0.65,
        "recall_std_le_0p12": recall_std_mean <= 0.12,
        "no_seed_rejected": no_seed_rejected,
        "specificity_ge_0p50": specificity_mean >= 0.50,
        "precision_ge_0p30": precision_mean >= PRECISION_FLOOR,
        "not_threshold_fragile": not threshold_fragile,
        "matches_or_exceeds_phase4_balanced_accuracy": balanced_accuracy_mean >= phase4_bal_acc,
    }
    usable = all(usability.values())
    usability_df = pd.DataFrame([{"usable": usable, **usability}])
    usability_df.to_csv(OUT_DIR / "usability_criteria.csv", index=False)

    failure_sentence = "False negatives tend to be lower-probability positive clips; no SNR estimate is available in the current metadata."
    summary = {
        "single_sentence_conclusion": "Usable" if usable else "Not usable",
        "chosen_seed": chosen_seed,
        "threshold_fragile": threshold_fragile,
        "number_of_seeds_rejected": int(seed_rejections["rejected"].sum()),
        "failure_pattern_summary": failure_sentence,
    }
    (OUT_DIR / "decision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reliability-first frozen fusion experiment once.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    run_experiment()
    print(f"Reliability-first frozen fusion complete: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
