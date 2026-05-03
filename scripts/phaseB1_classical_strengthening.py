"""Run Phase B1 controlled classical-baseline diagnostics.

This script intentionally writes only Phase B1 artifacts. It does not modify
Phase 4, Tier 1, Phase A, score-band, app, deployment, or feature-pipeline
outputs.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

PHASE4_SCRIPT = ROOT / "scripts/phase4_baseline_experiments.py"
FEATURE_CSV = ROOT / "data/features/features_phase3_governed_pruned.csv"
TRUSTED_MANIFEST = ROOT / "reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv"
GOVERNED_MANIFEST = ROOT / "reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv"
PHASE65_SEED_SUMMARY = ROOT / "reports/tables/phase6_5_cleaned_trusted_vs_governed_seed_summary.csv"

SEEDS = [42, 43, 44, 45, 46]
THRESHOLDS = [round(float(x), 2) for x in np.arange(0.10, 0.901, 0.05)]
OFFICIAL_LOCKED_THRESHOLD = 0.30
CURRENT_ANCHOR_C_GRID = [0.01, 0.1, 1.0, 10.0]
PHASEB1_C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
POSITIVE_LABEL = 1

READ_FIRST_FILES = [
    "reports/phaseA_classical_baseline_diagnostics.md",
    "reports/tables/phaseA_class_balance_audit.csv",
    "reports/tables/phaseA_speaker_distribution_audit.csv",
    "reports/tables/phaseA_fold_composition_audit.csv",
    "reports/tables/phaseA_duration_quality_audit.csv",
    "reports/tables/phaseA_feature_quality_audit.csv",
    "reports/tables/phaseA_feature_univariate_signal.csv",
    "reports/tables/phaseA_diagnostic_risk_ranking.csv",
    "reports/phase6_5_cleaned_trusted_vs_governed_rerun.md",
    "reports/tables/phase6_5_cleaned_trusted_vs_governed_seed_summary.csv",
    "reports/tables/phase6_5_cleaned_trusted_vs_governed_comparison.csv",
    "reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv",
    "reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv",
    "reports/phase4_official_winner.md",
    "reports/phase4_threshold_analysis.md",
    "scripts/phase4_baseline_experiments.py",
    "data/features/features_phase3_governed_pruned.csv",
]


@dataclass
class FoldSpec:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_speakers: set[str]


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def load_phase4_module() -> Any:
    spec = importlib.util.spec_from_file_location("phase4_baseline_experiments", PHASE4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {PHASE4_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bool_series(series: pd.Series, default: bool) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default).astype(bool)
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}

    def parse(value: Any) -> bool:
        if pd.isna(value):
            return default
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        return default

    return series.map(parse).astype(bool)


def normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["metadata_index"] = out["metadata_index"].astype(str)
    out["speaker_id"] = out["speaker_id"].astype(str)
    out["label"] = out["label"].astype(int)
    for col, default in [
        ("phase4_training_allowed", True),
        ("phase4_eval_allowed", True),
        ("manual_review_required", False),
        ("phase4_caution_flag", False),
    ]:
        out[col] = bool_series(out[col], default=default) if col in out.columns else default
    return out


def mode_int(series: pd.Series) -> int:
    return int(series.astype(int).mode().iloc[0])


def speaker_label_counts(df: pd.DataFrame) -> dict[int, int]:
    labels = df.groupby("speaker_id")["label"].agg(mode_int)
    return {int(k): int(v) for k, v in labels.value_counts().sort_index().to_dict().items()}


def build_outer_folds(df: pd.DataFrame, feature_cols: list[str], seed: int) -> list[FoldSpec]:
    eval_df = df[df["phase4_eval_allowed"]].copy()
    if eval_df.empty:
        raise ValueError("No Phase 4 evaluation-allowed rows are available.")
    class_speaker_counts = speaker_label_counts(eval_df)
    min_class_speakers = min(class_speaker_counts.values())
    n_splits = int(min(5, min_class_speakers))
    if n_splits < 2:
        raise ValueError(f"Grouped CV infeasible with speaker counts: {class_speaker_counts}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldSpec] = []
    for fold, (_, val_rel_idx) in enumerate(
        splitter.split(
            eval_df[feature_cols],
            eval_df["label"].astype(int).to_numpy(),
            groups=eval_df["speaker_id"].astype(str).to_numpy(),
        ),
        start=1,
    ):
        val_idx = eval_df.iloc[val_rel_idx].index.to_numpy()
        val_speakers = set(df.loc[val_idx, "speaker_id"].astype(str))
        train_mask = df["phase4_training_allowed"] & ~df["speaker_id"].astype(str).isin(val_speakers)
        train_idx = df.index[train_mask].to_numpy()
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        overlap = set(train_df["speaker_id"].astype(str)) & val_speakers
        if overlap:
            raise ValueError(f"Speaker leakage in fold {fold}: {sorted(overlap)[:5]}")
        if train_df["label"].nunique() < 2 or val_df["label"].nunique() < 2:
            raise ValueError(f"Fold {fold} lacks both classes in train or validation.")
        folds.append(FoldSpec(fold=fold, train_idx=train_idx, val_idx=val_idx, val_speakers=val_speakers))
    return folds


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


def make_logistic_pipeline(seed: int, c_value: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def speaker_inverse_weights(df: pd.DataFrame) -> np.ndarray:
    speaker_counts = df["speaker_id"].astype(str).map(df["speaker_id"].astype(str).value_counts())
    weights = 1.0 / speaker_counts.astype(float).to_numpy()
    return weights / np.mean(weights)


def fit_pipeline(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    c_value: float | None,
    speaker_weighted: bool,
) -> tuple[BaseEstimator, dict[str, Any]]:
    x_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int).to_numpy()
    groups = train_df["speaker_id"].astype(str).to_numpy()
    sample_weight = speaker_inverse_weights(train_df) if speaker_weighted else None
    fit_params = {"model__sample_weight": sample_weight} if speaker_weighted else {}

    details: dict[str, Any] = {
        "search_status": "not_run",
        "best_params": "",
        "best_score": np.nan,
        "inner_cv": "none",
        "speaker_weighted": speaker_weighted,
    }
    if c_value is not None:
        model = make_logistic_pipeline(seed=seed, c_value=c_value)
        model.fit(x_train, y_train, **fit_params)
        details.update({"search_status": "fixed_C", "best_params": f"model__C={c_value}"})
        return model, details

    cv = inner_group_cv(y_train, groups, seed=seed)
    model = make_logistic_pipeline(seed=seed, c_value=1.0)
    if cv is None:
        model.fit(x_train, y_train, **fit_params)
        details["search_status"] = "skipped_inner_group_cv_infeasible"
        return model, details

    search = GridSearchCV(
        estimator=model,
        param_grid={"model__C": CURRENT_ANCHOR_C_GRID},
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score=np.nan,
    )
    search.fit(x_train, y_train, groups=groups, **fit_params)
    details.update(
        {
            "search_status": "success",
            "best_params": str(search.best_params_),
            "best_score": float(search.best_score_) if np.isfinite(search.best_score_) else np.nan,
            "inner_cv": "StratifiedGroupKFold",
        }
    )
    return search.best_estimator_, details


def raw_calibration_scores(model: BaseEstimator, x: pd.DataFrame) -> tuple[np.ndarray, str]:
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=float)
        if decision.ndim == 2:
            decision = decision[:, -1]
        return decision.ravel(), "decision_function"
    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(x), dtype=float)
        if prob.ndim == 2 and prob.shape[1] > 1:
            return prob[:, 1].ravel(), "predict_proba_positive_class"
        return prob.ravel(), "predict_proba"
    pred = np.asarray(model.predict(x), dtype=float)
    return pred.ravel(), "hard_prediction_fallback"


def fit_platt_calibrator(
    base_model: BaseEstimator,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    speaker_weighted: bool,
) -> tuple[LogisticRegression | None, dict[str, Any]]:
    y_train = train_df["label"].astype(int).to_numpy()
    groups = train_df["speaker_id"].astype(str).to_numpy()
    cv = inner_group_cv(y_train, groups, seed=seed)
    details: dict[str, Any] = {
        "calibration_status": "not_run",
        "calibration_inner_cv": "none",
        "calibration_oof_rows": 0,
        "calibration_score_type": "",
    }
    if cv is None:
        details["calibration_status"] = "skipped_inner_group_cv_infeasible"
        return None, details

    oof_scores = np.full(len(train_df), np.nan, dtype=float)
    for inner_train_idx, inner_cal_idx in cv.split(train_df[feature_cols], y_train, groups=groups):
        inner_train = train_df.iloc[inner_train_idx]
        inner_cal = train_df.iloc[inner_cal_idx]
        if inner_train["label"].nunique() < 2 or inner_cal["label"].nunique() < 2:
            details["calibration_status"] = "skipped_inner_fold_class_support"
            return None, details
        inner_model = clone(base_model)
        fit_params = (
            {"model__sample_weight": speaker_inverse_weights(inner_train)}
            if speaker_weighted
            else {}
        )
        inner_model.fit(inner_train[feature_cols], inner_train["label"].astype(int).to_numpy(), **fit_params)
        fold_scores, score_type = raw_calibration_scores(inner_model, inner_cal[feature_cols])
        oof_scores[inner_cal_idx] = fold_scores
        details["calibration_score_type"] = score_type

    valid_mask = np.isfinite(oof_scores)
    if valid_mask.sum() < 10 or np.unique(y_train[valid_mask]).size < 2:
        details["calibration_status"] = "skipped_insufficient_oof_support"
        details["calibration_oof_rows"] = int(valid_mask.sum())
        return None, details

    platt = LogisticRegression(max_iter=2000, random_state=seed)
    platt.fit(oof_scores[valid_mask].reshape(-1, 1), y_train[valid_mask])
    details.update(
        {
            "calibration_status": "success",
            "calibration_inner_cv": "StratifiedGroupKFold",
            "calibration_oof_rows": int(valid_mask.sum()),
            "calibration_oof_positive": int(y_train[valid_mask].sum()),
            "calibration_oof_negative": int((y_train[valid_mask] == 0).sum()),
        }
    )
    return platt, details


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    negative_count = tn + fp
    positive_count = tp + fn
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / negative_count) if negative_count else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "false_negative_rate": float(fn / positive_count) if positive_count else np.nan,
        "confusion_matrix_tn_fp_fn_tp": f"{tn},{fp},{fn},{tp}",
    }
    if np.unique(y_true).size >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    try:
        out["brier_score"] = float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0)))
    except Exception:
        out["brier_score"] = np.nan
    return out


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any], str]:
    y_true = np.asarray(y_true).astype(int)
    prevalence = float(np.mean(y_true == POSITIVE_LABEL))
    minimum_precision = max(prevalence, 0.25)
    minimum_specificity = 0.20
    rows = []
    for threshold in THRESHOLDS:
        metrics = compute_metrics(y_true, scores, threshold)
        rows.append({"threshold": threshold, **metrics})
    table = pd.DataFrame(rows)
    candidates = table[
        (table["precision"] >= minimum_precision) & (table["specificity"] >= minimum_specificity)
    ].copy()
    if candidates.empty:
        candidates = table.copy()
        note = "fallback_no_threshold_met_screening_rule"
    else:
        note = "max_recall_with_precision_at_least_prevalence_and_specificity_at_least_0.20"
    selected = candidates.sort_values(
        ["recall", "specificity", "balanced_accuracy", "precision", "threshold"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return float(selected["threshold"]), selected.to_dict(), note


def fold_threshold_stats(predictions: pd.DataFrame) -> dict[str, Any]:
    thresholds = []
    for _, group in predictions.groupby("fold"):
        y = group["true_label"].astype(int).to_numpy()
        scores = group["score"].astype(float).to_numpy()
        threshold, _, _ = select_threshold(y, scores)
        thresholds.append(threshold)
    return {
        "threshold_fold_min": float(np.min(thresholds)) if thresholds else np.nan,
        "threshold_fold_max": float(np.max(thresholds)) if thresholds else np.nan,
        "threshold_fold_mean": float(np.mean(thresholds)) if thresholds else np.nan,
        "threshold_fold_std": float(np.std(thresholds, ddof=1)) if len(thresholds) > 1 else 0.0,
    }


def speaker_metrics(predictions: pd.DataFrame, threshold: float) -> dict[str, Any]:
    rows = []
    pred = predictions.copy()
    pred["clip_prediction"] = (pred["score"].astype(float) >= threshold).astype(int)
    for speaker_id, group in pred.groupby("speaker_id"):
        true_label = mode_int(group["true_label"])
        majority_score = float(group["clip_prediction"].mean())
        max_score = float(group["score"].astype(float).max())
        rows.append(
            {
                "speaker_id": speaker_id,
                "true_label": true_label,
                "majority_vote_prediction": int(majority_score >= 0.5),
                "max_risk_score_prediction": int(max_score >= threshold),
                "majority_vote_score": majority_score,
                "max_risk_score": max_score,
            }
        )
    speaker_df = pd.DataFrame(rows)
    out: dict[str, Any] = {"n_eval_speakers": int(len(speaker_df))}
    for rule, pred_col in [
        ("speaker_majority_vote", "majority_vote_prediction"),
        ("speaker_max_risk_score", "max_risk_score_prediction"),
    ]:
        metrics = compute_metrics(
            speaker_df["true_label"].astype(int).to_numpy(),
            speaker_df[pred_col].astype(float).to_numpy(),
            threshold=0.5,
        )
        for key in ["recall", "specificity", "precision", "f1", "balanced_accuracy", "false_negatives", "false_positives", "true_negatives", "true_positives"]:
            out[f"{rule}_{key}"] = metrics[key]
    return out


def one_clip_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Select one deterministic evaluation-safe clip per speaker.

    Rule:
    1. Prefer Phase 4 evaluation-allowed rows for that speaker.
    2. Within that set choose the longest duration_sec.
    3. Break ties by metadata_index, then audio_file.
    4. If a speaker has no evaluation-allowed row, use the same rule among
       Phase 4 training-allowed rows.
    """
    selected_rows = []
    for _, group in df.groupby("speaker_id", sort=True):
        eval_allowed = group[group["phase4_eval_allowed"]].copy()
        candidates = eval_allowed if not eval_allowed.empty else group[group["phase4_training_allowed"]].copy()
        candidates = candidates.sort_values(
            ["duration_sec", "metadata_index", "audio_file"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        row = candidates.iloc[0].copy()
        row["phaseB1_one_clip_selection_rule"] = (
            "prefer_phase4_eval_allowed_then_longest_duration_then_metadata_index_then_audio_file"
        )
        row["phaseB1_one_clip_eval_preferred"] = bool(not eval_allowed.empty)
        selected_rows.append(row)
    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    return out


def run_platt_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    condition: str,
    source_dataset: str,
    intervention: str,
    c_value: float | None,
    speaker_weighted: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folds = build_outer_folds(df, feature_cols, seed=seed)
    rows: list[dict[str, Any]] = []
    search_statuses: list[str] = []
    calibration_statuses: list[str] = []
    train_sizes: list[int] = []
    train_speakers: list[int] = []
    validation_sizes: list[int] = []
    validation_speakers: list[int] = []

    for fold_spec in folds:
        train_df = df.loc[fold_spec.train_idx].copy()
        val_df = df.loc[fold_spec.val_idx].copy()
        model, search_details = fit_pipeline(
            train_df=train_df,
            feature_cols=feature_cols,
            seed=seed + fold_spec.fold,
            c_value=c_value,
            speaker_weighted=speaker_weighted,
        )
        calibrator, calibration_details = fit_platt_calibrator(
            base_model=model,
            train_df=train_df,
            feature_cols=feature_cols,
            seed=seed + 1000 + fold_spec.fold,
            speaker_weighted=speaker_weighted,
        )
        fit_params = (
            {"model__sample_weight": speaker_inverse_weights(train_df)}
            if speaker_weighted
            else {}
        )
        # Ensure final model fit reflects the selected C and B1 weighting.
        model.fit(train_df[feature_cols], train_df["label"].astype(int).to_numpy(), **fit_params)
        raw_scores, raw_score_type = raw_calibration_scores(model, val_df[feature_cols])
        if calibrator is not None:
            scores = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
            score_type = f"platt_calibrated_{raw_score_type}"
            model_name = "logistic_regression_platt"
        else:
            scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -50, 50)))
            score_type = f"platt_unavailable_sigmoid_{raw_score_type}"
            model_name = "logistic_regression_platt_fallback"
        search_statuses.append(str(search_details["search_status"]))
        calibration_statuses.append(str(calibration_details["calibration_status"]))
        train_sizes.append(len(train_df))
        train_speakers.append(train_df["speaker_id"].nunique())
        validation_sizes.append(len(val_df))
        validation_speakers.append(val_df["speaker_id"].nunique())
        for pos, (_, row) in enumerate(val_df.iterrows()):
            rows.append(
                {
                    "dataset_version": condition,
                    "source_dataset": source_dataset,
                    "intervention_name": intervention,
                    "seed": seed,
                    "fold": fold_spec.fold,
                    "model_name": model_name,
                    "c_value": c_value if c_value is not None else np.nan,
                    "speaker_weighted": speaker_weighted,
                    "metadata_index": row["metadata_index"],
                    "speaker_id": row["speaker_id"],
                    "true_label": int(row["label"]),
                    "score": float(scores[pos]),
                    "score_type": score_type,
                    "duration_sec": row.get("duration_sec", np.nan),
                    "phase4_caution_flag": bool(row.get("phase4_caution_flag", False)),
                }
            )
    predictions = pd.DataFrame(rows)
    run_details = {
        "n_folds": len(folds),
        "search_status_counts": counts_string(search_statuses),
        "calibration_status_counts": counts_string(calibration_statuses),
        "n_train_mean": float(np.mean(train_sizes)),
        "n_validation_mean": float(np.mean(validation_sizes)),
        "n_train_speakers_mean": float(np.mean(train_speakers)),
        "n_validation_speakers_mean": float(np.mean(validation_speakers)),
    }
    return predictions, run_details


def counts_string(values: list[str]) -> str:
    if not values:
        return ""
    counts = pd.Series(values).value_counts().sort_index()
    return "; ".join([f"{k}:{int(v)}" for k, v in counts.items()])


def summarize_predictions(
    predictions: pd.DataFrame,
    threshold_mode: str,
    run_details: dict[str, Any] | None = None,
    locked_threshold: float | None = None,
) -> dict[str, Any]:
    y = predictions["true_label"].astype(int).to_numpy()
    scores = predictions["score"].astype(float).to_numpy()
    if locked_threshold is None:
        threshold, selected_metrics, selection_note = select_threshold(y, scores)
        fold_stats = fold_threshold_stats(predictions)
    else:
        threshold = float(locked_threshold)
        selected_metrics = compute_metrics(y, scores, threshold)
        selection_note = "locked_global_official_phase4_threshold_0.30"
        fold_stats = {
            "threshold_fold_min": threshold,
            "threshold_fold_max": threshold,
            "threshold_fold_mean": threshold,
            "threshold_fold_std": 0.0,
        }
    metrics = compute_metrics(y, scores, threshold)
    out = {
        "dataset_version": predictions["dataset_version"].iloc[0],
        "source_dataset": predictions["source_dataset"].iloc[0],
        "intervention_name": predictions["intervention_name"].iloc[0],
        "seed": int(predictions["seed"].iloc[0]),
        "model_name": predictions["model_name"].iloc[0],
        "c_value": predictions["c_value"].iloc[0],
        "threshold_mode": threshold_mode,
        "threshold_selected": threshold,
        "selection_note": selection_note,
        "n_eval_clips": int(len(predictions)),
        "n_eval_speakers": int(predictions["speaker_id"].nunique()),
        **metrics,
        **fold_stats,
        **speaker_metrics(predictions, threshold),
    }
    if run_details:
        out.update(run_details)
    return out


def baseline_reference_rows() -> pd.DataFrame:
    seed_df = pd.read_csv(PHASE65_SEED_SUMMARY)
    rows: list[dict[str, Any]] = []
    for _, row in seed_df.iterrows():
        source_dataset = "trusted" if str(row["dataset_version"]).replace(" ", "_") == "cleaned_trusted" else "governed"
        condition = f"{source_dataset}_baseline_reference"
        out = {
            "dataset_version": condition,
            "source_dataset": source_dataset,
            "intervention_name": "baseline_reference",
            "seed": int(row["seed"]),
            "model_name": row["model_name"],
            "c_value": np.nan,
            "threshold_mode": "existing_phase4_threshold_selection",
            "threshold_selected": float(row["threshold_selected"]),
            "selection_note": "copied_from_phase6_5_cleaned_rerun",
            "n_eval_clips": int(row["n_eval_clips"]),
            "n_eval_speakers": int(row["n_eval_speakers"]),
            "accuracy": row.get("accuracy", np.nan),
            "precision": row.get("precision", np.nan),
            "recall": row.get("recall", np.nan),
            "f1": row.get("f1", np.nan),
            "specificity": row.get("specificity", np.nan),
            "balanced_accuracy": row.get("balanced_accuracy", np.nan),
            "false_negatives": row.get("false_negatives", np.nan),
            "false_positives": row.get("false_positives", np.nan),
            "true_negatives": row.get("true_negatives", np.nan),
            "true_positives": row.get("true_positives", np.nan),
            "confusion_matrix_tn_fp_fn_tp": row.get("confusion_matrix_tn_fp_fn_tp", ""),
            "roc_auc": row.get("roc_auc", np.nan),
            "pr_auc": row.get("pr_auc", np.nan),
            "brier_score": row.get("brier_score", np.nan),
            "threshold_fold_min": row.get("threshold_fold_min", np.nan),
            "threshold_fold_max": row.get("threshold_fold_max", np.nan),
            "threshold_fold_mean": row.get("threshold_fold_mean", np.nan),
            "threshold_fold_std": row.get("threshold_fold_std", np.nan),
        }
        for prefix in ["speaker_majority_vote", "speaker_max_risk_score"]:
            for metric in [
                "recall",
                "specificity",
                "precision",
                "f1",
                "balanced_accuracy",
                "false_negatives",
                "false_positives",
                "true_negatives",
                "true_positives",
            ]:
                col = f"{prefix}_{metric}"
                out[col] = row.get(col, np.nan)
        rows.append(out)
    return pd.DataFrame(rows)


def cleaned_oof_predictions(source_dataset: str, seed: int) -> pd.DataFrame:
    suffix = "cleaned_trusted" if source_dataset == "trusted" else "cleaned_governed"
    path = TABLES_DIR / f"phase4_oof_predictions_{suffix}_seed{seed}.csv"
    df = pd.read_csv(path)
    df = df[df["model"].eq("logistic_regression_platt")].copy()
    df["dataset_version"] = f"{source_dataset}_locked_threshold_diagnostic"
    df["source_dataset"] = source_dataset
    df["intervention_name"] = "locked_threshold_diagnostic"
    df["seed"] = seed
    df["model_name"] = "logistic_regression_platt"
    df["c_value"] = np.nan
    df = df.rename(columns={"true_label": "true_label"})
    return df[
        [
            "dataset_version",
            "source_dataset",
            "intervention_name",
            "seed",
            "fold",
            "model_name",
            "c_value",
            "metadata_index",
            "speaker_id",
            "true_label",
            "score",
            "duration_sec",
            "phase4_caution_flag",
        ]
    ].copy()


def aggregate_seed_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_version", "source_dataset", "intervention_name", "model_name", "threshold_mode", "c_value"]
    df = seed_summary.copy()
    df["c_value"] = df["c_value"].fillna("current_anchor")
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_recall=("recall", "mean"),
            min_recall=("recall", "min"),
            max_recall=("recall", "max"),
            std_recall=("recall", "std"),
            mean_specificity=("specificity", "mean"),
            min_specificity=("specificity", "min"),
            max_specificity=("specificity", "max"),
            std_specificity=("specificity", "std"),
            mean_precision=("precision", "mean"),
            mean_f1=("f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            min_balanced_accuracy=("balanced_accuracy", "min"),
            max_balanced_accuracy=("balanced_accuracy", "max"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_threshold=("threshold_selected", "mean"),
            min_threshold=("threshold_selected", "min"),
            max_threshold=("threshold_selected", "max"),
            std_threshold=("threshold_selected", "std"),
            mean_threshold_fold_std=("threshold_fold_std", "mean"),
            n_eval_clips_mean=("n_eval_clips", "mean"),
            n_eval_speakers_mean=("n_eval_speakers", "mean"),
        )
        .reset_index()
    )
    return agg


def threshold_stability_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_version", "source_dataset", "intervention_name", "threshold_mode", "c_value"]
    df = seed_summary.copy()
    df["c_value"] = df["c_value"].fillna("current_anchor")
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            threshold_mean=("threshold_selected", "mean"),
            threshold_min=("threshold_selected", "min"),
            threshold_max=("threshold_selected", "max"),
            threshold_std=("threshold_selected", "std"),
            fold_threshold_std_mean=("threshold_fold_std", "mean"),
            recall_std=("recall", "std"),
            specificity_std=("specificity", "std"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .reset_index()
    )


def assessment_text(std_recall: float, std_specificity: float) -> str:
    combined = np.nanmean([std_recall, std_specificity])
    if combined <= 0.08:
        return "stable"
    if combined <= 0.18:
        return "moderate_variability"
    return "unstable"


def recommendation_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = comparison.copy()
    for _, row in work.iterrows():
        if row["intervention_name"] == "baseline_reference":
            continue
        stability = assessment_text(row["std_recall"], row["std_specificity"])
        if row["threshold_mode"].startswith("locked"):
            threshold_assessment = "locked_threshold_zero_selection_variance"
        elif row["std_threshold"] <= 0.03:
            threshold_assessment = "low_threshold_variance"
        elif row["std_threshold"] <= 0.08:
            threshold_assessment = "moderate_threshold_variance"
        else:
            threshold_assessment = "high_threshold_variance"
        defensibility = "high" if row["mean_specificity"] >= 0.20 and row["seeds"] == 5 else "limited"
        score = (
            float(row["mean_balanced_accuracy"])
            - 0.20 * float(row["std_recall"] if pd.notna(row["std_recall"]) else 0.0)
            - 0.10 * float(row["std_specificity"] if pd.notna(row["std_specificity"]) else 0.0)
        )
        rows.append(
            {
                "rank_score": score,
                "trusted_priority": 1 if row["source_dataset"] == "trusted" else 0,
                "intervention_name": row["intervention_name"],
                "dataset_version": row["dataset_version"],
                "model_name": row["model_name"],
                "c_value": row["c_value"],
                "mean_recall": row["mean_recall"],
                "mean_specificity": row["mean_specificity"],
                "mean_balanced_accuracy": row["mean_balanced_accuracy"],
                "stability_assessment": stability,
                "threshold_assessment": threshold_assessment,
                "defensibility_assessment": defensibility,
                "recommended_next_action": "",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["rank_score", "trusted_priority", "mean_recall", "mean_specificity"],
        ascending=[False, False, False, False],
    )
    actions = []
    for _, row in out.iterrows():
        intervention = row["intervention_name"]
        if intervention == "one_clip_per_speaker":
            actions.append("Use as a required Phase B2 stress test before any Tier 1 claim.")
        elif intervention == "speaker_balanced_weighting":
            actions.append("Carry forward only if it improves stability without collapsing specificity.")
        elif intervention == "locked_threshold_diagnostic":
            actions.append("Use to quantify threshold fragility; do not promote as production policy yet.")
        elif intervention == "logistic_C_sweep":
            actions.append("Carry best moderate C value into Phase B2 only if gains are stable.")
        else:
            actions.append("Review manually.")
    out["recommended_next_action"] = actions
    out = out.drop(columns=["rank_score", "trusted_priority"]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "(empty)"
    show = df.head(max_rows).copy() if max_rows is not None else df.copy()
    show = show.where(pd.notna(show), "")
    lines = [
        "| " + " | ".join([str(c) for c in show.columns]) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        values = []
        for col in show.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append("| ... | ... | ... |")
    return "\n".join(lines)


def compare_to_baseline(comparison: pd.DataFrame, source_dataset: str, intervention_name: str) -> pd.DataFrame:
    baseline = comparison[
        comparison["dataset_version"].eq(f"{source_dataset}_baseline_reference")
    ].iloc[0]
    subset = comparison[
        comparison["source_dataset"].eq(source_dataset)
        & comparison["intervention_name"].eq(intervention_name)
    ].copy()
    subset["delta_recall_vs_baseline"] = subset["mean_recall"] - baseline["mean_recall"]
    subset["delta_specificity_vs_baseline"] = subset["mean_specificity"] - baseline["mean_specificity"]
    subset["delta_recall_std_vs_baseline"] = subset["std_recall"] - baseline["std_recall"]
    subset["delta_specificity_std_vs_baseline"] = subset["std_specificity"] - baseline["std_specificity"]
    return subset


def write_report(seed_summary: pd.DataFrame, comparison: pd.DataFrame, threshold_summary: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    used_files = [f"- `{path}` - used" for path in READ_FIRST_FILES]
    baseline_rows = comparison[comparison["intervention_name"].eq("baseline_reference")][
        ["dataset_version", "mean_recall", "mean_specificity", "std_recall", "std_specificity", "mean_threshold", "std_threshold"]
    ]
    one_clip_rows = comparison[comparison["intervention_name"].eq("one_clip_per_speaker")]
    weighted_rows = comparison[comparison["intervention_name"].eq("speaker_balanced_weighting")]
    locked_rows = comparison[comparison["intervention_name"].eq("locked_threshold_diagnostic")]
    c_rows = comparison[comparison["intervention_name"].eq("logistic_C_sweep")]
    best_rows = recommendations.head(6)

    lines: list[str] = []
    lines.append("# Phase B1 Classical Strengthening Results")
    lines.append("")
    lines.append("Phase B1 ran only the requested controlled classical interventions. No Tier 2, deployment, hardware, demo-app, feature-engineering, score-band, label, speaker-ID, or governance changes were made.")
    lines.append("")
    lines.append("## 1. Phase A Diagnosis Being Tested")
    lines.append("")
    lines.append("Phase A identified repeated-speaker/composition-sensitive folds, weak score separation, and fragile thresholding as the dominant failure modes. One-clip-per-speaker tests repeated-speaker dependence; speaker-balanced weighting tests whether heavy speakers are over-weighting training; locked thresholding tests threshold-selection instability; the small C sweep tests whether modest logistic regularization changes reduce underfit-like weakness.")
    lines.append("")
    lines.append("## 2. Data And Evaluation Contract")
    lines.append("")
    lines.extend(used_files)
    lines.append("")
    lines.append(f"- Canonical feature source used: `{rel(FEATURE_CSV)}`.")
    lines.append(f"- Trusted manifest used: `{rel(TRUSTED_MANIFEST)}`.")
    lines.append(f"- Governed manifest used: `{rel(GOVERNED_MANIFEST)}`.")
    lines.append("- Grouped CV remained `StratifiedGroupKFold` by `speaker_id`; every validation speaker was excluded from that fold's training rows.")
    lines.append("- Seeds used: 42, 43, 44, 45, 46.")
    lines.append("- Governed remains sensitivity-only because governed-only caution rows remain training-only and governance-caution flagged.")
    lines.append("- Baseline-reference rows were copied from the existing cleaned Phase 6.5 rerun to avoid overwriting or rerunning prior Phase 4/Tier 1 outputs.")
    lines.append("- Locked-threshold diagnostic used the pre-existing official Phase 4 threshold 0.300; it was not selected from B1 evaluation folds.")
    lines.append("- Optional linear SVM comparison was skipped to keep Phase B1 limited to the official anchor and requested interventions; adding a comparable weighted/locked SVM branch would expand scope.")
    lines.append("")
    lines.append("## 3. Intervention Definitions")
    lines.append("")
    lines.append("- `one_clip_per_speaker`: selected at most one row per speaker by preferring Phase 4 evaluation-allowed clips, then longest duration, then `metadata_index`, then `audio_file`. Model family, features, labels, grouped CV, calibration type, and threshold-selection rule stayed unchanged.")
    lines.append("- `speaker_balanced_weighting`: kept full cleaned manifests and grouped CV; logistic training used the existing `class_weight='balanced'` plus training-fold-only inverse speaker-frequency sample weights. Validation data never entered weight computation.")
    lines.append("- `locked_threshold_diagnostic`: reused existing cleaned OOF anchor scores and applied the pre-existing global threshold 0.300 for every seed. This changes only the operating threshold, not scores or folds.")
    lines.append("- `logistic_C_sweep`: evaluated fixed `C in [0.1, 0.3, 1.0, 3.0, 10.0]` with Platt calibration and existing threshold-selection behavior. No other knobs were swept.")
    lines.append("")
    lines.append("## 4. Results By Intervention")
    lines.append("")
    lines.append("Baseline reference:")
    lines.append(markdown_table(baseline_rows))
    lines.append("")
    lines.append("Top aggregate intervention rows:")
    lines.append(markdown_table(best_rows, max_rows=12))
    lines.append("")
    lines.append("Full aggregate comparison is in `reports/tables/phaseB1_intervention_comparison.csv`; seed-level rows are in `reports/tables/phaseB1_seed_summary.csv`.")
    lines.append("")
    lines.append("## 5. One-Clip-Per-Speaker Interpretation")
    lines.append("")
    lines.append(markdown_table(one_clip_rows[["dataset_version", "mean_recall", "mean_specificity", "std_recall", "std_specificity", "mean_threshold", "std_threshold"]]))
    lines.append("")
    for source in ["trusted", "governed"]:
        delta = compare_to_baseline(comparison, source, "one_clip_per_speaker")
        if not delta.empty:
            row = delta.iloc[0]
            lines.append(f"- `{source}`: recall delta vs baseline {row['delta_recall_vs_baseline']:.3f}; specificity delta {row['delta_specificity_vs_baseline']:.3f}; recall-std delta {row['delta_recall_std_vs_baseline']:.3f}.")
    lines.append("")
    lines.append("Interpretation: if one-clip performance stabilizes or improves, repeated speaker weighting was hurting the anchor. If it collapses, the original anchor was leaning on repeated-speaker representation. The governed one-clip subset is evaluation-safe and may converge toward trusted because governed-only caution rows are training-only.")
    lines.append("")
    lines.append("## 6. Speaker-Balanced Weighting Interpretation")
    lines.append("")
    lines.append(markdown_table(weighted_rows[["dataset_version", "mean_recall", "mean_specificity", "std_recall", "std_specificity", "mean_threshold", "std_threshold"]]))
    lines.append("")
    for source in ["trusted", "governed"]:
        delta = compare_to_baseline(comparison, source, "speaker_balanced_weighting")
        if not delta.empty:
            row = delta.iloc[0]
            lines.append(f"- `{source}`: recall delta vs baseline {row['delta_recall_vs_baseline']:.3f}; specificity delta {row['delta_specificity_vs_baseline']:.3f}; specificity-std delta {row['delta_specificity_std_vs_baseline']:.3f}.")
    lines.append("")
    lines.append("Interpretation should prioritize stability and specificity preservation, not recall alone. Weighting is defensible only if it reduces repeated-speaker sensitivity without driving specificity below the screening floor.")
    lines.append("")
    lines.append("## 7. Locked-Threshold Interpretation")
    lines.append("")
    lines.append(markdown_table(locked_rows[["dataset_version", "mean_recall", "mean_specificity", "std_recall", "std_specificity", "mean_threshold", "std_threshold"]]))
    lines.append("")
    lines.append("The locked threshold sets threshold-selection variance to zero by design. If recall/specificity remain highly unstable, fold composition and score weakness dominate; if stability improves materially without large metric loss, threshold fragility is a leading failure mode.")
    lines.append("")
    lines.append("## 8. Logistic C Sweep Interpretation")
    lines.append("")
    lines.append(markdown_table(c_rows[["dataset_version", "c_value", "mean_recall", "mean_specificity", "std_recall", "std_specificity", "mean_balanced_accuracy", "mean_threshold"]], max_rows=20))
    lines.append("")
    lines.append("Moderate C changes can indicate whether the official anchor is too rigid, but a C value should only move forward if gains are stable across seeds and do not trade recall for unusable specificity.")
    lines.append("")
    lines.append("## 9. Optional Linear SVM Interpretation")
    lines.append("")
    lines.append("Not run in Phase B1. The existing Phase 4 script has a linear SVM candidate, but Phase B1 stayed limited to the official `logistic_regression_platt` anchor plus the requested controlled interventions.")
    lines.append("")
    lines.append("## 10. Final Recommendation")
    lines.append("")
    lines.append(markdown_table(recommendations, max_rows=15))
    lines.append("")
    top = recommendations.iloc[0]
    lines.append(f"Most helpful B1 condition by the aggregate stability-adjusted ranking: `{top['dataset_version']}` (`{top['intervention_name']}`).")
    lines.append("Tier 1 should not be treated as salvaged unless the best trusted condition improves stability and preserves a defensible specificity/recall tradeoff. Phase B2 should take only the strongest trusted intervention forward for a narrower confirmation run. Deployment and hardware remain blocked.")
    lines.append("")
    lines.append("## Threshold Stability Summary")
    lines.append("")
    lines.append(markdown_table(threshold_summary, max_rows=25))
    lines.append("")
    (REPORTS_DIR / "phaseB1_classical_strengthening_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    phase4 = load_phase4_module()
    feature_df = pd.read_csv(FEATURE_CSV)
    feature_cols = phase4.select_feature_columns(feature_df)
    if len(feature_cols) != 256:
        raise RuntimeError(f"Expected 256 canonical Phase 4 feature columns, found {len(feature_cols)}")

    trusted = normalize_manifest(pd.read_csv(TRUSTED_MANIFEST))
    governed = normalize_manifest(pd.read_csv(GOVERNED_MANIFEST))
    missing_trusted = sorted(set(feature_cols) - set(trusted.columns))
    missing_governed = sorted(set(feature_cols) - set(governed.columns))
    if missing_trusted or missing_governed:
        raise RuntimeError(f"Manifest feature mismatch: trusted={missing_trusted[:5]}, governed={missing_governed[:5]}")

    trusted_one = one_clip_manifest(trusted)
    governed_one = one_clip_manifest(governed)
    trusted_one.to_csv(TABLES_DIR / "phaseB1_one_clip_per_speaker_manifest_trusted.csv", index=False)
    governed_one.to_csv(TABLES_DIR / "phaseB1_one_clip_per_speaker_manifest_governed.csv", index=False)

    seed_rows = [baseline_reference_rows()]

    for source_dataset, manifest in [("trusted", trusted_one), ("governed", governed_one)]:
        condition = f"{source_dataset}_one_clip_per_speaker"
        for seed in SEEDS:
            print(f"Running {condition} seed {seed}")
            preds, details = run_platt_oof(
                df=manifest,
                feature_cols=feature_cols,
                seed=seed,
                condition=condition,
                source_dataset=source_dataset,
                intervention="one_clip_per_speaker",
                c_value=None,
                speaker_weighted=False,
            )
            seed_rows.append(pd.DataFrame([summarize_predictions(preds, "existing_phase4_threshold_selection", details)]))

    for source_dataset, manifest in [("trusted", trusted), ("governed", governed)]:
        condition = f"{source_dataset}_speaker_balanced_weighting"
        for seed in SEEDS:
            print(f"Running {condition} seed {seed}")
            preds, details = run_platt_oof(
                df=manifest,
                feature_cols=feature_cols,
                seed=seed,
                condition=condition,
                source_dataset=source_dataset,
                intervention="speaker_balanced_weighting",
                c_value=None,
                speaker_weighted=True,
            )
            seed_rows.append(pd.DataFrame([summarize_predictions(preds, "existing_phase4_threshold_selection", details)]))

    for source_dataset in ["trusted", "governed"]:
        for seed in SEEDS:
            print(f"Running {source_dataset}_locked_threshold_diagnostic seed {seed}")
            preds = cleaned_oof_predictions(source_dataset, seed)
            seed_rows.append(
                pd.DataFrame(
                    [
                        summarize_predictions(
                            preds,
                            "locked_global_official_phase4_threshold_0.30",
                            locked_threshold=OFFICIAL_LOCKED_THRESHOLD,
                        )
                    ]
                )
            )

    for source_dataset, manifest in [("trusted", trusted), ("governed", governed)]:
        condition = f"{source_dataset}_logistic_C_sweep"
        for c_value in PHASEB1_C_GRID:
            for seed in SEEDS:
                print(f"Running {condition} C={c_value} seed {seed}")
                preds, details = run_platt_oof(
                    df=manifest,
                    feature_cols=feature_cols,
                    seed=seed,
                    condition=condition,
                    source_dataset=source_dataset,
                    intervention="logistic_C_sweep",
                    c_value=c_value,
                    speaker_weighted=False,
                )
                seed_rows.append(pd.DataFrame([summarize_predictions(preds, "existing_phase4_threshold_selection", details)]))

    seed_summary = pd.concat(seed_rows, ignore_index=True, sort=False)
    seed_summary.to_csv(TABLES_DIR / "phaseB1_seed_summary.csv", index=False)
    comparison = aggregate_seed_summary(seed_summary)
    comparison.to_csv(TABLES_DIR / "phaseB1_intervention_comparison.csv", index=False)
    threshold_summary = threshold_stability_summary(seed_summary)
    threshold_summary.to_csv(TABLES_DIR / "phaseB1_threshold_stability_summary.csv", index=False)
    recommendations = recommendation_table(comparison)
    recommendations.to_csv(TABLES_DIR / "phaseB1_recommendation_table.csv", index=False)
    write_report(seed_summary, comparison, threshold_summary, recommendations)
    print("Phase B1 complete.")


if __name__ == "__main__":
    main()
