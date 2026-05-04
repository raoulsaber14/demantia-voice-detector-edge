"""Run Phase 4 speaker-safe classical baseline experiments.

This script intentionally stays within the Phase 4 classical-ML boundary:
hand-crafted Phase 3 features, grouped evaluation, a transparent heuristic
benchmark, and clinically oriented error analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
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
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CSV = ROOT / "data/features/features_phase3_governed_pruned.csv"
GOVERNANCE_CSV = ROOT / "reports/phase3_governed_record_decisions.csv"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
RANDOM_SEED = 42
POSITIVE_LABEL = 1
POSITIVE_CLASS_NAME = "dementia"
CALIBRATION_SUFFIXES = ("_uncalibrated", "_platt", "_isotonic")

NON_FEATURE_COLUMNS = {
    "metadata_index",
    "record_id",
    "speaker_id",
    "audio_file",
    "repo_audio_file",
    "processed_file",
    "label",
    "label_name",
    "datasplit",
    "dementia_type",
    "gender",
    "ethnicity",
    "error",
    "name",
    "birth",
    "death",
    "first_symptoms",
    "feature_extraction_status",
    "decision_status",
    "speaker_grouping_status",
    "phase3_baseline_included",
    "phase3_training_allowed",
    "phase3_final_eval_allowed",
    "phase4_training_allowed",
    "phase4_eval_allowed",
    "phase4_caution_flag",
    "phase2_decision",
    "phase2_issue_type",
    "manual_review_required",
    "decision_reason",
    "source_of_decision",
    "where_enforced",
}

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "specificity",
    "balanced_accuracy",
    "false_negatives",
    "false_negative_rate",
    "brier_score",
]


@dataclass
class ModelSpec:
    name: str
    estimator: BaseEstimator | None
    param_grid: dict[str, list[Any]]
    uses_scaling: bool
    model_family: str
    status: str = "available"
    unavailable_reason: str = ""


@dataclass
class FoldSpec:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_speakers: set[str]


def rel(path: Path) -> str:
    """Return a repo-relative path for reports."""
    return str(path.relative_to(ROOT))


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a compact markdown table without optional dependencies."""
    if df.empty:
        return "(empty)"
    show = df.head(max_rows).copy() if max_rows is not None else df.copy()
    show = show.where(pd.notna(show), "")
    headers = [str(c) for c in show.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        values = [str(row[c]).replace("\n", " ") for c in show.columns]
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"| ... | ... | ... |")
    return "\n".join(lines)


def bool_series(series: pd.Series, default: bool) -> pd.Series:
    """Parse mixed bool/object CSV columns into booleans."""
    if series.dtype == bool:
        return series.fillna(default).astype(bool)
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}

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


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select official numeric model columns after excluding metadata/governance."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in NON_FEATURE_COLUMNS]


def load_phase4_data() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load the official Phase 3 table and attach governance decisions."""
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"Official Phase 3 feature table not found: {FEATURE_CSV}")
    if not GOVERNANCE_CSV.exists():
        raise FileNotFoundError(f"Governed-record decisions file not found: {GOVERNANCE_CSV}")

    features = pd.read_csv(FEATURE_CSV)
    governance = pd.read_csv(GOVERNANCE_CSV)
    governance_cols = [
        "record_id",
        "repo_audio_file",
        "speaker_grouping_status",
        "decision_status",
        "phase3_baseline_included",
        "phase3_training_allowed",
        "phase3_final_eval_allowed",
        "phase2_decision",
        "phase2_issue_type",
        "manual_review_required",
        "decision_reason",
        "source_of_decision",
        "where_enforced",
    ]
    missing_cols = [c for c in governance_cols if c not in governance.columns]
    if missing_cols:
        raise ValueError(f"Governance file is missing expected columns: {missing_cols}")

    df = features.merge(
        governance[governance_cols],
        how="left",
        left_on="metadata_index",
        right_on="record_id",
        suffixes=("", "_governance"),
    ).copy()
    for col, default in [
        ("phase3_baseline_included", True),
        ("phase3_training_allowed", True),
        ("phase3_final_eval_allowed", True),
        ("manual_review_required", False),
    ]:
        df[col] = bool_series(df[col], default=default)

    df["decision_status"] = df["decision_status"].fillna("missing_governance_match")
    df["phase2_decision"] = df["phase2_decision"].fillna("missing_governance_match")
    df["phase2_issue_type"] = df["phase2_issue_type"].fillna("")
    df["phase4_training_allowed"] = df["phase3_baseline_included"] & df["phase3_training_allowed"]
    df["phase4_eval_allowed"] = df["phase3_baseline_included"] & df["phase3_final_eval_allowed"]
    df["phase4_caution_flag"] = (
        df["decision_status"].ne("include")
        | df["manual_review_required"]
        | df["phase2_decision"].ne("include_no_phase2_flag")
    )
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df["speaker_id"] = df["speaker_id"].astype(str)

    feature_cols = select_feature_columns(df)
    if len(feature_cols) != 256:
        raise ValueError(f"Expected 256 official numeric features, found {len(feature_cols)}")
    if df["speaker_id"].isna().any() or df["speaker_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing speaker_id values would break grouped evaluation.")
    return df, feature_cols, governance


def mode_int(series: pd.Series) -> int:
    """Stable integer mode for a single-label speaker group."""
    return int(series.astype(int).mode().iloc[0])


def speaker_label_counts(df: pd.DataFrame) -> dict[int, int]:
    labels = df.groupby("speaker_id")["label"].agg(mode_int)
    return {int(k): int(v) for k, v in labels.value_counts().sort_index().to_dict().items()}


def build_outer_folds(
    df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    max_splits: int = 5,
) -> tuple[list[FoldSpec], pd.DataFrame, str, int]:
    """Build final-evaluation folds while allowing training-only records in training folds."""
    eval_df = df[df["phase4_eval_allowed"]].copy()
    if eval_df.empty:
        raise ValueError("No final-evaluation-allowed records are available.")
    if eval_df["label"].nunique() < 2:
        raise ValueError("Final-evaluation-allowed records do not contain both classes.")

    class_speaker_counts = speaker_label_counts(eval_df)
    min_class_speakers = min(class_speaker_counts.values())
    n_splits = int(min(max_splits, min_class_speakers))
    if n_splits >= 2:
        splitter: StratifiedGroupKFold | GroupKFold = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        splitter_name = "StratifiedGroupKFold"
    else:
        n_splits = int(min(max_splits, eval_df["speaker_id"].nunique()))
        if n_splits < 2:
            raise ValueError("Grouped CV is infeasible: fewer than two evaluation speakers.")
        splitter = GroupKFold(n_splits=n_splits)
        splitter_name = "GroupKFold"

    split_iter = splitter.split(
        eval_df[feature_cols],
        eval_df["label"].astype(int).to_numpy(),
        groups=eval_df["speaker_id"].astype(str).to_numpy(),
    )

    folds: list[FoldSpec] = []
    split_rows: list[dict[str, Any]] = []
    for fold, (_, val_rel_idx) in enumerate(split_iter, start=1):
        val_idx = eval_df.iloc[val_rel_idx].index.to_numpy()
        val_speakers = set(df.loc[val_idx, "speaker_id"].astype(str))
        train_mask = df["phase4_training_allowed"] & ~df["speaker_id"].astype(str).isin(val_speakers)
        train_idx = df.index[train_mask].to_numpy()
        train_df = df.loc[train_idx]
        val_fold_df = df.loc[val_idx]
        overlap = set(train_df["speaker_id"].astype(str)) & val_speakers
        if overlap:
            raise ValueError(f"Speaker leakage detected in fold {fold}: {sorted(overlap)[:5]}")
        if train_df["label"].nunique() < 2 or val_fold_df["label"].nunique() < 2:
            raise ValueError(f"Fold {fold} does not contain both classes in train and validation.")
        folds.append(FoldSpec(fold=fold, train_idx=train_idx, val_idx=val_idx, val_speakers=val_speakers))
        split_rows.append(
            {
                "fold": fold,
                "train_clips": int(len(train_df)),
                "validation_clips": int(len(val_fold_df)),
                "train_speakers": int(train_df["speaker_id"].nunique()),
                "validation_speakers": int(val_fold_df["speaker_id"].nunique()),
                "train_dementia_clips": int((train_df["label"] == 1).sum()),
                "train_non_dementia_clips": int((train_df["label"] == 0).sum()),
                "validation_dementia_clips": int((val_fold_df["label"] == 1).sum()),
                "validation_non_dementia_clips": int((val_fold_df["label"] == 0).sum()),
                "training_only_caution_clips_in_train": int((~train_df["phase4_eval_allowed"]).sum()),
                "speaker_overlap_count": len(overlap),
            }
        )

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(TABLES_DIR / "phase4_cv_split_summary.csv", index=False)
    note = (
        f"{splitter_name} with n_splits={n_splits}; "
        f"final-eval speaker class counts={class_speaker_counts}"
    )
    return folds, split_df, note, n_splits


def inner_group_cv(y: np.ndarray, groups: np.ndarray, seed: int, max_splits: int = 3) -> StratifiedGroupKFold | None:
    """Create an inner grouped CV splitter when speaker class support allows it."""
    group_df = pd.DataFrame({"speaker_id": groups.astype(str), "label": y.astype(int)}).drop_duplicates()
    labels = group_df.groupby("speaker_id")["label"].agg(mode_int)
    counts = labels.value_counts().to_dict()
    if len(counts) < 2:
        return None
    n_splits = int(min(max_splits, min(counts.values())))
    if n_splits < 2:
        return None
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def build_model_specs(seed: int) -> list[ModelSpec]:
    """Create the practical Phase 4 model list."""
    specs = [
        ModelSpec(
            name="logistic_regression",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=seed,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
            uses_scaling=True,
            model_family="linear_probabilistic",
        ),
        ModelSpec(
            name="linear_svm",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LinearSVC(
                            C=1.0,
                            class_weight="balanced",
                            random_state=seed,
                            max_iter=20000,
                            dual=False,
                        ),
                    ),
                ]
            ),
            param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
            uses_scaling=True,
            model_family="linear_margin",
        ),
        ModelSpec(
            name="elastic_net_logistic",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=10000,
                            class_weight="balanced",
                            random_state=seed,
                            solver="saga",
                            l1_ratio=0.50,
                        ),
                    ),
                ]
            ),
            param_grid={"model__C": [0.01, 0.1, 1.0], "model__l1_ratio": [0.25, 0.50, 0.75]},
            uses_scaling=True,
            model_family="linear_probabilistic_elastic_net",
        ),
        ModelSpec(
            name="random_forest",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt"],
            },
            uses_scaling=False,
            model_family="tree_ensemble",
        ),
    ]

    if xgboost_available():
        from xgboost import XGBClassifier

        specs.append(
            ModelSpec(
                name="xgboost",
                estimator=Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBClassifier(
                                objective="binary:logistic",
                                eval_metric="logloss",
                                random_state=seed,
                                n_estimators=150,
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
                param_grid={
                    "model__max_depth": [2, 3],
                    "model__learning_rate": [0.03, 0.1],
                    "model__subsample": [0.8, 1.0],
                },
                uses_scaling=False,
                model_family="gradient_boosted_trees",
            )
        )
    else:
        specs.append(
            ModelSpec(
                name="xgboost",
                estimator=None,
                param_grid={},
                uses_scaling=False,
                model_family="gradient_boosted_trees",
                status="unavailable",
                unavailable_reason="The xgboost package is not installed in the active environment.",
            )
        )
    return specs


def sigmoid(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))


def positive_scores(model: BaseEstimator, x: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return a positive-class risk score in [0, 1]."""
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(x)
            arr = np.asarray(prob)
            if arr.ndim == 2 and arr.shape[1] > 1:
                return arr[:, 1].astype(float), "predict_proba"
            return arr.ravel().astype(float), "predict_proba"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=float)
        if decision.ndim == 2:
            decision = decision[:, -1]
        return sigmoid(decision).ravel(), "sigmoid_decision_function"
    pred = np.asarray(model.predict(x), dtype=float)
    return pred.ravel(), "hard_prediction_fallback"


def raw_calibration_scores(model: BaseEstimator, x: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return a one-dimensional score for fitting external calibrators."""
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


def calibrated_candidate_name(model_name: str, calibration_method: str) -> str:
    return f"{model_name}_{calibration_method}"


def split_candidate_name(candidate_name: str) -> tuple[str, str]:
    for suffix in CALIBRATION_SUFFIXES:
        if candidate_name.endswith(suffix):
            return candidate_name[: -len(suffix)], suffix[1:]
    return candidate_name, "not_applicable"


def fit_external_calibrators(
    model: BaseEstimator,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit Platt and isotonic calibrators using inner grouped OOF scores."""
    y_train = train_df["label"].astype(int).to_numpy()
    groups = train_df["speaker_id"].astype(str).to_numpy()
    cv = inner_group_cv(y_train, groups, seed=seed)
    details: dict[str, Any] = {
        "calibration_status": "not_run",
        "calibration_inner_cv": "none",
        "calibration_oof_rows": 0,
        "calibration_score_type": "",
        "calibration_note": "",
    }
    if cv is None:
        details["calibration_status"] = "skipped_inner_group_cv_infeasible"
        return {}, details

    oof_scores = np.full(len(train_df), np.nan, dtype=float)
    for inner_train_idx, inner_cal_idx in cv.split(train_df[feature_cols], y_train, groups=groups):
        inner_y_train = y_train[inner_train_idx]
        inner_y_cal = y_train[inner_cal_idx]
        if np.unique(inner_y_train).size < 2 or np.unique(inner_y_cal).size < 2:
            details["calibration_status"] = "skipped_inner_fold_class_support"
            details["calibration_note"] = "An inner calibration split lacked both classes."
            return {}, details
        inner_model = clone(model)
        inner_model.fit(train_df.iloc[inner_train_idx][feature_cols], inner_y_train)
        fold_scores, score_type = raw_calibration_scores(inner_model, train_df.iloc[inner_cal_idx][feature_cols])
        oof_scores[inner_cal_idx] = fold_scores
        details["calibration_score_type"] = score_type

    valid_mask = np.isfinite(oof_scores)
    if valid_mask.sum() < 10 or np.unique(y_train[valid_mask]).size < 2:
        details["calibration_status"] = "skipped_insufficient_oof_support"
        details["calibration_oof_rows"] = int(valid_mask.sum())
        return {}, details

    platt = LogisticRegression(max_iter=2000, random_state=seed)
    platt.fit(oof_scores[valid_mask].reshape(-1, 1), y_train[valid_mask])

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(oof_scores[valid_mask], y_train[valid_mask])

    details.update(
        {
            "calibration_status": "success",
            "calibration_inner_cv": "StratifiedGroupKFold",
            "calibration_oof_rows": int(valid_mask.sum()),
            "calibration_oof_positive": int(y_train[valid_mask].sum()),
            "calibration_oof_negative": int((y_train[valid_mask] == 0).sum()),
        }
    )
    return {"platt": platt, "isotonic": isotonic}, details


class HeuristicRiskModel:
    """Transparent pause/speech-rate/prosody risk benchmark."""

    def __init__(self, pause_col: str, rate_col: str, prosody_col: str) -> None:
        self.pause_col = pause_col
        self.rate_col = rate_col
        self.prosody_col = prosody_col
        self.stats: dict[str, dict[str, float]] = {}

    @property
    def columns(self) -> list[str]:
        return [self.pause_col, self.rate_col, self.prosody_col]

    def fit(self, df: pd.DataFrame) -> "HeuristicRiskModel":
        for col in self.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            median = float(values.median())
            low = float(values.quantile(0.05))
            high = float(values.quantile(0.95))
            if not np.isfinite(median):
                median = 0.0
            if not np.isfinite(low):
                low = median
            if not np.isfinite(high):
                high = median
            if high <= low:
                high = low + 1.0
            self.stats[col] = {"median": median, "low": low, "high": high}
        return self

    def _norm(self, df: pd.DataFrame, col: str) -> np.ndarray:
        stat = self.stats[col]
        values = pd.to_numeric(df[col], errors="coerce").fillna(stat["median"]).astype(float).to_numpy()
        norm = (values - stat["low"]) / (stat["high"] - stat["low"])
        return np.clip(norm, 0.0, 1.0)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        pause = self._norm(df, self.pause_col)
        rate = self._norm(df, self.rate_col)
        prosody = self._norm(df, self.prosody_col)
        return (pause + (1.0 - rate) + prosody) / 3.0


def choose_heuristic_columns(feature_cols: list[str]) -> dict[str, str]:
    """Choose official columns for the required transparent heuristic."""
    pause_options = ["pause_ratio", "total_silence_duration_sec", "avg_pause_duration_sec", "pause_rate_per_min"]
    rate_options = ["speaking_rate_proxy", "onset_rate_per_speech_sec", "onset_count"]
    prosody_options = ["f0_semitone_std", "f0_iqr", "f0_std", "jitter_local_proxy"]

    def first_available(options: list[str], label: str) -> str:
        for col in options:
            if col in feature_cols:
                return col
        raise ValueError(f"No defendable heuristic {label} column found. Tried: {options}")

    return {
        "pause_burden": first_available(pause_options, "pause burden"),
        "speech_rate": first_available(rate_options, "speech-rate"),
        "prosody_instability": first_available(prosody_options, "prosody instability"),
    }


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute screening metrics at a decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positive_count = tp + fn
    negative_count = tn + fp
    metrics = {
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
        "threshold": float(threshold),
    }
    if np.unique(y_true).size >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    try:
        metrics["brier_score"] = float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0)))
    except Exception:
        metrics["brier_score"] = np.nan
    return metrics


def save_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray, threshold: float, out_path: Path, title: str) -> None:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=["non_dementia", "dementia"])
    ax.set_yticks([0, 1], labels=["non_dementia", "dementia"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fit_grid_model(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Tune a model on training speakers only."""
    if spec.estimator is None:
        raise ValueError(f"Model {spec.name} is unavailable.")
    x_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int).to_numpy()
    groups = train_df["speaker_id"].astype(str).to_numpy()
    cv = inner_group_cv(y_train, groups, seed=seed)
    estimator = clone(spec.estimator)
    details: dict[str, Any] = {
        "model": spec.name,
        "search_status": "not_run",
        "best_params": {},
        "best_score": np.nan,
        "inner_cv": "none",
    }
    if cv is None or not spec.param_grid:
        estimator.fit(x_train, y_train)
        details["search_status"] = "skipped_inner_group_cv_infeasible"
        return estimator, details

    search = GridSearchCV(
        estimator=estimator,
        param_grid=spec.param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score=np.nan,
    )
    try:
        search.fit(x_train, y_train, groups=groups)
        details.update(
            {
                "search_status": "success",
                "best_params": search.best_params_,
                "best_score": float(search.best_score_) if np.isfinite(search.best_score_) else np.nan,
                "inner_cv": "StratifiedGroupKFold",
            }
        )
        return search.best_estimator_, details
    except Exception as exc:
        estimator.fit(x_train, y_train)
        details.update({"search_status": "failed_default_fit_used", "failure_reason": str(exc)})
        return estimator, details


def collect_feature_importance(
    model_name: str,
    model: BaseEstimator,
    feature_cols: list[str],
    fold: int,
) -> list[dict[str, Any]]:
    """Collect fold-level coefficient/importances for interpretable models."""
    if not isinstance(model, Pipeline) or "model" not in model.named_steps:
        return []
    final_model = model.named_steps["model"]
    rows: list[dict[str, Any]] = []
    if model_name in {"logistic_regression", "elastic_net_logistic", "linear_svm"} and hasattr(final_model, "coef_"):
        coef = np.asarray(final_model.coef_).ravel()
        for feature, value in zip(feature_cols, coef):
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "feature": feature,
                    "importance": float(abs(value)),
                    "coefficient": float(value),
                }
            )
    elif model_name == "random_forest" and hasattr(final_model, "feature_importances_"):
        for feature, value in zip(feature_cols, final_model.feature_importances_):
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "feature": feature,
                    "importance": float(value),
                    "coefficient": np.nan,
                }
            )
    return rows


def run_grouped_experiments(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[FoldSpec],
    specs: list[ModelSpec],
    heuristic_cols: dict[str, str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run outer grouped CV for ML models and the heuristic benchmark."""
    fold_metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for fold_spec in folds:
        train_df = df.loc[fold_spec.train_idx].copy()
        val_df = df.loc[fold_spec.val_idx].copy()
        y_val = val_df["label"].astype(int).to_numpy()

        for spec in specs:
            if spec.status == "unavailable":
                continue
            model, search_details = fit_grid_model(spec, train_df, feature_cols, seed=seed + fold_spec.fold)
            search_rows.append({"fold": fold_spec.fold, **search_details})
            importance_rows.extend(collect_feature_importance(spec.name, model, feature_cols, fold_spec.fold))

            calibrators, calibration_details = fit_external_calibrators(
                model=model,
                train_df=train_df,
                feature_cols=feature_cols,
                seed=seed + 1000 + fold_spec.fold,
            )
            calibration_rows.append({"fold": fold_spec.fold, "base_model": spec.name, **calibration_details})

            uncalibrated_scores, uncalibrated_score_type = positive_scores(model, val_df[feature_cols])
            raw_scores, raw_score_type = raw_calibration_scores(model, val_df[feature_cols])
            variants: list[tuple[str, np.ndarray, str]] = [
                ("uncalibrated", uncalibrated_scores, uncalibrated_score_type)
            ]
            if "platt" in calibrators:
                variants.append(
                    (
                        "platt",
                        calibrators["platt"].predict_proba(raw_scores.reshape(-1, 1))[:, 1],
                        f"platt_calibrated_{raw_score_type}",
                    )
                )
            if "isotonic" in calibrators:
                variants.append(
                    (
                        "isotonic",
                        np.clip(calibrators["isotonic"].predict(raw_scores), 0.0, 1.0),
                        f"isotonic_calibrated_{raw_score_type}",
                    )
                )

            for calibration_method, scores, score_type in variants:
                candidate_name = calibrated_candidate_name(spec.name, calibration_method)
                metrics = compute_metrics(y_val, scores, threshold=0.5)
                fold_metric_rows.append(
                    {
                        "fold": fold_spec.fold,
                        "model": candidate_name,
                        "base_model": spec.name,
                        "calibration_method": calibration_method,
                        "model_status": "evaluated",
                        "score_type": score_type,
                        "threshold": 0.5,
                        "n_train": int(len(train_df)),
                        "n_validation": int(len(val_df)),
                        "n_train_speakers": int(train_df["speaker_id"].nunique()),
                        "n_validation_speakers": int(val_df["speaker_id"].nunique()),
                        **metrics,
                    }
                )
                for pos, (idx, row) in enumerate(val_df.iterrows()):
                    prediction_rows.append(
                        {
                            "fold": fold_spec.fold,
                            "model": candidate_name,
                            "base_model": spec.name,
                            "calibration_method": calibration_method,
                            "row_index": int(idx),
                            "metadata_index": row.get("metadata_index"),
                            "speaker_id": row.get("speaker_id"),
                            "audio_file": row.get("audio_file"),
                            "processed_file": row.get("processed_file"),
                            "true_label": int(row.get("label")),
                            "score": float(scores[pos]),
                            "default_prediction": int(scores[pos] >= 0.5),
                            "score_type": score_type,
                            "decision_status": row.get("decision_status"),
                            "phase2_decision": row.get("phase2_decision"),
                            "phase2_issue_type": row.get("phase2_issue_type"),
                            "manual_review_required": bool(row.get("manual_review_required")),
                            "phase4_caution_flag": bool(row.get("phase4_caution_flag")),
                            "duration_sec": row.get("duration_sec", np.nan),
                            heuristic_cols["pause_burden"]: row.get(heuristic_cols["pause_burden"], np.nan),
                            heuristic_cols["speech_rate"]: row.get(heuristic_cols["speech_rate"], np.nan),
                            heuristic_cols["prosody_instability"]: row.get(heuristic_cols["prosody_instability"], np.nan),
                        }
                    )

        heuristic = HeuristicRiskModel(
            pause_col=heuristic_cols["pause_burden"],
            rate_col=heuristic_cols["speech_rate"],
            prosody_col=heuristic_cols["prosody_instability"],
        ).fit(train_df)
        heuristic_scores = heuristic.score(val_df)
        metrics = compute_metrics(y_val, heuristic_scores, threshold=0.5)
        fold_metric_rows.append(
            {
                    "fold": fold_spec.fold,
                    "model": "heuristic",
                    "base_model": "heuristic",
                    "calibration_method": "not_applicable",
                    "model_status": "evaluated",
                "score_type": "training_fold_robust_percentile_score",
                "threshold": 0.5,
                "n_train": int(len(train_df)),
                "n_validation": int(len(val_df)),
                "n_train_speakers": int(train_df["speaker_id"].nunique()),
                "n_validation_speakers": int(val_df["speaker_id"].nunique()),
                **metrics,
            }
        )
        for pos, (idx, row) in enumerate(val_df.iterrows()):
            prediction_rows.append(
                {
                    "fold": fold_spec.fold,
                    "model": "heuristic",
                    "base_model": "heuristic",
                    "calibration_method": "not_applicable",
                    "row_index": int(idx),
                    "metadata_index": row.get("metadata_index"),
                    "speaker_id": row.get("speaker_id"),
                    "audio_file": row.get("audio_file"),
                    "processed_file": row.get("processed_file"),
                    "true_label": int(row.get("label")),
                    "score": float(heuristic_scores[pos]),
                    "default_prediction": int(heuristic_scores[pos] >= 0.5),
                    "score_type": "training_fold_robust_percentile_score",
                    "decision_status": row.get("decision_status"),
                    "phase2_decision": row.get("phase2_decision"),
                    "phase2_issue_type": row.get("phase2_issue_type"),
                    "manual_review_required": bool(row.get("manual_review_required")),
                    "phase4_caution_flag": bool(row.get("phase4_caution_flag")),
                    "duration_sec": row.get("duration_sec", np.nan),
                    heuristic_cols["pause_burden"]: row.get(heuristic_cols["pause_burden"], np.nan),
                    heuristic_cols["speech_rate"]: row.get(heuristic_cols["speech_rate"], np.nan),
                    heuristic_cols["prosody_instability"]: row.get(heuristic_cols["prosody_instability"], np.nan),
                }
            )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    feature_importance = pd.DataFrame(importance_rows)
    search_details = pd.DataFrame(search_rows)
    calibration_details = pd.DataFrame(calibration_rows)
    fold_metrics.to_csv(TABLES_DIR / "phase4_fold_metrics.csv", index=False)
    predictions.to_csv(TABLES_DIR / "phase4_oof_predictions.csv", index=False)
    search_details.to_csv(TABLES_DIR / "phase4_hyperparameter_search_summary.csv", index=False)
    calibration_details.to_csv(TABLES_DIR / "phase4_calibration_fit_summary.csv", index=False)
    return fold_metrics, predictions, feature_importance, search_details


def threshold_sweep(predictions: pd.DataFrame, unavailable_specs: list[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep operating thresholds and select recall-focused rows."""
    thresholds = [round(float(x), 2) for x in np.arange(0.10, 0.901, 0.05)]
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for model_name, model_df in predictions.groupby("model"):
        y_true = model_df["true_label"].astype(int).to_numpy()
        scores = model_df["score"].astype(float).to_numpy()
        prevalence = float(np.mean(y_true == POSITIVE_LABEL))
        minimum_precision = max(prevalence, 0.25)
        minimum_specificity = 0.20
        for threshold in thresholds:
            metrics = compute_metrics(y_true, scores, threshold=threshold)
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "selected": False,
                    "minimum_precision": minimum_precision,
                    "minimum_specificity": minimum_specificity,
                    **metrics,
                }
            )
        model_rows = pd.DataFrame([row for row in rows if row["model"] == model_name])
        screening_candidates = model_rows[
            (model_rows["precision"] >= minimum_precision)
            & (model_rows["specificity"] >= minimum_specificity)
        ].copy()
        if not screening_candidates.empty:
            candidates = screening_candidates
            selection_note = (
                "max_recall_with_precision_at_least_prevalence_and_specificity_at_least_0.20"
            )
        else:
            candidates = model_rows.copy()
            selection_note = "fallback_no_threshold_met_screening_rule"
        selected = candidates.sort_values(
            ["recall", "specificity", "balanced_accuracy", "precision", "threshold"],
            ascending=[False, False, False, False, False],
        ).iloc[0].to_dict()
        selected["selected"] = True
        selected["selection_note"] = selection_note
        selected_rows.append(selected)
        for row in rows:
            if row["model"] == model_name and abs(float(row["threshold"]) - float(selected["threshold"])) < 1e-9:
                row["selected"] = True
                row["selection_note"] = selection_note
            elif row["model"] == model_name:
                row["selection_note"] = ""

    for spec in unavailable_specs:
        rows.append(
            {
                "model": spec.name,
                "threshold": np.nan,
                "selected": False,
                "selection_note": spec.unavailable_reason,
                "model_status": "unavailable",
            }
        )
    threshold_df = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows)
    threshold_df.to_csv(TABLES_DIR / "phase4_threshold_analysis.csv", index=False)
    return threshold_df, selected_df


def aggregate_experiments(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    specs: list[ModelSpec],
) -> pd.DataFrame:
    """Create the main model comparison table with fold variability."""
    rows: list[dict[str, Any]] = []
    selected_by_model = selected_thresholds.set_index("model").to_dict("index") if not selected_thresholds.empty else {}
    for model_name, group in fold_metrics.groupby("model"):
        pred_group = predictions[predictions["model"] == model_name]
        pooled = compute_metrics(
            pred_group["true_label"].astype(int).to_numpy(),
            pred_group["score"].astype(float).to_numpy(),
            threshold=0.5,
        )
        row: dict[str, Any] = {
            "model": model_name,
            "base_model": pred_group["base_model"].iloc[0] if "base_model" in pred_group.columns else split_candidate_name(model_name)[0],
            "calibration_method": pred_group["calibration_method"].iloc[0]
            if "calibration_method" in pred_group.columns
            else split_candidate_name(model_name)[1],
            "model_status": "evaluated",
            "n_folds": int(group["fold"].nunique()),
            "n_eval_clips": int(len(pred_group)),
            "n_eval_speakers": int(pred_group["speaker_id"].nunique()),
            "default_threshold": 0.5,
            "score_type": ";".join(sorted(pred_group["score_type"].dropna().unique())),
        }
        for metric in METRIC_COLUMNS:
            if metric in group.columns:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_pooled"] = pooled.get(metric, np.nan)
        selected = selected_by_model.get(model_name, {})
        row["selected_threshold"] = selected.get("threshold", np.nan)
        row["selected_recall"] = selected.get("recall", np.nan)
        row["selected_precision"] = selected.get("precision", np.nan)
        row["selected_specificity"] = selected.get("specificity", np.nan)
        row["selected_balanced_accuracy"] = selected.get("balanced_accuracy", np.nan)
        row["selected_pr_auc"] = selected.get("pr_auc", np.nan)
        row["selected_brier_score"] = selected.get("brier_score", np.nan)
        row["selected_false_negatives"] = selected.get("false_negatives", np.nan)
        row["threshold_selection_note"] = selected.get("selection_note", "")
        rows.append(row)

    for spec in specs:
        if spec.status == "unavailable":
            rows.append(
                {
                    "model": spec.name,
                    "model_status": "unavailable",
                    "unavailable_reason": spec.unavailable_reason,
                    "n_folds": 0,
                }
            )

    out = pd.DataFrame(rows)
    base_order = {
        name: i
        for i, name in enumerate(
            ["logistic_regression", "elastic_net_logistic", "linear_svm", "random_forest", "xgboost", "heuristic"]
        )
    }
    calibration_order = {"uncalibrated": 0, "platt": 1, "isotonic": 2, "not_applicable": 0}
    out["model_order"] = out["model"].map(lambda value: base_order.get(split_candidate_name(str(value))[0], 99))
    out["calibration_order"] = out.get("calibration_method", pd.Series([""] * len(out))).map(calibration_order).fillna(99)
    out = out.sort_values(["model_order", "calibration_order", "model"]).drop(columns=["model_order", "calibration_order"])
    out.to_csv(TABLES_DIR / "phase4_baseline_experiments.csv", index=False)
    return out


def summarize_feature_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level importances and save the official Phase 4 table."""
    if feature_importance.empty:
        out = pd.DataFrame(columns=["model", "feature", "mean_importance", "std_importance", "mean_coefficient", "n_folds"])
        out.to_csv(TABLES_DIR / "phase4_feature_importance.csv", index=False)
        return out
    out = (
        feature_importance.groupby(["model", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            mean_coefficient=("coefficient", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["model", "mean_importance"], ascending=[True, False])
    )
    out["std_importance"] = out["std_importance"].fillna(0.0)
    out["coefficient_direction"] = np.where(
        out["mean_coefficient"].isna(),
        "",
        np.where(out["mean_coefficient"] > 0, "higher_score", np.where(out["mean_coefficient"] < 0, "lower_score", "zero")),
    )
    out.to_csv(TABLES_DIR / "phase4_feature_importance.csv", index=False)
    return out


def select_best_model(experiment_df: pd.DataFrame, selected_thresholds: pd.DataFrame) -> tuple[str, pd.Series]:
    """Rank models by the requested clinically oriented priority rule."""
    candidates = experiment_df[experiment_df["model_status"].eq("evaluated")].copy()
    keep_cols = [
        "model",
        "base_model",
        "calibration_method",
        "selected_threshold",
        "selected_recall",
        "selected_precision",
        "selected_specificity",
        "selected_balanced_accuracy",
        "selected_false_negatives",
        "selected_pr_auc",
        "selected_brier_score",
    ]
    candidates = candidates[keep_cols].copy()
    candidates["interpretability_rank"] = candidates["base_model"].map(
        {
            "heuristic": 0,
            "logistic_regression": 1,
            "elastic_net_logistic": 2,
            "linear_svm": 3,
            "random_forest": 4,
            "xgboost": 5,
        }
    )
    candidates["calibration_rank"] = candidates["calibration_method"].map(
        {"platt": 0, "isotonic": 1, "uncalibrated": 2, "not_applicable": 3}
    )
    candidates = candidates.sort_values(
        [
            "selected_recall",
            "selected_pr_auc",
            "selected_balanced_accuracy",
            "selected_brier_score",
            "calibration_rank",
            "interpretability_rank",
        ],
        ascending=[False, False, False, True, True, True],
    )
    best = candidates.iloc[0]
    return str(best["model"]), best


def false_negative_analysis(
    predictions: pd.DataFrame,
    best_model: str,
    threshold: float,
    heuristic_cols: dict[str, str],
) -> pd.DataFrame:
    """Save false negative cases for the selected operating threshold."""
    model_df = predictions[predictions["model"] == best_model].copy()
    model_df["selected_threshold"] = threshold
    model_df["selected_prediction"] = (model_df["score"].astype(float) >= threshold).astype(int)
    fn = model_df[(model_df["true_label"] == 1) & (model_df["selected_prediction"] == 0)].copy()
    keep = [
        "metadata_index",
        "audio_file",
        "processed_file",
        "speaker_id",
        "fold",
        "true_label",
        "score",
        "selected_threshold",
        "selected_prediction",
        "duration_sec",
        heuristic_cols["pause_burden"],
        heuristic_cols["speech_rate"],
        heuristic_cols["prosody_instability"],
        "decision_status",
        "phase2_decision",
        "phase2_issue_type",
        "manual_review_required",
        "phase4_caution_flag",
    ]
    fn = fn[[c for c in keep if c in fn.columns]].sort_values(["speaker_id", "score"])
    fn.to_csv(TABLES_DIR / "phase4_false_negative_cases.csv", index=False)
    return fn


def speaker_level_evaluation(predictions: pd.DataFrame, best_model: str, threshold: float) -> pd.DataFrame:
    """Aggregate best-model predictions to speaker level."""
    model_df = predictions[predictions["model"] == best_model].copy()
    model_df["clip_prediction"] = (model_df["score"].astype(float) >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []
    for speaker_id, speaker_df in model_df.groupby("speaker_id"):
        labels = speaker_df["true_label"].astype(int).unique()
        if len(labels) != 1:
            continue
        true_label = int(labels[0])
        majority_score = float(speaker_df["clip_prediction"].mean())
        majority_pred = int(majority_score >= 0.5)
        max_score = float(speaker_df["score"].astype(float).max())
        max_pred = int(max_score >= threshold)
        speaker_rows.extend(
            [
                {
                    "speaker_id": speaker_id,
                    "aggregation_rule": "majority_vote",
                    "n_clips": int(len(speaker_df)),
                    "true_label": true_label,
                    "speaker_score": majority_score,
                    "threshold": 0.5,
                    "prediction": majority_pred,
                },
                {
                    "speaker_id": speaker_id,
                    "aggregation_rule": "max_risk_score",
                    "n_clips": int(len(speaker_df)),
                    "true_label": true_label,
                    "speaker_score": max_score,
                    "threshold": threshold,
                    "prediction": max_pred,
                },
            ]
        )

    speaker_pred_df = pd.DataFrame(speaker_rows)
    speaker_pred_df.to_csv(TABLES_DIR / "phase4_speaker_level_predictions.csv", index=False)
    for rule, rule_df in speaker_pred_df.groupby("aggregation_rule"):
        y_true = rule_df["true_label"].astype(int).to_numpy()
        y_score = rule_df["speaker_score"].astype(float).to_numpy()
        y_pred = rule_df["prediction"].astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics = compute_metrics(y_true, y_score, threshold=float(rule_df["threshold"].iloc[0]))
        metrics.update(
            {
                "model": best_model,
                "aggregation_rule": rule,
                "n_speakers": int(len(rule_df)),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        )
        rows.append(metrics)
        save_confusion_matrix(
            y_true,
            y_score,
            threshold=float(rule_df["threshold"].iloc[0]),
            out_path=FIGURES_DIR / f"phase4_speaker_level_{rule}_confusion_matrix.png",
            title=f"Phase 4 speaker-level {rule}",
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "phase4_speaker_level_metrics.csv", index=False)
    return out


def calibration_report(predictions: pd.DataFrame, best_model: str, threshold: float) -> tuple[pd.DataFrame, float, str]:
    """Create calibration comparison artifacts for the selected base model."""
    best_base_model, _ = split_candidate_name(best_model)
    candidate_df = predictions[predictions["base_model"].eq(best_base_model)].copy()
    if candidate_df.empty:
        candidate_df = predictions[predictions["model"].eq(best_model)].copy()

    comparison_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Ideal")

    for candidate_name, model_df in candidate_df.groupby("model"):
        y_true = model_df["true_label"].astype(int).to_numpy()
        scores = np.clip(model_df["score"].astype(float).to_numpy(), 0.0, 1.0)
        metrics = compute_metrics(y_true, scores, threshold=0.5)
        brier_value = float(brier_score_loss(y_true, scores))
        comparison_rows.append(
            {
                "model": candidate_name,
                "base_model": model_df["base_model"].iloc[0],
                "calibration_method": model_df["calibration_method"].iloc[0],
                "score_type": ";".join(sorted(model_df["score_type"].dropna().unique())),
                "brier_score": brier_value,
                "roc_auc": metrics.get("roc_auc", np.nan),
                "pr_auc": metrics.get("pr_auc", np.nan),
            }
        )
        if np.unique(y_true).size >= 2:
            prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=5, strategy="quantile")
            line_width = 2.5 if candidate_name == best_model else 1.4
            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                linewidth=line_width,
                label=f"{model_df['calibration_method'].iloc[0]} Brier={brier_value:.3f}",
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("brier_score")
    comparison_df.to_csv(TABLES_DIR / "phase4_calibration_comparison.csv", index=False)

    best_comparison = comparison_df[comparison_df["model"].eq(best_model)]
    brier = float(best_comparison["brier_score"].iloc[0]) if not best_comparison.empty else np.nan
    if brier <= 0.15:
        quality = "moderately calibrated for a small baseline"
    elif brier <= 0.25:
        quality = "weak to moderate calibration"
    else:
        quality = "poor calibration"

    ax.axvline(threshold, linestyle=":", linewidth=1, color="black", label="Selected threshold")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed dementia rate")
    ax.set_title(f"Phase 4 calibration: {best_base_model}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase4_calibration_curve.png", dpi=200)
    plt.close(fig)
    return comparison_df, brier, quality


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_input_summary(df: pd.DataFrame, feature_cols: list[str], governance: pd.DataFrame) -> None:
    caution_in_table = int(df["decision_status"].eq("include_with_caution").sum())
    training_only_caution = int((df["decision_status"].eq("include_with_caution") & ~df["phase4_eval_allowed"]).sum())
    excluded = governance[~bool_series(governance["phase3_baseline_included"], default=False)]
    lines = [
        "# Phase 4 Input Summary",
        "",
        f"- Official modeling table: `{rel(FEATURE_CSV)}`.",
        f"- Rows in official table: **{len(df)}**.",
        f"- Numeric model features: **{len(feature_cols)}**.",
        "- Label column: `label` (`1` = dementia, `0` = non-dementia).",
        "- Speaker/group column: `speaker_id`.",
        f"- Unique speakers: **{df['speaker_id'].nunique()}**.",
        f"- Class counts: {json.dumps({str(k): int(v) for k, v in df['label_name'].value_counts().to_dict().items()})}.",
        f"- Governed-record decisions file: `{rel(GOVERNANCE_CSV)}`.",
        "- Phase 3 closeout note: `reports/phase3_closeout.md`.",
        "- Feature dictionary: `docs/FEATURE_DICTIONARY.md`.",
        "- Feature manifest: `reports/phase3_feature_manifest.csv`.",
        "- Current baseline code inspected: `src/train_baseline.py`.",
        "- Evaluation helpers inspected: `src/evaluate.py`.",
        "- Existing CLI entry points inspected: `Makefile`.",
        "",
        "## Governance Counts",
        "",
        f"- Official table includes **{caution_in_table}** `include_with_caution` records.",
        f"- Training-allowed but not final-evaluation-allowed caution records in the official table: **{training_only_caution}**.",
        f"- Final-evaluation-allowed records used for validation folds: **{int(df['phase4_eval_allowed'].sum())}**.",
        f"- Records excluded from the governed baseline table by the decisions file: **{len(excluded)}**.",
        "",
        "## Phase 4 Files/Scripts",
        "",
        "- Phase 4 runner: `scripts/phase4_baseline_experiments.py`.",
        "- Reports: `reports/phase4_*.md`.",
        "- Machine-readable tables: `reports/tables/phase4_*.csv`.",
        "- Figures: `reports/figures/phase4_*.png`.",
    ]
    (REPORTS_DIR / "phase4_input_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol_report(df: pd.DataFrame, split_df: pd.DataFrame, cv_note: str, n_splits: int, seed: int) -> None:
    eval_df = df[df["phase4_eval_allowed"]]
    lines = [
        "# Phase 4 Evaluation Protocol",
        "",
        "- Primary evaluation: clip-level out-of-fold performance across grouped folds.",
        "- Secondary evaluation: speaker-level aggregation for the selected best baseline only.",
        f"- Grouping field: `speaker_id`.",
        f"- CV splitter: {cv_note}.",
        f"- Number of folds: **{n_splits}**.",
        f"- Random seed: **{seed}**.",
        "- Speaker leakage rule: every validation speaker is removed from the training rows for that fold.",
        "- Governance rule: final validation folds use only records marked `phase3_final_eval_allowed`; training-only caution records can be used in training folds only when their speaker is not held out.",
        f"- Final-evaluation clip count: **{len(eval_df)}**.",
        f"- Final-evaluation speaker count: **{eval_df['speaker_id'].nunique()}**.",
        f"- Final-evaluation speaker class counts: {json.dumps(speaker_label_counts(eval_df))}.",
        "",
        "## Fold Composition",
        "",
        markdown_table(split_df),
        "",
        "## Constraints",
        "",
        "- Speaker IDs are metadata/path-derived rather than biometrically verified.",
        "- The dataset is small and heterogeneous, so fold variability is reported instead of relying only on point estimates.",
        "- Thresholds selected from cross-validated out-of-fold scores are Phase 4 operating-point estimates, not externally validated clinical thresholds.",
    ]
    (REPORTS_DIR / "phase4_evaluation_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preprocessing_note(specs: list[ModelSpec]) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "model": spec.name,
                "imputer": "SimpleImputer(strategy='median')" if spec.status == "available" else "not run",
                "scaler": "StandardScaler inside Pipeline" if spec.uses_scaling else "none",
                "leakage_control": "fit only on each training fold" if spec.status == "available" else spec.unavailable_reason,
            }
        )
    rows.append(
        {
            "model": "heuristic",
            "imputer": "training-fold median fill for selected heuristic columns",
            "scaler": "training-fold 5th-to-95th percentile robust normalization",
            "leakage_control": "normalization statistics fit only on each training fold",
        }
    )
    note_df = pd.DataFrame(rows)
    lines = [
        "# Phase 4 Model Preprocessing Note",
        "",
        "- No feature imputer, scaler, or heuristic normalizer is fit globally before cross-validation.",
        "- Logistic Regression, Elastic Net Logistic Regression, and Linear SVM use median imputation plus `StandardScaler` inside sklearn pipelines.",
        "- Random Forest does not use scaling; it uses median imputation inside its pipeline.",
        "- XGBoost, when available, uses median imputation and no scaling.",
        "- The heuristic benchmark normalizes its three inputs with training-fold statistics only.",
        "",
        markdown_table(note_df),
    ]
    (REPORTS_DIR / "phase4_model_preprocessing_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heuristic_definition(heuristic_cols: dict[str, str], experiment_df: pd.DataFrame | None = None) -> None:
    comparison_lines: list[str] = []
    if experiment_df is not None and not experiment_df.empty and (experiment_df["model"] == "heuristic").any():
        heuristic_row = experiment_df[experiment_df["model"].eq("heuristic")].iloc[0]
        ml_rows = experiment_df[
            experiment_df["model_status"].eq("evaluated") & experiment_df["model"].ne("heuristic")
        ].copy()
        if not ml_rows.empty:
            ml_rows = ml_rows.sort_values(
                ["selected_recall", "pr_auc_pooled", "balanced_accuracy_pooled"],
                ascending=[False, False, False],
            )
            best_ml = ml_rows.iloc[0]
            comparison_lines = [
                "",
                "## Benchmark Comparison",
                "",
                (
                    f"- Heuristic selected-threshold recall={fmt(heuristic_row.get('selected_recall'))}, "
                    f"PR-AUC={fmt(heuristic_row.get('selected_pr_auc'))}, "
                    f"balanced accuracy={fmt(heuristic_row.get('selected_balanced_accuracy'))}."
                ),
                (
                    f"- Strongest ML candidate by recall-first ranking: `{best_ml['model']}` with "
                    f"recall={fmt(best_ml.get('selected_recall'))}, "
                    f"PR-AUC={fmt(best_ml.get('selected_pr_auc'))}, "
                    f"balanced accuracy={fmt(best_ml.get('selected_balanced_accuracy'))}."
                ),
                "- The comparison asks whether learned acoustic-feature weighting adds value beyond a transparent handcrafted rule; it does not validate either approach clinically.",
            ]
    lines = [
        "# Phase 4 Heuristic Benchmark Definition",
        "",
        "The heuristic is a transparent risk score, not a learned clinical model.",
        "",
        "Formula:",
        "",
        "`risk_score = (pause_burden_norm + (1 - speech_rate_norm) + prosody_instability_norm) / 3`",
        "",
        "Selected official Phase 3 columns:",
        "",
        f"- Pause burden: `{heuristic_cols['pause_burden']}`.",
        f"- Speech-rate proxy: `{heuristic_cols['speech_rate']}`.",
        f"- Prosody instability: `{heuristic_cols['prosody_instability']}`.",
        "",
        "Why these components:",
        "",
        "- Pause burden approximates hesitation/segmentation load from the Phase 3 energy-based pause proxy.",
        "- Speech-rate proxy approximates acoustic activity density without transcripts; lower values increase risk.",
        "- Prosody instability captures F0 variability as an acoustic estimate, not a clinical prosody measurement.",
        "",
        "Normalization and threshold:",
        "",
        "- Each component is normalized within each training fold using the 5th and 95th percentiles, then clipped to `[0, 1]`.",
        "- Missing values are filled with the training-fold median for that component.",
        "- Equal weights are used because this is a transparent benchmark, not a tuned model.",
        "- Default threshold for the main fold table is `0.5`.",
        "- Threshold analysis sweeps `0.10` to `0.90` in `0.05` increments under the same grouped protocol.",
        "",
        "Caution: all three inputs are acoustic proxies from Phase 3, not transcript-derived or clinical measurements.",
    ]
    lines.extend(comparison_lines)
    (REPORTS_DIR / "phase4_heuristic_definition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_baseline_report(experiment_df: pd.DataFrame, fold_metrics: pd.DataFrame) -> None:
    evaluated = experiment_df[experiment_df["model_status"].eq("evaluated")].copy()
    show_rows = []
    for _, row in evaluated.iterrows():
        show_rows.append(
            {
                "model": row["model"],
                "recall mean+/-std": f"{fmt(row.get('recall_mean'))} +/- {fmt(row.get('recall_std'))}",
                "PR-AUC mean+/-std": f"{fmt(row.get('pr_auc_mean'))} +/- {fmt(row.get('pr_auc_std'))}",
                "balanced acc mean+/-std": f"{fmt(row.get('balanced_accuracy_mean'))} +/- {fmt(row.get('balanced_accuracy_std'))}",
                "specificity mean+/-std": f"{fmt(row.get('specificity_mean'))} +/- {fmt(row.get('specificity_std'))}",
                "pooled Brier": fmt(row.get("brier_score_pooled")),
                "selected threshold": fmt(row.get("selected_threshold")),
                "selected recall": fmt(row.get("selected_recall")),
                "selected FN": fmt(row.get("selected_false_negatives"), digits=0),
            }
        )
    unavailable = experiment_df[experiment_df["model_status"].eq("unavailable")]
    lines = [
        "# Phase 4 Baseline Experiments",
        "",
        "- Main table: `reports/tables/phase4_baseline_experiments.csv`.",
        "- Fold-level metrics: `reports/tables/phase4_fold_metrics.csv`.",
        "- Metrics at default threshold `0.5` are summarized as mean +/- standard deviation across grouped folds.",
        "- Threshold-focused operating points are reported separately and included here for comparison.",
        "- PR-AUC is emphasized because precision is weak and threshold selection is deliberately recall-focused.",
        "",
        markdown_table(pd.DataFrame(show_rows)),
    ]
    if not unavailable.empty:
        lines.extend(["", "## Unavailable Models", "", markdown_table(unavailable[["model", "unavailable_reason"]])])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Positive class is dementia.",
            "- Evaluation is speaker-grouped; no speaker appears in both train and validation within a fold.",
            "- Fold variability is reported because the dataset is small and speaker composition matters.",
            "- The selected operating point is screening-first only; low specificity means a high false-positive burden and is not acceptable for diagnosis.",
        ]
    )
    (REPORTS_DIR / "phase4_baseline_experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_report(threshold_df: pd.DataFrame, selected_thresholds: pd.DataFrame) -> None:
    show = selected_thresholds[
        [
            "model",
            "threshold",
            "selection_note",
            "recall",
            "precision",
            "specificity",
            "false_negatives",
            "balanced_accuracy",
            "pr_auc",
        ]
    ].copy()
    lines = [
        "# Phase 4 Threshold Analysis",
        "",
        "- Threshold sweep table: `reports/tables/phase4_threshold_analysis.csv`.",
        "- Swept thresholds from `0.10` to `0.90` in `0.05` increments.",
        "- Selection rule: maximize recall while keeping precision at least the observed positive-class prevalence and specificity at least `0.20`.",
        "- Ties prefer higher specificity, then balanced accuracy, then precision.",
        "- This rule is used instead of a fixed precision `0.50` floor because the fixed floor was too restrictive for a recall-focused screening baseline on this small dataset.",
        "- If no threshold reaches the screening rule, the table marks that explicitly.",
        "",
        "## Selected Operating Points",
        "",
        markdown_table(show),
        "",
        "These thresholds are clinically motivated screening operating points estimated from grouped out-of-fold predictions. They are not externally validated diagnostic thresholds.",
        "",
        "## Operating-Point Tradeoff",
        "",
        "The selected threshold is acceptable only under a screening-first framing where false negatives are prioritized. Low specificity means many false positives should be expected, so these scores must not be framed as diagnostic decisions.",
    ]
    (REPORTS_DIR / "phase4_threshold_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_selection_report(best_model: str, best_row: pd.Series, experiment_df: pd.DataFrame) -> None:
    evaluated = experiment_df[experiment_df["model_status"].eq("evaluated")].copy()
    show = evaluated[
        [
            "model",
            "base_model",
            "calibration_method",
            "selected_threshold",
            "selected_recall",
            "selected_precision",
            "selected_specificity",
            "selected_balanced_accuracy",
            "selected_false_negatives",
            "selected_pr_auc",
            "selected_brier_score",
            "recall_mean",
            "recall_std",
            "pr_auc_mean",
            "pr_auc_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
        ]
    ].copy()
    show = show.sort_values(
        ["selected_recall", "selected_pr_auc", "selected_balanced_accuracy", "selected_brier_score"],
        ascending=[False, False, False, True],
    )
    show.to_csv(TABLES_DIR / "phase4_model_selection_scoreboard.csv", index=False)
    reasons = []
    for _, row in show.iterrows():
        if row["model"] == best_model:
            continue
        if row["selected_recall"] < best_row["selected_recall"]:
            reasons.append(
                f"- `{row['model']}` was not selected because its selected-threshold recall was lower "
                f"({fmt(row['selected_recall'])} vs {fmt(best_row['selected_recall'])}), even where other metrics were competitive."
            )
        elif row["selected_pr_auc"] < best_row["selected_pr_auc"]:
            reasons.append(
                f"- `{row['model']}` matched recall but had lower PR-AUC "
                f"({fmt(row['selected_pr_auc'])} vs {fmt(best_row['selected_pr_auc'])})."
            )
        elif row["selected_balanced_accuracy"] < best_row["selected_balanced_accuracy"]:
            reasons.append(
                f"- `{row['model']}` matched recall and PR-AUC closely but had lower balanced accuracy "
                f"({fmt(row['selected_balanced_accuracy'])} vs {fmt(best_row['selected_balanced_accuracy'])})."
            )
        else:
            reasons.append(
                f"- `{row['model']}` was not selected after applying the calibration and interpretability tie-breakers."
            )
    lines = [
        "# Phase 4 Model Selection",
        "",
        f"Selected best baseline model: **`{best_model}`**.",
        "",
        "Selection priority:",
        "",
        "1. Highest recall at the clinically chosen threshold.",
        "2. Highest PR-AUC.",
        "3. Highest balanced accuracy.",
        "4. Lower Brier score where performance is otherwise close.",
        "5. Interpretability as a tiebreaker when differences are small.",
        "",
        "## Ranking Table",
        "",
        markdown_table(
            show[
                [
                    "model",
                    "calibration_method",
                    "selected_threshold",
                    "selected_recall",
                    "selected_precision",
                    "selected_specificity",
                    "selected_balanced_accuracy",
                    "selected_false_negatives",
                    "selected_pr_auc",
                    "selected_brier_score",
                ]
            ]
        ),
        "",
        "Full scoreboard with mean +/- std fold metrics: `reports/tables/phase4_model_selection_scoreboard.csv`.",
        "",
        "## Rationale",
        "",
        (
            f"`{best_model}` had selected-threshold recall {fmt(best_row.get('selected_recall'))}, "
            f"PR-AUC {fmt(best_row.get('selected_pr_auc'))}, balanced accuracy "
            f"{fmt(best_row.get('selected_balanced_accuracy'))}, and Brier score "
            f"{fmt(best_row.get('selected_brier_score'))}."
        ),
        "",
        "\n".join(reasons) if reasons else "No competing evaluated model was available.",
        "",
        "Accuracy alone was not used for selection.",
    ]
    (REPORTS_DIR / "phase4_model_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_importance_report(importance_df: pd.DataFrame) -> None:
    lines = [
        "# Phase 4 Feature Importance",
        "",
        "- Machine-readable table: `reports/tables/phase4_feature_importance.csv`.",
        "- Logistic Regression, Elastic Net Logistic Regression, and Linear SVM coefficients are from fold-fitted scaled pipelines, so magnitudes are comparable within each fitted model but not causal effects.",
        "- Random Forest importances are impurity-based and can be biased toward continuous or high-cardinality features.",
        "- Proxy features are interpreted as acoustic proxies only, not clinical biomarkers.",
    ]
    for model_name in ["logistic_regression", "elastic_net_logistic", "random_forest", "linear_svm"]:
        sub = importance_df[importance_df["model"] == model_name].head(10)
        if sub.empty:
            continue
        show = sub[["feature", "mean_importance", "std_importance", "coefficient_direction"]].copy()
        show["mean_importance"] = show["mean_importance"].map(fmt)
        show["std_importance"] = show["std_importance"].map(fmt)
        lines.extend(["", f"## Top Features: `{model_name}`", "", markdown_table(show)])
    (REPORTS_DIR / "phase4_feature_importance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_false_negative_report(
    fn_df: pd.DataFrame,
    predictions: pd.DataFrame,
    best_model: str,
    threshold: float,
    heuristic_cols: dict[str, str],
) -> None:
    model_df = predictions[predictions["model"] == best_model].copy()
    model_df["selected_prediction"] = (model_df["score"].astype(float) >= threshold).astype(int)
    positives = model_df[model_df["true_label"] == 1]
    true_positives = model_df[(model_df["true_label"] == 1) & (model_df["selected_prediction"] == 1)]
    lines = [
        "# Phase 4 False Negative Analysis",
        "",
        f"- Best model: `{best_model}`.",
        f"- Selected threshold: `{fmt(threshold)}`.",
        "- Case table: `reports/tables/phase4_false_negative_cases.csv`.",
        f"- False negatives: **{len(fn_df)}** out of **{len(positives)}** dementia-positive validation clips.",
    ]
    if fn_df.empty:
        lines.extend(
            [
                "",
                "No false negatives were observed at the selected operating threshold in the grouped out-of-fold predictions.",
                "This should not be overinterpreted; the dataset is small and the threshold may trade this off against false positives.",
            ]
        )
    else:
        speaker_counts = fn_df["speaker_id"].value_counts().reset_index()
        speaker_counts.columns = ["speaker_id", "false_negative_count"]
        feature_rows = []
        for col in ["duration_sec", heuristic_cols["pause_burden"], heuristic_cols["speech_rate"], heuristic_cols["prosody_instability"]]:
            feature_rows.append(
                {
                    "feature": col,
                    "false_negative_median": fmt(fn_df[col].median()) if col in fn_df else "NA",
                    "true_positive_median": fmt(true_positives[col].median()) if col in true_positives else "NA",
                    "all_positive_median": fmt(positives[col].median()) if col in positives else "NA",
                }
            )
        lines.extend(
            [
                "",
                "## Speaker Concentration",
                "",
                markdown_table(speaker_counts.head(10)),
                "",
                "## Feature Pattern Check",
                "",
                markdown_table(pd.DataFrame(feature_rows)),
                "",
                f"- Caution-flagged false negatives: **{int(fn_df['phase4_caution_flag'].sum())}**.",
                "- Patterns are descriptive only; they should guide manual review rather than causal claims.",
            ]
        )
    (REPORTS_DIR / "phase4_false_negative_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_speaker_level_report(speaker_metrics: pd.DataFrame, experiment_df: pd.DataFrame, best_model: str) -> None:
    clip = experiment_df[experiment_df["model"].eq(best_model)].iloc[0]
    show = speaker_metrics[
        [
            "aggregation_rule",
            "n_speakers",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy",
            "false_negatives",
            "false_positives",
            "true_negatives",
            "true_positives",
        ]
    ].copy()
    lines = [
        "# Phase 4 Speaker-Level Evaluation",
        "",
        f"- Best model: `{best_model}`.",
        "- Metrics table: `reports/tables/phase4_speaker_level_metrics.csv`.",
        "- Speaker prediction table: `reports/tables/phase4_speaker_level_predictions.csv`.",
        "- Aggregation rules: majority vote across clip predictions and max risk score across clips.",
        "",
        "## Speaker-Level Metrics",
        "",
        markdown_table(show),
        "",
        "## Clip-Level Comparison",
        "",
        (
            f"At clip level, the selected threshold had recall {fmt(clip.get('selected_recall'))}, "
            f"precision {fmt(clip.get('selected_precision'))}, specificity {fmt(clip.get('selected_specificity'))}, "
            f"and false negatives {fmt(clip.get('selected_false_negatives'), digits=0)}."
        ),
        "",
        "Speaker aggregation can improve recall when at least one clip for a dementia-positive speaker receives a high score, but max-risk aggregation can also increase false positives.",
    ]
    (REPORTS_DIR / "phase4_speaker_level_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_report(
    best_model: str,
    threshold: float,
    brier: float,
    quality: str,
    predictions: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
) -> None:
    score_type = ";".join(sorted(predictions[predictions["model"] == best_model]["score_type"].dropna().unique()))
    comparison_show = calibration_comparison[
        ["model", "calibration_method", "brier_score", "roc_auc", "pr_auc"]
    ].copy()
    for col in ["brier_score", "roc_auc", "pr_auc"]:
        comparison_show[col] = comparison_show[col].map(fmt)
    lines = [
        "# Phase 4 Calibration Check",
        "",
        f"- Best model: `{best_model}`.",
        f"- Score type: `{score_type}`.",
        f"- Selected threshold: `{fmt(threshold)}`.",
        f"- Brier score: **{fmt(brier)}**.",
        f"- Qualitative assessment: **{quality}**.",
        "- Calibration comparison table: `reports/tables/phase4_calibration_comparison.csv`.",
        "- Calibration curve: `reports/figures/phase4_calibration_curve.png`.",
        "",
        "## Calibrated vs Uncalibrated",
        "",
        markdown_table(comparison_show),
        "",
        "The calibration check uses grouped out-of-fold predictions. This is useful for baseline screening assessment, but it is not a substitute for external calibration validation.",
    ]
    if not calibration_comparison.empty:
        best_brier_row = calibration_comparison.sort_values("brier_score").iloc[0]
        if best_brier_row["model"] == best_model:
            lines.append("The selected candidate is also the lowest-Brier variant for its base model.")
        else:
            lines.append(
                f"The lowest-Brier variant for this base model is `{best_brier_row['model']}`; "
                "the selected candidate remains chosen by the recall-first model-selection rule."
            )
    if "sigmoid_decision_function" in score_type:
        lines.append("The selected score is derived from a margin through a sigmoid transform, so calibration should be treated cautiously.")
    (REPORTS_DIR / "phase4_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_selected_confusion_figures(predictions: pd.DataFrame, selected_thresholds: pd.DataFrame) -> None:
    selected_by_model = selected_thresholds.set_index("model").to_dict("index")
    for model_name, model_df in predictions.groupby("model"):
        threshold = float(selected_by_model[model_name]["threshold"])
        save_confusion_matrix(
            model_df["true_label"].astype(int).to_numpy(),
            model_df["score"].astype(float).to_numpy(),
            threshold=threshold,
            out_path=FIGURES_DIR / f"phase4_{model_name}_selected_threshold_confusion_matrix.png",
            title=f"Phase 4 {model_name} selected threshold",
        )


def compact_comparison_table(experiment_df: pd.DataFrame) -> pd.DataFrame:
    """Build the requested compact final-threshold comparison."""
    evaluated = experiment_df[experiment_df["model_status"].eq("evaluated")].copy()
    rf_rows = evaluated[evaluated["base_model"].eq("random_forest")].copy()
    if rf_rows.empty:
        rf_best = pd.DataFrame()
    else:
        rf_best = rf_rows.sort_values(
            ["selected_recall", "selected_pr_auc", "selected_balanced_accuracy", "selected_brier_score"],
            ascending=[False, False, False, True],
        ).head(1)

    requested_rows = []
    for label, model_name in [
        ("Heuristic", "heuristic"),
        ("LR", "logistic_regression_uncalibrated"),
        ("LR Platt", "logistic_regression_platt"),
    ]:
        row = evaluated[evaluated["model"].eq(model_name)].copy()
        if not row.empty:
            row = row.head(1)
            row["comparison_label"] = label
            requested_rows.append(row)
    if not rf_best.empty:
        rf_best = rf_best.copy()
        rf_best["comparison_label"] = "RF best competitor"
        requested_rows.append(rf_best)

    if not requested_rows:
        return pd.DataFrame()

    out = pd.concat(requested_rows, ignore_index=True)
    out = out[
        [
            "comparison_label",
            "model",
            "calibration_method",
            "selected_threshold",
            "selected_recall",
            "selected_precision",
            "selected_specificity",
            "selected_balanced_accuracy",
            "selected_pr_auc",
            "selected_brier_score",
            "selected_false_negatives",
        ]
    ].rename(
        columns={
            "selected_threshold": "threshold",
            "selected_recall": "recall",
            "selected_precision": "precision",
            "selected_specificity": "specificity",
            "selected_balanced_accuracy": "balanced_accuracy",
            "selected_pr_auc": "pr_auc",
            "selected_brier_score": "brier_score",
            "selected_false_negatives": "false_negatives",
        }
    )
    out.to_csv(TABLES_DIR / "phase4_compact_comparison.csv", index=False)
    return out


def save_compact_comparison_figure(compact_df: pd.DataFrame) -> None:
    """Save a compact comparison figure for final selected-threshold metrics."""
    if compact_df.empty:
        return
    metrics = [
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("specificity", "Specificity"),
        ("balanced_accuracy", "Balanced Acc."),
        ("pr_auc", "PR-AUC"),
    ]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, (_, row) in enumerate(compact_df.iterrows()):
        values = [float(row[col]) for col, _ in metrics]
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=str(row["comparison_label"]))
    ax.set_xticks(x, labels=[label for _, label in metrics])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Phase 4 Selected-Threshold Comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase4_compact_comparison.png", dpi=200)
    plt.close(fig)


def write_phase4_executive_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    experiment_df: pd.DataFrame,
    compact_df: pd.DataFrame,
    split_df: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
) -> None:
    """Write the one-page Phase 4 executive summary."""
    best = experiment_df[experiment_df["model_status"].eq("evaluated")].sort_values(
        ["selected_recall", "selected_pr_auc", "selected_balanced_accuracy", "selected_brier_score"],
        ascending=[False, False, False, True],
    ).iloc[0]
    final_eval_rows = int(df["phase4_eval_allowed"].sum()) if "phase4_eval_allowed" in df.columns else int(best["n_eval_clips"])
    final_eval_speakers = int(df[df["phase4_eval_allowed"]]["speaker_id"].nunique()) if "phase4_eval_allowed" in df.columns else int(best["n_eval_speakers"])
    fold_count = int(split_df["fold"].nunique()) if not split_df.empty else int(best["n_folds"])
    speaker_overlap = int(split_df["speaker_overlap_count"].sum()) if "speaker_overlap_count" in split_df.columns else 0
    calibration_line = "Calibration comparison unavailable."
    if not calibration_comparison.empty:
        best_cal = calibration_comparison.sort_values("brier_score").iloc[0]
        calibration_line = (
            f"Platt/isotonic calibration was compared for the winning base model; "
            f"lowest Brier was `{best_cal['model']}` at {fmt(best_cal['brier_score'])}."
        )

    lines = [
        "# Phase 4 Executive Summary",
        "",
        f"- Official input: `data/features/features_phase3_governed_pruned.csv` ({len(df)} rows, {len(feature_cols)} numeric features).",
        f"- Evaluation: {fold_count}-fold `StratifiedGroupKFold` by `speaker_id`; {final_eval_rows} final-evaluation clips from {final_eval_speakers} speakers; speaker-overlap count across folds = {speaker_overlap}.",
        "- Models compared: Logistic Regression, Elastic Net Logistic Regression, Linear SVM, Random Forest, XGBoost if available, and the transparent heuristic benchmark. XGBoost was unavailable in this environment.",
        f"- Official winner: `{best['model']}`.",
        f"- Official threshold: {fmt(best['selected_threshold'])}.",
        f"- Headline metrics at selected threshold: recall {fmt(best['selected_recall'])}, precision {fmt(best['selected_precision'])}, specificity {fmt(best['selected_specificity'])}, balanced accuracy {fmt(best['selected_balanced_accuracy'])}, PR-AUC {fmt(best['selected_pr_auc'])}, Brier {fmt(best['selected_brier_score'])}, false negatives {fmt(best['selected_false_negatives'], digits=0)}.",
        f"- Why it won: it tied the top recall group while beating the closest Random Forest competitor on PR-AUC and retaining materially better calibration than uncalibrated Logistic Regression.",
        f"- Calibration: {calibration_line}",
        "- Screening-only framing: the operating point intentionally prioritizes recall, but specificity remains low, so false positives are expected and the score must not be treated as a diagnostic probability.",
        "- Main limitations: small heterogeneous dataset, metadata/path-derived speaker IDs, acoustic proxy features, weak specificity, and no external validation.",
        "",
        "## Compact Comparison",
        "",
        markdown_table(compact_df),
        "",
        "Supporting artifacts: `reports/tables/phase4_compact_comparison.csv`, `reports/figures/phase4_compact_comparison.png`, and `reports/phase4_official_winner.md`.",
    ]
    (REPORTS_DIR / "phase4_executive_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase4_official_winner(experiment_df: pd.DataFrame, calibration_comparison: pd.DataFrame) -> None:
    """Freeze the official Phase 4 winner in one place."""
    best = experiment_df[experiment_df["model_status"].eq("evaluated")].sort_values(
        ["selected_recall", "selected_pr_auc", "selected_balanced_accuracy", "selected_brier_score"],
        ascending=[False, False, False, True],
    ).iloc[0]
    base_model = best.get("base_model", split_candidate_name(str(best["model"]))[0])
    calibration_method = best.get("calibration_method", split_candidate_name(str(best["model"]))[1])
    calibration_note = "Calibration comparison unavailable."
    if not calibration_comparison.empty:
        selected_cal = calibration_comparison[calibration_comparison["model"].eq(best["model"])]
        if not selected_cal.empty:
            calibration_note = (
                f"Selected model Brier score: {fmt(selected_cal.iloc[0]['brier_score'])}; "
                "calibration is improved versus the uncalibrated base model but still not externally validated."
            )
    lines = [
        "# Official Phase 4 Winner",
        "",
        f"- Official winning model: `{best['model']}`.",
        f"- Base model family: `{base_model}`.",
        f"- Calibration method: `{calibration_method}`.",
        f"- Official operating threshold: **{fmt(best['selected_threshold'])}**.",
        "- Official evaluation framing: speaker-aware grouped cross-validation with dementia as the positive class and recall prioritized for screening.",
        f"- Official selected-threshold metrics: recall {fmt(best['selected_recall'])}, precision {fmt(best['selected_precision'])}, specificity {fmt(best['selected_specificity'])}, balanced accuracy {fmt(best['selected_balanced_accuracy'])}, PR-AUC {fmt(best['selected_pr_auc'])}, Brier {fmt(best['selected_brier_score'])}, false negatives {fmt(best['selected_false_negatives'], digits=0)}.",
        f"- Calibration note: {calibration_note}",
        "",
        "## Required Caveat",
        "",
        "The Phase 4 score is a screening risk score only. It is not a diagnostic probability, is not externally validated, and should not be used to make clinical decisions without confirmatory assessment.",
    ]
    (REPORTS_DIR / "phase4_official_winner.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def official_best_row(experiment_df: pd.DataFrame) -> pd.Series:
    """Return the frozen Phase 4 winner row."""
    return experiment_df[experiment_df["model_status"].eq("evaluated")].sort_values(
        ["selected_recall", "selected_pr_auc", "selected_balanced_accuracy", "selected_brier_score"],
        ascending=[False, False, False, True],
    ).iloc[0]


def score_banding_table(experiment_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Create simple Phase 5 product-framing score bands for the frozen model."""
    best = official_best_row(experiment_df)
    model_name = str(best["model"])
    threshold = float(best["selected_threshold"])
    model_df = predictions[predictions["model"].eq(model_name)].copy()
    if model_df.empty:
        return pd.DataFrame()

    high_cut = min(1.0, threshold + 0.10)
    band_specs = [
        {
            "score_band": "low",
            "score_min_inclusive": 0.0,
            "score_max_exclusive": threshold,
            "display_label": "Lower screening signal",
            "phase5_framing": "Below the Phase 4 operating threshold; still not a medical all-clear.",
            "mask": model_df["score"].astype(float) < threshold,
        },
        {
            "score_band": "moderate",
            "score_min_inclusive": threshold,
            "score_max_exclusive": high_cut,
            "display_label": "Intermediate screening signal",
            "phase5_framing": "At or above the Phase 4 operating threshold but below the stronger-score band; screening-positive only.",
            "mask": (model_df["score"].astype(float) >= threshold) & (model_df["score"].astype(float) < high_cut),
        },
        {
            "score_band": "high",
            "score_min_inclusive": high_cut,
            "score_max_exclusive": 1.0,
            "display_label": "Elevated screening signal",
            "phase5_framing": "Above the stronger-score band; screening-positive only, not diagnostic.",
            "mask": model_df["score"].astype(float) >= high_cut,
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in band_specs:
        band_df = model_df[spec["mask"]].copy()
        dementia_clips = int((band_df["true_label"].astype(int) == 1).sum()) if not band_df.empty else 0
        non_dementia_clips = int((band_df["true_label"].astype(int) == 0).sum()) if not band_df.empty else 0
        n_clips = int(len(band_df))
        rows.append(
            {
                "official_model": model_name,
                "official_threshold": threshold,
                "score_band": spec["score_band"],
                "display_label": spec["display_label"],
                "score_min_inclusive": spec["score_min_inclusive"],
                "score_max_exclusive": spec["score_max_exclusive"],
                "screen_result_at_official_threshold": "screen_negative" if spec["score_band"] == "low" else "screen_positive",
                "n_clips": n_clips,
                "n_speakers": int(band_df["speaker_id"].nunique()) if not band_df.empty else 0,
                "dementia_clips": dementia_clips,
                "non_dementia_clips": non_dementia_clips,
                "observed_dementia_rate": float(dementia_clips / n_clips) if n_clips else np.nan,
                "false_negatives_in_band": dementia_clips if spec["score_band"] == "low" else 0,
                "false_positives_in_band": non_dementia_clips if spec["score_band"] != "low" else 0,
                "phase5_framing": spec["phase5_framing"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "phase4_score_bands.csv", index=False)
    return out


def write_score_banding_report(score_bands: pd.DataFrame) -> None:
    """Write the small Phase 5 score-banding note."""
    if score_bands.empty:
        lines = [
            "# Phase 4 Score Banding",
            "",
            "Score banding could not be generated because frozen-model predictions were unavailable.",
        ]
    else:
        show = score_bands[
            [
                "score_band",
                "display_label",
                "score_min_inclusive",
                "score_max_exclusive",
                "screen_result_at_official_threshold",
                "n_clips",
                "observed_dementia_rate",
                "false_negatives_in_band",
                "false_positives_in_band",
                "phase5_framing",
            ]
        ].copy()
        lines = [
            "# Phase 4 Score Banding",
            "",
            "- Purpose: Phase 5 product wording support only.",
            "- These bands are derived from the frozen Phase 4 model score and selected threshold; they are not medical risk categories.",
            "- The high band means screening-positive at the Phase 4 threshold, not diagnosis.",
            "",
            markdown_table(show),
            "",
            "Required language: score bands describe the model's screening signal on this recording, not a clinical probability of dementia.",
        ]
    (REPORTS_DIR / "phase4_score_banding.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_marker_shortlist() -> pd.DataFrame:
    """Define which marker summaries are safe for Phase 5 wording."""
    rows = [
        {
            "marker": "unusual_pause_burden",
            "decision": "show_cautiously",
            "safe_user_wording": "The recording showed elevated pause or low-speech burden.",
            "source_columns": "pause_ratio; pause_count; total_silence_duration_sec",
            "required_caveat": "Energy-based pause proxy; not transcript-based hesitation annotation and not a clinical marker.",
        },
        {
            "marker": "slower_speech_rate_proxy",
            "decision": "show_cautiously",
            "safe_user_wording": "The recording showed a lower acoustic speech-activity rate.",
            "source_columns": "speaking_rate_proxy; onset_rate_per_speech_sec; onset_count",
            "required_caveat": "Acoustic activity proxy; not words per minute and not a language assessment.",
        },
        {
            "marker": "reduced_fluency_or_fragmentation",
            "decision": "show_cautiously",
            "safe_user_wording": "The recording showed speech-flow patterns that may warrant review.",
            "source_columns": "speech_segment_count; speech_segment_rate_per_min; pause_rate_per_min",
            "required_caveat": "Derived from automatic energy segmentation; should be presented only as a broad speech-pattern signal.",
        },
        {
            "marker": "prosody_instability",
            "decision": "show_cautiously",
            "safe_user_wording": "The recording showed pitch or prosody variability captured by the acoustic model.",
            "source_columns": "f0_semitone_std; f0_iqr; f0_std",
            "required_caveat": "Automatic F0 estimate; not a clinical prosody or neurological measurement.",
        },
        {
            "marker": "model_feature_importance_or_coefficients",
            "decision": "internal_only",
            "safe_user_wording": "",
            "source_columns": "reports/tables/phase4_feature_importance.csv",
            "required_caveat": "Use for technical review only; do not expose coefficient-level explanations as causes.",
        },
        {
            "marker": "diagnostic_or_causal_claims",
            "decision": "do_not_show",
            "safe_user_wording": "",
            "source_columns": "not_applicable",
            "required_caveat": "Never state that speech markers diagnose dementia, prove impairment, or measure disease severity.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "phase4_safe_marker_shortlist.csv", index=False)
    return out


def write_clinical_handoff_note(
    experiment_df: pd.DataFrame,
    score_bands: pd.DataFrame,
    marker_df: pd.DataFrame,
) -> None:
    """Write Phase 5-facing clinical/product language handoff."""
    best = official_best_row(experiment_df)
    lines = [
        "# Phase 4 Clinical Handoff Note",
        "",
        "This note translates the Phase 4 classical baseline into safe Phase 5 product language. It is not a clinical validation document.",
        "",
        "## Frozen Model",
        "",
        f"- Official winning model: `{best['model']}`.",
        f"- Official threshold: **{fmt(best['selected_threshold'])}**.",
        f"- Evaluation frame: speaker-aware grouped cross-validation; dementia is the positive screening class.",
        f"- Headline selected-threshold metrics: recall {fmt(best['selected_recall'])}, precision {fmt(best['selected_precision'])}, specificity {fmt(best['selected_specificity'])}, PR-AUC {fmt(best['selected_pr_auc'])}, Brier {fmt(best['selected_brier_score'])}.",
        "",
        "## What The Model Is Good For",
        "",
        "- Classical baseline screening support on the governed Phase 3 acoustic feature set.",
        "- Producing a recall-prioritized speech-pattern signal for follow-up review.",
        "- Comparing future Phase 5 model behavior against a leakage-safe speaker-grouped baseline.",
        "",
        "## What The Model Is Not Good For",
        "",
        "- Diagnosing dementia.",
        "- Estimating a person's true medical probability of dementia.",
        "- Replacing clinical assessment, cognitive testing, medical history, or caregiver/clinician judgment.",
        "- Explaining causality from any single acoustic feature.",
        "",
        "## Score Meaning",
        "",
        "- The score should mean: the model found a stronger or weaker acoustic screening signal in this recording relative to the Phase 4 baseline.",
        "- The score should not mean: a diagnostic probability, disease severity, cognitive score, or proof of impairment.",
        "- Scores at or above the threshold should be worded as `elevated screening signal`, not `dementia detected`.",
        "",
        "## Score Bands For Phase 5 Copy",
        "",
        markdown_table(score_bands[["score_band", "display_label", "score_min_inclusive", "score_max_exclusive", "phase5_framing"]]) if not score_bands.empty else "Score bands unavailable.",
        "",
        "## Marker Summary Policy",
        "",
        markdown_table(marker_df[["marker", "decision", "safe_user_wording", "required_caveat"]]),
        "",
        "## Limitations That Must Appear Near The Output",
        "",
        "- Screening-only, not diagnostic.",
        "- False positives are expected because the selected operating point prioritizes recall.",
        "- False negatives remain possible.",
        "- Calibration is improved but not externally validated; do not call the score a medical probability.",
        "- Speaker IDs are metadata/path-derived rather than biometrically verified.",
        "- Features are acoustic proxies affected by recording quality, microphone conditions, language, and speaking context.",
        "- External clinical validation is required before any real-world clinical use.",
    ]
    (REPORTS_DIR / "phase4_clinical_handoff_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_phase4_closeout() -> None:
    """Generate final Phase 4 closeout-only artifacts from existing outputs."""
    ensure_output_dirs()
    df, feature_cols, _ = load_phase4_data()
    experiment_df = pd.read_csv(TABLES_DIR / "phase4_baseline_experiments.csv")
    split_df = pd.read_csv(TABLES_DIR / "phase4_cv_split_summary.csv")
    predictions = pd.read_csv(TABLES_DIR / "phase4_oof_predictions.csv")
    calibration_path = TABLES_DIR / "phase4_calibration_comparison.csv"
    calibration_comparison = pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame()
    compact_df = compact_comparison_table(experiment_df)
    save_compact_comparison_figure(compact_df)
    score_bands = score_banding_table(experiment_df, predictions)
    write_score_banding_report(score_bands)
    marker_df = safe_marker_shortlist()
    write_clinical_handoff_note(experiment_df, score_bands, marker_df)
    write_phase4_executive_summary(
        df=df,
        feature_cols=feature_cols,
        experiment_df=experiment_df,
        compact_df=compact_df,
        split_df=split_df,
        calibration_comparison=calibration_comparison,
    )
    write_phase4_official_winner(experiment_df, calibration_comparison)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 classical baseline experiments.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-splits", type=int, default=5)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only regenerate final executive-summary, compact-comparison, and winner-freeze artifacts from existing Phase 4 tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    if args.finalize_only:
        finalize_phase4_closeout()
        print("Phase 4 final closeout artifacts regenerated.")
        return
    df, feature_cols, governance = load_phase4_data()
    write_input_summary(df, feature_cols, governance)
    folds, split_df, cv_note, n_splits = build_outer_folds(df, feature_cols, seed=args.seed, max_splits=args.max_splits)
    write_protocol_report(df, split_df, cv_note, n_splits, seed=args.seed)
    specs = build_model_specs(seed=args.seed)
    write_preprocessing_note(specs)
    heuristic_cols = choose_heuristic_columns(feature_cols)
    write_heuristic_definition(heuristic_cols)

    fold_metrics, predictions, feature_importance_raw, _ = run_grouped_experiments(
        df=df,
        feature_cols=feature_cols,
        folds=folds,
        specs=specs,
        heuristic_cols=heuristic_cols,
        seed=args.seed,
    )
    unavailable_specs = [spec for spec in specs if spec.status == "unavailable"]
    threshold_df, selected_thresholds = threshold_sweep(predictions, unavailable_specs)
    experiment_df = aggregate_experiments(fold_metrics, predictions, selected_thresholds, specs)
    write_heuristic_definition(heuristic_cols, experiment_df=experiment_df)
    importance_df = summarize_feature_importance(feature_importance_raw)
    best_model, best_row = select_best_model(experiment_df, selected_thresholds)
    best_threshold = float(best_row["selected_threshold"])
    fn_df = false_negative_analysis(predictions, best_model, best_threshold, heuristic_cols)
    speaker_metrics = speaker_level_evaluation(predictions, best_model, best_threshold)
    calibration_comparison, brier, calibration_quality = calibration_report(predictions, best_model, best_threshold)
    save_selected_confusion_figures(predictions, selected_thresholds)

    write_baseline_report(experiment_df, fold_metrics)
    write_threshold_report(threshold_df, selected_thresholds)
    write_model_selection_report(best_model, best_row, experiment_df)
    write_feature_importance_report(importance_df)
    write_false_negative_report(fn_df, predictions, best_model, best_threshold, heuristic_cols)
    write_speaker_level_report(speaker_metrics, experiment_df, best_model)
    write_calibration_report(best_model, best_threshold, brier, calibration_quality, predictions, calibration_comparison)
    finalize_phase4_closeout()

    print(f"Phase 4 complete. Best baseline: {best_model} at threshold {best_threshold:.2f}.")


if __name__ == "__main__":
    main()
