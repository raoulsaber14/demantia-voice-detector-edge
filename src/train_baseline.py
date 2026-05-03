"""Train and evaluate classical ML baselines on hand-crafted acoustic features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover - scipy is present in the project env, but keep the CLI robust.
    mannwhitneyu = None

from src.config import CFG
from src.evaluate import (
    bootstrap_metric_cis,
    classification_metrics,
    positive_class_probability,
    save_calibration_curve_figure,
    save_confusion_matrix_figure,
    save_pr_curve_figure,
    save_roc_curve_figure,
)


NON_FEATURE_COLUMNS = {
    "metadata_index",
    "speaker_id",
    "audio_file",
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
}

MODEL_ORDER = ["logistic_regression", "svm_rbf", "random_forest"]
UPGRADE_REPORT_DIR = Path("reports/classical_baseline_upgrade")
UPGRADE_FIGURE_DIR = UPGRADE_REPORT_DIR / "figures"
STANDARD_TABLE_DIR = Path("reports/tables")
STANDARD_REPORT_DIR = Path("reports")
RANDOM_SEED = CFG.random_seed


def load_feature_table(path: Path, allowed_splits: Tuple[str, ...] = ("train", "valid")) -> pd.DataFrame:
    """Load features and keep rows valid for training/evaluation."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature CSV not found: {path}. "
            "Run feature extraction first: python -m src.features ..."
        )
    if path.stat().st_size == 0:
        raise ValueError(
            f"Feature CSV is empty: {path}. "
            "Run preprocessing and feature extraction before baseline training."
        )

    try:
        df = pd.read_csv(path)
    except EmptyDataError as exc:
        raise ValueError(
            f"Feature CSV has no columns/rows: {path}. "
            "This usually means no processed audio was found."
        ) from exc

    if df.empty:
        raise ValueError(
            f"Feature CSV has zero rows after load: {path}. "
            "Check preprocessing output under data/processed_audio."
        )

    df = df[df["label"].isin([0, 1])].copy()
    df = df[df["datasplit"].isin(allowed_splits)].copy()
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    if "feature_extraction_status" in df.columns:
        df = df[df["feature_extraction_status"].fillna("success").eq("success")].copy()
    if df.empty:
        raise ValueError(
            "No rows remain after filtering label/split/error columns. "
            "Check features and metadata splits."
        )
    return ensure_speaker_column(df)


def ensure_speaker_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every row has a stable speaker/group key for grouped validation."""
    out = df.copy()
    if "speaker_id" not in out.columns:
        out["speaker_id"] = ""

    speaker = out["speaker_id"].fillna("").astype(str).str.strip()
    missing = speaker.eq("")
    if missing.any() and "audio_file" in out.columns:
        derived = out.loc[missing, "audio_file"].fillna("").astype(str).map(
            lambda p: Path(p).parent.name.strip().lower().replace(" ", "_") if p else ""
        )
        speaker.loc[missing] = derived
        missing = speaker.eq("")

    if missing.any():
        fallback_ids = [f"missing_speaker_row_{idx}" for idx in out.index[missing]]
        speaker.loc[missing] = fallback_ids

    out["speaker_id"] = speaker.astype(str)
    return out


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select numeric features excluding metadata and labels."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric_cols if c not in NON_FEATURE_COLUMNS and c != "label"]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional dependencies."""
    if df.empty:
        return "(empty)"
    text_df = df.copy().astype(object)
    text_df = text_df.where(pd.notna(text_df), "")
    headers = [str(col) for col in text_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text_df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in text_df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_models(seed: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """Create baseline model pipelines.

    SVM intentionally uses probability=False. Probability=True runs libsvm's
    internal Platt scaling, and the enhanced workflow applies one explicit
    external calibrator on a held-out validation speaker subset.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=False,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        random_state=seed,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def build_search_spaces(seed: int = RANDOM_SEED) -> Dict[str, Tuple[Pipeline, Dict[str, Sequence[Any]]]]:
    """Return practical group-CV search spaces for the three classical baselines."""
    models = build_models(seed=seed)
    return {
        "logistic_regression": (
            models["logistic_regression"],
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        "svm_rbf": (
            models["svm_rbf"],
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", 0.01, 0.1],
            },
        ),
        "random_forest": (
            models["random_forest"],
            {
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", "log2", None],
            },
        ),
    }


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))


def predict_positive_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    """Return a dementia-positive probability-like vector."""
    if hasattr(model, "predict_proba"):
        try:
            return positive_class_probability(model.predict_proba(x))
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        if scores.ndim == 2:
            scores = scores[:, -1]
        return _sigmoid(scores)
    return np.asarray(model.predict(x), dtype=float)


def model_score_for_calibration(model: Pipeline, x: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Return the raw score used by the external logistic calibrator."""
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        if scores.ndim == 2:
            scores = scores[:, -1]
        return scores.ravel(), "decision_function"
    if hasattr(model, "predict_proba"):
        return positive_class_probability(model.predict_proba(x)), "predict_proba_positive_class"
    return np.asarray(model.predict(x), dtype=float).ravel(), "hard_prediction_fallback"


def split_verification_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize clip/class/speaker counts by split."""
    rows: List[Dict[str, object]] = []
    splits = [s for s in ["train", "valid", "test"] if s in set(df["datasplit"].astype(str))]
    splits.extend(sorted(set(df["datasplit"].astype(str)) - set(splits)))
    for split in splits:
        sub = df[df["datasplit"].astype(str) == split]
        rows.append(
            {
                "split": split,
                "n_clips": int(len(sub)),
                "n_dementia_clips": int((sub["label"].astype(int) == 1).sum()),
                "n_non_dementia_clips": int((sub["label"].astype(int) == 0).sum()),
                "n_unique_speakers": int(sub["speaker_id"].nunique()),
                "has_both_classes": bool(sub["label"].nunique() == 2),
            }
        )
    return pd.DataFrame(rows)


def speaker_overlap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize speaker overlap across all split pairs."""
    split_to_speakers = {
        split: set(sub["speaker_id"].astype(str))
        for split, sub in df.groupby(df["datasplit"].astype(str), dropna=False)
    }
    rows: List[Dict[str, object]] = []
    splits = sorted(split_to_speakers)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = sorted(split_to_speakers[left] & split_to_speakers[right])
            rows.append(
                {
                    "split_a": left,
                    "split_b": right,
                    "overlap_speaker_count": len(overlap),
                    "overlap_speaker_sample": ";".join(overlap[:20]),
                    "status": "pass" if len(overlap) == 0 else "fail",
                }
            )
    return pd.DataFrame(rows)


def evaluation_split_validity(df: pd.DataFrame, split: str) -> Tuple[bool, str]:
    """Check class support and leakage for a candidate evaluation split."""
    eval_df = df[df["datasplit"].astype(str) == split]
    if eval_df.empty:
        return False, f"`{split}` split is empty."
    if eval_df["label"].nunique() < 2:
        return False, f"`{split}` split does not contain both classes."

    eval_speakers = set(eval_df["speaker_id"].astype(str))
    train_speakers = set(df[df["datasplit"].astype(str) == "train"]["speaker_id"].astype(str))
    if split == "test":
        valid_speakers = set(df[df["datasplit"].astype(str) == "valid"]["speaker_id"].astype(str))
        overlap = eval_speakers & (train_speakers | valid_speakers)
    else:
        overlap = eval_speakers & train_speakers
    if overlap:
        return False, f"`{split}` has speaker leakage with training/calibration data: {sorted(overlap)[:10]}"
    return True, f"`{split}` has both classes and no detected speaker overlap."


def choose_evaluation_split(df: pd.DataFrame, requested_split: str = "auto") -> Tuple[str, str, bool]:
    """Select the final evaluation split, preferring a valid held-out test set."""
    if requested_split != "auto":
        ok, reason = evaluation_split_validity(df, requested_split)
        if not ok:
            raise ValueError(f"Requested evaluation split is invalid: {reason}")
        return requested_split, reason, requested_split == "test"

    test_ok, test_reason = evaluation_split_validity(df, "test")
    if test_ok:
        return "test", test_reason, True

    valid_ok, valid_reason = evaluation_split_validity(df, "valid")
    if valid_ok:
        return (
            "valid",
            f"Held-out test rejected: {test_reason} Fallback uses validation-based evaluation: {valid_reason}",
            False,
        )
    raise ValueError(f"No valid evaluation split is available. Test: {test_reason} Valid: {valid_reason}")


def write_precheck_summary(
    df: pd.DataFrame,
    feature_csv: Path,
    out_dir: Path,
    eval_split: str,
    eval_reason: str,
    heldout_test_valid: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Write split precheck tables and markdown summary before training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split_df = split_verification_table(df)
    overlap_df = speaker_overlap_table(df)
    split_df.to_csv(out_dir / "split_verification.csv", index=False)
    overlap_df.to_csv(out_dir / "speaker_overlap_check.csv", index=False)

    lines = [
        "# Classical Baseline Upgrade Precheck",
        "",
        f"- Feature table inspected: `{feature_csv}`",
        f"- Rows available for enhanced baseline: **{len(df)}**",
        f"- Unique speakers: **{df['speaker_id'].nunique()}**",
        f"- Selected evaluation split: **{eval_split}**",
        f"- Held-out test valid: **{heldout_test_valid}**",
        f"- Split decision note: {eval_reason}",
        "",
        "## Split Composition",
        "",
        dataframe_to_markdown(split_df),
        "",
        "## Speaker Overlap",
        "",
        dataframe_to_markdown(overlap_df) if not overlap_df.empty else "Only one split was present.",
        "",
        "## Verdict",
        "",
    ]
    if heldout_test_valid:
        lines.append("The enhanced test split is valid for held-out evaluation: both classes are present and no speaker overlap was detected.")
    else:
        lines.append("The held-out test split is not valid; the pipeline will not silently evaluate on it.")
    (out_dir / "precheck_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return split_df, overlap_df


def validate_training_splits(df: pd.DataFrame, eval_split: str) -> None:
    """Fail fast if required train/valid/eval splits are one-class or leak speakers."""
    required = ["train", "valid", eval_split]
    for split in dict.fromkeys(required):
        split_df = df[df["datasplit"].astype(str) == split]
        counts = split_df["label"].value_counts().to_dict()
        if split_df.empty or split_df["label"].nunique() < 2:
            raise ValueError(f"Split `{split}` must contain both classes. Observed counts: {counts}")

    overlap_df = speaker_overlap_table(df[df["datasplit"].isin(["train", "valid", "test"])])
    leaking = overlap_df[overlap_df["overlap_speaker_count"] > 0]
    if not leaking.empty:
        raise ValueError(f"Speaker leakage detected across splits:\n{leaking.to_string(index=False)}")


def split_validation_for_calibration_threshold(
    valid_df: pd.DataFrame,
    seed: int,
    threshold_fraction: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Split validation speakers into disjoint calibration and threshold-selection subsets."""
    speaker_rows: List[Dict[str, object]] = []
    mixed_speakers: List[str] = []
    for speaker_id, sub in valid_df.groupby("speaker_id"):
        labels = sorted(sub["label"].astype(int).unique().tolist())
        if len(labels) > 1:
            mixed_speakers.append(str(speaker_id))
        speaker_rows.append(
            {
                "speaker_id": str(speaker_id),
                "speaker_label": int(sub["label"].astype(int).mode().iloc[0]),
                "n_clips": int(len(sub)),
            }
        )
    speaker_df = pd.DataFrame(speaker_rows)
    notes: Dict[str, object] = {
        "split_method": "speaker_stratified_validation_subsplit",
        "mixed_label_speakers_in_valid": len(mixed_speakers),
        "mixed_label_speaker_sample": ";".join(mixed_speakers[:10]),
        "fallback_used": False,
        "fallback_reason": "",
    }

    class_speaker_counts = speaker_df["speaker_label"].value_counts().to_dict()
    feasible = len(class_speaker_counts) == 2 and min(class_speaker_counts.values()) >= 2
    if feasible:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=threshold_fraction,
            random_state=seed,
        )
        try:
            idx = np.arange(len(speaker_df))
            cal_idx, thr_idx = next(splitter.split(idx, speaker_df["speaker_label"].to_numpy()))
            cal_speakers = set(speaker_df.iloc[cal_idx]["speaker_id"])
            thr_speakers = set(speaker_df.iloc[thr_idx]["speaker_id"])
            cal_df = valid_df[valid_df["speaker_id"].astype(str).isin(cal_speakers)].copy()
            thr_df = valid_df[valid_df["speaker_id"].astype(str).isin(thr_speakers)].copy()
            if cal_df["label"].nunique() == 2 and thr_df["label"].nunique() == 2 and cal_speakers.isdisjoint(thr_speakers):
                notes.update(
                    {
                        "n_calibration_clips": len(cal_df),
                        "n_threshold_selection_clips": len(thr_df),
                        "n_calibration_speakers": len(cal_speakers),
                        "n_threshold_selection_speakers": len(thr_speakers),
                    }
                )
                return cal_df, thr_df, notes
        except Exception as exc:
            notes["fallback_reason"] = f"Stratified validation speaker split failed: {exc}"

    notes["fallback_used"] = True
    if not notes["fallback_reason"]:
        notes["fallback_reason"] = (
            "Too few validation speakers per class to split calibration and threshold "
            f"selection cleanly. Class speaker counts: {class_speaker_counts}"
        )
    notes["split_method"] = "validation_reused_for_calibration_and_threshold"
    notes.update(
        {
            "n_calibration_clips": len(valid_df),
            "n_threshold_selection_clips": len(valid_df),
            "n_calibration_speakers": valid_df["speaker_id"].nunique(),
            "n_threshold_selection_speakers": valid_df["speaker_id"].nunique(),
        }
    )
    return valid_df.copy(), valid_df.copy(), notes


def select_high_recall_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
) -> Tuple[float, Dict[str, float]]:
    """Pick a threshold that prioritizes dementia recall on threshold-selection data."""
    y_prob = positive_class_probability(y_prob)
    thresholds = np.unique(np.concatenate(([0.0, 1.0], y_prob)))
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = classification_metrics(y_true, y_pred, y_prob)
        rows.append({"threshold": float(threshold), **metrics})

    candidates = [row for row in rows if row["recall"] >= target_recall]
    if candidates:
        selected = sorted(
            candidates,
            key=lambda row: (row["specificity"], row["precision"], row["balanced_accuracy"], row["threshold"]),
            reverse=True,
        )[0]
        selected["selection_note"] = "met_target_recall"
    else:
        selected = sorted(rows, key=lambda row: (row["recall"], row["balanced_accuracy"], row["threshold"]), reverse=True)[0]
        selected["selection_note"] = "target_recall_not_reached"
    return float(selected["threshold"]), selected


def speaker_group_cv(y: np.ndarray, groups: np.ndarray, seed: int, max_splits: int = 5) -> Tuple[Optional[StratifiedGroupKFold], str, int]:
    """Build a StratifiedGroupKFold object when speaker/class support allows it."""
    group_df = pd.DataFrame({"group": groups.astype(str), "label": y.astype(int)}).drop_duplicates()
    mixed_group_count = int(group_df.groupby("group")["label"].nunique().gt(1).sum())
    group_label = group_df.groupby("group")["label"].agg(lambda s: int(s.mode().iloc[0]))
    class_group_counts = group_label.value_counts().to_dict()
    if len(class_group_counts) < 2:
        return None, f"Grouped CV unavailable: only one class at speaker level: {class_group_counts}", 0
    n_splits = int(min(max_splits, min(class_group_counts.values())))
    if n_splits < 2:
        return None, f"Grouped CV unavailable: too few speakers per class: {class_group_counts}", 0
    note = (
        f"StratifiedGroupKFold with n_splits={n_splits}; "
        f"speaker class counts={class_group_counts}; mixed_label_groups={mixed_group_count}"
    )
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), note, n_splits


def tune_models_with_group_cv(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    out_dir: Path,
    seed: int,
    scoring: str = "average_precision",
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, pd.DataFrame]:
    """Tune model hyperparameters with speaker-grouped CV and save search artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cv, cv_note, n_splits = speaker_group_cv(y_train, groups, seed=seed)
    tuned_models: Dict[str, Pipeline] = {}
    search_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []

    for model_name, (estimator, param_grid) in build_search_spaces(seed=seed).items():
        if cv is None:
            estimator.fit(x_train, y_train)
            tuned_models[model_name] = estimator
            search_rows.append(
                {
                    "model": model_name,
                    "search_status": "skipped",
                    "cv_note": cv_note,
                    "params": json.dumps({}),
                    "mean_test_score": np.nan,
                    "std_test_score": np.nan,
                    "rank_test_score": np.nan,
                }
            )
            best_rows.append(
                {
                    "model": model_name,
                    "search_status": "skipped",
                    "best_score": np.nan,
                    "best_params": json.dumps({}),
                    "scoring": scoring,
                    "cv_n_splits": n_splits,
                    "cv_note": cv_note,
                }
            )
            continue

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score=np.nan,
            return_train_score=True,
        )
        try:
            search.fit(x_train, y_train, groups=groups)
            tuned_models[model_name] = search.best_estimator_
            cv_results = pd.DataFrame(search.cv_results_)
            param_cols = [c for c in cv_results.columns if c.startswith("param_")]
            keep_cols = param_cols + [
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
                "mean_train_score",
                "std_train_score",
            ]
            for _, row in cv_results[keep_cols].iterrows():
                params = {col.replace("param_", ""): row[col] for col in param_cols}
                search_rows.append(
                    {
                        "model": model_name,
                        "search_status": "success",
                        "cv_note": cv_note,
                        "params": json.dumps(params, default=str),
                        "mean_test_score": row.get("mean_test_score", np.nan),
                        "std_test_score": row.get("std_test_score", np.nan),
                        "rank_test_score": row.get("rank_test_score", np.nan),
                        "mean_train_score": row.get("mean_train_score", np.nan),
                        "std_train_score": row.get("std_train_score", np.nan),
                    }
                )
            best_rows.append(
                {
                    "model": model_name,
                    "search_status": "success",
                    "best_score": float(search.best_score_) if search.best_score_ is not None else np.nan,
                    "best_params": json.dumps(search.best_params_, default=str),
                    "scoring": scoring,
                    "cv_n_splits": n_splits,
                    "cv_note": cv_note,
                }
            )
        except Exception as exc:
            fallback = estimator.fit(x_train, y_train)
            tuned_models[model_name] = fallback
            search_rows.append(
                {
                    "model": model_name,
                    "search_status": "failed_default_fit_used",
                    "cv_note": cv_note,
                    "failure_reason": str(exc),
                    "params": json.dumps({}),
                    "mean_test_score": np.nan,
                    "std_test_score": np.nan,
                    "rank_test_score": np.nan,
                }
            )
            best_rows.append(
                {
                    "model": model_name,
                    "search_status": "failed_default_fit_used",
                    "best_score": np.nan,
                    "best_params": json.dumps({}),
                    "scoring": scoring,
                    "cv_n_splits": n_splits,
                    "cv_note": f"{cv_note}; failure={exc}",
                }
            )

    search_df = pd.DataFrame(search_rows)
    best_df = pd.DataFrame(best_rows)
    search_df.to_csv(out_dir / "hyperparameter_search_results.csv", index=False)
    best_df.to_csv(out_dir / "best_model_configs.csv", index=False)
    return tuned_models, search_df, best_df


def save_feature_importances(model_name: str, model: Pipeline, feat_cols: List[str], feature_rows: List[Dict[str, object]]) -> None:
    """Collect feature importance values where available."""
    if model_name == "random_forest":
        importances = model.named_steps["model"].feature_importances_
        for col, imp in zip(feat_cols, importances):
            feature_rows.append({"model": model_name, "feature": col, "importance": float(imp)})
    elif model_name == "logistic_regression":
        coeffs = model.named_steps["model"].coef_[0]
        for col, coef in zip(feat_cols, coeffs):
            feature_rows.append({"model": model_name, "feature": col, "importance": float(abs(coef))})


def build_comparison_table(enhanced_metrics: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Compare original baseline metrics to the upgraded corrected-split metrics."""
    original_path = Path("reports/tables/baseline_metrics.csv")
    comparison_rows: List[Dict[str, object]] = []
    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    original = pd.read_csv(original_path) if original_path.exists() else pd.DataFrame()

    for model_name in MODEL_ORDER:
        old_row = original[original["model"] == model_name].iloc[0].to_dict() if not original.empty and (original["model"] == model_name).any() else {}
        uncalibrated = enhanced_metrics[
            (enhanced_metrics["model"] == model_name) & (enhanced_metrics["variant"] == "uncalibrated")
        ]
        calibrated = enhanced_metrics[
            (enhanced_metrics["model"] == model_name) & (enhanced_metrics["variant"] == "calibrated")
        ]
        uncal_row = uncalibrated.iloc[0].to_dict() if not uncalibrated.empty else {}
        cal_row = calibrated.iloc[0].to_dict() if not calibrated.empty else {}
        for metric in metric_cols:
            comparison_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "original_baseline": old_row.get(metric, np.nan),
                    "upgraded_uncalibrated": uncal_row.get(metric, np.nan),
                    "upgraded_calibrated": cal_row.get(metric, np.nan),
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_csv, index=False)
    return comparison_df


def make_wide_prediction_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long variant predictions into the previous wide prediction shape."""
    if long_df.empty:
        return pd.DataFrame()
    id_cols = ["model", "metadata_index", "speaker_id", "audio_file", "true_label", "eval_split"]
    wide_base = long_df[id_cols].drop_duplicates()
    for variant in ["uncalibrated", "calibrated"]:
        sub = long_df[long_df["variant"] == variant][
            id_cols + ["probability", "prediction", "threshold"]
        ].rename(
            columns={
                "probability": f"{variant}_probability",
                "prediction": f"{variant}_prediction",
                "threshold": f"{variant}_threshold",
            }
        )
        wide_base = wide_base.merge(sub, on=id_cols, how="left")
    return wide_base


def speaker_level_outputs(
    clip_predictions: pd.DataFrame,
    out_dir: Path,
    figure_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate clip predictions to speaker-level metrics."""
    prediction_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    aggregation_methods = {
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
    }

    for (model_name, variant), model_df in clip_predictions.groupby(["model", "variant"]):
        for agg_name, agg_func in aggregation_methods.items():
            rows: List[Dict[str, object]] = []
            for speaker_id, speaker_df in model_df.groupby("speaker_id"):
                labels = sorted(speaker_df["true_label"].astype(int).unique().tolist())
                mixed = len(labels) > 1
                true_label = int(labels[0]) if not mixed else np.nan
                prob = float(agg_func(speaker_df["probability"].astype(float).to_numpy()))
                threshold = float(speaker_df["threshold"].astype(float).iloc[0])
                pred = int(prob >= threshold) if not mixed else np.nan
                row = {
                    "model": model_name,
                    "variant": variant,
                    "aggregation_method": agg_name,
                    "speaker_id": speaker_id,
                    "n_clips": int(len(speaker_df)),
                    "true_label": true_label,
                    "mixed_label_speaker": mixed,
                    "probability": prob,
                    "threshold": threshold,
                    "prediction": pred,
                }
                prediction_rows.append(row)
                rows.append(row)

            agg_df = pd.DataFrame(rows)
            eval_df = agg_df[~agg_df["mixed_label_speaker"]].copy()
            if not eval_df.empty and eval_df["true_label"].nunique() == 2:
                y_true = eval_df["true_label"].astype(int).to_numpy()
                y_prob = eval_df["probability"].astype(float).to_numpy()
                y_pred = eval_df["prediction"].astype(int).to_numpy()
                metrics = classification_metrics(y_true, y_pred, y_prob)
                metric_rows.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "aggregation_method": agg_name,
                        "n_speakers": int(len(eval_df)),
                        "n_mixed_label_speakers_excluded": int(agg_df["mixed_label_speaker"].sum()),
                        **metrics,
                    }
                )
                if agg_name == "mean":
                    save_confusion_matrix_figure(
                        y_true,
                        y_pred,
                        figure_dir / f"{model_name}_{variant}_speaker_mean_confusion_matrix.png",
                        title=f"{model_name} {variant} speaker mean confusion matrix",
                    )
            else:
                metric_rows.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "aggregation_method": agg_name,
                        "n_speakers": int(len(eval_df)),
                        "n_mixed_label_speakers_excluded": int(agg_df["mixed_label_speaker"].sum()) if not agg_df.empty else 0,
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "specificity": np.nan,
                        "f1": np.nan,
                        "roc_auc": np.nan,
                        "pr_auc": np.nan,
                        "balanced_accuracy": np.nan,
                        "brier_score": np.nan,
                    }
                )

    pred_df = pd.DataFrame(prediction_rows)
    metrics_df = pd.DataFrame(metric_rows)
    pred_df.to_csv(out_dir / "speaker_level_predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "speaker_level_metrics.csv", index=False)

    lines = [
        "# Speaker-Level Evaluation Notes",
        "",
        "- Speaker-level scores aggregate clip probabilities per speaker.",
        "- Aggregation methods: mean, median, max.",
        "- A speaker-level true label is used only when all clips for that speaker have the same label.",
        "- Mixed-label speakers are excluded from speaker-level metrics and counted in the table.",
        "- The main screening interpretation should prioritize the calibrated mean-probability rows.",
    ]
    (out_dir / "speaker_level_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pred_df, metrics_df


def confound_feature_columns(feat_cols: Sequence[str]) -> List[str]:
    """Identify explicit duration/energy/silence proxy columns for sanity checks."""
    keywords = [
        "duration",
        "rms",
        "silence",
        "energy",
        "pause_ratio",
        "unvoiced",
        "speech_duration",
    ]
    return [col for col in feat_cols if any(key in col for key in keywords)]


VOICE_QUALITY_PROXY_FEATURES = {
    "jitter_local_proxy",
    "jitter_rap_proxy",
    "pitch_period_std_proxy",
    "shimmer_local_proxy",
    "shimmer_db_proxy",
}


def feature_ablation_groups(feat_cols: Sequence[str]) -> Dict[str, List[str]]:
    """Return requested Phase 3 feature-family removal groups."""
    cols = list(feat_cols)
    groups = {
        "mfcc_family": [
            col
            for col in cols
            if col.startswith("mfcc_") or col.startswith("delta_mfcc_") or col.startswith("delta2_mfcc_")
        ],
        "f0_prosody": [col for col in cols if col.startswith("f0_") or col == "voiced_ratio"],
        "pause_fluency": [
            col
            for col in cols
            if col.startswith("pause_")
            or col.startswith("silence_")
            or col.startswith("avg_pause_")
            or col.startswith("std_pause_")
            or col.startswith("median_pause_")
            or col.startswith("p10_pause_")
            or col.startswith("p25_pause_")
            or col.startswith("p75_pause_")
            or col.startswith("p90_pause_")
            or col.startswith("avg_silence_")
            or col.startswith("std_silence_")
            or col.startswith("median_silence_")
            or col.startswith("p10_silence_")
            or col.startswith("p25_silence_")
            or col.startswith("p75_silence_")
            or col.startswith("p90_silence_")
            or col.startswith("speech_segment_")
            or col.startswith("avg_speech_segment_")
            or col.startswith("std_speech_segment_")
            or col.startswith("median_speech_segment_")
            or col.startswith("p10_speech_segment_")
            or col.startswith("p25_speech_segment_")
            or col.startswith("p75_speech_segment_")
            or col.startswith("p90_speech_segment_")
            or col
            in {
                "unvoiced_ratio",
                "energy_voiced_ratio",
                "speech_duration_sec",
                "speaking_rate_proxy",
                "onset_count",
                "onset_rate_per_speech_sec",
                "pause_to_speech_transition_count",
                "speech_to_pause_transition_count",
                "pause_speech_transition_rate_per_min",
            }
        ],
        "voice_quality_proxies": [col for col in cols if col in VOICE_QUALITY_PROXY_FEATURES],
        "energy_features": [
            col
            for col in cols
            if col.startswith("rms_")
            or col == "low_energy_ratio"
            or col.startswith("energy_modulation_")
            or col == "log_energy_modulation_std"
        ],
    }
    return {name: sorted(set(values)) for name, values in groups.items() if values}


def evaluate_simple_model(
    feature_set_name: str,
    columns: List[str],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    seed: int,
) -> Dict[str, object]:
    """Fit a small logistic sanity model and evaluate it on the evaluation split."""
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            ),
        ]
    )
    model.fit(train_df[columns], train_df["label"].astype(int).to_numpy())
    y_true = eval_df["label"].astype(int).to_numpy()
    y_prob = predict_positive_probability(model, eval_df[columns])
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "check_type": "simple_model",
        "feature_or_model": feature_set_name,
        "n_features": len(columns),
        "feature_columns": ";".join(columns),
        **classification_metrics(y_true, y_pred, y_prob),
    }


def run_confound_checks(
    df: pd.DataFrame,
    feat_cols: List[str],
    tuned_models: Dict[str, Pipeline],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run lightweight confound checks and feature-set ablations."""
    confound_cols = confound_feature_columns(feat_cols)
    check_rows: List[Dict[str, object]] = []

    for col in confound_cols:
        dem = df[df["label"].astype(int) == 1][col].dropna().astype(float)
        non = df[df["label"].astype(int) == 0][col].dropna().astype(float)
        p_value = np.nan
        if mannwhitneyu is not None and len(dem) > 0 and len(non) > 0:
            try:
                p_value = float(mannwhitneyu(dem, non, alternative="two-sided").pvalue)
            except Exception:
                p_value = np.nan
        check_rows.append(
            {
                "check_type": "univariate_class_comparison",
                "feature_or_model": col,
                "n_dementia": int(len(dem)),
                "n_non_dementia": int(len(non)),
                "dementia_mean": float(dem.mean()) if len(dem) else np.nan,
                "non_dementia_mean": float(non.mean()) if len(non) else np.nan,
                "dementia_median": float(dem.median()) if len(dem) else np.nan,
                "non_dementia_median": float(non.median()) if len(non) else np.nan,
                "mannwhitney_p": p_value,
            }
        )

    if "duration_sec" in feat_cols:
        check_rows.append(evaluate_simple_model("duration_only", ["duration_sec"], train_df, eval_df, seed))

    small_confound_cols = [
        col
        for col in [
            "duration_sec",
            "rms_mean",
            "rms_std",
            "rms_min",
            "rms_max",
            "silence_ratio",
            "energy_voiced_ratio",
            "total_silence_duration_sec",
            "speech_duration_sec",
        ]
        if col in feat_cols
    ]
    if small_confound_cols:
        check_rows.append(
            evaluate_simple_model(
                "duration_energy_silence_subset",
                small_confound_cols,
                train_df,
                eval_df,
                seed,
            )
        )

    no_confound_cols = [col for col in feat_cols if col not in confound_cols]
    ablation_specs: List[Tuple[str, str, List[str], str]] = [
        ("all_features", "none", feat_cols, "baseline"),
        (
            "without_duration_energy_silence_features",
            "duration_energy_silence",
            no_confound_cols,
            "shortcut_check",
        ),
    ]
    for family_name, family_cols in feature_ablation_groups(feat_cols).items():
        remaining = [col for col in feat_cols if col not in set(family_cols)]
        ablation_specs.append(
            (
                f"without_{family_name}",
                family_name,
                remaining,
                "feature_family_ablation",
            )
        )

    ablation_rows: List[Dict[str, object]] = []
    for model_name, fitted_model in tuned_models.items():
        for feature_set_name, removed_group, columns, ablation_type in ablation_specs:
            if not columns:
                continue
            removed_cols = [col for col in feat_cols if col not in set(columns)]
            model = clone(fitted_model)
            model.fit(train_df[columns], train_df["label"].astype(int).to_numpy())
            y_true = eval_df["label"].astype(int).to_numpy()
            y_prob = predict_positive_probability(model, eval_df[columns])
            y_pred = (y_prob >= 0.5).astype(int)
            ablation_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "ablation_type": ablation_type,
                    "n_features": len(columns),
                    "removed_feature_count": len(removed_cols),
                    "removed_feature_group": removed_group,
                    "removed_feature_sample": ";".join(removed_cols[:40]),
                    **classification_metrics(y_true, y_pred, y_prob),
                }
            )

    checks_df = pd.DataFrame(check_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    if not ablation_df.empty:
        for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "recall", "specificity"]:
            baseline_by_model = (
                ablation_df[ablation_df["feature_set"] == "all_features"]
                .set_index("model")[metric]
                .to_dict()
                if metric in ablation_df.columns
                else {}
            )
            ablation_df[f"delta_{metric}_vs_all"] = ablation_df.apply(
                lambda row: row.get(metric, np.nan) - baseline_by_model.get(row["model"], np.nan),
                axis=1,
            )

    family_ablation_df = (
        ablation_df[ablation_df["ablation_type"].isin(["baseline", "feature_family_ablation"])].copy()
        if not ablation_df.empty and "ablation_type" in ablation_df.columns
        else pd.DataFrame()
    )
    checks_df.to_csv(out_dir / "confound_checks.csv", index=False)
    ablation_df.to_csv(out_dir / "ablation_results.csv", index=False)
    family_ablation_df.to_csv(out_dir / "feature_family_ablation_results.csv", index=False)

    duration_row = checks_df[
        (checks_df["check_type"] == "simple_model") & (checks_df["feature_or_model"] == "duration_only")
    ]
    duration_line = "Duration-only model was not run because `duration_sec` was unavailable."
    if not duration_row.empty:
        row = duration_row.iloc[0]
        duration_line = (
            f"Duration-only sanity model: ROC-AUC={row.get('roc_auc', np.nan):.3f}, "
            f"PR-AUC={row.get('pr_auc', np.nan):.3f}, recall={row.get('recall', np.nan):.3f}."
        )

    subset_row = checks_df[
        (checks_df["check_type"] == "simple_model")
        & (checks_df["feature_or_model"] == "duration_energy_silence_subset")
    ]
    subset_line = "Duration/energy/silence subset model was not run because proxy features were unavailable."
    if not subset_row.empty:
        row = subset_row.iloc[0]
        subset_line = (
            f"Duration/energy/silence subset sanity model: ROC-AUC={row.get('roc_auc', np.nan):.3f}, "
            f"PR-AUC={row.get('pr_auc', np.nan):.3f}, recall={row.get('recall', np.nan):.3f}."
        )

    ablation_line = "Ablation interpretation unavailable."
    if not ablation_df.empty:
        comparable = ablation_df.pivot_table(
            index="model",
            columns="feature_set",
            values="roc_auc",
            aggfunc="first",
        )
        if {"all_features", "without_duration_energy_silence_features"}.issubset(set(comparable.columns)):
            deltas = comparable["without_duration_energy_silence_features"] - comparable["all_features"]
            max_delta_model = deltas.abs().idxmax()
            ablation_line = (
                "Removing explicit duration/energy/silence features did not collapse performance; "
                f"largest ROC-AUC change was {deltas[max_delta_model]:.3f} for `{max_delta_model}`."
            )

    family_line = "Feature-family ablations were not available."
    if not family_ablation_df.empty:
        family_only = family_ablation_df[family_ablation_df["ablation_type"] == "feature_family_ablation"].copy()
        if not family_only.empty and "delta_roc_auc_vs_all" in family_only.columns:
            family_only["abs_delta_roc_auc"] = family_only["delta_roc_auc_vs_all"].abs()
            row = family_only.sort_values("abs_delta_roc_auc", ascending=False).iloc[0]
            family_line = (
                "Largest feature-family ROC-AUC shift: "
                f"`{row['model']}` `{row['removed_feature_group']}` "
                f"delta={row.get('delta_roc_auc_vs_all', np.nan):.3f}."
            )

    lines = [
        "# Confound Analysis",
        "",
        "- This is a shortcut-learning risk check, not a causal analysis.",
        "- Checked explicit duration, RMS/energy, silence, pause-ratio, and speech-duration proxy features.",
        "- Simple sanity models are trained only on the training split and evaluated on the selected evaluation split.",
        "",
        duration_line,
        subset_line,
        ablation_line,
        family_line,
        "",
        "Ablation table:",
        "",
        "`reports/classical_baseline_upgrade/ablation_results.csv`",
        "`reports/classical_baseline_upgrade/feature_family_ablation_results.csv`",
        "",
        "If a small confound subset performs close to the full model, treat the baseline as less dementia-specific and prioritize better data controls.",
    ]
    (out_dir / "confound_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    family_report_lines = [
        "# Phase 3 Feature Family Ablation Report",
        "",
        "- Each ablation refits the already selected classical model configuration on the training split after removing one feature family.",
        "- Hyperparameters are not retuned per ablation; this keeps the comparison practical and stable, but it is a diagnostic rather than a definitive causal importance test.",
        "- Evaluation uses the same held-out split and the same 0.5 uncalibrated decision threshold used in the ablation diagnostics.",
        "- Families removed one at a time: MFCC, F0/prosody, pause/fluency, voice-quality proxies, and energy features.",
        "",
        family_line,
        "",
        "Detailed table: `reports/classical_baseline_upgrade/feature_family_ablation_results.csv`.",
    ]
    (STANDARD_REPORT_DIR / "phase3_feature_family_ablation_report.md").write_text(
        "\n".join(family_report_lines) + "\n",
        encoding="utf-8",
    )
    return checks_df, ablation_df


def write_calibration_method_notes(out_dir: Path, validation_split_notes: Dict[str, object]) -> None:
    """Document calibration and threshold workflow."""
    lines = [
        "# Calibration Method Notes",
        "",
        "- SVM RBF is trained with `probability=False`, so libsvm's internal Platt scaling is not used.",
        "- All models use one explicit external logistic calibrator.",
        "- Logistic Regression and SVM calibrators use the model `decision_function`; Random Forest uses positive-class `predict_proba`.",
        "- The calibrator is fit on a validation-speaker calibration subset.",
        "- The high-recall operating threshold is selected on a disjoint validation-speaker threshold-selection subset.",
        "- The frozen calibrated model and threshold are then evaluated on the selected evaluation split.",
        "",
        "Validation split separation:",
        "",
        dataframe_to_markdown(pd.DataFrame([validation_split_notes])),
    ]
    (out_dir / "calibration_method_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_screening_best_row(df: pd.DataFrame, target_recall: float) -> Optional[pd.Series]:
    """Choose a defensible high-recall row without ignoring false positives."""
    if df.empty:
        return None
    candidates = df[df["recall"] >= target_recall].copy()
    if candidates.empty:
        candidates = df.copy()
    candidates = candidates.sort_values(
        ["balanced_accuracy", "specificity", "pr_auc", "roc_auc", "recall"],
        ascending=False,
    )
    return candidates.iloc[0]


def write_reports(
    metrics_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    speaker_metrics_df: pd.DataFrame,
    confound_checks_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    split_df: pd.DataFrame,
    eval_split: str,
    eval_reason: str,
    heldout_test_valid: bool,
    feature_csv: Path,
    model_out_dir: Path,
    calibrated_model_out_dir: Path,
    target_recall: float,
    out_dir: Path,
) -> None:
    """Write standard and upgrade markdown reports."""
    STANDARD_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_rows = metrics_df[metrics_df["variant"] == "calibrated"].copy()
    best = choose_screening_best_row(cal_rows, target_recall)
    if best is None:
        best_line = "No calibrated model result was produced."
        best_model_name = "n/a"
    else:
        best_model_name = str(best["model"])
        best_line = (
            f"Best calibrated clip-level screening row: `{best['model']}` with "
            f"recall={best['recall']:.3f}, specificity={best['specificity']:.3f}, "
            f"PR-AUC={best['pr_auc']:.3f}, ROC-AUC={best['roc_auc']:.3f}."
        )

    speaker_best_line = "No speaker-level calibrated mean result was produced."
    speaker_mean = speaker_metrics_df[
        (speaker_metrics_df["variant"] == "calibrated")
        & (speaker_metrics_df["aggregation_method"] == "mean")
    ].copy()
    sp_best = choose_screening_best_row(speaker_mean, target_recall)
    if sp_best is not None:
        speaker_best_line = (
            f"Best calibrated speaker-mean row: `{sp_best['model']}` with "
            f"recall={sp_best['recall']:.3f}, specificity={sp_best['specificity']:.3f}, "
            f"PR-AUC={sp_best['pr_auc']:.3f}, ROC-AUC={sp_best['roc_auc']:.3f}."
        )

    duration_model = confound_checks_df[
        (confound_checks_df.get("check_type") == "simple_model")
        & (confound_checks_df.get("feature_or_model") == "duration_only")
    ]
    duration_line = "Duration-only sanity model was not available."
    if not duration_model.empty:
        row = duration_model.iloc[0]
        duration_line = (
            f"Duration-only sanity model: ROC-AUC={row.get('roc_auc', np.nan):.3f}, "
            f"PR-AUC={row.get('pr_auc', np.nan):.3f}, recall={row.get('recall', np.nan):.3f}."
        )
    confound_subset = confound_checks_df[
        (confound_checks_df.get("check_type") == "simple_model")
        & (confound_checks_df.get("feature_or_model") == "duration_energy_silence_subset")
    ]
    subset_line = "Duration/energy/silence subset sanity model was not available."
    if not confound_subset.empty:
        row = confound_subset.iloc[0]
        subset_line = (
            f"Duration/energy/silence subset sanity model: ROC-AUC={row.get('roc_auc', np.nan):.3f}, "
            f"PR-AUC={row.get('pr_auc', np.nan):.3f}, recall={row.get('recall', np.nan):.3f}."
        )

    family_ablation_line = "Feature-family ablation results were not available."
    if not ablation_df.empty and "ablation_type" in ablation_df.columns:
        family_rows = ablation_df[ablation_df["ablation_type"] == "feature_family_ablation"].copy()
        if not family_rows.empty and "delta_roc_auc_vs_all" in family_rows.columns:
            family_rows["abs_delta_roc_auc"] = family_rows["delta_roc_auc_vs_all"].abs()
            row = family_rows.sort_values("abs_delta_roc_auc", ascending=False).iloc[0]
            family_ablation_line = (
                f"Largest feature-family ablation ROC-AUC shift: `{row.get('model')}` without "
                f"`{row.get('removed_feature_group')}` changed ROC-AUC by "
                f"{row.get('delta_roc_auc_vs_all', np.nan):.3f}."
            )

    evaluation_lines = [
        "# Phase 1 Evaluation Report",
        "",
        "Classical baseline upgrade run.",
        "",
        "- Positive class: `dementia`",
        f"- Evaluation split: `{eval_split}`",
        f"- Held-out test valid: `{heldout_test_valid}`",
        f"- Split note: {eval_reason}",
        f"- Feature table: `{feature_csv}`",
        "- Hyperparameter tuning: speaker-grouped cross-validation on training speakers.",
        "- Metrics include accuracy, precision, dementia recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, balanced accuracy, and Brier score.",
        "- Bootstrap confidence intervals sample speakers with replacement when speaker IDs are available.",
        "",
        best_line,
        speaker_best_line,
        "",
        "Main tables:",
        "",
        "- `reports/tables/baseline_metrics_enhanced.csv`",
        "- `reports/classical_baseline_upgrade/speaker_level_metrics.csv`",
        "- `reports/classical_baseline_upgrade/split_verification.csv`",
        "",
        "Curve figures are saved under `reports/classical_baseline_upgrade/figures/`.",
        "",
        "This remains a screening baseline, not a diagnostic decision system.",
    ]
    (STANDARD_REPORT_DIR / "phase1_evaluation_report.md").write_text("\n".join(evaluation_lines) + "\n", encoding="utf-8")

    calibration_lines = [
        "# Phase 1 Calibration Report",
        "",
        "- Calibration method: external Platt/logistic calibration.",
        "- SVM fix: `SVC(probability=False)` avoids libsvm's internal Platt scaling, so SVM is not double-calibrated.",
        "- Calibration and threshold selection are separated by validation speakers.",
        f"- Target recall for threshold selection: {target_recall:.2f}",
        f"- Calibrated model artifacts: `{calibrated_model_out_dir}`",
        "",
        "Threshold summary:",
        "",
        dataframe_to_markdown(threshold_df),
        "",
        "This calibrated score is a screening risk score, not a medical diagnosis.",
    ]
    (STANDARD_REPORT_DIR / "phase1_calibration_report.md").write_text("\n".join(calibration_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Phase 1 Final Summary",
        "",
        "Enhanced classical baselines were upgraded with valid split checks, speaker-grouped tuning, separated calibration/threshold selection, speaker-level evaluation, and confound checks.",
        "",
        f"- Feature input: `{feature_csv}`",
        f"- Baseline models: `{model_out_dir}`",
        f"- Calibrated models: `{calibrated_model_out_dir}`",
        "- Enhanced metrics: `reports/tables/baseline_metrics_enhanced.csv`",
        "- Baseline comparison: `reports/tables/baseline_comparison.csv`",
        "- Threshold analysis: `reports/tables/threshold_analysis.csv`",
        "- Upgrade folder: `reports/classical_baseline_upgrade/`",
        "",
        best_line,
        speaker_best_line,
        duration_line,
        subset_line,
        family_ablation_line,
        "",
        "Known limitation: this is still a small observational speech dataset. Speaker grouping reduces leakage but does not prove clinical generalization.",
    ]
    (STANDARD_REPORT_DIR / "phase1_final_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    final_upgrade_lines = [
        "# Classical Baseline Upgrade Summary",
        "",
        "## Split validity",
        "",
        f"- Held-out test valid: **{heldout_test_valid}**",
        f"- Evaluation split used: **{eval_split}**",
        f"- Split note: {eval_reason}",
        "",
        dataframe_to_markdown(split_df),
        "",
        "## Calibration",
        "",
        "- SVM now uses `probability=False`; external calibration is applied once.",
        "- Calibration and threshold selection use disjoint validation speaker subsets when feasible.",
        "",
        "## Tuning and evaluation",
        "",
        "- Added speaker-grouped hyperparameter tuning for Logistic Regression, SVM RBF, and Random Forest.",
        "- Added speaker-level evaluation by mean, median, and max aggregation.",
        "",
        best_line,
        speaker_best_line,
        "",
        "## Confound checks",
        "",
        duration_line,
        subset_line,
        family_ablation_line,
        "- See `confound_checks.csv` and `ablation_results.csv` for duration/energy/silence shortcut checks.",
        "- See `feature_family_ablation_results.csv` for MFCC, F0/prosody, pause/fluency, voice-quality proxy, and energy-family removal diagnostics.",
        "",
        "## Best classical baseline recommendation",
        "",
        f"Use `{best_model_name}` calibrated with the frozen high-recall threshold for the current classical baseline, while reporting both clip-level and speaker-level metrics.",
        "",
        "## Remaining limitations",
        "",
        "- Small dataset and moderate class imbalance.",
        "- Speaker IDs are metadata/path-based, not biometric verification.",
        "- Language/accent metadata is incomplete, so fairness/generalization cannot be fully audited.",
        "- Thresholds are screening-oriented and will increase false positives when recall is prioritized.",
    ]
    (out_dir / "final_upgrade_summary.md").write_text("\n".join(final_upgrade_lines) + "\n", encoding="utf-8")


def train_and_evaluate(
    feature_csv: Path = Path("data/features/file_features.csv"),
    model_out_dir: Path = CFG.baseline_model_dir,
) -> pd.DataFrame:
    """Train the legacy validation baseline models and save artifacts."""
    df = load_feature_table(feature_csv)
    feat_cols = select_feature_columns(df)
    if not feat_cols:
        raise ValueError("No usable feature columns found.")

    train_df = df[df["datasplit"] == "train"].copy()
    valid_df = df[df["datasplit"] == "valid"].copy()

    if train_df["label"].nunique() < 2:
        raise ValueError(f"Train split must contain both classes. Observed: {train_df['label'].value_counts().to_dict()}")
    if valid_df["label"].nunique() < 2:
        raise ValueError(f"Valid split must contain both classes. Observed: {valid_df['label'].value_counts().to_dict()}")

    x_train = train_df[feat_cols]
    y_train = train_df["label"].astype(int).to_numpy()
    x_valid = valid_df[feat_cols]
    y_valid = valid_df["label"].astype(int).to_numpy()

    model_out_dir.mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    metrics_by_model: Dict[str, Dict[str, float]] = {}
    feature_rows: List[Dict[str, object]] = []

    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        y_prob = predict_positive_probability(model, x_valid)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics_by_model[model_name] = classification_metrics(y_valid, y_pred, y_prob)

        joblib.dump(model, model_out_dir / f"{model_name}.joblib")
        save_confusion_matrix_figure(
            y_valid,
            y_pred,
            out_path=Path("reports/figures") / f"{model_name}_confusion_matrix.png",
            title=f"{model_name} - Valid confusion matrix",
        )
        save_feature_importances(model_name, model, feat_cols, feature_rows)

    metrics_df = pd.DataFrame.from_dict(metrics_by_model, orient="index").reset_index(names="model")
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(Path("reports/tables/baseline_metrics.csv"), index=False)
    if feature_rows:
        pd.DataFrame(feature_rows).sort_values("importance", ascending=False).to_csv(
            Path("reports/tables/feature_importance_baseline.csv"), index=False
        )
    return metrics_df


def train_and_evaluate_enhanced(
    feature_csv: Path = Path("data/features/features_enhanced.csv"),
    model_out_dir: Path = Path("models/baseline_enhanced_v2"),
    calibrated_model_out_dir: Path = Path("models/calibrated_v2"),
    metrics_out: Path = Path("reports/tables/baseline_metrics_enhanced.csv"),
    eval_split_request: str = "auto",
    target_recall: float = 0.80,
    bootstrap_iterations: int = 500,
    random_seed: int = RANDOM_SEED,
    upgrade_report_dir: Path = UPGRADE_REPORT_DIR,
) -> pd.DataFrame:
    """Train upgraded classical baselines with leakage-aware evaluation."""
    df = load_feature_table(feature_csv, allowed_splits=("train", "valid", "test"))
    feat_cols = select_feature_columns(df)
    if not feat_cols:
        raise ValueError("No usable feature columns found.")

    eval_split, eval_reason, heldout_test_valid = choose_evaluation_split(df, eval_split_request)
    validate_training_splits(df, eval_split)

    upgrade_report_dir.mkdir(parents=True, exist_ok=True)
    UPGRADE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    STANDARD_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model_out_dir.mkdir(parents=True, exist_ok=True)
    calibrated_model_out_dir.mkdir(parents=True, exist_ok=True)

    split_df, _ = write_precheck_summary(
        df=df,
        feature_csv=feature_csv,
        out_dir=upgrade_report_dir,
        eval_split=eval_split,
        eval_reason=eval_reason,
        heldout_test_valid=heldout_test_valid,
    )

    train_df = df[df["datasplit"] == "train"].copy()
    valid_df = df[df["datasplit"] == "valid"].copy()
    eval_df = df[df["datasplit"] == eval_split].copy()

    calibration_df, threshold_df_internal, split_notes = split_validation_for_calibration_threshold(
        valid_df,
        seed=random_seed,
    )
    write_calibration_method_notes(upgrade_report_dir, split_notes)

    x_train = train_df[feat_cols]
    y_train = train_df["label"].astype(int).to_numpy()
    train_groups = train_df["speaker_id"].astype(str).to_numpy()
    x_cal = calibration_df[feat_cols]
    y_cal = calibration_df["label"].astype(int).to_numpy()
    x_thr = threshold_df_internal[feat_cols]
    y_thr = threshold_df_internal["label"].astype(int).to_numpy()
    x_eval = eval_df[feat_cols]
    y_eval = eval_df["label"].astype(int).to_numpy()
    eval_speakers = eval_df["speaker_id"].astype(str).to_numpy()

    tuned_models, search_df, best_config_df = tune_models_with_group_cv(
        x_train=x_train,
        y_train=y_train,
        groups=train_groups,
        out_dir=upgrade_report_dir,
        seed=random_seed,
    )

    metrics_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    clip_prediction_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []

    for model_name in MODEL_ORDER:
        model = tuned_models[model_name]
        save_feature_importances(model_name, model, feat_cols, feature_rows)
        joblib.dump(model, model_out_dir / f"{model_name}.joblib")

        cal_scores, calibration_score_type = model_score_for_calibration(model, x_cal)
        calibrator = LogisticRegression(max_iter=2000, random_state=random_seed)
        calibrator.fit(cal_scores.reshape(-1, 1), y_cal)

        thr_scores, _ = model_score_for_calibration(model, x_thr)
        thr_prob_calibrated = calibrator.predict_proba(thr_scores.reshape(-1, 1))[:, 1]
        threshold, threshold_metrics = select_high_recall_threshold(
            y_thr,
            thr_prob_calibrated,
            target_recall=target_recall,
        )

        eval_prob_uncalibrated = predict_positive_probability(model, x_eval)
        eval_pred_uncalibrated = (eval_prob_uncalibrated >= 0.5).astype(int)
        eval_scores, _ = model_score_for_calibration(model, x_eval)
        eval_prob_calibrated = calibrator.predict_proba(eval_scores.reshape(-1, 1))[:, 1]
        eval_pred_calibrated = (eval_prob_calibrated >= threshold).astype(int)

        cal_prob_uncal = predict_positive_probability(model, x_cal)
        cal_prob_cal = calibrator.predict_proba(cal_scores.reshape(-1, 1))[:, 1]
        cal_uncal_metrics = classification_metrics(y_cal, (cal_prob_uncal >= 0.5).astype(int), cal_prob_uncal)
        cal_cal_metrics = classification_metrics(y_cal, (cal_prob_cal >= 0.5).astype(int), cal_prob_cal)

        threshold_rows.append(
            {
                "model": model_name,
                "calibration_method": "external_platt_logistic",
                "calibration_score_type": calibration_score_type,
                "selected_threshold": threshold,
                "target_recall": target_recall,
                "selection_note": threshold_metrics.get("selection_note", ""),
                "threshold_selection_recall": threshold_metrics.get("recall", np.nan),
                "threshold_selection_specificity": threshold_metrics.get("specificity", np.nan),
                "threshold_selection_precision": threshold_metrics.get("precision", np.nan),
                "threshold_selection_f1": threshold_metrics.get("f1", np.nan),
                "threshold_selection_pr_auc": threshold_metrics.get("pr_auc", np.nan),
                "threshold_selection_roc_auc": threshold_metrics.get("roc_auc", np.nan),
                "calibration_brier_uncalibrated": cal_uncal_metrics.get("brier_score", np.nan),
                "calibration_brier_calibrated": cal_cal_metrics.get("brier_score", np.nan),
                "n_calibration": len(calibration_df),
                "n_threshold_selection": len(threshold_df_internal),
                "n_calibration_speakers": calibration_df["speaker_id"].nunique(),
                "n_threshold_selection_speakers": threshold_df_internal["speaker_id"].nunique(),
                "validation_split_method": split_notes.get("split_method", ""),
                "validation_split_fallback_used": split_notes.get("fallback_used", False),
                "validation_split_fallback_reason": split_notes.get("fallback_reason", ""),
            }
        )

        best_config = best_config_df[best_config_df["model"] == model_name]
        best_params = best_config.iloc[0].get("best_params", "{}") if not best_config.empty else "{}"
        best_score = best_config.iloc[0].get("best_score", np.nan) if not best_config.empty else np.nan

        calibrated_package = {
            "base_model": model,
            "calibrator": calibrator,
            "feature_columns": feat_cols,
            "positive_class": "dementia",
            "positive_label": 1,
            "threshold": threshold,
            "target_recall": target_recall,
            "calibration_method": "external_platt_logistic",
            "calibration_score_type": calibration_score_type,
            "validation_split_method": split_notes.get("split_method", ""),
            "eval_split": eval_split,
            "random_seed": random_seed,
            "best_params": best_params,
        }
        joblib.dump(calibrated_package, calibrated_model_out_dir / f"{model_name}_calibrated.joblib")

        variants = [
            ("uncalibrated", eval_prob_uncalibrated, eval_pred_uncalibrated, 0.5, "none"),
            ("calibrated", eval_prob_calibrated, eval_pred_calibrated, threshold, "external_platt_logistic"),
        ]
        for variant, eval_prob, eval_pred, used_threshold, calibration_method in variants:
            metrics = classification_metrics(y_eval, eval_pred, eval_prob)
            metrics.update(
                bootstrap_metric_cis(
                    y_eval,
                    eval_pred,
                    eval_prob,
                    speaker_ids=eval_speakers,
                    n_bootstrap=bootstrap_iterations,
                    seed=random_seed,
                )
            )
            metrics_rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "eval_split": eval_split,
                    "eval_level": "clip",
                    "positive_class": "dementia",
                    "threshold": used_threshold,
                    "calibration_method": calibration_method,
                    "calibration_score_type": calibration_score_type if variant == "calibrated" else "not_calibrated",
                    "n_train": len(train_df),
                    "n_valid": len(valid_df),
                    "n_calibration": len(calibration_df),
                    "n_threshold_selection": len(threshold_df_internal),
                    "n_eval": len(eval_df),
                    "n_eval_speakers": eval_df["speaker_id"].nunique(),
                    "tuning_scoring": "average_precision",
                    "tuning_best_score": best_score,
                    "tuning_best_params": best_params,
                    **metrics,
                }
            )

            figure_prefix = f"{model_name}_{variant}_v2"
            save_confusion_matrix_figure(
                y_eval,
                eval_pred,
                out_path=UPGRADE_FIGURE_DIR / f"{figure_prefix}_confusion_matrix.png",
                title=f"{model_name} {variant} - {eval_split} confusion matrix",
            )
            save_roc_curve_figure(
                y_eval,
                eval_prob,
                out_path=UPGRADE_FIGURE_DIR / f"{figure_prefix}_roc_curve.png",
                title=f"{model_name} {variant} - {eval_split} ROC",
            )
            save_pr_curve_figure(
                y_eval,
                eval_prob,
                out_path=UPGRADE_FIGURE_DIR / f"{figure_prefix}_pr_curve.png",
                title=f"{model_name} {variant} - {eval_split} precision-recall",
            )
            save_calibration_curve_figure(
                y_eval,
                eval_prob,
                out_path=UPGRADE_FIGURE_DIR / f"{figure_prefix}_calibration_curve.png",
                title=f"{model_name} {variant} - {eval_split} calibration",
            )

            for row_idx, (_, eval_row) in enumerate(eval_df.iterrows()):
                clip_prediction_rows.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "metadata_index": eval_row.get("metadata_index"),
                        "speaker_id": eval_row.get("speaker_id"),
                        "audio_file": eval_row.get("audio_file"),
                        "true_label": int(eval_row.get("label")),
                        "eval_split": eval_split,
                        "probability": float(eval_prob[row_idx]),
                        "prediction": int(eval_pred[row_idx]),
                        "threshold": float(used_threshold),
                        "calibration_method": calibration_method,
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(STANDARD_TABLE_DIR / "threshold_analysis.csv", index=False)
    threshold_df.to_csv(upgrade_report_dir / "threshold_selection_summary.csv", index=False)

    clip_predictions_df = pd.DataFrame(clip_prediction_rows)
    clip_predictions_df.to_csv(upgrade_report_dir / "clip_level_predictions.csv", index=False)
    make_wide_prediction_table(clip_predictions_df).to_csv(
        STANDARD_TABLE_DIR / "baseline_enhanced_predictions.csv",
        index=False,
    )

    if feature_rows:
        pd.DataFrame(feature_rows).sort_values("importance", ascending=False).to_csv(
            STANDARD_TABLE_DIR / "feature_importance_baseline_enhanced.csv",
            index=False,
        )

    speaker_predictions_df, speaker_metrics_df = speaker_level_outputs(
        clip_predictions=clip_predictions_df,
        out_dir=upgrade_report_dir,
        figure_dir=UPGRADE_FIGURE_DIR,
    )

    confound_checks_df, ablation_df = run_confound_checks(
        df=df,
        feat_cols=feat_cols,
        tuned_models=tuned_models,
        train_df=train_df,
        eval_df=eval_df,
        out_dir=upgrade_report_dir,
        seed=random_seed,
    )

    comparison_df = build_comparison_table(metrics_df, STANDARD_TABLE_DIR / "baseline_comparison.csv")
    comparison_df.to_csv(upgrade_report_dir / "baseline_comparison_v2.csv", index=False)

    write_reports(
        metrics_df=metrics_df,
        threshold_df=threshold_df,
        comparison_df=comparison_df,
        speaker_metrics_df=speaker_metrics_df,
        confound_checks_df=confound_checks_df,
        ablation_df=ablation_df,
        split_df=split_df,
        eval_split=eval_split,
        eval_reason=eval_reason,
        heldout_test_valid=heldout_test_valid,
        feature_csv=feature_csv,
        model_out_dir=model_out_dir,
        calibrated_model_out_dir=calibrated_model_out_dir,
        target_recall=target_recall,
        out_dir=upgrade_report_dir,
    )

    return metrics_df


def parse_args() -> argparse.Namespace:
    """CLI arguments for baseline training."""
    parser = argparse.ArgumentParser(description="Train classical ML baseline models on extracted features.")
    parser.add_argument("--feature-csv", type=str, default="data/features/file_features.csv")
    parser.add_argument("--model-out-dir", type=str, default=str(CFG.baseline_model_dir))
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Run corrected-split enhanced classical baseline training, calibration, and evaluation.",
    )
    parser.add_argument("--calibrated-model-out-dir", type=str, default="models/calibrated_v2")
    parser.add_argument("--metrics-out", type=str, default="reports/tables/baseline_metrics_enhanced.csv")
    parser.add_argument("--eval-split", type=str, default="auto", choices=["auto", "valid", "test"])
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=CFG.random_seed)
    parser.add_argument("--upgrade-report-dir", type=str, default=str(UPGRADE_REPORT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.enhanced:
        results = train_and_evaluate_enhanced(
            feature_csv=Path(args.feature_csv),
            model_out_dir=Path(args.model_out_dir),
            calibrated_model_out_dir=Path(args.calibrated_model_out_dir),
            metrics_out=Path(args.metrics_out),
            eval_split_request=args.eval_split,
            target_recall=args.target_recall,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.random_seed,
            upgrade_report_dir=Path(args.upgrade_report_dir),
        )
    else:
        results = train_and_evaluate(Path(args.feature_csv), Path(args.model_out_dir))
    print(results)
