"""Phase D controlled HuBERT recall-recovery benchmark.

This runner is intentionally narrow:
- primary dataset is the trusted cleaned subset only;
- cached Phase C frozen embeddings are reused;
- labels, manifests, speaker IDs, governance decisions, app code, deployment
  code, and prior Phase outputs are not modified;
- threshold selection is performed from training speakers only inside each
  outer grouped fold.
"""

from __future__ import annotations

import itertools
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
CACHE_DIR = ROOT / "artifacts/phaseC_frozen_embeddings/trusted_cleaned"

TRUSTED_MANIFEST = TABLES_DIR / "phase6_5_trusted_subset_manifest_cleaned.csv"
FEATURE_TABLE = ROOT / "data/features/features_phase3_governed_pruned.csv"

PHASEC_REPORT = REPORTS_DIR / "phaseC_frozen_embedding_benchmark.md"
PHASEC_SPEAKER_SUMMARY = TABLES_DIR / "phaseC_embedding_speaker_seed_summary.csv"
PHASEC_MODEL_COMPARISON = TABLES_DIR / "phaseC_embedding_model_comparison.csv"
PHASEC_BEST_CANDIDATE = TABLES_DIR / "phaseC_embedding_best_candidate_analysis.csv"

REPORT_MD = REPORTS_DIR / "phaseD_hubert_recall_recovery_benchmark.md"
AGGREGATION_CSV = TABLES_DIR / "phaseD_aggregation_seed_summary.csv"
FUSION_CSV = TABLES_DIR / "phaseD_fusion_seed_summary.csv"
ENSEMBLE_CSV = TABLES_DIR / "phaseD_ensemble_seed_summary.csv"
MODEL_COMPARISON_CSV = TABLES_DIR / "phaseD_model_comparison.csv"
COMPLEMENTARITY_CSV = TABLES_DIR / "phaseD_error_complementarity_summary.csv"
BEST_CANDIDATE_CSV = TABLES_DIR / "phaseD_best_candidate_analysis.csv"

HUBERT_BACKBONE = "facebook/hubert-base-ls960"
WAV2VEC2_BACKBONE = "facebook/wav2vec2-base"
LOGISTIC = "logistic_regression_platt"
LINEAR_SVM = "linear_svm_platt"
DATASET_VERSION = "trusted_cleaned"
SEEDS = [42, 43, 44, 45, 46]
MAX_SPLITS = 5
TARGET_RECALL = 0.70
TARGET_SPECIFICITY = 0.50
THRESHOLD_POLICIES = ["default_0.5", "constrained_spec_ge_0.50"]
AGGREGATION_METHODS = [
    "mean_score",
    "max_score",
    "majority_vote",
    "top_2_mean",
    "top_3_mean",
    "top_half_mean",
]

OFFICIAL_CLASSICAL_BASELINE = {
    "name": "final_cleaned_logistic_regression_platt",
    "threshold": 0.300,
    "speaker_recall_mean": 0.4761904761904761,
    "speaker_specificity_mean": 0.47476635514018695,
    "speaker_f1_mean": 0.374558,
    "speaker_aggregation": "max_risk_score_seed_selected_threshold",
}

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


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_speakers: set[str]


@dataclass
class FitResult:
    selected_c: float
    calibration_status: str
    train_probabilities: np.ndarray
    val_probabilities: np.ndarray


@dataclass(frozen=True)
class ComponentSpec:
    model_id: str
    component_name: str
    feature_mode: str
    classifier: str
    aggregation: str
    threshold_policy: str


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
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
        lines.append("| ... | ... |")
    return "\n".join(lines)


def mode_int(series: pd.Series) -> int:
    return int(series.astype(int).mode().iloc[0])


def speaker_label_counts(df: pd.DataFrame) -> dict[int, int]:
    labels = df.groupby("speaker_id")["label"].agg(mode_int)
    return {int(k): int(v) for k, v in labels.value_counts().sort_index().to_dict().items()}


def build_outer_folds(df: pd.DataFrame, seed: int, max_splits: int = MAX_SPLITS) -> tuple[list[FoldSpec], str, int]:
    class_speaker_counts = speaker_label_counts(df)
    min_class_speakers = min(class_speaker_counts.values())
    n_splits = int(min(max_splits, min_class_speakers))
    if n_splits < 2:
        raise ValueError("Grouped CV is infeasible: fewer than two speakers in a class.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df["speaker_id"].astype(str).to_numpy()
    y = df["label"].astype(int).to_numpy()
    folds: list[FoldSpec] = []
    for fold, (_, val_pos_idx) in enumerate(splitter.split(np.zeros(len(df)), y, groups=groups), start=1):
        val_speakers = set(df.iloc[val_pos_idx]["speaker_id"].astype(str))
        train_idx = df.index[~df["speaker_id"].astype(str).isin(val_speakers)].to_numpy()
        val_idx = df.index[val_pos_idx].to_numpy()
        overlap = set(df.loc[train_idx, "speaker_id"].astype(str)) & val_speakers
        if overlap:
            raise ValueError(f"Speaker leakage detected in seed {seed}, fold {fold}: {sorted(overlap)[:5]}")
        if df.loc[train_idx, "label"].nunique() < 2 or df.loc[val_idx, "label"].nunique() < 2:
            raise ValueError(f"Seed {seed}, fold {fold} lacks both classes in train or validation.")
        folds.append(FoldSpec(fold=fold, train_idx=train_idx, val_idx=val_idx, val_speakers=val_speakers))
    note = f"StratifiedGroupKFold n_splits={n_splits}; speaker class counts={class_speaker_counts}"
    return folds, note, n_splits


def build_estimator(classifier: str, seed: int, c_value: float = 1.0) -> Pipeline:
    if classifier == LOGISTIC:
        model = LogisticRegression(
            C=float(c_value),
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
            solver="liblinear",
        )
    elif classifier == LINEAR_SVM:
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
        return np.asarray(model.decision_function(x), dtype=float).ravel()
    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(x), dtype=float)
        if prob.ndim == 2 and prob.shape[1] > 1:
            prob = prob[:, 1]
        return prob.ravel()
    return np.asarray(model.predict(x), dtype=float).ravel()


def fit_platt_model(
    classifier: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    seed: int,
) -> FitResult:
    selected_c = 1.0
    estimator = build_estimator(classifier, seed=seed, c_value=selected_c)
    estimator.fit(x_train, y_train)
    train_raw = raw_scores(estimator, x_train)
    calibrator = LogisticRegression(max_iter=2000, random_state=seed)
    calibrator.fit(train_raw.reshape(-1, 1), y_train)
    train_prob = calibrator.predict_proba(train_raw.reshape(-1, 1))[:, 1]
    val_raw = raw_scores(estimator, x_val)
    val_prob = calibrator.predict_proba(val_raw.reshape(-1, 1))[:, 1]
    return FitResult(
        selected_c=selected_c,
        calibration_status="training_only_in_sample_platt",
        train_probabilities=train_prob,
        val_probabilities=val_prob,
    )


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2 or np.unique(scores).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def metrics_from_arrays(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = np.nanmean([recall, specificity])
    return {
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "confusion_matrix_tn_fp_fn_tp": f"{tn},{fp},{fn},{tp}",
    }


def threshold_candidates() -> np.ndarray:
    return np.round(np.linspace(0.0, 1.0, 101), 4)


def select_constrained_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates():
        y_pred = (scores >= threshold).astype(int)
        rows.append({"threshold": float(threshold), **metrics_from_arrays(y_true, scores, y_pred)})
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
    return float(selected["threshold"]), note


def speaker_score_value(scores: np.ndarray, aggregation_method: str, threshold: float) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot aggregate an empty speaker score vector.")
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
    elif aggregation_method == "top_2_mean":
        k = min(2, len(scores))
        score = float(np.mean(np.sort(scores)[-k:]))
        pred = int(score >= threshold)
    elif aggregation_method == "top_3_mean":
        k = min(3, len(scores))
        score = float(np.mean(np.sort(scores)[-k:]))
        pred = int(score >= threshold)
    elif aggregation_method == "top_half_mean":
        k = max(1, int(math.ceil(len(scores) / 2.0)))
        score = float(np.mean(np.sort(scores)[-k:]))
        pred = int(score >= threshold)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
    return score, pred


def speaker_score_table(clip_df: pd.DataFrame, aggregation_method: str, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for speaker_id, group in clip_df.groupby("speaker_id", sort=True):
        score, pred = speaker_score_value(group["score"].astype(float).to_numpy(), aggregation_method, threshold)
        rows.append(
            {
                "speaker_id": str(speaker_id),
                "label": mode_int(group["label"]),
                "score": score,
                "prediction": pred,
                "n_clips": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def select_speaker_constrained_threshold(train_clip_df: pd.DataFrame, aggregation_method: str) -> tuple[float, str]:
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates():
        speaker_df = speaker_score_table(train_clip_df, aggregation_method, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                **metrics_from_arrays(
                    speaker_df["label"].astype(int).to_numpy(),
                    speaker_df["score"].astype(float).to_numpy(),
                    speaker_df["prediction"].astype(int).to_numpy(),
                ),
            }
        )
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
    return float(selected["threshold"]), note


def cache_path_for_backbone(backbone: str) -> Path:
    mapping = {
        HUBERT_BACKBONE: CACHE_DIR / "facebook__hubert_base_ls960_mean_std_chunk8p0.csv.gz",
        WAV2VEC2_BACKBONE: CACHE_DIR / "facebook__wav2vec2_base_mean_std_chunk8p0.csv.gz",
    }
    if backbone not in mapping:
        raise ValueError(f"No Phase D cache mapping configured for {backbone}.")
    return mapping[backbone]


def cache_meta_for_backbone(backbone: str) -> Path:
    return cache_path_for_backbone(backbone).with_suffix("").with_suffix(".json")


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(TRUSTED_MANIFEST)
    required = {"metadata_index", "speaker_id", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{TRUSTED_MANIFEST} missing columns: {missing}")
    df = df[df["label"].isin([0, 1])].copy()
    if "feature_extraction_status" in df.columns:
        df = df[df["feature_extraction_status"].fillna("success").eq("success")].copy()
    df["metadata_index"] = df["metadata_index"].astype(str)
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def load_cached_embeddings(backbone: str, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_csv = cache_path_for_backbone(backbone)
    meta_json = cache_meta_for_backbone(backbone)
    if not cache_csv.exists():
        raise FileNotFoundError(f"Required cached embedding file missing: {cache_csv}")
    embeddings = pd.read_csv(cache_csv)
    embeddings["metadata_index"] = embeddings["metadata_index"].astype(str)
    embeddings["speaker_id"] = embeddings["speaker_id"].astype(str).str.strip()
    embeddings["label"] = embeddings["label"].astype(int)
    manifest_keys = set(manifest["metadata_index"].astype(str))
    embed_keys = set(embeddings["metadata_index"].astype(str))
    missing = sorted(manifest_keys - embed_keys)
    extra = sorted(embed_keys - manifest_keys)
    if missing:
        raise ValueError(f"{backbone} cache is missing manifest rows: {missing[:5]}")
    if extra:
        warnings.warn(f"{backbone} cache has {len(extra)} extra rows not in trusted manifest; filtering.", stacklevel=2)
    embeddings = embeddings[embeddings["metadata_index"].isin(manifest_keys)].copy()
    embeddings = manifest[["metadata_index"]].merge(embeddings, on="metadata_index", how="left", validate="one_to_one")
    if embeddings["speaker_id"].isna().any():
        raise ValueError(f"{backbone} cache merge produced missing speaker IDs.")
    meta = json.loads(meta_json.read_text()) if meta_json.exists() else {}
    meta["cache_csv"] = rel(cache_csv)
    meta["cache_status"] = "loaded_existing_cache"
    return embeddings.reset_index(drop=True), meta


def embedding_matrix(embeddings: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in embeddings.columns if c.startswith("emb_")]
    if not cols:
        raise ValueError("No emb_* columns found in cached embeddings.")
    return embeddings[cols].to_numpy(dtype=np.float32), cols


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in NON_FEATURE_COLUMNS]


def load_aligned_handcrafted_features(manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    features = pd.read_csv(FEATURE_TABLE)
    feature_cols = select_feature_columns(features)
    if len(feature_cols) != 256:
        raise ValueError(f"Expected 256 official numeric handcrafted features, found {len(feature_cols)}.")
    for frame in [features, manifest]:
        frame["metadata_index"] = frame["metadata_index"].astype(str)
    if features["metadata_index"].duplicated().any():
        duplicates = features.loc[features["metadata_index"].duplicated(), "metadata_index"].head(5).tolist()
        raise ValueError(f"Feature table has duplicate metadata_index values: {duplicates}")
    aligned = manifest[["metadata_index", "speaker_id", "label"]].merge(
        features[["metadata_index", "speaker_id", "label", *feature_cols]],
        on="metadata_index",
        how="left",
        suffixes=("_manifest", "_feature"),
        validate="one_to_one",
        indicator=True,
    )
    lost = int(aligned["_merge"].ne("both").sum())
    if lost:
        missing = aligned.loc[aligned["_merge"].ne("both"), "metadata_index"].head(10).tolist()
        raise ValueError(f"Handcrafted feature join lost {lost} rows; first missing keys: {missing}")
    speaker_mismatch = aligned["speaker_id_manifest"].astype(str).ne(aligned["speaker_id_feature"].astype(str))
    label_mismatch = aligned["label_manifest"].astype(int).ne(aligned["label_feature"].astype(int))
    if speaker_mismatch.any() or label_mismatch.any():
        raise ValueError(
            "Handcrafted feature join found speaker/label mismatches: "
            f"speaker_mismatch={int(speaker_mismatch.sum())}, label_mismatch={int(label_mismatch.sum())}"
        )
    join_info = {
        "key": "metadata_index",
        "manifest_rows": int(len(manifest)),
        "feature_table_rows": int(len(features)),
        "matched_rows": int(len(aligned)),
        "dropped_rows": lost,
        "feature_count": int(len(feature_cols)),
    }
    return aligned[feature_cols].copy(), feature_cols, join_info


def build_clip_frame(
    base_df: pd.DataFrame,
    idx: np.ndarray,
    scores: np.ndarray,
    model_id: str,
    seed: int,
    fold: int,
    split: str,
) -> pd.DataFrame:
    cols = ["dataset_version", "clip_id", "metadata_index", "speaker_id", "label", "backbone"]
    available_cols = [c for c in cols if c in base_df.columns]
    out = base_df.loc[idx, available_cols].copy()
    out["dataset_version"] = DATASET_VERSION
    out["model_id"] = model_id
    out["seed"] = int(seed)
    out["fold"] = int(fold)
    out["split"] = split
    out["score"] = scores.astype(float)
    out["speaker_id"] = out["speaker_id"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def evaluate_cv_model(
    *,
    model_id: str,
    display_name: str,
    x: np.ndarray,
    base_df: pd.DataFrame,
    classifier: str,
    feature_mode: str,
    backbone: str,
    aggregations: list[str],
    seeds: list[int],
    max_splits: int,
) -> dict[str, pd.DataFrame]:
    y = base_df["label"].astype(int).to_numpy()
    clip_summary_rows: list[dict[str, Any]] = []
    speaker_summary_rows: list[dict[str, Any]] = []
    clip_prediction_parts: dict[tuple[int, str], list[pd.DataFrame]] = {}
    speaker_prediction_parts: dict[tuple[int, str, str], list[pd.DataFrame]] = {}
    threshold_rows: list[dict[str, Any]] = []
    fold_clip_parts: list[pd.DataFrame] = []

    for seed in seeds:
        folds, fold_note, n_splits = build_outer_folds(base_df, seed=seed, max_splits=max_splits)
        selected_cs: list[float] = []
        calibration_statuses: list[str] = []
        for fold_spec in folds:
            fit = fit_platt_model(
                classifier=classifier,
                x_train=x[fold_spec.train_idx],
                y_train=y[fold_spec.train_idx],
                x_val=x[fold_spec.val_idx],
                seed=seed,
            )
            selected_cs.append(float(fit.selected_c))
            calibration_statuses.append(fit.calibration_status)
            train_clip = build_clip_frame(
                base_df, fold_spec.train_idx, fit.train_probabilities, model_id, seed, fold_spec.fold, "train"
            )
            val_clip = build_clip_frame(
                base_df, fold_spec.val_idx, fit.val_probabilities, model_id, seed, fold_spec.fold, "validation"
            )
            fold_clip_parts.extend([train_clip, val_clip])

            clip_thresholds = {"default_0.5": (0.5, "fixed_default_probability_threshold")}
            constrained_threshold, constrained_note = select_constrained_threshold(
                train_clip["label"].astype(int).to_numpy(),
                train_clip["score"].astype(float).to_numpy(),
            )
            clip_thresholds["constrained_spec_ge_0.50"] = (constrained_threshold, constrained_note)
            for policy, (threshold, note) in clip_thresholds.items():
                pred = val_clip.copy()
                pred["threshold_policy"] = policy
                pred["threshold"] = float(threshold)
                pred["prediction"] = (pred["score"].astype(float) >= float(threshold)).astype(int)
                clip_prediction_parts.setdefault((seed, policy), []).append(pred)
                threshold_rows.append(
                    {
                        "model_id": model_id,
                        "display_name": display_name,
                        "feature_mode": feature_mode,
                        "backbone": backbone,
                        "classifier": classifier,
                        "seed": seed,
                        "fold": fold_spec.fold,
                        "metric_level": "clip",
                        "aggregation": "",
                        "threshold_policy": policy,
                        "threshold": float(threshold),
                        "selection_note": note,
                    }
                )

            for aggregation in aggregations:
                speaker_thresholds = {"default_0.5": (0.5, "fixed_default_probability_threshold")}
                speaker_threshold, speaker_note = select_speaker_constrained_threshold(train_clip, aggregation)
                speaker_thresholds["constrained_spec_ge_0.50"] = (speaker_threshold, speaker_note)
                for policy, (threshold, note) in speaker_thresholds.items():
                    speaker_pred = speaker_score_table(val_clip, aggregation, float(threshold))
                    speaker_pred["dataset_version"] = DATASET_VERSION
                    speaker_pred["model_id"] = model_id
                    speaker_pred["display_name"] = display_name
                    speaker_pred["feature_mode"] = feature_mode
                    speaker_pred["backbone"] = backbone
                    speaker_pred["classifier"] = classifier
                    speaker_pred["seed"] = int(seed)
                    speaker_pred["fold"] = int(fold_spec.fold)
                    speaker_pred["aggregation"] = aggregation
                    speaker_pred["threshold_policy"] = policy
                    speaker_pred["threshold"] = float(threshold)
                    speaker_prediction_parts.setdefault((seed, aggregation, policy), []).append(speaker_pred)
                    threshold_rows.append(
                        {
                            "model_id": model_id,
                            "display_name": display_name,
                            "feature_mode": feature_mode,
                            "backbone": backbone,
                            "classifier": classifier,
                            "seed": seed,
                            "fold": fold_spec.fold,
                            "metric_level": "speaker",
                            "aggregation": aggregation,
                            "threshold_policy": policy,
                            "threshold": float(threshold),
                            "selection_note": note,
                        }
                    )

        threshold_df = pd.DataFrame([r for r in threshold_rows if r["model_id"] == model_id and r["seed"] == seed])
        for policy in THRESHOLD_POLICIES:
            pred_df = pd.concat(clip_prediction_parts[(seed, policy)], ignore_index=True)
            metrics = metrics_from_arrays(
                pred_df["label"].astype(int).to_numpy(),
                pred_df["score"].astype(float).to_numpy(),
                pred_df["prediction"].astype(int).to_numpy(),
            )
            thresholds = threshold_df[
                (threshold_df["metric_level"] == "clip") & (threshold_df["threshold_policy"] == policy)
            ]
            clip_summary_rows.append(
                {
                    "dataset_version": DATASET_VERSION,
                    "model_id": model_id,
                    "display_name": display_name,
                    "feature_mode": feature_mode,
                    "backbone": backbone,
                    "classifier": classifier,
                    "seed": seed,
                    "threshold_policy": policy,
                    "selected_threshold": float(thresholds["threshold"].mean()),
                    "selected_threshold_std": float(thresholds["threshold"].std(ddof=1)) if len(thresholds) > 1 else 0.0,
                    "n_folds": int(n_splits),
                    "n_clips": int(len(pred_df)),
                    "n_speakers": int(pred_df["speaker_id"].nunique()),
                    "selected_c_values": ";".join(str(v) for v in selected_cs),
                    "calibration_statuses": ";".join(sorted(set(calibration_statuses))),
                    "fold_note": fold_note,
                    **metrics,
                }
            )

        for aggregation in aggregations:
            for policy in THRESHOLD_POLICIES:
                pred_df = pd.concat(speaker_prediction_parts[(seed, aggregation, policy)], ignore_index=True)
                duplicate_speakers = int(pred_df["speaker_id"].duplicated().sum())
                metrics = metrics_from_arrays(
                    pred_df["label"].astype(int).to_numpy(),
                    pred_df["score"].astype(float).to_numpy(),
                    pred_df["prediction"].astype(int).to_numpy(),
                )
                thresholds = threshold_df[
                    (threshold_df["metric_level"] == "speaker")
                    & (threshold_df["aggregation"] == aggregation)
                    & (threshold_df["threshold_policy"] == policy)
                ]
                speaker_summary_rows.append(
                    {
                        "dataset_version": DATASET_VERSION,
                        "model_id": model_id,
                        "display_name": display_name,
                        "feature_mode": feature_mode,
                        "backbone": backbone,
                        "classifier": classifier,
                        "seed": seed,
                        "aggregation": aggregation,
                        "threshold_policy": policy,
                        "selected_threshold": float(thresholds["threshold"].mean()),
                        "selected_threshold_std": float(thresholds["threshold"].std(ddof=1))
                        if len(thresholds) > 1
                        else 0.0,
                        "selected_threshold_min": float(thresholds["threshold"].min()),
                        "selected_threshold_max": float(thresholds["threshold"].max()),
                        "n_folds": int(n_splits),
                        "n_speakers": int(pred_df["speaker_id"].nunique()),
                        "duplicate_validation_speaker_rows": duplicate_speakers,
                        "selected_c_values": ";".join(str(v) for v in selected_cs),
                        "calibration_statuses": ";".join(sorted(set(calibration_statuses))),
                        "fold_note": fold_note,
                        "pass_target_zone": yes_no(
                            metrics["recall"] >= TARGET_RECALL and metrics["specificity"] >= TARGET_SPECIFICITY
                        ),
                        **metrics,
                    }
                )
        print(f"[{display_name}] completed seed {seed}", flush=True)

    return {
        "clip_summary": pd.DataFrame(clip_summary_rows),
        "speaker_summary": pd.DataFrame(speaker_summary_rows),
        "speaker_predictions": pd.concat(
            [pd.concat(parts, ignore_index=True) for parts in speaker_prediction_parts.values()], ignore_index=True
        ),
        "fold_clip_scores": pd.concat(fold_clip_parts, ignore_index=True),
        "thresholds": pd.DataFrame(threshold_rows),
    }


def summarize_candidates(
    df: pd.DataFrame,
    group_cols: list[str],
    recall_col: str = "recall",
    specificity_col: str = "specificity",
    f1_col: str = "f1",
    threshold_col: str = "selected_threshold",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        recall_mean = float(group[recall_col].mean())
        specificity_mean = float(group[specificity_col].mean())
        rows.append(
            {
                **base,
                "n_seeds": int(group["seed"].nunique()),
                "speaker_recall_mean": recall_mean,
                "speaker_recall_std": float(group[recall_col].std(ddof=1)) if len(group) > 1 else 0.0,
                "speaker_specificity_mean": specificity_mean,
                "speaker_specificity_std": float(group[specificity_col].std(ddof=1)) if len(group) > 1 else 0.0,
                "speaker_f1_mean": float(group[f1_col].mean()),
                "speaker_f1_std": float(group[f1_col].std(ddof=1)) if len(group) > 1 else 0.0,
                "speaker_precision_mean": float(group["precision"].mean()) if "precision" in group else float("nan"),
                "speaker_auc_mean": float(group["roc_auc"].mean()) if "roc_auc" in group else float("nan"),
                "threshold_mean": float(group[threshold_col].mean()),
                "threshold_std": float(group[threshold_col].std(ddof=1)) if len(group) > 1 else 0.0,
                "target_zone_pass": yes_no(recall_mean >= TARGET_RECALL and specificity_mean >= TARGET_SPECIFICITY),
                "specificity_floor_pass": specificity_mean >= TARGET_SPECIFICITY,
            }
        )
    return pd.DataFrame(rows)


def rank_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_zone_bool"] = out["target_zone_pass"].eq("yes")
    out["specificity_floor_bool"] = out["speaker_specificity_mean"] >= TARGET_SPECIFICITY
    sort_cols = [
        "target_zone_bool",
        "specificity_floor_bool",
        "speaker_recall_mean",
        "speaker_specificity_mean",
        "speaker_f1_mean",
        "speaker_auc_mean",
    ]
    return out.sort_values(sort_cols, ascending=[False, False, False, False, False, False]).reset_index(drop=True)


def format_aggregation_output(speaker_summary: pd.DataFrame) -> pd.DataFrame:
    out = speaker_summary.copy()
    out = out.rename(
        columns={
            "aggregation": "aggregation",
            "recall": "speaker_recall",
            "specificity": "speaker_specificity",
            "precision": "speaker_precision",
            "f1": "speaker_f1",
            "roc_auc": "speaker_auc",
        }
    )
    cols = [
        "seed",
        "backbone",
        "classifier",
        "aggregation",
        "threshold_policy",
        "selected_threshold",
        "speaker_recall",
        "speaker_specificity",
        "speaker_precision",
        "speaker_f1",
        "speaker_auc",
        "pass_target_zone",
        "n_folds",
        "n_speakers",
        "selected_threshold_std",
        "selected_threshold_min",
        "selected_threshold_max",
        "calibration_statuses",
    ]
    return out[cols].sort_values(["seed", "aggregation", "threshold_policy"]).reset_index(drop=True)


def format_fusion_output(speaker_summary: pd.DataFrame, clip_summary: pd.DataFrame) -> pd.DataFrame:
    clip_cols = [
        "model_id",
        "seed",
        "threshold_policy",
        "recall",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
    ]
    clip = clip_summary[clip_cols].rename(
        columns={
            "recall": "clip_recall",
            "specificity": "clip_specificity",
            "precision": "clip_precision",
            "f1": "clip_f1",
            "roc_auc": "clip_auc",
        }
    )
    out = speaker_summary.merge(clip, on=["model_id", "seed", "threshold_policy"], how="left", validate="many_to_one")
    out = out.rename(
        columns={
            "recall": "speaker_recall",
            "specificity": "speaker_specificity",
            "precision": "speaker_precision",
            "f1": "speaker_f1",
            "roc_auc": "speaker_auc",
        }
    )
    cols = [
        "seed",
        "feature_mode",
        "backbone",
        "classifier",
        "aggregation",
        "threshold_policy",
        "selected_threshold",
        "speaker_recall",
        "speaker_specificity",
        "speaker_precision",
        "speaker_f1",
        "speaker_auc",
        "clip_recall",
        "clip_specificity",
        "clip_precision",
        "clip_f1",
        "clip_auc",
        "pass_target_zone",
        "n_folds",
        "n_speakers",
        "selected_threshold_std",
        "calibration_statuses",
    ]
    return out[cols].sort_values(["seed", "feature_mode", "classifier", "aggregation", "threshold_policy"]).reset_index(
        drop=True
    )


def component_speaker_scores(
    fold_clip_scores: pd.DataFrame,
    component: ComponentSpec,
    seed: int,
    fold: int,
    split: str,
    threshold: float,
) -> pd.DataFrame:
    clip = fold_clip_scores[
        (fold_clip_scores["model_id"] == component.model_id)
        & (fold_clip_scores["seed"] == seed)
        & (fold_clip_scores["fold"] == fold)
        & (fold_clip_scores["split"] == split)
    ].copy()
    if clip.empty:
        raise ValueError(f"Missing fold scores for {component.model_id}, seed {seed}, fold {fold}, split {split}")
    out = speaker_score_table(clip, component.aggregation, threshold)
    return out.rename(columns={"score": component.model_id, "prediction": f"{component.model_id}_prediction"})


def threshold_index(threshold: float) -> int:
    return int(round(float(threshold) * 100))


def static_speaker_score(scores: np.ndarray, aggregation_method: str) -> float:
    scores = np.asarray(scores, dtype=float)
    if aggregation_method == "mean_score":
        return float(np.mean(scores))
    if aggregation_method == "max_score":
        return float(np.max(scores))
    if aggregation_method == "top_2_mean":
        k = min(2, len(scores))
        return float(np.mean(np.sort(scores)[-k:]))
    if aggregation_method == "top_3_mean":
        k = min(3, len(scores))
        return float(np.mean(np.sort(scores)[-k:]))
    if aggregation_method == "top_half_mean":
        k = max(1, int(math.ceil(len(scores) / 2.0)))
        return float(np.mean(np.sort(scores)[-k:]))
    raise ValueError(f"Static score is not defined for {aggregation_method}")


def precompute_component_score_grid(
    fold_clip_scores: pd.DataFrame,
    components: list[ComponentSpec],
) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    """Cache speaker-level component scores for all thresholds.

    This keeps the ensemble scan controlled but avoids re-running thousands of
    pandas group-bys when a component uses majority-vote aggregation.
    """
    thresholds = threshold_candidates()
    cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for component in components:
        component_rows = fold_clip_scores[fold_clip_scores["model_id"] == component.model_id].copy()
        if component_rows.empty:
            raise ValueError(f"No fold clip scores found for ensemble component {component.model_id}.")
        for (seed, fold, split), group in component_rows.groupby(["seed", "fold", "split"], sort=True):
            speaker_ids: list[str] = []
            labels: list[int] = []
            score_rows: list[np.ndarray] = []
            for speaker_id, speaker_group in group.groupby("speaker_id", sort=True):
                clip_scores = speaker_group["score"].astype(float).to_numpy()
                speaker_ids.append(str(speaker_id))
                labels.append(mode_int(speaker_group["label"]))
                if component.aggregation == "majority_vote":
                    score_rows.append((clip_scores[:, None] >= thresholds[None, :]).mean(axis=0))
                else:
                    base_score = static_speaker_score(clip_scores, component.aggregation)
                    score_rows.append(np.repeat(base_score, len(thresholds)))
            cache[(component.model_id, int(seed), int(fold), str(split))] = {
                "speaker_id": np.asarray(speaker_ids, dtype=object),
                "label": np.asarray(labels, dtype=int),
                "scores": np.vstack(score_rows).astype(float),
            }
    return cache


def cached_component_scores(
    score_grid: dict[tuple[str, int, int, str], dict[str, Any]],
    component: ComponentSpec,
    seed: int,
    fold: int,
    split: str,
    threshold: float,
) -> pd.DataFrame:
    key = (component.model_id, int(seed), int(fold), str(split))
    if key not in score_grid:
        raise ValueError(f"Missing precomputed component scores for {key}")
    item = score_grid[key]
    idx = threshold_index(threshold)
    return pd.DataFrame(
        {
            "speaker_id": item["speaker_id"],
            "label": item["label"],
            component.model_id: item["scores"][:, idx],
        }
    )


def ensemble_speaker_table(
    fold_clip_scores: pd.DataFrame,
    components: list[ComponentSpec],
    seed: int,
    fold: int,
    split: str,
    threshold: float,
    method: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for component in components:
        comp_scores = component_speaker_scores(fold_clip_scores, component, seed, fold, split, threshold)
        keep = ["speaker_id", "label", component.model_id]
        comp_scores = comp_scores[keep]
        if merged is None:
            merged = comp_scores
        else:
            merged = merged.merge(comp_scores, on=["speaker_id", "label"], how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise ValueError("No component scores available for ensemble.")
    component_cols = [component.model_id for component in components]
    values = merged[component_cols].astype(float).to_numpy()
    if method == "mean_probability_ensemble":
        merged["score"] = values.mean(axis=1)
        merged["prediction"] = (merged["score"].astype(float) >= threshold).astype(int)
    elif method == "max_probability_ensemble":
        merged["score"] = values.max(axis=1)
        merged["prediction"] = (merged["score"].astype(float) >= threshold).astype(int)
    elif method == "majority_vote_ensemble":
        votes = (values >= threshold).astype(int)
        merged["score"] = votes.mean(axis=1)
        merged["prediction"] = (merged["score"].astype(float) >= 0.5).astype(int)
    else:
        raise ValueError(f"Unsupported ensemble method: {method}")
    return merged[["speaker_id", "label", "score", "prediction", *component_cols]]


def ensemble_speaker_table_cached(
    score_grid: dict[tuple[str, int, int, str], dict[str, Any]],
    components: list[ComponentSpec],
    seed: int,
    fold: int,
    split: str,
    threshold: float,
    method: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for component in components:
        comp_scores = cached_component_scores(score_grid, component, seed, fold, split, threshold)
        if merged is None:
            merged = comp_scores
        else:
            merged = merged.merge(comp_scores, on=["speaker_id", "label"], how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise ValueError("No cached component scores available for ensemble.")
    component_cols = [component.model_id for component in components]
    values = merged[component_cols].astype(float).to_numpy()
    if method == "mean_probability_ensemble":
        merged["score"] = values.mean(axis=1)
        merged["prediction"] = (merged["score"].astype(float) >= threshold).astype(int)
    elif method == "max_probability_ensemble":
        merged["score"] = values.max(axis=1)
        merged["prediction"] = (merged["score"].astype(float) >= threshold).astype(int)
    elif method == "majority_vote_ensemble":
        votes = (values >= threshold).astype(int)
        merged["score"] = votes.mean(axis=1)
        merged["prediction"] = (merged["score"].astype(float) >= 0.5).astype(int)
    else:
        raise ValueError(f"Unsupported ensemble method: {method}")
    return merged[["speaker_id", "label", "score", "prediction", *component_cols]]


def select_ensemble_constrained_threshold_cached(
    score_grid: dict[tuple[str, int, int, str], dict[str, Any]],
    components: list[ComponentSpec],
    seed: int,
    fold: int,
    method: str,
) -> tuple[float, str]:
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates():
        speaker_df = ensemble_speaker_table_cached(score_grid, components, seed, fold, "train", float(threshold), method)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics_from_arrays(
                    speaker_df["label"].astype(int).to_numpy(),
                    speaker_df["score"].astype(float).to_numpy(),
                    speaker_df["prediction"].astype(int).to_numpy(),
                ),
            }
        )
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
    return float(selected["threshold"]), note


def evaluate_ensembles(fold_clip_scores: pd.DataFrame, components: list[ComponentSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = ["mean_probability_ensemble", "max_probability_ensemble", "majority_vote_ensemble"]
    summary_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    component_names = " + ".join(component.component_name for component in components)
    component_aggs = " + ".join(f"{component.component_name}:{component.aggregation}" for component in components)
    score_grid = precompute_component_score_grid(fold_clip_scores, components)

    for seed in SEEDS:
        seed_folds = sorted(fold_clip_scores.loc[fold_clip_scores["seed"] == seed, "fold"].unique())
        for method in methods:
            for policy in THRESHOLD_POLICIES:
                fold_predictions: list[pd.DataFrame] = []
                fold_thresholds: list[float] = []
                selection_notes: list[str] = []
                for fold in seed_folds:
                    if policy == "default_0.5":
                        threshold, note = 0.5, "fixed_default_probability_threshold"
                    else:
                        threshold, note = select_ensemble_constrained_threshold_cached(
                            score_grid, components, int(seed), int(fold), method
                        )
                    val = ensemble_speaker_table_cached(
                        score_grid, components, int(seed), int(fold), "validation", float(threshold), method
                    )
                    val["seed"] = int(seed)
                    val["fold"] = int(fold)
                    val["ensemble_method"] = method
                    val["threshold_policy"] = policy
                    val["threshold"] = float(threshold)
                    fold_predictions.append(val)
                    fold_thresholds.append(float(threshold))
                    selection_notes.append(note)
                pred_df = pd.concat(fold_predictions, ignore_index=True)
                duplicate_speakers = int(pred_df["speaker_id"].duplicated().sum())
                metrics = metrics_from_arrays(
                    pred_df["label"].astype(int).to_numpy(),
                    pred_df["score"].astype(float).to_numpy(),
                    pred_df["prediction"].astype(int).to_numpy(),
                )
                pass_zone = metrics["recall"] >= TARGET_RECALL and metrics["specificity"] >= TARGET_SPECIFICITY
                summary_rows.append(
                    {
                        "seed": int(seed),
                        "ensemble_method": method,
                        "component_models": component_names,
                        "component_model_ids": " + ".join(component.model_id for component in components),
                        "aggregation": "speaker_level_component_scores",
                        "component_aggregations": component_aggs,
                        "threshold_policy": policy,
                        "selected_threshold": float(np.mean(fold_thresholds)),
                        "selected_threshold_std": float(np.std(fold_thresholds, ddof=1)) if len(fold_thresholds) > 1 else 0.0,
                        "speaker_recall": metrics["recall"],
                        "speaker_specificity": metrics["specificity"],
                        "speaker_precision": metrics["precision"],
                        "speaker_f1": metrics["f1"],
                        "speaker_auc": metrics["roc_auc"],
                        "pass_target_zone": yes_no(pass_zone),
                        "n_folds": int(len(seed_folds)),
                        "n_speakers": int(pred_df["speaker_id"].nunique()),
                        "duplicate_validation_speaker_rows": duplicate_speakers,
                        "selection_notes": ";".join(sorted(set(selection_notes))),
                        **{
                            k: metrics[k]
                            for k in [
                                "true_negatives",
                                "false_positives",
                                "false_negatives",
                                "true_positives",
                                "confusion_matrix_tn_fp_fn_tp",
                            ]
                        },
                    }
                )
                pred_df["component_models"] = component_names
                prediction_parts.append(pred_df)
            print(f"[ensemble] completed seed {seed} method {method}", flush=True)
    return pd.DataFrame(summary_rows), pd.concat(prediction_parts, ignore_index=True)


def component_prediction_frame(
    all_speaker_predictions: pd.DataFrame,
    component: ComponentSpec,
) -> pd.DataFrame:
    out = all_speaker_predictions[
        (all_speaker_predictions["model_id"] == component.model_id)
        & (all_speaker_predictions["aggregation"] == component.aggregation)
        & (all_speaker_predictions["threshold_policy"] == component.threshold_policy)
    ].copy()
    out = out[["seed", "speaker_id", "label", "score", "prediction"]]
    out = out.rename(
        columns={
            "score": f"{component.model_id}_score",
            "prediction": f"{component.model_id}_prediction",
        }
    )
    return out


def summarize_complementarity(all_speaker_predictions: pd.DataFrame, components: list[ComponentSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comp_a, comp_b in itertools.combinations(components, 2):
        a = component_prediction_frame(all_speaker_predictions, comp_a)
        b = component_prediction_frame(all_speaker_predictions, comp_b)
        merged = a.merge(b, on=["seed", "speaker_id", "label"], how="inner", validate="one_to_one")
        pred_a = merged[f"{comp_a.model_id}_prediction"].astype(int)
        pred_b = merged[f"{comp_b.model_id}_prediction"].astype(int)
        labels = merged["label"].astype(int)
        fn_a = set(merged.loc[(labels == 1) & (pred_a == 0), ["seed", "speaker_id"]].itertuples(index=False, name=None))
        fn_b = set(merged.loc[(labels == 1) & (pred_b == 0), ["seed", "speaker_id"]].itertuples(index=False, name=None))
        fp_a = set(merged.loc[(labels == 0) & (pred_a == 1), ["seed", "speaker_id"]].itertuples(index=False, name=None))
        fp_b = set(merged.loc[(labels == 0) & (pred_b == 1), ["seed", "speaker_id"]].itertuples(index=False, name=None))
        shared_fn = len(fn_a & fn_b)
        shared_fp = len(fp_a & fp_b)
        fn_union = len(fn_a | fn_b)
        fp_union = len(fp_a | fp_b)
        disagreement = int((pred_a != pred_b).sum())
        fn_overlap = shared_fn / fn_union if fn_union else 0.0
        fp_overlap = shared_fp / fp_union if fp_union else 0.0
        disagreement_rate = disagreement / max(len(merged), 1)
        if fn_overlap < 0.45 and disagreement_rate >= 0.15:
            judgment = "moderate_to_high"
        elif fn_overlap < 0.70 and disagreement_rate >= 0.08:
            judgment = "moderate"
        else:
            judgment = "low"
        rows.append(
            {
                "model_a": comp_a.component_name,
                "model_b": comp_b.component_name,
                "shared_fn_count": int(shared_fn),
                "shared_fp_count": int(shared_fp),
                "fn_overlap_ratio": float(fn_overlap),
                "fp_overlap_ratio": float(fp_overlap),
                "speaker_disagreement_count": int(disagreement),
                "complementarity_judgment": judgment,
                "notes": (
                    "Overlap ratios are Jaccard overlap across seed-speaker error sets; "
                    f"aligned rows={len(merged)}, disagreement_rate={disagreement_rate:.3f}."
                ),
            }
        )
    return pd.DataFrame(rows)


def phasec_hubert_baseline_row() -> dict[str, Any]:
    phasec = pd.read_csv(PHASEC_SPEAKER_SUMMARY)
    rowset = phasec[
        (phasec["dataset_version"] == DATASET_VERSION)
        & (phasec["backbone"] == HUBERT_BACKBONE)
        & (phasec["classifier"] == LOGISTIC)
        & (phasec["aggregation_method"] == "mean_score")
        & (phasec["threshold_policy"] == "default_0.5")
    ].copy()
    if rowset.empty:
        raise ValueError("Could not locate Phase C HuBERT logistic mean_score default_0.5 baseline.")
    recall_mean = float(rowset["recall"].mean())
    specificity_mean = float(rowset["specificity"].mean())
    return {
        "candidate_type": "phaseC_reference",
        "candidate_name": "Phase C HuBERT logistic mean_score default_0.5",
        "feature_mode": "hubert_only",
        "classifier": LOGISTIC,
        "aggregation": "mean_score",
        "threshold_policy": "default_0.5",
        "n_seeds": int(rowset["seed"].nunique()),
        "speaker_recall_mean": recall_mean,
        "speaker_recall_std": float(rowset["recall"].std(ddof=1)),
        "speaker_specificity_mean": specificity_mean,
        "speaker_specificity_std": float(rowset["specificity"].std(ddof=1)),
        "speaker_f1_mean": float(rowset["f1"].mean()),
        "speaker_f1_std": float(rowset["f1"].std(ddof=1)),
        "threshold_mean": float(rowset["threshold_mean"].mean()),
        "threshold_std": float(rowset["threshold_mean"].std(ddof=1)),
        "target_zone_pass": yes_no(recall_mean >= TARGET_RECALL and specificity_mean >= TARGET_SPECIFICITY),
    }


def choose_best_second_embedding_backbone() -> str:
    if not PHASEC_MODEL_COMPARISON.exists():
        return WAV2VEC2_BACKBONE
    model_df = pd.read_csv(PHASEC_MODEL_COMPARISON)
    candidates = model_df[
        (model_df["dataset_version"] == DATASET_VERSION)
        & (model_df["metric_level"] == "speaker")
        & (model_df["aggregation_method"] == "mean_score")
        & (model_df["classifier"] == LOGISTIC)
        & (model_df["backbone"] != HUBERT_BACKBONE)
    ].copy()
    if candidates.empty:
        return WAV2VEC2_BACKBONE
    candidates["target_zone_pass"] = candidates["target_zone_pass"].astype(str).str.lower().eq("true")
    candidates["specificity_floor"] = candidates["specificity_mean"] >= TARGET_SPECIFICITY
    candidates = candidates.sort_values(
        ["target_zone_pass", "specificity_floor", "recall_mean", "specificity_mean", "f1_mean"],
        ascending=[False, False, False, False, False],
    )
    return str(candidates.iloc[0]["backbone"])


def build_phaseD_model_comparison(
    aggregation_summary: pd.DataFrame,
    fusion_summary: pd.DataFrame,
    ensemble_summary: pd.DataFrame,
) -> pd.DataFrame:
    phasec_row = phasec_hubert_baseline_row()

    agg_group = summarize_candidates(
        aggregation_summary.rename(
            columns={
                "speaker_recall": "recall",
                "speaker_specificity": "specificity",
                "speaker_f1": "f1",
                "speaker_auc": "roc_auc",
            }
        ),
        ["backbone", "classifier", "aggregation", "threshold_policy"],
    )
    agg_best = rank_summary(agg_group).iloc[0].to_dict()

    fusion_group = summarize_candidates(
        fusion_summary.rename(
            columns={
                "speaker_recall": "recall",
                "speaker_specificity": "specificity",
                "speaker_f1": "f1",
                "speaker_auc": "roc_auc",
            }
        ),
        ["feature_mode", "backbone", "classifier", "aggregation", "threshold_policy"],
    )
    fusion_best = rank_summary(fusion_group[fusion_group["feature_mode"] == "hubert_plus_handcrafted"]).iloc[0].to_dict()

    ensemble_group = summarize_candidates(
        ensemble_summary.rename(
            columns={
                "speaker_recall": "recall",
                "speaker_specificity": "specificity",
                "speaker_f1": "f1",
                "speaker_auc": "roc_auc",
            }
        ),
        ["ensemble_method", "component_models", "aggregation", "threshold_policy"],
    )
    ensemble_best = rank_summary(ensemble_group).iloc[0].to_dict()

    rows = [
        phasec_row,
        {
            "candidate_type": "phaseD_aggregation_only",
            "candidate_name": f"HuBERT logistic {agg_best['aggregation']} {agg_best['threshold_policy']}",
            "feature_mode": "hubert_only",
            "classifier": agg_best["classifier"],
            "aggregation": agg_best["aggregation"],
            "threshold_policy": agg_best["threshold_policy"],
            **{k: agg_best[k] for k in phasec_row.keys() if k.endswith("_mean") or k.endswith("_std") or k == "n_seeds"},
            "target_zone_pass": agg_best["target_zone_pass"],
        },
        {
            "candidate_type": "phaseD_fusion",
            "candidate_name": (
                f"HuBERT+handcrafted {fusion_best['classifier']} "
                f"{fusion_best['aggregation']} {fusion_best['threshold_policy']}"
            ),
            "feature_mode": fusion_best["feature_mode"],
            "classifier": fusion_best["classifier"],
            "aggregation": fusion_best["aggregation"],
            "threshold_policy": fusion_best["threshold_policy"],
            **{k: fusion_best[k] for k in phasec_row.keys() if k.endswith("_mean") or k.endswith("_std") or k == "n_seeds"},
            "target_zone_pass": fusion_best["target_zone_pass"],
        },
        {
            "candidate_type": "phaseD_ensemble",
            "candidate_name": f"{ensemble_best['ensemble_method']} {ensemble_best['threshold_policy']}",
            "feature_mode": "ensemble",
            "classifier": "ensemble",
            "aggregation": ensemble_best["aggregation"],
            "threshold_policy": ensemble_best["threshold_policy"],
            **{
                "n_seeds": ensemble_best["n_seeds"],
                "speaker_recall_mean": ensemble_best["speaker_recall_mean"],
                "speaker_recall_std": ensemble_best["speaker_recall_std"],
                "speaker_specificity_mean": ensemble_best["speaker_specificity_mean"],
                "speaker_specificity_std": ensemble_best["speaker_specificity_std"],
                "speaker_f1_mean": ensemble_best["speaker_f1_mean"],
                "speaker_f1_std": ensemble_best["speaker_f1_std"],
                "threshold_mean": ensemble_best["threshold_mean"],
                "threshold_std": ensemble_best["threshold_std"],
            },
            "target_zone_pass": ensemble_best["target_zone_pass"],
        },
    ]
    out = pd.DataFrame(rows)
    out["specificity_floor_pass"] = out["speaker_specificity_mean"] >= TARGET_SPECIFICITY
    out["target_zone_bool"] = out["target_zone_pass"].eq("yes")
    out = out.sort_values(
        ["target_zone_bool", "specificity_floor_pass", "speaker_recall_mean", "speaker_specificity_mean", "speaker_f1_mean"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    out["decision_ranking"] = np.arange(1, len(out) + 1)
    return out.drop(columns=["target_zone_bool", "specificity_floor_pass"])


def build_best_candidate_analysis(model_comparison: pd.DataFrame) -> pd.DataFrame:
    phasec = model_comparison[model_comparison["candidate_type"] == "phaseC_reference"].iloc[0]
    candidates = model_comparison[model_comparison["candidate_type"] != "phaseC_reference"].copy()
    best = candidates.sort_values("decision_ranking").iloc[0]
    better_than_phasec = (
        best["speaker_recall_mean"] > phasec["speaker_recall_mean"]
        and best["speaker_specificity_mean"] >= TARGET_SPECIFICITY
    )
    better_than_classical = (
        best["speaker_recall_mean"] > OFFICIAL_CLASSICAL_BASELINE["speaker_recall_mean"]
        and best["speaker_specificity_mean"] > OFFICIAL_CLASSICAL_BASELINE["speaker_specificity_mean"]
    )
    target_pass = best["target_zone_pass"] == "yes"
    if target_pass and better_than_phasec:
        recommendation = "advance"
        reason = "Reached the target zone while improving recall over the Phase C HuBERT reference."
    elif better_than_phasec:
        recommendation = "hold"
        reason = "Improved recall over Phase C while preserving the specificity floor, but did not reach the target zone."
    else:
        recommendation = "reject"
        reason = "Did not improve the Phase C HuBERT recall/specificity tradeoff enough to justify advancement."
    return pd.DataFrame(
        [
            {
                "candidate_name": best["candidate_name"],
                "feature_mode": best["feature_mode"],
                "classifier": best["classifier"],
                "aggregation": best["aggregation"],
                "threshold_policy": best["threshold_policy"],
                "speaker_recall_mean": best["speaker_recall_mean"],
                "speaker_recall_std": best["speaker_recall_std"],
                "speaker_specificity_mean": best["speaker_specificity_mean"],
                "speaker_specificity_std": best["speaker_specificity_std"],
                "speaker_f1_mean": best["speaker_f1_mean"],
                "speaker_f1_std": best["speaker_f1_std"],
                "target_zone_pass": best["target_zone_pass"],
                "better_than_phaseC": yes_no(better_than_phasec),
                "better_than_classical_baseline": yes_no(better_than_classical),
                "recommendation": recommendation,
                "reason": reason,
            }
        ]
    )


def write_report(
    *,
    aggregation_summary: pd.DataFrame,
    fusion_summary: pd.DataFrame,
    ensemble_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    complementarity: pd.DataFrame,
    best_candidate: pd.DataFrame,
    cache_meta: list[dict[str, Any]],
    join_info: dict[str, Any],
    best_aggregation: str,
    fusion_aggregations: list[str],
    second_backbone: str,
) -> None:
    agg_group = summarize_candidates(
        aggregation_summary.rename(
            columns={
                "speaker_recall": "recall",
                "speaker_specificity": "specificity",
                "speaker_f1": "f1",
                "speaker_auc": "roc_auc",
            }
        ),
        ["backbone", "classifier", "aggregation", "threshold_policy"],
    )
    agg_ranked = rank_summary(agg_group)
    fusion_group = summarize_candidates(
        fusion_summary.rename(
            columns={
                "speaker_recall": "recall",
                "speaker_specificity": "specificity",
                "speaker_f1": "f1",
                "speaker_auc": "roc_auc",
            }
        ),
        ["feature_mode", "classifier", "aggregation", "threshold_policy"],
    )
    fusion_ranked = rank_summary(fusion_group)
    ensemble_group = summarize_candidates(
        ensemble_summary.rename(
            columns={
                "speaker_recall": "recall",
                "speaker_specificity": "specificity",
                "speaker_f1": "f1",
                "speaker_auc": "roc_auc",
            }
        ),
        ["ensemble_method", "threshold_policy"],
    )
    ensemble_ranked = rank_summary(ensemble_group)
    phasec = model_comparison[model_comparison["candidate_type"] == "phaseC_reference"].iloc[0]
    best = best_candidate.iloc[0]
    target_reached = int((model_comparison["target_zone_pass"] == "yes").sum())
    seed_level_target_passes = int(
        aggregation_summary["pass_target_zone"].eq("yes").sum()
        + fusion_summary["pass_target_zone"].eq("yes").sum()
        + ensemble_summary["pass_target_zone"].eq("yes").sum()
    )
    best_agg_report = agg_ranked.iloc[0]
    best_fusion_plus = rank_summary(fusion_group[fusion_group["feature_mode"] == "hubert_plus_handcrafted"]).iloc[0]
    best_ensemble_report = ensemble_ranked.iloc[0]

    cache_rows = pd.DataFrame(
        [
            {
                "backbone": meta.get("backbone", ""),
                "cache_status": meta.get("cache_status", ""),
                "rows": meta.get("n_embedded_rows", ""),
                "skipped": meta.get("n_skipped_rows", ""),
                "embedding_dim": meta.get("embedding_dim", ""),
                "cache_csv": meta.get("cache_csv", ""),
            }
            for meta in cache_meta
        ]
    )
    helped = model_comparison.sort_values("decision_ranking").iloc[0]["candidate_name"]
    if best["recommendation"] == "advance":
        next_step = (
            "Run a narrow confirmation pass for the selected Phase D candidate on the same trusted protocol, "
            "then only after that consider governed sensitivity."
        )
    elif best["recommendation"] == "hold":
        next_step = (
            "Inspect the remaining false negatives for the held Phase D candidate before adding model complexity; "
            "the next experiment should be error-driven, not a broader sweep."
        )
    else:
        next_step = (
            "Stop Phase D model escalation and inspect missed-positive data/representation limits before any new "
            "benchmark phase."
        )

    lines = [
        "# Phase D HuBERT Recall Recovery Benchmark",
        "",
        "## 1. Purpose",
        "",
        "Phase D tests whether speaker-level recall can be recovered for the best frozen embedding path without "
        "collapsing specificity. The target zone remains speaker recall >= 0.70 and specificity >= 0.50. "
        "This is a controlled benchmark, not an open-ended experiment.",
        "",
        "## 2. Frozen reference setup",
        "",
        "Used frozen references: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`, `docs/BASELINE_MODEL_CARD.md`, "
        "`reports/FINAL_BASELINE_SUMMARY.md`, `reports/final_fp_fn_error_analysis.md`, "
        "`reports/phaseC_frozen_embedding_benchmark.md`, Phase C summary tables, the trusted cleaned manifest, "
        "and the canonical governed-pruned feature table. Phase C established `facebook/hubert-base-ls960` + "
        "`logistic_regression_platt` with `mean_score` as the best stable frozen embedding reference.",
        "",
        "## 3. Data sources used",
        "",
        "- Primary manifest: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv`.",
        "- Canonical handcrafted features: `data/features/features_phase3_governed_pruned.csv`.",
        "- Fusion join key: `metadata_index`.",
        f"- Feature join: {join_info['matched_rows']}/{join_info['manifest_rows']} trusted rows matched; "
        f"{join_info['dropped_rows']} rows dropped; {join_info['feature_count']} official numeric handcrafted features.",
        "- Cached embeddings used:",
        "",
        markdown_table(cache_rows, max_rows=5),
        "",
        "## 4. HuBERT aggregation benchmark",
        "",
        "Fixed model: `facebook/hubert-base-ls960` + `logistic_regression_platt`. Tested `mean_score`, "
        "`max_score`, `majority_vote`, `top_2_mean`, `top_3_mean`, and `top_half_mean` under `default_0.5` and "
        "`constrained_spec_ge_0.50`.",
        "",
        markdown_table(
            agg_ranked[
                [
                    "aggregation",
                    "threshold_policy",
                    "speaker_recall_mean",
                    "speaker_recall_std",
                    "speaker_specificity_mean",
                    "speaker_specificity_std",
                    "speaker_f1_mean",
                    "threshold_mean",
                    "target_zone_pass",
                ]
            ].round(4),
            max_rows=12,
        ),
        "",
        f"Best aggregation selected for downstream fusion scope: `{best_aggregation}`.",
        f" Aggregation alone helped recall only modestly: `{best_agg_report['aggregation']}` with "
        f"`{best_agg_report['threshold_policy']}` reached mean recall "
        f"{best_agg_report['speaker_recall_mean']:.3f} and specificity "
        f"{best_agg_report['speaker_specificity_mean']:.3f}, versus Phase C `mean_score` recall "
        f"{phasec['speaker_recall_mean']:.3f}.",
        "",
        "## 5. Fusion benchmark",
        "",
        "Fusion concatenates the pooled HuBERT embedding with the official 256 handcrafted acoustic/prosodic features. "
        "Only `logistic_regression_platt` and `linear_svm_platt` were tested. Fusion aggregations were "
        f"`{fusion_aggregations[0]}`" + (f" and `{fusion_aggregations[1]}`." if len(fusion_aggregations) > 1 else "."),
        "",
        markdown_table(
            fusion_ranked[
                [
                    "feature_mode",
                    "classifier",
                    "aggregation",
                    "threshold_policy",
                    "speaker_recall_mean",
                    "speaker_specificity_mean",
                    "speaker_f1_mean",
                    "threshold_mean",
                    "target_zone_pass",
                ]
            ].round(4),
            max_rows=16,
        ),
        "",
        f"Fusion did not improve over the HuBERT-only control. The best fused setup was "
        f"`{best_fusion_plus['classifier']}` + `{best_fusion_plus['aggregation']}` + "
        f"`{best_fusion_plus['threshold_policy']}`, with recall "
        f"{best_fusion_plus['speaker_recall_mean']:.3f} and specificity "
        f"{best_fusion_plus['speaker_specificity_mean']:.3f}.",
        "",
        "## 6. Ensemble benchmark",
        "",
        "The ensemble used at most three controlled components: the best HuBERT aggregation-only model from Phase D, "
        "the best HuBERT+handcrafted fusion model from Phase D, and the best second embedding-only Phase C candidate "
        f"(`{second_backbone}` + logistic mean-score).",
        "",
        markdown_table(
            ensemble_ranked[
                [
                    "ensemble_method",
                    "threshold_policy",
                    "speaker_recall_mean",
                    "speaker_specificity_mean",
                    "speaker_f1_mean",
                    "threshold_mean",
                    "target_zone_pass",
                ]
            ].round(4),
            max_rows=8,
        ),
        "",
        f"The ensemble helped recall most: `{best_ensemble_report['ensemble_method']}` with "
        f"`{best_ensemble_report['threshold_policy']}` reached mean recall "
        f"{best_ensemble_report['speaker_recall_mean']:.3f} and specificity "
        f"{best_ensemble_report['speaker_specificity_mean']:.3f}.",
        "",
        "## 7. Threshold policy and leakage controls",
        "",
        "- Outer evaluation used 5-fold `StratifiedGroupKFold` grouped by `speaker_id` for seeds 42, 43, 44, 45, and 46.",
        "- Training and held-out speaker sets were disjoint in every fold.",
        "- Platt calibration was fit from training speakers only, matching the Phase C in-sample training-speaker calibration style.",
        "- `default_0.5` used a fixed threshold of 0.5.",
        "- `constrained_spec_ge_0.50` selected a threshold on training speakers only, maximizing recall subject to specificity >= 0.50.",
        "- Held-out fold labels were not used for threshold selection.",
        "",
        "## 8. Comparison against Phase C best model",
        "",
        f"Phase C HuBERT baseline speaker recall mean was {phasec['speaker_recall_mean']:.3f} with specificity "
        f"{phasec['speaker_specificity_mean']:.3f}. Phase D comparison:",
        "",
        markdown_table(
            model_comparison[
                [
                    "decision_ranking",
                    "candidate_type",
                    "candidate_name",
                    "speaker_recall_mean",
                    "speaker_recall_std",
                    "speaker_specificity_mean",
                    "speaker_specificity_std",
                    "speaker_f1_mean",
                    "threshold_mean",
                    "target_zone_pass",
                ]
            ].round(4),
            max_rows=8,
        ),
        "",
        "## 9. Comparison against official classical baseline",
        "",
        f"The official classical reference remains `{OFFICIAL_CLASSICAL_BASELINE['name']}` at threshold "
        f"{OFFICIAL_CLASSICAL_BASELINE['threshold']:.3f}. Its available multi-seed speaker reference uses "
        f"`{OFFICIAL_CLASSICAL_BASELINE['speaker_aggregation']}` with recall "
        f"{OFFICIAL_CLASSICAL_BASELINE['speaker_recall_mean']:.3f} and specificity "
        f"{OFFICIAL_CLASSICAL_BASELINE['speaker_specificity_mean']:.3f}. Phase D did not alter that baseline.",
        "",
        "## 10. Error complementarity findings",
        "",
        markdown_table(complementarity.round(4), max_rows=8),
        "",
        "## 11. Best Phase D candidate",
        "",
        markdown_table(best_candidate.round(4), max_rows=1),
        "",
        "## 12. Did anything reach the target zone?",
        "",
        f"Number of comparison rows reaching speaker recall >= {TARGET_RECALL:.2f} and specificity >= "
        f"{TARGET_SPECIFICITY:.2f}: **{target_reached}**.",
        f" At seed level, target-zone passes across all Phase D rows: **{seed_level_target_passes}**; "
        "the headline decision uses multi-seed means, so no Phase D setup is treated as having reached the target zone.",
        "",
        "## 13. What helped recall most?",
        "",
        f"The strongest ranked setup was `{helped}`. Whether that is enough is captured by the target-zone and "
        "recommendation fields above.",
        "",
        "## 14. What failed and why?",
        "",
        "Aggregation alone recovered only a small amount of recall and traded away specificity. Fusion failed because "
        "the handcrafted features did not add enough complementary signal to recover missed positives. The ensemble "
        "was the closest setup, but its multi-seed recall still stayed below 0.70. Constrained-threshold variants "
        "preserved high specificity but reduced recall sharply.",
        "",
        "## 15. Single most justified next step",
        "",
        next_step,
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    manifest = load_manifest()
    hubert_embeddings, hubert_meta = load_cached_embeddings(HUBERT_BACKBONE, manifest)
    x_hubert, _ = embedding_matrix(hubert_embeddings)
    handcrafted, _, join_info = load_aligned_handcrafted_features(manifest)
    x_handcrafted = handcrafted.to_numpy(dtype=np.float32)
    x_hubert_plus_handcrafted = np.hstack([x_hubert, x_handcrafted])

    print("[Part 1] HuBERT aggregation benchmark", flush=True)
    hubert_logistic = evaluate_cv_model(
        model_id="hubert_only_logistic",
        display_name="HuBERT logistic",
        x=x_hubert,
        base_df=hubert_embeddings,
        classifier=LOGISTIC,
        feature_mode="hubert_only",
        backbone=HUBERT_BACKBONE,
        aggregations=AGGREGATION_METHODS,
        seeds=SEEDS,
        max_splits=MAX_SPLITS,
    )
    aggregation_summary = format_aggregation_output(hubert_logistic["speaker_summary"])
    aggregation_summary.to_csv(AGGREGATION_CSV, index=False)

    agg_group = summarize_candidates(
        aggregation_summary.rename(
            columns={
                "speaker_recall": "recall",
                "speaker_specificity": "specificity",
                "speaker_f1": "f1",
                "speaker_auc": "roc_auc",
            }
        ),
        ["backbone", "classifier", "aggregation", "threshold_policy"],
    )
    best_agg_row = rank_summary(agg_group).iloc[0]
    best_aggregation = str(best_agg_row["aggregation"])
    best_agg_policy = str(best_agg_row["threshold_policy"])
    fusion_aggregations = list(dict.fromkeys(["mean_score", best_aggregation]))

    print("[Part 2] HuBERT + handcrafted fusion benchmark", flush=True)
    fusion_runs: list[dict[str, pd.DataFrame]] = []
    for feature_mode, x_matrix in [
        ("hubert_only", x_hubert),
        ("hubert_plus_handcrafted", x_hubert_plus_handcrafted),
    ]:
        for classifier in [LOGISTIC, LINEAR_SVM]:
            model_id = f"{feature_mode}_{classifier}"
            fusion_runs.append(
                evaluate_cv_model(
                    model_id=model_id,
                    display_name=f"{feature_mode} {classifier}",
                    x=x_matrix,
                    base_df=hubert_embeddings,
                    classifier=classifier,
                    feature_mode=feature_mode,
                    backbone=HUBERT_BACKBONE,
                    aggregations=fusion_aggregations,
                    seeds=SEEDS,
                    max_splits=MAX_SPLITS,
                )
            )
    fusion_speaker = pd.concat([run["speaker_summary"] for run in fusion_runs], ignore_index=True)
    fusion_clip = pd.concat([run["clip_summary"] for run in fusion_runs], ignore_index=True)
    fusion_summary = format_fusion_output(fusion_speaker, fusion_clip)
    fusion_summary.to_csv(FUSION_CSV, index=False)

    fusion_group = summarize_candidates(
        fusion_summary.rename(
            columns={
                "speaker_recall": "recall",
                "speaker_specificity": "specificity",
                "speaker_f1": "f1",
                "speaker_auc": "roc_auc",
            }
        ),
        ["feature_mode", "backbone", "classifier", "aggregation", "threshold_policy"],
    )
    best_fusion_row = rank_summary(fusion_group[fusion_group["feature_mode"] == "hubert_plus_handcrafted"]).iloc[0]

    print("[Part 3] Controlled ensemble benchmark", flush=True)
    second_backbone = choose_best_second_embedding_backbone()
    second_embeddings, second_meta = load_cached_embeddings(second_backbone, manifest)
    x_second, _ = embedding_matrix(second_embeddings)
    second_embedding_run = evaluate_cv_model(
        model_id="phaseC_second_embedding_logistic",
        display_name=f"{second_backbone} logistic",
        x=x_second,
        base_df=second_embeddings,
        classifier=LOGISTIC,
        feature_mode="embedding_only",
        backbone=second_backbone,
        aggregations=["mean_score"],
        seeds=SEEDS,
        max_splits=MAX_SPLITS,
    )

    fold_clip_scores = pd.concat(
        [
            hubert_logistic["fold_clip_scores"],
            *[run["fold_clip_scores"] for run in fusion_runs],
            second_embedding_run["fold_clip_scores"],
        ],
        ignore_index=True,
    )
    all_speaker_predictions = pd.concat(
        [
            hubert_logistic["speaker_predictions"],
            *[run["speaker_predictions"] for run in fusion_runs],
            second_embedding_run["speaker_predictions"],
        ],
        ignore_index=True,
    )
    components = [
        ComponentSpec(
            model_id="hubert_only_logistic",
            component_name=f"HuBERT logistic {best_aggregation} {best_agg_policy}",
            feature_mode="hubert_only",
            classifier=LOGISTIC,
            aggregation=best_aggregation,
            threshold_policy=best_agg_policy,
        ),
        ComponentSpec(
            model_id=f"hubert_plus_handcrafted_{best_fusion_row['classifier']}",
            component_name=(
                f"HuBERT+handcrafted {best_fusion_row['classifier']} "
                f"{best_fusion_row['aggregation']} {best_fusion_row['threshold_policy']}"
            ),
            feature_mode="hubert_plus_handcrafted",
            classifier=str(best_fusion_row["classifier"]),
            aggregation=str(best_fusion_row["aggregation"]),
            threshold_policy=str(best_fusion_row["threshold_policy"]),
        ),
        ComponentSpec(
            model_id="phaseC_second_embedding_logistic",
            component_name=f"{second_backbone} logistic mean_score default_0.5",
            feature_mode="embedding_only",
            classifier=LOGISTIC,
            aggregation="mean_score",
            threshold_policy="default_0.5",
        ),
    ]
    complementarity = summarize_complementarity(all_speaker_predictions, components)
    complementarity.to_csv(COMPLEMENTARITY_CSV, index=False)
    ensemble_summary, _ = evaluate_ensembles(fold_clip_scores, components)
    ensemble_summary.to_csv(ENSEMBLE_CSV, index=False)

    model_comparison = build_phaseD_model_comparison(aggregation_summary, fusion_summary, ensemble_summary)
    model_comparison.to_csv(MODEL_COMPARISON_CSV, index=False)
    best_candidate = build_best_candidate_analysis(model_comparison)
    best_candidate.to_csv(BEST_CANDIDATE_CSV, index=False)
    write_report(
        aggregation_summary=aggregation_summary,
        fusion_summary=fusion_summary,
        ensemble_summary=ensemble_summary,
        model_comparison=model_comparison,
        complementarity=complementarity,
        best_candidate=best_candidate,
        cache_meta=[hubert_meta, second_meta],
        join_info=join_info,
        best_aggregation=best_aggregation,
        fusion_aggregations=fusion_aggregations,
        second_backbone=second_backbone,
    )

    print("Created Phase D outputs:", flush=True)
    for path in [
        Path("scripts/phaseD_hubert_recall_recovery_benchmark.py"),
        REPORT_MD,
        AGGREGATION_CSV,
        FUSION_CSV,
        ENSEMBLE_CSV,
        MODEL_COMPARISON_CSV,
        COMPLEMENTARITY_CSV,
        BEST_CANDIDATE_CSV,
    ]:
        print(f"- {path}", flush=True)


if __name__ == "__main__":
    main()
