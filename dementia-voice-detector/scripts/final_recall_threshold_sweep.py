"""Final controlled threshold sweep for the held Phase D ensemble.

This script intentionally stays narrow:
- trusted cleaned Phase D subset only;
- speaker-grouped outer folds with seeds 42-46 only;
- existing selected ensemble components only;
- no new architectures, feature engineering, or data changes.

It reconstructs the three held component models, validates fold integrity and
component alignment, then runs:
1. a fixed global threshold sweep over the requested grid; and
2. a separate training-selected-threshold evaluation using only training
   speakers inside each outer fold.
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
import sklearn


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
REPORT_PATH = ROOT / "reports" / "final_recall_threshold_sweep_report.md"

SUMMARY_CSV = TABLES_DIR / "final_recall_threshold_sweep_summary.csv"
BY_SEED_CSV = TABLES_DIR / "final_recall_threshold_sweep_by_seed.csv"
BY_FOLD_CSV = TABLES_DIR / "final_recall_threshold_sweep_by_fold.csv"
PER_SPEAKER_CSV = TABLES_DIR / "final_recall_threshold_sweep_per_speaker.csv"
FALSE_NEGATIVES_CSV = TABLES_DIR / "final_recall_threshold_sweep_false_negatives.csv"
FALSE_POSITIVES_CSV = TABLES_DIR / "final_recall_threshold_sweep_false_positives.csv"
COMPONENT_SCORES_CSV = TABLES_DIR / "final_recall_threshold_sweep_component_scores.csv"

BASELINE_PHASED_ENSEMBLE_CSV = TABLES_DIR / "phaseD_ensemble_seed_summary.csv"

THRESHOLD_GRID = [0.500, 0.475, 0.450, 0.425, 0.400, 0.375, 0.350, 0.325, 0.300, 0.275, 0.250, 0.225, 0.200]
FIXED_SWEEP = "fixed_global_sweep"
TRAIN_SELECTED = "train_selected_threshold"
TARGET_RECALL = 0.70
TARGET_SPECIFICITY = 0.50


@dataclass(frozen=True)
class ComponentRunConfig:
    model_id: str
    component_label: str
    display_name: str
    feature_mode: str
    backbone: str
    classifier: str
    aggregation: str
    base_df: pd.DataFrame
    x_matrix: np.ndarray


def load_phase_d_module() -> Any:
    module_path = ROOT / "scripts" / "phaseD_hubert_recall_recovery_benchmark.py"
    spec = importlib.util.spec_from_file_location("phase_d_benchmark", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Phase D module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


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
        lines.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in show.columns) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append("| ... | ... |")
    return "\n".join(lines)


def threshold_grid() -> list[float]:
    return [float(f"{value:.3f}") for value in THRESHOLD_GRID]


def outcome_label(true_label: int, predicted_label: int) -> str:
    if true_label == 1 and predicted_label == 1:
        return "true_positive"
    if true_label == 1 and predicted_label == 0:
        return "false_negative"
    if true_label == 0 and predicted_label == 1:
        return "false_positive"
    return "true_negative"


def miss_band(threshold: float, score: float) -> str:
    margin = float(threshold) - float(score)
    if margin <= 0.05:
        return "borderline"
    if margin <= 0.15:
        return "moderate"
    return "deep"


def support_type(n_components_at_or_above_threshold: int) -> str:
    if n_components_at_or_above_threshold >= 2:
        return "multi_component"
    if n_components_at_or_above_threshold == 1:
        return "single_component"
    return "none"


def choose_threshold(
    metrics_df: pd.DataFrame,
    *,
    threshold_col: str = "threshold",
    specificity_col: str = "specificity",
    recall_col: str = "recall",
    f1_col: str = "f1",
) -> tuple[float, str]:
    viable = metrics_df[metrics_df[specificity_col] >= TARGET_SPECIFICITY].copy()
    if viable.empty:
        selected = metrics_df.sort_values(
            [specificity_col, recall_col, f1_col, threshold_col],
            ascending=[False, False, False, False],
        ).iloc[0]
        return float(selected[threshold_col]), "fallback_no_threshold_met_specificity_floor"
    selected = viable.sort_values(
        [recall_col, f1_col, specificity_col, threshold_col],
        ascending=[False, False, False, False],
    ).iloc[0]
    return float(selected[threshold_col]), "max_recall_subject_to_specificity_ge_0.50"


def rounded_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def metrics_from_speaker_df(phase_d: Any, speaker_df: pd.DataFrame) -> dict[str, Any]:
    return rounded_metrics(
        phase_d.metrics_from_arrays(
            speaker_df["label"].astype(int).to_numpy(),
            speaker_df["score"].astype(float).to_numpy(),
            speaker_df["prediction"].astype(int).to_numpy(),
        )
    )


def build_component_configs(phase_d: Any) -> tuple[pd.DataFrame, list[ComponentRunConfig], dict[str, Any]]:
    manifest = phase_d.load_manifest()
    hubert_embeddings, hubert_meta = phase_d.load_cached_embeddings(phase_d.HUBERT_BACKBONE, manifest)
    wav2vec_embeddings, wav2vec_meta = phase_d.load_cached_embeddings(phase_d.WAV2VEC2_BACKBONE, manifest)
    handcrafted, handcrafted_cols, join_info = phase_d.load_aligned_handcrafted_features(manifest)

    for frame_name, frame in {
        "hubert_embeddings": hubert_embeddings,
        "wav2vec2_embeddings": wav2vec_embeddings,
    }.items():
        if len(frame) != len(manifest):
            raise ValueError(f"{frame_name} row count mismatch vs manifest.")
        if not frame["metadata_index"].astype(str).equals(manifest["metadata_index"].astype(str)):
            raise ValueError(f"{frame_name} metadata_index order does not match trusted manifest.")

    x_hubert, _ = phase_d.embedding_matrix(hubert_embeddings)
    x_wav2vec2, _ = phase_d.embedding_matrix(wav2vec_embeddings)
    x_handcrafted = handcrafted.to_numpy(dtype=np.float32)
    x_hubert_plus_handcrafted = np.hstack([x_hubert, x_handcrafted])

    configs = [
        ComponentRunConfig(
            model_id="hubert_only_logistic",
            component_label="HuBERT-only",
            display_name="HuBERT logistic",
            feature_mode="hubert_only",
            backbone=phase_d.HUBERT_BACKBONE,
            classifier=phase_d.LOGISTIC,
            aggregation="majority_vote",
            base_df=hubert_embeddings,
            x_matrix=x_hubert,
        ),
        ComponentRunConfig(
            model_id="hubert_plus_handcrafted_logistic_regression_platt",
            component_label="HuBERT + handcrafted",
            display_name="hubert_plus_handcrafted logistic_regression_platt",
            feature_mode="hubert_plus_handcrafted",
            backbone=phase_d.HUBERT_BACKBONE,
            classifier=phase_d.LOGISTIC,
            aggregation="majority_vote",
            base_df=hubert_embeddings,
            x_matrix=x_hubert_plus_handcrafted,
        ),
        ComponentRunConfig(
            model_id="phaseC_second_embedding_logistic",
            component_label="wav2vec2",
            display_name=f"{phase_d.WAV2VEC2_BACKBONE} logistic",
            feature_mode="embedding_only",
            backbone=phase_d.WAV2VEC2_BACKBONE,
            classifier=phase_d.LOGISTIC,
            aggregation="mean_score",
            base_df=wav2vec_embeddings,
            x_matrix=x_wav2vec2,
        ),
    ]

    meta = {
        "hubert_cache": hubert_meta.get("cache_csv"),
        "wav2vec2_cache": wav2vec_meta.get("cache_csv"),
        "handcrafted_feature_count": len(handcrafted_cols),
        "feature_join_info": join_info,
    }
    return manifest, configs, meta


def run_component_reconstruction(
    phase_d: Any,
    manifest: pd.DataFrame,
    configs: list[ComponentRunConfig],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = manifest["label"].astype(int).to_numpy()
    fold_rows: list[dict[str, Any]] = []
    clip_parts: list[pd.DataFrame] = []

    for seed in phase_d.SEEDS:
        folds, fold_note, n_splits = phase_d.build_outer_folds(manifest, seed=seed, max_splits=phase_d.MAX_SPLITS)
        for fold_spec in folds:
            train_speakers = set(manifest.loc[fold_spec.train_idx, "speaker_id"].astype(str))
            val_speakers = set(manifest.loc[fold_spec.val_idx, "speaker_id"].astype(str))
            overlap = train_speakers & val_speakers
            if overlap:
                raise ValueError(f"Leakage detected for seed={seed}, fold={fold_spec.fold}: {sorted(overlap)[:5]}")
            fold_rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fold_spec.fold),
                    "train_speakers": int(len(train_speakers)),
                    "validation_speakers": int(len(val_speakers)),
                    "train_clips": int(len(fold_spec.train_idx)),
                    "validation_clips": int(len(fold_spec.val_idx)),
                    "speaker_overlap_count": int(len(overlap)),
                    "fold_note": fold_note,
                    "n_splits": int(n_splits),
                }
            )
            for cfg in configs:
                fit = phase_d.fit_platt_model(
                    classifier=cfg.classifier,
                    x_train=cfg.x_matrix[fold_spec.train_idx],
                    y_train=y[fold_spec.train_idx],
                    x_val=cfg.x_matrix[fold_spec.val_idx],
                    seed=seed,
                )
                train_clip = phase_d.build_clip_frame(
                    cfg.base_df,
                    fold_spec.train_idx,
                    fit.train_probabilities,
                    cfg.model_id,
                    seed,
                    fold_spec.fold,
                    "train",
                )
                val_clip = phase_d.build_clip_frame(
                    cfg.base_df,
                    fold_spec.val_idx,
                    fit.val_probabilities,
                    cfg.model_id,
                    seed,
                    fold_spec.fold,
                    "validation",
                )
                clip_parts.extend([train_clip, val_clip])

    fold_info = pd.DataFrame(fold_rows).drop_duplicates().sort_values(["seed", "fold"]).reset_index(drop=True)
    fold_clip_scores = pd.concat(clip_parts, ignore_index=True)
    return fold_info, fold_clip_scores


def validate_reconstruction(
    phase_d: Any,
    manifest: pd.DataFrame,
    configs: list[ComponentRunConfig],
    fold_info: pd.DataFrame,
    fold_clip_scores: pd.DataFrame,
    component_specs: list[Any],
) -> tuple[dict[str, Any], dict[tuple[str, int, int, str], dict[str, Any]], pd.Series]:
    validation: dict[str, Any] = {}

    speaker_label_nunique = manifest.groupby("speaker_id")["label"].nunique()
    inconsistent_speakers = int((speaker_label_nunique > 1).sum())
    if inconsistent_speakers:
        raise ValueError(f"Manifest has {inconsistent_speakers} speakers with inconsistent labels.")
    validation["manifest_speaker_label_inconsistencies"] = inconsistent_speakers

    if int(fold_info["speaker_overlap_count"].sum()) != 0:
        raise ValueError("Fold registry reported speaker overlap between train and validation.")
    validation["speaker_train_validation_separation_pass"] = True
    validation["speaker_leakage_overlap_total"] = int(fold_info["speaker_overlap_count"].sum())

    non_numeric_scores = int(pd.to_numeric(fold_clip_scores["score"], errors="coerce").isna().sum())
    if non_numeric_scores:
        raise ValueError(f"Found {non_numeric_scores} non-numeric component probabilities.")
    out_of_bounds = int(((fold_clip_scores["score"] < 0.0) | (fold_clip_scores["score"] > 1.0)).sum())
    if out_of_bounds:
        raise ValueError(f"Found {out_of_bounds} component probabilities outside [0, 1].")
    validation["component_probability_non_numeric_count"] = non_numeric_scores
    validation["component_probability_out_of_bounds_count"] = out_of_bounds

    clip_count_frame = (
        fold_clip_scores.groupby(["seed", "fold", "split", "speaker_id", "model_id"]).size().unstack("model_id")
    )
    if clip_count_frame.isna().any().any():
        missing = int(clip_count_frame.isna().sum().sum())
        raise ValueError(f"Missing component clip counts for {missing} seed/fold/split/speaker rows.")
    inconsistent_clip_counts = int((clip_count_frame.nunique(axis=1) != 1).sum())
    if inconsistent_clip_counts:
        raise ValueError(f"Clip counts differ across components for {inconsistent_clip_counts} speakers.")
    clip_count_lookup = clip_count_frame.iloc[:, 0].astype(int)
    validation["component_clip_count_mismatch_rows"] = inconsistent_clip_counts

    score_grid = phase_d.precompute_component_score_grid(fold_clip_scores, component_specs)
    reference_by_split: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    missing_component_scores = 0
    for cfg in configs:
        for seed in phase_d.SEEDS:
            seed_folds = sorted(fold_clip_scores.loc[fold_clip_scores["seed"] == seed, "fold"].unique())
            for fold in seed_folds:
                for split in ["train", "validation"]:
                    key = (cfg.model_id, int(seed), int(fold), split)
                    if key not in score_grid:
                        raise ValueError(f"Missing speaker score grid for {key}.")
                    item = score_grid[key]
                    speaker_ids = np.asarray(item["speaker_id"], dtype=object)
                    labels = np.asarray(item["label"], dtype=int)
                    scores = np.asarray(item["scores"], dtype=float)
                    if speaker_ids.size != np.unique(speaker_ids).size:
                        raise ValueError(f"Duplicate aggregated speaker rows found for {key}.")
                    if not np.isfinite(scores).all():
                        raise ValueError(f"Non-finite cached speaker scores found for {key}.")
                    if ((scores < 0.0) | (scores > 1.0)).any():
                        raise ValueError(f"Cached speaker scores outside [0,1] found for {key}.")
                    ref_key = (int(seed), int(fold), split)
                    if ref_key not in reference_by_split:
                        reference_by_split[ref_key] = (speaker_ids, labels)
                    else:
                        ref_speakers, ref_labels = reference_by_split[ref_key]
                        if not np.array_equal(ref_speakers, speaker_ids):
                            raise ValueError(f"Speaker alignment mismatch across components for {ref_key}.")
                        if not np.array_equal(ref_labels, labels):
                            raise ValueError(f"Label alignment mismatch across components for {ref_key}.")
                    missing_component_scores += int(np.isnan(scores).sum())
    validation["speaker_alignment_pass"] = True
    validation["duplicate_aggregated_speaker_rows"] = 0
    validation["missing_component_probability_cells"] = missing_component_scores

    validation["validated_components"] = ", ".join(cfg.component_label for cfg in configs)
    validation["validated_seed_count"] = int(len(phase_d.SEEDS))
    validation["validated_fold_count"] = int(fold_info["fold"].nunique())
    validation["validated_outer_fold_rows"] = int(len(fold_info))

    return validation, score_grid, clip_count_lookup


def build_component_specs(phase_d: Any) -> list[Any]:
    return [
        phase_d.ComponentSpec(
            model_id="hubert_only_logistic",
            component_name="HuBERT logistic majority_vote default_0.5",
            feature_mode="hubert_only",
            classifier=phase_d.LOGISTIC,
            aggregation="majority_vote",
            threshold_policy="default_0.5",
        ),
        phase_d.ComponentSpec(
            model_id="hubert_plus_handcrafted_logistic_regression_platt",
            component_name="HuBERT+handcrafted logistic_regression_platt majority_vote default_0.5",
            feature_mode="hubert_plus_handcrafted",
            classifier=phase_d.LOGISTIC,
            aggregation="majority_vote",
            threshold_policy="default_0.5",
        ),
        phase_d.ComponentSpec(
            model_id="phaseC_second_embedding_logistic",
            component_name="facebook/wav2vec2-base logistic mean_score default_0.5",
            feature_mode="embedding_only",
            classifier=phase_d.LOGISTIC,
            aggregation="mean_score",
            threshold_policy="default_0.5",
        ),
    ]


def per_speaker_frame(
    speaker_df: pd.DataFrame,
    *,
    analysis_type: str,
    threshold: float,
    seed: int,
    fold: int,
    clip_count_lookup: pd.Series,
) -> pd.DataFrame:
    renamed = speaker_df.rename(
        columns={
            "label": "true_label",
            "score": "ensemble_probability",
            "prediction": "predicted_label",
            "hubert_only_logistic": "hubert_only_probability",
            "hubert_plus_handcrafted_logistic_regression_platt": "hubert_plus_handcrafted_probability",
            "phaseC_second_embedding_logistic": "wav2vec2_probability",
        }
    ).copy()
    renamed["analysis_type"] = analysis_type
    renamed["seed"] = int(seed)
    renamed["fold"] = int(fold)
    renamed["threshold"] = float(threshold)
    renamed["predicted_label"] = renamed["predicted_label"].astype(int)
    renamed["true_label"] = renamed["true_label"].astype(int)
    renamed["outcome_label"] = [
        outcome_label(int(y_true), int(y_pred))
        for y_true, y_pred in zip(renamed["true_label"], renamed["predicted_label"])
    ]
    component_cols = [
        "hubert_only_probability",
        "hubert_plus_handcrafted_probability",
        "wav2vec2_probability",
    ]
    renamed["max_component_name"] = renamed[component_cols].idxmax(axis=1)
    renamed["max_component_probability"] = renamed[component_cols].max(axis=1)
    renamed["components_at_or_above_threshold"] = (
        renamed[component_cols].ge(float(threshold)).sum(axis=1).astype(int)
    )
    renamed["component_support_type"] = renamed["components_at_or_above_threshold"].map(support_type)
    renamed["clip_count_for_speaker"] = [
        int(clip_count_lookup.loc[(int(seed), int(fold), "validation", str(speaker_id))])
        for speaker_id in renamed["speaker_id"]
    ]
    renamed["miss_margin"] = np.where(
        renamed["outcome_label"].eq("false_negative"),
        float(threshold) - renamed["ensemble_probability"].astype(float),
        np.nan,
    )
    renamed["false_negative_band"] = [
        miss_band(float(threshold), float(score))
        if label == "false_negative"
        else ""
        for label, score in zip(renamed["outcome_label"], renamed["ensemble_probability"])
    ]
    return renamed[
        [
            "analysis_type",
            "seed",
            "fold",
            "speaker_id",
            "true_label",
            "threshold",
            "ensemble_probability",
            "predicted_label",
            "outcome_label",
            "hubert_only_probability",
            "hubert_plus_handcrafted_probability",
            "wav2vec2_probability",
            "max_component_name",
            "max_component_probability",
            "components_at_or_above_threshold",
            "component_support_type",
            "clip_count_for_speaker",
            "miss_margin",
            "false_negative_band",
        ]
    ]


def summarize_per_fold(phase_d: Any, per_speaker: pd.DataFrame, analysis_type: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["analysis_type", "seed", "fold", "threshold"]
    for keys, group in per_speaker[per_speaker["analysis_type"] == analysis_type].groupby(group_cols, dropna=False):
        metrics = phase_d.metrics_from_arrays(
            group["true_label"].astype(int).to_numpy(),
            group["ensemble_probability"].astype(float).to_numpy(),
            group["predicted_label"].astype(int).to_numpy(),
        )
        analysis, seed, fold, threshold = keys
        rows.append(
            {
                "analysis_type": analysis,
                "seed": int(seed),
                "fold": int(fold),
                "threshold": float(threshold),
                "n_speakers": int(group["speaker_id"].nunique()),
                **rounded_metrics(metrics),
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis_type", "threshold", "seed", "fold"]).reset_index(drop=True)


def summarize_per_seed(phase_d: Any, per_speaker: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in per_speaker.groupby(["analysis_type", "seed", "threshold"], dropna=False):
        analysis_type, seed, threshold = keys
        metrics = phase_d.metrics_from_arrays(
            group["true_label"].astype(int).to_numpy(),
            group["ensemble_probability"].astype(float).to_numpy(),
            group["predicted_label"].astype(int).to_numpy(),
        )
        rows.append(
            {
                "analysis_type": analysis_type,
                "seed": int(seed),
                "threshold": float(threshold),
                "n_speakers": int(group["speaker_id"].nunique()),
                **rounded_metrics(metrics),
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis_type", "threshold", "seed"]).reset_index(drop=True)


def summarize_thresholds(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold, group in by_seed[by_seed["analysis_type"] == FIXED_SWEEP].groupby("threshold", dropna=False):
        rows.append(
            {
                "analysis_type": FIXED_SWEEP,
                "threshold": float(threshold),
                "n_seeds": int(group["seed"].nunique()),
                "speaker_recall_mean": float(group["recall"].mean()),
                "speaker_recall_std": float(group["recall"].std(ddof=1)) if len(group) > 1 else 0.0,
                "speaker_specificity_mean": float(group["specificity"].mean()),
                "speaker_specificity_std": float(group["specificity"].std(ddof=1)) if len(group) > 1 else 0.0,
                "speaker_precision_mean": float(group["precision"].mean()),
                "speaker_precision_std": float(group["precision"].std(ddof=1)) if len(group) > 1 else 0.0,
                "speaker_f1_mean": float(group["f1"].mean()),
                "speaker_f1_std": float(group["f1"].std(ddof=1)) if len(group) > 1 else 0.0,
                "true_negatives_mean": float(group["true_negatives"].mean()),
                "true_negatives_total": int(group["true_negatives"].sum()),
                "false_positives_mean": float(group["false_positives"].mean()),
                "false_positives_total": int(group["false_positives"].sum()),
                "false_negatives_mean": float(group["false_negatives"].mean()),
                "false_negatives_total": int(group["false_negatives"].sum()),
                "true_positives_mean": float(group["true_positives"].mean()),
                "true_positives_total": int(group["true_positives"].sum()),
                "target_zone_pass": "yes"
                if float(group["recall"].mean()) >= TARGET_RECALL and float(group["specificity"].mean()) >= TARGET_SPECIFICITY
                else "no",
                "specificity_floor_pass": "yes" if float(group["specificity"].mean()) >= TARGET_SPECIFICITY else "no",
            }
        )
    summary = pd.DataFrame(rows).sort_values("threshold", ascending=False).reset_index(drop=True)
    if summary.empty:
        return summary
    best_threshold, selection_note = choose_threshold(
        summary.rename(
            columns={
                "speaker_recall_mean": "recall",
                "speaker_specificity_mean": "specificity",
                "speaker_f1_mean": "f1",
            }
        )
    )
    summary["selection_note"] = ""
    summary.loc[np.isclose(summary["threshold"], best_threshold), "selection_note"] = selection_note
    return summary


def training_selected_summary(by_seed: pd.DataFrame, by_fold: pd.DataFrame) -> pd.DataFrame:
    seed_rows = by_seed[by_seed["analysis_type"] == TRAIN_SELECTED].copy()
    fold_rows = by_fold[by_fold["analysis_type"] == TRAIN_SELECTED].copy()
    if seed_rows.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "analysis_type": TRAIN_SELECTED,
                "threshold": float("nan"),
                "n_seeds": int(seed_rows["seed"].nunique()),
                "speaker_recall_mean": float(seed_rows["recall"].mean()),
                "speaker_recall_std": float(seed_rows["recall"].std(ddof=1)) if len(seed_rows) > 1 else 0.0,
                "speaker_specificity_mean": float(seed_rows["specificity"].mean()),
                "speaker_specificity_std": float(seed_rows["specificity"].std(ddof=1)) if len(seed_rows) > 1 else 0.0,
                "speaker_precision_mean": float(seed_rows["precision"].mean()),
                "speaker_precision_std": float(seed_rows["precision"].std(ddof=1)) if len(seed_rows) > 1 else 0.0,
                "speaker_f1_mean": float(seed_rows["f1"].mean()),
                "speaker_f1_std": float(seed_rows["f1"].std(ddof=1)) if len(seed_rows) > 1 else 0.0,
                "true_negatives_mean": float(seed_rows["true_negatives"].mean()),
                "true_negatives_total": int(seed_rows["true_negatives"].sum()),
                "false_positives_mean": float(seed_rows["false_positives"].mean()),
                "false_positives_total": int(seed_rows["false_positives"].sum()),
                "false_negatives_mean": float(seed_rows["false_negatives"].mean()),
                "false_negatives_total": int(seed_rows["false_negatives"].sum()),
                "true_positives_mean": float(seed_rows["true_positives"].mean()),
                "true_positives_total": int(seed_rows["true_positives"].sum()),
                "target_zone_pass": "yes"
                if float(seed_rows["recall"].mean()) >= TARGET_RECALL
                and float(seed_rows["specificity"].mean()) >= TARGET_SPECIFICITY
                else "no",
                "specificity_floor_pass": "yes"
                if float(seed_rows["specificity"].mean()) >= TARGET_SPECIFICITY
                else "no",
                "selection_note": "threshold_selected_on_training_speakers_only",
                "selected_threshold_mean": float(fold_rows["threshold"].mean()),
                "selected_threshold_min": float(fold_rows["threshold"].min()),
                "selected_threshold_max": float(fold_rows["threshold"].max()),
            }
        ]
    )


def reproduce_baseline_check(summary_df: pd.DataFrame) -> dict[str, Any]:
    phase_d_summary = pd.read_csv(BASELINE_PHASED_ENSEMBLE_CSV)
    historical = phase_d_summary[
        (phase_d_summary["ensemble_method"] == "max_probability_ensemble")
        & (phase_d_summary["threshold_policy"] == "default_0.5")
    ].copy()
    if historical.empty:
        raise ValueError("Could not locate historical Phase D max_probability_ensemble default_0.5 rows.")
    current = summary_df[np.isclose(summary_df["threshold"], 0.5)].copy()
    if current.empty:
        raise ValueError("Fixed sweep summary is missing threshold 0.500.")
    row = current.iloc[0]
    historical_means = {
        "speaker_recall_mean": float(historical["speaker_recall"].mean()),
        "speaker_specificity_mean": float(historical["speaker_specificity"].mean()),
        "speaker_f1_mean": float(historical["speaker_f1"].mean()),
    }
    current_means = {
        "speaker_recall_mean": float(row["speaker_recall_mean"]),
        "speaker_specificity_mean": float(row["speaker_specificity_mean"]),
        "speaker_f1_mean": float(row["speaker_f1_mean"]),
    }
    deltas = {key: current_means[key] - historical_means[key] for key in historical_means}
    max_abs_delta = max(abs(value) for value in deltas.values())
    if max_abs_delta > 1e-12:
        raise ValueError(f"Reconstructed default_0.5 metrics do not match historical Phase D baseline: {deltas}")
    return {
        "historical_reproduction_pass": True,
        "historical_default_0_5_recall": historical_means["speaker_recall_mean"],
        "historical_default_0_5_specificity": historical_means["speaker_specificity_mean"],
        "historical_default_0_5_f1": historical_means["speaker_f1_mean"],
        "historical_reproduction_max_abs_delta": max_abs_delta,
    }


def build_fixed_sweep_outputs(
    phase_d: Any,
    score_grid: dict[tuple[str, int, int, str], dict[str, Any]],
    component_specs: list[Any],
    clip_count_lookup: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_speaker_parts: list[pd.DataFrame] = []
    by_fold_rows: list[dict[str, Any]] = []
    for threshold in threshold_grid():
        for seed in phase_d.SEEDS:
            seed_folds = sorted({int(k[2]) for k in score_grid if k[1] == int(seed) and k[3] == "validation"})
            for fold in seed_folds:
                val = phase_d.ensemble_speaker_table_cached(
                    score_grid, component_specs, int(seed), int(fold), "validation", float(threshold), "max_probability_ensemble"
                )
                speaker_frame = per_speaker_frame(
                    val,
                    analysis_type=FIXED_SWEEP,
                    threshold=float(threshold),
                    seed=int(seed),
                    fold=int(fold),
                    clip_count_lookup=clip_count_lookup,
                )
                per_speaker_parts.append(speaker_frame)
                metrics = metrics_from_speaker_df(phase_d, val)
                by_fold_rows.append(
                    {
                        "analysis_type": FIXED_SWEEP,
                        "seed": int(seed),
                        "fold": int(fold),
                        "threshold": float(threshold),
                        "n_speakers": int(val["speaker_id"].nunique()),
                        **metrics,
                    }
                )
    per_speaker = pd.concat(per_speaker_parts, ignore_index=True)
    by_fold = pd.DataFrame(by_fold_rows).sort_values(["threshold", "seed", "fold"], ascending=[False, True, True])
    return per_speaker, by_fold.reset_index(drop=True)


def build_training_selected_outputs(
    phase_d: Any,
    score_grid: dict[tuple[str, int, int, str], dict[str, Any]],
    component_specs: list[Any],
    clip_count_lookup: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_speaker_parts: list[pd.DataFrame] = []
    by_fold_rows: list[dict[str, Any]] = []

    for seed in phase_d.SEEDS:
        seed_folds = sorted({int(k[2]) for k in score_grid if k[1] == int(seed) and k[3] == "validation"})
        for fold in seed_folds:
            train_metric_rows: list[dict[str, Any]] = []
            for threshold in threshold_grid():
                train = phase_d.ensemble_speaker_table_cached(
                    score_grid, component_specs, int(seed), int(fold), "train", float(threshold), "max_probability_ensemble"
                )
                metrics = metrics_from_speaker_df(phase_d, train)
                train_metric_rows.append({"threshold": float(threshold), **metrics})
            train_metrics = pd.DataFrame(train_metric_rows)
            selected_threshold, selection_note = choose_threshold(train_metrics)
            val = phase_d.ensemble_speaker_table_cached(
                score_grid,
                component_specs,
                int(seed),
                int(fold),
                "validation",
                float(selected_threshold),
                "max_probability_ensemble",
            )
            speaker_frame = per_speaker_frame(
                val,
                analysis_type=TRAIN_SELECTED,
                threshold=float(selected_threshold),
                seed=int(seed),
                fold=int(fold),
                clip_count_lookup=clip_count_lookup,
            )
            speaker_frame["selection_note"] = selection_note
            per_speaker_parts.append(speaker_frame)

            train_selected_metrics = train_metrics[np.isclose(train_metrics["threshold"], selected_threshold)].iloc[0]
            val_metrics = metrics_from_speaker_df(phase_d, val)
            by_fold_rows.append(
                {
                    "analysis_type": TRAIN_SELECTED,
                    "seed": int(seed),
                    "fold": int(fold),
                    "threshold": float(selected_threshold),
                    "selection_note": selection_note,
                    "n_speakers": int(val["speaker_id"].nunique()),
                    "train_recall": float(train_selected_metrics["recall"]),
                    "train_specificity": float(train_selected_metrics["specificity"]),
                    "train_precision": float(train_selected_metrics["precision"]),
                    "train_f1": float(train_selected_metrics["f1"]),
                    **val_metrics,
                }
            )
    per_speaker = pd.concat(per_speaker_parts, ignore_index=True)
    by_fold = pd.DataFrame(by_fold_rows).sort_values(["seed", "fold"]).reset_index(drop=True)
    if "selection_note" not in per_speaker.columns:
        per_speaker["selection_note"] = ""
    return per_speaker, by_fold


def component_score_table(per_speaker: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "analysis_type",
        "seed",
        "fold",
        "speaker_id",
        "true_label",
        "threshold",
        "ensemble_probability",
        "hubert_only_probability",
        "hubert_plus_handcrafted_probability",
        "wav2vec2_probability",
        "max_component_name",
        "max_component_probability",
        "components_at_or_above_threshold",
        "component_support_type",
        "predicted_label",
        "outcome_label",
    ]
    return per_speaker[keep].copy()


def best_threshold_row(summary_df: pd.DataFrame) -> pd.Series:
    viable = summary_df[summary_df["speaker_specificity_mean"] >= TARGET_SPECIFICITY].copy()
    if viable.empty:
        return summary_df.sort_values(
            ["speaker_specificity_mean", "speaker_recall_mean", "speaker_f1_mean", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
    return viable.sort_values(
        ["speaker_recall_mean", "speaker_f1_mean", "speaker_specificity_mean", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]


def build_report(
    *,
    validation_checks: dict[str, Any],
    reconstruction_meta: dict[str, Any],
    summary_df: pd.DataFrame,
    train_selected_summary_df: pd.DataFrame,
    per_speaker_df: pd.DataFrame,
) -> None:
    best_fixed = best_threshold_row(summary_df)
    default_row = summary_df[np.isclose(summary_df["threshold"], 0.5)].iloc[0]
    best_threshold = float(best_fixed["threshold"])
    best_selected = per_speaker_df[
        (per_speaker_df["analysis_type"] == FIXED_SWEEP) & np.isclose(per_speaker_df["threshold"], best_threshold)
    ].copy()
    target_hits = summary_df[
        (summary_df["speaker_recall_mean"] >= TARGET_RECALL) & (summary_df["speaker_specificity_mean"] >= TARGET_SPECIFICITY)
    ].copy()

    fn_selected = best_selected[best_selected["outcome_label"] == "false_negative"].copy()
    fp_selected = best_selected[best_selected["outcome_label"] == "false_positive"].copy()
    borderline_fn = int(fn_selected["false_negative_band"].eq("borderline").sum())
    moderate_fn = int(fn_selected["false_negative_band"].eq("moderate").sum())
    deep_fn = int(fn_selected["false_negative_band"].eq("deep").sum())
    mostly_borderline = borderline_fn >= max(moderate_fn + deep_fn, 1)

    fp_single = int(fp_selected["component_support_type"].eq("single_component").sum())
    fp_multi = int(fp_selected["component_support_type"].eq("multi_component").sum())
    fp_support_statement = "multi-component" if fp_multi > fp_single else "single-component"

    train_section = "(not run)"
    if not train_selected_summary_df.empty:
        train_section = markdown_table(
            train_selected_summary_df[
                [
                    "analysis_type",
                    "speaker_recall_mean",
                    "speaker_specificity_mean",
                    "speaker_precision_mean",
                    "speaker_f1_mean",
                    "selected_threshold_mean",
                    "selected_threshold_min",
                    "selected_threshold_max",
                    "target_zone_pass",
                ]
            ].round(4)
        )

    train_selected_pass = False
    if not train_selected_summary_df.empty:
        train_selected_pass = str(train_selected_summary_df.iloc[0]["target_zone_pass"]).strip().lower() == "yes"

    answer_1 = (
        f"Yes. Thresholds reaching the target zone in the fixed global sweep: "
        f"{', '.join(f'{value:.3f}' for value in target_hits['threshold'].tolist())}."
        if not target_hits.empty
        else "No. No fixed global threshold reached recall >= 0.70 while keeping specificity >= 0.50."
    )
    if target_hits.empty:
        answer_7 = "No."
    elif train_selected_pass:
        answer_7 = "Yes. Both the fixed sweep and the training-selected evaluation reached the target zone."
    else:
        answer_7 = (
            "Exploratory fixed-sweep yes, final training-selected-threshold no. "
            "The target zone was reached only in the fixed global sweep, not in the separate training-selected evaluation."
        )

    if mostly_borderline:
        fn_statement = (
            f"Mostly borderline. At the selected threshold there were {borderline_fn} borderline, "
            f"{moderate_fn} moderate, and {deep_fn} deep false negatives."
        )
    else:
        fn_statement = (
            f"Not mostly borderline. At the selected threshold there were {borderline_fn} borderline, "
            f"{moderate_fn} moderate, and {deep_fn} deep false negatives."
        )

    replacement_statement = (
        f"The fixed-sweep operating point `{best_threshold:.3f}` is the strongest exploratory threshold under the "
        "predefined selection rule."
    )
    if not train_selected_summary_df.empty:
        train_row = train_selected_summary_df.iloc[0]
        train_recall = float(train_row["speaker_recall_mean"])
        train_specificity = float(train_row["speaker_specificity_mean"])
        train_f1 = float(train_row["speaker_f1_mean"])
        default_recall = float(default_row["speaker_recall_mean"])
        default_specificity = float(default_row["speaker_specificity_mean"])
        if train_recall > default_recall and train_specificity >= TARGET_SPECIFICITY:
            replacement_statement += (
                f" Because the separate training-selected-threshold evaluation also improved on `default_0.5` "
                f"(recall {train_recall:.4f}, specificity {train_specificity:.4f}, F1 {train_f1:.4f}), "
                "that training-selected version is the cleaner replacement candidate."
            )
        else:
            replacement_statement += (
                f" The separate training-selected-threshold evaluation reached recall {train_recall:.4f}, "
                f"specificity {train_specificity:.4f}, and F1 {train_f1:.4f}. Because the fixed sweep scans held-out "
                "labels directly, and because this training-selected result did not move beyond `default_0.5`, "
                "`max_probability_ensemble default_0.5` should remain the final held candidate while `0.350` is "
                "reported as an exploratory post-hoc threshold finding."
            )
        if best_threshold == 0.5 and math.isclose(train_recall, default_recall) and math.isclose(train_specificity, default_specificity):
            replacement_statement += " In this run, that evidence does not support replacing `default_0.5`."
    elif math.isclose(best_threshold, 0.5):
        replacement_statement += " The selected fixed threshold is still `0.500`, so the held candidate remains unchanged."
    else:
        replacement_statement += (
            " No separate training-selected-threshold result was available, so the fixed sweep alone should be treated "
            "as exploratory rather than a clean held-candidate replacement basis."
        )

    sweep_view = summary_df[
        [
            "threshold",
            "speaker_recall_mean",
            "speaker_specificity_mean",
            "speaker_precision_mean",
            "speaker_f1_mean",
            "false_negatives_mean",
            "false_positives_mean",
            "target_zone_pass",
            "selection_note",
        ]
    ].round(4)

    validation_view = pd.DataFrame(
        [
            {"check": key, "value": value}
            for key, value in validation_checks.items()
            if key
            in {
                "speaker_train_validation_separation_pass",
                "speaker_leakage_overlap_total",
                "manifest_speaker_label_inconsistencies",
                "speaker_alignment_pass",
                "duplicate_aggregated_speaker_rows",
                "component_probability_non_numeric_count",
                "component_probability_out_of_bounds_count",
                "component_clip_count_mismatch_rows",
                "missing_component_probability_cells",
                "historical_reproduction_pass",
                "historical_reproduction_max_abs_delta",
            }
        ]
    )

    lines = [
        "# Final Recall Threshold Sweep Report",
        "",
        "## 1. Scope",
        "",
        "This run stayed inside the locked Phase D frame: trusted cleaned subset, speaker-grouped outer folds, "
        "seeds 42-46, and the held three-component max-probability ensemble only. "
        "No new architectures, features, folds, cached embeddings, or labels were introduced.",
        "",
        "Important implementation note: the HuBERT-only and HuBERT + handcrafted components use "
        "`majority_vote` speaker aggregation, so their speaker-level component scores are threshold-conditioned by "
        "design. This sweep preserves the original Phase D implementation rather than inventing a new decoupled rule.",
        "",
        "## 2. Reconstruction Inputs",
        "",
        f"- Trusted manifest: `{phase_safe(reconstruction_meta.get('trusted_manifest', 'reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv'))}`.",
        f"- HuBERT cache: `{reconstruction_meta['hubert_cache']}`.",
        f"- wav2vec2 cache: `{reconstruction_meta['wav2vec2_cache']}`.",
        f"- Handcrafted feature count: **{reconstruction_meta['handcrafted_feature_count']}**.",
        f"- Execution stack: Python `{reconstruction_meta['python_version']}`, scikit-learn `{reconstruction_meta['sklearn_version']}`, pandas `{reconstruction_meta['pandas_version']}`, numpy `{reconstruction_meta['numpy_version']}`.",
        "",
        "## 3. Validation Checks",
        "",
        markdown_table(validation_view),
        "",
        "## 4. Fixed Global Sweep",
        "",
        markdown_table(sweep_view),
        "",
        "## 5. Training-Selected Threshold Evaluation",
        "",
        train_section,
        "",
        "## 6. Required Answers",
        "",
        f"1. Did any threshold reach recall >= 0.70 and specificity >= 0.50? {answer_1}",
        f"2. Best threshold under the predefined selection rule: `{best_threshold:.3f}`.",
        f"3. Recall improvement vs `default_0.5`: {float(best_fixed['speaker_recall_mean']) - float(default_row['speaker_recall_mean']):+.4f}.",
        f"4. Specificity change vs `default_0.5`: {float(best_fixed['speaker_specificity_mean']) - float(default_row['speaker_specificity_mean']):+.4f}.",
        f"5. Were the remaining false negatives borderline or deeply missed? {fn_statement}",
        f"6. Are false positives mostly single-component or multi-component supported? Mostly {fp_support_statement} "
        f"({fp_single} single-component vs {fp_multi} multi-component false positives at the selected threshold).",
        f"7. Should the final project claim target-zone pass? {answer_7.rstrip('.')}.",
        f"8. Should `max_probability_ensemble default_0.5` remain the final held candidate, or should the new selected threshold replace it? {replacement_statement}",
        "",
        "## 7. Recommended Final Framing",
        "",
        (
            f"The fixed-sweep best threshold is `{best_threshold:.3f}` with mean speaker recall "
            f"{float(best_fixed['speaker_recall_mean']):.4f}, specificity {float(best_fixed['speaker_specificity_mean']):.4f}, "
            f"precision {float(best_fixed['speaker_precision_mean']):.4f}, and F1 {float(best_fixed['speaker_f1_mean']):.4f}."
        ),
        (
            "The result should still be reported as screening-support evidence, not diagnosis. "
            "If a target-zone threshold was not achieved, that should be stated explicitly."
            if target_hits.empty
            else "Because at least one fixed threshold reached the target zone, the report can state that the target zone "
            "was achieved in the exploratory fixed sweep. However, the training-selected-threshold evaluation stayed at "
            "`0.500`, so the held candidate should remain `default_0.5` unless a future clean validation run also "
            "supports the lower threshold."
        ),
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def phase_safe(value: str) -> str:
    return str(value).replace("`", "")


def main() -> None:
    ensure_dirs()
    phase_d = load_phase_d_module()
    manifest, configs, reconstruction_meta = build_component_configs(phase_d)
    reconstruction_meta["trusted_manifest"] = str(phase_d.TRUSTED_MANIFEST.relative_to(ROOT))
    reconstruction_meta["python_version"] = sys.version.split()[0]
    reconstruction_meta["sklearn_version"] = sklearn.__version__
    reconstruction_meta["pandas_version"] = pd.__version__
    reconstruction_meta["numpy_version"] = np.__version__

    print("[1/5] Reconstructing held component models", flush=True)
    fold_info, fold_clip_scores = run_component_reconstruction(phase_d, manifest, configs)
    component_specs = build_component_specs(phase_d)

    print("[2/5] Validating fold integrity and component alignment", flush=True)
    validation_checks, score_grid, clip_count_lookup = validate_reconstruction(
        phase_d,
        manifest,
        configs,
        fold_info,
        fold_clip_scores,
        component_specs,
    )

    print("[3/5] Running fixed threshold sweep", flush=True)
    fixed_per_speaker, fixed_by_fold = build_fixed_sweep_outputs(
        phase_d,
        score_grid,
        component_specs,
        clip_count_lookup,
    )
    fixed_by_seed = summarize_per_seed(phase_d, fixed_per_speaker)
    summary_df = summarize_thresholds(fixed_by_seed)
    validation_checks.update(reproduce_baseline_check(summary_df))

    print("[4/5] Running training-selected threshold evaluation", flush=True)
    train_per_speaker, train_by_fold = build_training_selected_outputs(
        phase_d,
        score_grid,
        component_specs,
        clip_count_lookup,
    )
    train_by_seed = summarize_per_seed(phase_d, train_per_speaker)
    train_summary_df = training_selected_summary(train_by_seed, train_by_fold)

    print("[5/5] Writing artifacts", flush=True)
    all_per_speaker = pd.concat([fixed_per_speaker, train_per_speaker], ignore_index=True)
    all_per_speaker = all_per_speaker.sort_values(
        ["analysis_type", "threshold", "seed", "fold", "speaker_id"], ascending=[True, False, True, True, True]
    ).reset_index(drop=True)

    fixed_by_fold = fixed_by_fold.copy()
    fixed_by_fold["selection_note"] = ""
    all_by_fold = pd.concat([fixed_by_fold, train_by_fold], ignore_index=True, sort=False).sort_values(
        ["analysis_type", "threshold", "seed", "fold"], ascending=[True, False, True, True]
    )
    all_by_seed = pd.concat([fixed_by_seed, train_by_seed], ignore_index=True, sort=False).sort_values(
        ["analysis_type", "threshold", "seed"], ascending=[True, False, True]
    )
    all_summary = pd.concat([summary_df, train_summary_df], ignore_index=True, sort=False).sort_values(
        ["analysis_type", "threshold"], ascending=[True, False]
    )

    false_negatives = all_per_speaker[all_per_speaker["outcome_label"] == "false_negative"].copy()
    false_positives = all_per_speaker[all_per_speaker["outcome_label"] == "false_positive"].copy()
    component_scores = component_score_table(all_per_speaker)

    all_summary.to_csv(SUMMARY_CSV, index=False)
    all_by_seed.to_csv(BY_SEED_CSV, index=False)
    all_by_fold.to_csv(BY_FOLD_CSV, index=False)
    all_per_speaker.to_csv(PER_SPEAKER_CSV, index=False)
    false_negatives.to_csv(FALSE_NEGATIVES_CSV, index=False)
    false_positives.to_csv(FALSE_POSITIVES_CSV, index=False)
    component_scores.to_csv(COMPONENT_SCORES_CSV, index=False)

    build_report(
        validation_checks=validation_checks,
        reconstruction_meta=reconstruction_meta,
        summary_df=summary_df,
        train_selected_summary_df=train_summary_df,
        per_speaker_df=all_per_speaker,
    )

    print("Created artifacts:", flush=True)
    for path in [
        SUMMARY_CSV,
        BY_SEED_CSV,
        BY_FOLD_CSV,
        PER_SPEAKER_CSV,
        FALSE_NEGATIVES_CSV,
        FALSE_POSITIVES_CSV,
        COMPONENT_SCORES_CSV,
        REPORT_PATH,
    ]:
        print(f"- {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
