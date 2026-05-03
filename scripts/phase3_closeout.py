"""Create Phase 3 closeout artifacts for the classical ML baseline.

The script is intentionally narrow: it does not change raw audio or re-extract
features. It turns Phase 2 governance decisions and Phase 3 QC outputs into a
governed baseline-ready feature table plus auditable documentation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METADATA_CSV = ROOT / "data/metadata/metadata_fixed_splits.csv"
RAW_AUDIO_ROOT = ROOT / "data/raw_audio"
PROCESSED_AUDIO_ROOT = ROOT / "data/processed_audio"
FEATURE_CSV = ROOT / "data/features/features_enhanced.csv"
GOVERNED_FEATURE_CSV = ROOT / "data/features/features_phase3_governed.csv"
PRUNED_FEATURE_CSV = ROOT / "data/features/features_phase3_governed_pruned.csv"
DECISIONS_CSV = ROOT / "reports/dataset_audit/pre_phase3_data_decisions.csv"
PREPROCESS_LOG_CSV = ROOT / "data/metadata/preprocess_log.csv"

REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
DOCS_DIR = ROOT / "docs"

GOVERNANCE_MANIFEST_CSV = REPORTS_DIR / "phase3_governance_manifest.csv"
GOVERNED_RECORD_DECISIONS_CSV = REPORTS_DIR / "phase3_governed_record_decisions.csv"
GOVERNANCE_UNMATCHED_CSV = REPORTS_DIR / "phase3_governance_unmatched_decisions.csv"
FEATURE_MANIFEST_CSV = REPORTS_DIR / "phase3_feature_manifest.csv"
PROXY_FEATURE_STATUS_CSV = REPORTS_DIR / "phase3_proxy_feature_status.csv"
REDUNDANCY_PRUNING_MANIFEST_CSV = REPORTS_DIR / "phase3_redundancy_pruning_manifest.csv"
REDUNDANCY_PRUNING_REPORT_MD = REPORTS_DIR / "phase3_redundancy_pruning_report.md"
RETAINED_CORRELATION_APPENDIX_CSV = REPORTS_DIR / "phase3_retained_feature_correlation_pairs.csv"
RETAINED_CORRELATION_APPENDIX_MD = REPORTS_DIR / "phase3_retained_feature_correlation_appendix.md"
SPEAKER_LEVEL_EVALUATION_MD = REPORTS_DIR / "phase3_speaker_level_evaluation.md"
FINAL_BASELINE_TAKEAWAY_MD = REPORTS_DIR / "phase3_final_baseline_takeaway.md"
FEATURE_FAMILY_ABLATION_REPORT_MD = REPORTS_DIR / "phase3_feature_family_ablation_report.md"
GAP_ASSESSMENT_MD = REPORTS_DIR / "phase3_gap_assessment.md"
PRE_PHASE4_GAP_CHECK_MD = REPORTS_DIR / "phase3_pre_phase4_gap_check.md"
PRE_PHASE4_VERIFICATION_MD = REPORTS_DIR / "phase3_pre_phase4_verification.md"
PHASE4_READINESS_MD = REPORTS_DIR / "phase3_phase4_readiness.md"
RUN_MANIFEST_MD = REPORTS_DIR / "phase3_run_manifest.md"
RUN_MANIFEST_JSON = REPORTS_DIR / "phase3_run_manifest.json"
VAD_NOTE_MD = REPORTS_DIR / "phase3_vad_pause_robustness_note.md"
LEAKAGE_NOTE_MD = REPORTS_DIR / "phase3_scaling_and_leakage_note.md"
CLOSEOUT_VALIDATION_MD = REPORTS_DIR / "phase3_closeout_validation.md"
CLOSEOUT_MD = REPORTS_DIR / "phase3_closeout.md"
PIPELINE_SPEC_MD = DOCS_DIR / "PHASE3_PIPELINE_SPEC.md"
FEATURE_POLICY_MD = DOCS_DIR / "PHASE3_FEATURE_ACCEPTANCE_POLICY.md"
FEATURE_DICTIONARY_MD = DOCS_DIR / "FEATURE_DICTIONARY.md"

MAX_FEATURE_MISSING_PCT = 0.25
CLASS_MISSING_GAP_CAUTION = 0.10
IQR_OUTLIER_CAUTION_PCT = 0.05
REDUNDANCY_CORRELATION_THRESHOLD = 0.98

METADATA_COLUMNS = {
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

JITTER_SHIMMER_FEATURES = {
    "jitter_local_proxy",
    "jitter_rap_proxy",
    "pitch_period_std_proxy",
    "shimmer_local_proxy",
    "shimmer_db_proxy",
}

ENERGY_MODULATION_PROXY_FEATURES = {
    "energy_modulation_mean_abs",
    "energy_modulation_std",
    "log_energy_modulation_std",
}

VAD_PAUSE_PROXY_PREFIXES = (
    "silence_",
    "pause_",
    "avg_silence_",
    "std_silence_",
    "median_silence_",
    "p10_silence_",
    "p25_silence_",
    "p75_silence_",
    "p90_silence_",
    "max_silence_",
    "avg_pause_",
    "std_pause_",
    "median_pause_",
    "p10_pause_",
    "p25_pause_",
    "p75_pause_",
    "p90_pause_",
    "max_pause_",
    "total_silence_",
    "speech_segment_",
    "avg_speech_segment_",
    "std_speech_segment_",
    "median_speech_segment_",
    "p10_speech_segment_",
    "p25_speech_segment_",
    "p75_speech_segment_",
    "p90_speech_segment_",
    "max_speech_segment_",
)

VAD_PAUSE_PROXY_FEATURES = {
    "pause_ratio",
    "silence_ratio",
    "unvoiced_ratio",
    "voiced_unvoiced_ratio",
    "energy_voiced_ratio",
    "speech_duration_sec",
    "speaking_rate_proxy",
    "onset_count",
    "onset_rate_per_speech_sec",
    "pause_count",
    "pause_rate_per_min",
    "silence_event_count",
    "speech_segment_count",
    "speech_segment_rate_per_min",
    "pause_to_speech_transition_count",
    "speech_to_pause_transition_count",
    "pause_speech_transition_rate_per_min",
}


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty DataFrame when the file is absent."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def repo_rel(path_value: object) -> str:
    """Normalize absolute or relative project paths to repository-relative POSIX paths."""
    if pd.isna(path_value):
        return ""
    raw = str(path_value).strip()
    if not raw:
        return ""
    path = Path(raw)
    try:
        if path.is_absolute():
            return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        pass
    normalized = raw.replace("\\", "/")
    anchors = (
        "data",
        "docs",
        "reports",
        "artifacts",
        "configs",
        "models",
        "app",
        "src",
        "tests",
        "scripts",
    )
    for anchor in anchors:
        marker = f"/{anchor}/"
        if marker in normalized:
            return f"{anchor}/" + normalized.split(marker, 1)[1]
    return normalized.lstrip("./")


def strictest_allowed(values: Iterable[object]) -> bool:
    """Return false when any Phase 2 use directive starts with an exclusion."""
    for value in values:
        text = "" if pd.isna(value) else str(value).strip().lower()
        if text.startswith("exclude"):
            return False
    return True


def combine_unique(values: Iterable[object], default: str = "") -> str:
    """Join unique non-empty values in first-seen order."""
    seen: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return "; ".join(seen) if seen else default


def summarize_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Collapse Phase 2 decision rows to one row per raw audio file."""
    if decisions.empty:
        return pd.DataFrame()

    dec = decisions.copy()
    dec["repo_audio_file"] = dec["file_path"].map(repo_rel)
    rows: List[Dict[str, object]] = []
    for repo_audio_file, group in dec.groupby("repo_audio_file", dropna=False):
        training_values = group.get("phase3_training_use", pd.Series(dtype=object))
        eval_values = group.get("phase3_validation_test_use", pd.Series(dtype=object))
        rows.append(
            {
                "repo_audio_file": repo_audio_file,
                "phase2_issue_type": combine_unique(group.get("issue_type", [])),
                "phase2_decision": combine_unique(group.get("decision", []), default="include"),
                "phase2_training_use": combine_unique(training_values, default="allowed_for_training"),
                "phase2_validation_test_use": combine_unique(eval_values, default="allowed_for_validation_test"),
                "manual_review_required": combine_unique(group.get("manual_review_required", []), default="False"),
                "phase2_reason": combine_unique(group.get("reason", []), default="not_flagged"),
                "required_action_before_phase3_final_eval": combine_unique(
                    group.get("required_action_before_phase3_final_eval", []),
                    default="none",
                ),
                "raw_file_action": combine_unique(group.get("raw_file_action", []), default="none"),
                "phase3_training_allowed": strictest_allowed(training_values),
                "phase3_final_eval_allowed": strictest_allowed(eval_values),
            }
        )
    return pd.DataFrame(rows)


def governance_action(row: pd.Series) -> Tuple[bool, str, str]:
    """Determine whether a row belongs in the governed Phase 3 baseline table."""
    split = str(row.get("datasplit", "")).strip().lower()
    train_allowed = bool(row.get("phase3_training_allowed", True))
    eval_allowed = bool(row.get("phase3_final_eval_allowed", True))
    decision = str(row.get("phase2_decision", "include_no_phase2_flag"))
    issue = str(row.get("phase2_issue_type", ""))

    if split == "train":
        if not train_allowed:
            return False, "excluded_from_training", f"Phase 2 training directive excludes this record: {decision}"
        if issue:
            return True, "included_with_phase2_flag", f"Included for training with Phase 2 flag: {decision}"
        return True, "included", "Not flagged by Phase 2 decision table."

    if split in {"valid", "test"}:
        if not eval_allowed:
            return (
                False,
                "excluded_from_final_validation_test",
                f"Phase 2 final-evaluation directive excludes this record: {decision}",
            )
        if issue:
            return True, "included_with_phase2_flag", f"Included for final evaluation with Phase 2 flag: {decision}"
        return True, "included", "Not flagged by Phase 2 decision table."

    if not train_allowed or not eval_allowed:
        return False, "excluded_from_unknown_split", f"Phase 2 directive excludes this record: {decision}"
    return True, "included_unknown_split", "Split is not train/valid/test but no Phase 2 exclusion applied."


def governed_decision_status(action: object) -> str:
    """Map internal governance actions to project-management decision labels."""
    text = "" if pd.isna(action) else str(action)
    return {
        "included": "include",
        "included_with_phase2_flag": "include_with_caution",
        "excluded_from_final_validation_test": "eval_restricted_excluded_from_governed_baseline",
        "excluded_from_training": "exclude",
        "excluded_from_unknown_split": "exclude",
        "included_unknown_split": "include_with_caution",
    }.get(text, text or "include")


def write_governed_record_decisions(manifest: pd.DataFrame) -> pd.DataFrame:
    """Write the concise governed-record decision file requested for closeout."""
    out = manifest.copy()
    out["decision_status"] = out["phase3_governance_action"].map(governed_decision_status)
    out["source_of_decision"] = np.where(
        out["phase2_decision"].fillna("").astype(str).eq("include_no_phase2_flag"),
        "default_phase3_include_rule",
        "reports/dataset_audit/pre_phase3_data_decisions.csv",
    )
    out["where_enforced"] = (
        "scripts/phase3_closeout.py writes data/features/features_phase3_governed.csv; "
        "Makefile phase3-baseline trains/evaluates using that governed table"
    )
    out["decision_reason"] = out["phase3_governance_reason"]
    out["record_id"] = out.get("metadata_index", pd.Series([""] * len(out)))
    columns = [
        "record_id",
        "repo_audio_file",
        "audio_file",
        "speaker_id",
        "speaker_grouping_status",
        "label_name",
        "datasplit",
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
    decision_df = out[[col for col in columns if col in out.columns]].copy()
    decision_df.to_csv(GOVERNED_RECORD_DECISIONS_CSV, index=False)
    return decision_df


def apply_governance(feature_df: pd.DataFrame, decision_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the governed feature table and per-record governance manifest."""
    if feature_df.empty:
        raise ValueError(f"Feature table is missing or empty: {FEATURE_CSV}")
    if "audio_file" not in feature_df.columns:
        raise ValueError("Feature table must contain an `audio_file` column for governance matching.")

    features = feature_df.copy()
    features["repo_audio_file"] = features["audio_file"].map(repo_rel)

    if decision_summary.empty:
        merged = features.copy()
    else:
        merged = features.merge(decision_summary, on="repo_audio_file", how="left", indicator=True)

    defaults = {
        "phase2_issue_type": "",
        "phase2_decision": "include_no_phase2_flag",
        "phase2_training_use": "allowed_for_training",
        "phase2_validation_test_use": "allowed_for_validation_test",
        "manual_review_required": "False",
        "phase2_reason": "not flagged in Phase 2 decision table",
        "required_action_before_phase3_final_eval": "none",
        "raw_file_action": "none",
        "phase3_training_allowed": True,
        "phase3_final_eval_allowed": True,
    }
    for column, default in defaults.items():
        if column not in merged.columns:
            merged[column] = default
        else:
            merged[column] = merged[column].fillna(default)

    actions = merged.apply(governance_action, axis=1)
    merged["phase3_baseline_included"] = [item[0] for item in actions]
    merged["phase3_governance_action"] = [item[1] for item in actions]
    merged["phase3_governance_reason"] = [item[2] for item in actions]
    merged["speaker_grouping_status"] = np.where(
        merged.get("speaker_id", "").fillna("").astype(str).str.strip().ne(""),
        "speaker_id_available",
        "speaker_id_missing",
    )

    manifest_columns = [
        "metadata_index",
        "speaker_id",
        "speaker_grouping_status",
        "label",
        "label_name",
        "datasplit",
        "audio_file",
        "repo_audio_file",
        "phase2_issue_type",
        "phase2_decision",
        "phase2_training_use",
        "phase2_validation_test_use",
        "manual_review_required",
        "phase2_reason",
        "required_action_before_phase3_final_eval",
        "raw_file_action",
        "phase3_training_allowed",
        "phase3_final_eval_allowed",
        "phase3_baseline_included",
        "phase3_governance_action",
        "phase3_governance_reason",
    ]
    manifest = merged[[col for col in manifest_columns if col in merged.columns]].copy()
    manifest.to_csv(GOVERNANCE_MANIFEST_CSV, index=False)
    write_governed_record_decisions(manifest)

    included_audio = set(merged["repo_audio_file"].astype(str))
    unmatched = decision_summary[~decision_summary["repo_audio_file"].astype(str).isin(included_audio)].copy()
    unmatched.to_csv(GOVERNANCE_UNMATCHED_CSV, index=False)

    original_columns = [col for col in feature_df.columns]
    governed = merged[merged["phase3_baseline_included"]].copy()
    governed = governed[original_columns]
    GOVERNED_FEATURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    governed.to_csv(GOVERNED_FEATURE_CSV, index=False)
    return governed, manifest, unmatched


def feature_family(feature: str) -> str:
    """Assign a stable feature family label for the feature manifest."""
    if feature.startswith("delta2_mfcc_"):
        return "delta_delta_mfcc"
    if feature.startswith("delta_mfcc_"):
        return "delta_mfcc"
    if feature.startswith("mfcc_"):
        return "mfcc"
    if feature.startswith("f0_") or feature == "voiced_ratio":
        return "pitch_prosody"
    if feature in JITTER_SHIMMER_FEATURES:
        return "voice_quality_proxy"
    if (
        feature.startswith("rms_")
        or feature == "low_energy_ratio"
        or feature.startswith("energy_modulation_")
        or feature == "log_energy_modulation_std"
    ):
        return "energy_loudness"
    if feature.startswith("spectral_centroid_"):
        return "spectral_centroid"
    if feature.startswith("spectral_bandwidth_"):
        return "spectral_bandwidth"
    if feature.startswith("spectral_rolloff_"):
        return "spectral_rolloff"
    if feature.startswith("spectral_flatness_"):
        return "spectral_flatness"
    if feature.startswith("zcr_"):
        return "zero_crossing_rate"
    if feature in VAD_PAUSE_PROXY_FEATURES or feature.startswith(VAD_PAUSE_PROXY_PREFIXES):
        return "pause_vad_fluency_proxy"
    if feature == "duration_sec":
        return "duration_qc"
    return "other_acoustic"


def direct_or_proxy(feature: str) -> str:
    """Label whether a feature is direct, estimated, or a proxy."""
    family = feature_family(feature)
    if family in {"voice_quality_proxy", "pause_vad_fluency_proxy"}:
        return "proxy"
    if family == "pitch_prosody":
        return "acoustic_estimate"
    if family == "duration_qc":
        return "direct_qc"
    if feature == "low_energy_ratio":
        return "proxy"
    return "direct_acoustic_descriptor"


def build_feature_manifest(governed: pd.DataFrame) -> pd.DataFrame:
    """Apply the formal Phase 3 feature acceptance policy to candidate features."""
    dropped = read_csv(TABLES_DIR / "phase3_dropped_features.csv")

    dropped_by_feature = dropped.set_index("feature").to_dict("index") if "feature" in dropped.columns else {}

    governed_numeric = set(governed.select_dtypes(include=["number"]).columns) - METADATA_COLUMNS - {"label"}
    candidate_features = sorted(governed_numeric | set(dropped_by_feature))

    rows: List[Dict[str, object]] = []
    for feature in candidate_features:
        d = dropped_by_feature.get(feature, {})
        if feature in governed.columns:
            series = pd.to_numeric(governed[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            non_null = series.dropna()
            missing_pct = float(series.isna().mean())
            variance = float(non_null.var()) if len(non_null) > 1 else 0.0
            n_unique = int(non_null.nunique())
            near_zero = bool(n_unique <= 1 or np.isclose(variance, 0.0))
            q1 = float(non_null.quantile(0.25)) if len(non_null) else np.nan
            q3 = float(non_null.quantile(0.75)) if len(non_null) else np.nan
            iqr = q3 - q1 if np.isfinite(q1) and np.isfinite(q3) else np.nan
            outlier_count = int(((series < (q1 - 3 * iqr)) | (series > (q3 + 3 * iqr))).sum()) if iqr and iqr > 0 else 0
            outlier_pct = float(outlier_count / len(series)) if len(series) else np.nan
            if "label" in governed.columns:
                class_missing = governed.assign(_missing=series.isna()).groupby("label")["_missing"].mean()
                class_gap = float(class_missing.max() - class_missing.min()) if len(class_missing) > 1 else 0.0
            else:
                class_gap = 0.0
        else:
            missing_pct = float(d.get("missing_pct", np.nan))
            variance = float(d.get("variance", np.nan))
            n_unique = int(d.get("n_unique_non_null", 0))
            near_zero = bool(n_unique <= 1 or np.isclose(variance, 0.0, equal_nan=False))
            outlier_pct = np.nan
            class_gap = 0.0

        family = feature_family(feature)
        measure_type = direct_or_proxy(feature)
        final_feature = feature in governed_numeric
        status = "kept"
        reason = "Passes Phase 3 missingness, variance, and stability checks."

        if feature in dropped_by_feature:
            status = "dropped"
            reason = str(d.get("drop_reason", "dropped_by_feature_cleaning_policy"))
            final_feature = False
        elif missing_pct > MAX_FEATURE_MISSING_PCT:
            status = "dropped"
            reason = f"Missingness {missing_pct:.2%} exceeds {MAX_FEATURE_MISSING_PCT:.0%} policy limit."
            final_feature = False
        elif near_zero or n_unique <= 1 or np.isclose(variance, 0.0, equal_nan=False):
            status = "dropped"
            reason = "Near-zero variance or one unique non-null value."
            final_feature = False
        elif feature in JITTER_SHIMMER_FEATURES:
            status = "kept_with_caution"
            reason = "Kept as a frame-level proxy only; not clinical cycle-level jitter/shimmer."
        elif family == "pause_vad_fluency_proxy":
            status = "kept_with_caution"
            reason = "Kept as an energy-based VAD/pause or speaking-rate proxy."
        elif class_gap >= CLASS_MISSING_GAP_CAUTION:
            status = "kept_with_caution"
            reason = f"Class missingness gap {class_gap:.2%} meets caution threshold."
        elif outlier_pct > IQR_OUTLIER_CAUTION_PCT:
            status = "kept_with_caution"
            reason = f"IQR outlier rate {outlier_pct:.2%} exceeds caution threshold."
        elif missing_pct > 0:
            status = "kept_with_caution"
            reason = f"Low missingness remains ({missing_pct:.2%}); imputed inside model pipeline."

        rows.append(
            {
                "feature": feature,
                "family": family,
                "direct_or_proxy": measure_type,
                "status": status,
                "final_baseline_feature": bool(final_feature and status != "dropped"),
                "decision_reason": reason,
                "missing_pct": missing_pct,
                "variance": variance,
                "n_unique_non_null": n_unique,
                "near_zero_variance": bool(near_zero),
                "iqr_outlier_pct": outlier_pct,
                "class_missingness_gap": class_gap,
                "policy_missingness_limit": MAX_FEATURE_MISSING_PCT,
                "policy_class_gap_caution": CLASS_MISSING_GAP_CAUTION,
                "policy_outlier_caution_pct": IQR_OUTLIER_CAUTION_PCT,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(FEATURE_MANIFEST_CSV, index=False)
    return manifest


def numeric_model_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric model feature columns, excluding labels and metadata."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric_cols if col not in METADATA_COLUMNS and col != "label"]


def feature_interpretability_score(feature: str, manifest_lookup: Dict[str, Dict[str, object]]) -> float:
    """Score a feature for redundancy tie-breaking; higher means prefer to keep."""
    info = manifest_lookup.get(feature, {})
    measure = str(info.get("direct_or_proxy", direct_or_proxy(feature)))
    status = str(info.get("status", "kept"))
    missing_pct = float(info.get("missing_pct", 0.0) or 0.0)
    outlier_pct = float(info.get("iqr_outlier_pct", 0.0) or 0.0)

    measure_score = {
        "direct_acoustic_descriptor": 40.0,
        "acoustic_estimate": 34.0,
        "direct_qc": 25.0,
        "proxy": 18.0,
    }.get(measure, 20.0)
    suffix_score = 0.0
    if feature.endswith(("_mean", "_median")):
        suffix_score = 8.0
    elif feature.endswith(("_p25", "_p75")):
        suffix_score = 7.0
    elif feature.endswith(("_p10", "_p90")):
        suffix_score = 6.0
    elif feature.endswith(("_std", "_iqr", "_range")):
        suffix_score = 5.0
    elif feature.endswith(("_min", "_max")):
        suffix_score = 2.0

    caution_penalty = 3.0 if status == "kept_with_caution" else 0.0
    return measure_score + suffix_score - caution_penalty - (missing_pct * 10.0) - (outlier_pct * 5.0)


def choose_redundant_drop(
    left: str,
    right: str,
    manifest_lookup: Dict[str, Dict[str, object]],
) -> Tuple[str, str]:
    """Return `(keep, drop)` for a highly correlated pair."""
    left_score = feature_interpretability_score(left, manifest_lookup)
    right_score = feature_interpretability_score(right, manifest_lookup)
    if not np.isclose(left_score, right_score):
        return (left, right) if left_score > right_score else (right, left)

    left_missing = float(manifest_lookup.get(left, {}).get("missing_pct", 0.0) or 0.0)
    right_missing = float(manifest_lookup.get(right, {}).get("missing_pct", 0.0) or 0.0)
    if not np.isclose(left_missing, right_missing):
        return (left, right) if left_missing < right_missing else (right, left)

    # Stable deterministic tie-breaker: keep the shorter, then alphabetically earlier, feature name.
    left_key = (len(left), left)
    right_key = (len(right), right)
    return (left, right) if left_key <= right_key else (right, left)


def apply_redundancy_pruning(
    governed: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    threshold: float = REDUNDANCY_CORRELATION_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop redundant numeric features using train-split correlations only."""
    if governed.empty or feature_manifest.empty:
        governed.to_csv(PRUNED_FEATURE_CSV, index=False)
        empty_manifest = pd.DataFrame(
            columns=[
                "feature",
                "family",
                "redundancy_pruning_status",
                "used_in_pruned_baseline",
                "redundant_with",
                "abs_correlation",
                "decision_reason",
            ]
        )
        empty_manifest.to_csv(REDUNDANCY_PRUNING_MANIFEST_CSV, index=False)
        return governed.copy(), empty_manifest, feature_manifest.copy()

    accepted = feature_manifest[feature_manifest["final_baseline_feature"] == True].copy()
    candidate_features = [col for col in accepted["feature"].astype(str).tolist() if col in governed.columns]
    if "datasplit" in governed.columns:
        train_df = governed[governed["datasplit"].astype(str).eq("train")].copy()
    else:
        train_df = pd.DataFrame()
    if train_df.empty:
        train_df = governed.copy()

    numeric_train = train_df[candidate_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = numeric_train.corr(method="spearman", min_periods=20).abs()
    manifest_lookup = feature_manifest.set_index("feature").to_dict("index") if "feature" in feature_manifest.columns else {}

    dropped: Dict[str, Dict[str, object]] = {}
    pair_rows: List[Dict[str, object]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value) or float(value) < threshold:
                continue
            if left in dropped or right in dropped:
                continue
            keep, drop = choose_redundant_drop(left, right, manifest_lookup)
            dropped[drop] = {
                "redundant_with": keep,
                "abs_correlation": float(value),
                "decision_reason": (
                    f"Spearman correlation {float(value):.3f} on training split exceeds "
                    f"{threshold:.2f}; kept `{keep}` as the more interpretable/stable feature."
                ),
            }
            pair_rows.append(
                {
                    "kept_feature": keep,
                    "dropped_feature": drop,
                    "abs_correlation": float(value),
                    "correlation_method": "spearman",
                    "correlation_split": "train",
                    "threshold": threshold,
                }
            )

    pruning_rows: List[Dict[str, object]] = []
    for feature in candidate_features:
        dropped_info = dropped.get(feature)
        used = dropped_info is None
        pruning_rows.append(
            {
                "feature": feature,
                "family": feature_family(feature),
                "direct_or_proxy": direct_or_proxy(feature),
                "redundancy_pruning_status": "kept" if used else "dropped_redundant",
                "used_in_pruned_baseline": bool(used),
                "redundant_with": "" if used else dropped_info["redundant_with"],
                "abs_correlation": np.nan if used else dropped_info["abs_correlation"],
                "decision_reason": (
                    "Retained after train-split redundancy screen."
                    if used
                    else dropped_info["decision_reason"]
                ),
                "correlation_method": "spearman",
                "correlation_split": "train",
                "correlation_threshold": threshold,
            }
        )

    pruning_manifest = pd.DataFrame(pruning_rows)
    pruning_manifest.to_csv(REDUNDANCY_PRUNING_MANIFEST_CSV, index=False)
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(REPORTS_DIR / "phase3_redundant_feature_pairs.csv", index=False)
    else:
        pd.DataFrame(
            columns=["kept_feature", "dropped_feature", "abs_correlation", "correlation_method", "correlation_split", "threshold"]
        ).to_csv(REPORTS_DIR / "phase3_redundant_feature_pairs.csv", index=False)

    drop_cols = sorted(dropped)
    pruned = governed.drop(columns=[col for col in drop_cols if col in governed.columns]).copy()
    pruned.to_csv(PRUNED_FEATURE_CSV, index=False)

    updated_manifest = feature_manifest.copy()
    updated_manifest["accepted_by_quality_policy"] = updated_manifest["final_baseline_feature"]
    pruning_status = pruning_manifest.set_index("feature")["redundancy_pruning_status"].to_dict()
    used_status = pruning_manifest.set_index("feature")["used_in_pruned_baseline"].to_dict()
    updated_manifest["redundancy_pruning_status"] = updated_manifest["feature"].map(pruning_status).fillna("not_quality_accepted")
    updated_manifest["final_baseline_feature"] = updated_manifest.apply(
        lambda row: bool(row["accepted_by_quality_policy"] and used_status.get(row["feature"], False)),
        axis=1,
    )
    updated_manifest.to_csv(FEATURE_MANIFEST_CSV, index=False)

    lines = [
        "# Phase 3 Redundancy Pruning Report",
        "",
        "- Scope: train-split-only correlation pruning for the official classical baseline feature table.",
        f"- Correlation method: Spearman absolute correlation on `datasplit == train` rows.",
        f"- Drop threshold: `abs(correlation) >= {threshold:.2f}`.",
        "- The pruning rule keeps the more interpretable/stable feature by preferring direct descriptors over proxies, lower missingness, and deterministic name tie-breaks.",
        "- No class labels or held-out validation/test feature distributions are used for pruning decisions.",
        "",
        f"- Quality-accepted features before pruning: **{len(candidate_features)}**.",
        f"- Redundant features pruned: **{len(drop_cols)}**.",
        f"- Official pruned baseline features: **{len(candidate_features) - len(drop_cols)}**.",
        "",
        "Machine-readable decisions: `reports/phase3_redundancy_pruning_manifest.csv`.",
        "Retained-feature correlation appendix: `reports/phase3_retained_feature_correlation_appendix.md`.",
        "Official pruned feature table: `data/features/features_phase3_governed_pruned.csv`.",
    ]
    REDUNDANCY_PRUNING_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pruned, pruning_manifest, updated_manifest


def fmt_metric(value: object) -> str:
    """Format a metric value for short markdown summaries."""
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def fmt_corr(value: object) -> str:
    """Format correlations with enough precision to compare against thresholds."""
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


def summarize_feature_family_ablation() -> Dict[str, object]:
    """Create a compact interpretation of feature-family ablation results."""
    ablation = read_csv(REPORTS_DIR / "classical_baseline_upgrade/feature_family_ablation_results.csv")
    if ablation.empty or "ablation_type" not in ablation.columns:
        return {
            "feature_family_ablation_available": False,
            "best_ablation_takeaway": "Feature-family ablation results were not available for this closeout run.",
            "largest_abs_ablation_delta": np.nan,
        }

    family_only = ablation[ablation["ablation_type"].astype(str).eq("feature_family_ablation")].copy()
    if family_only.empty:
        return {
            "feature_family_ablation_available": False,
            "best_ablation_takeaway": "Feature-family ablation rows were not available for this closeout run.",
            "largest_abs_ablation_delta": np.nan,
        }

    family_only["abs_delta_roc_auc"] = pd.to_numeric(
        family_only.get("delta_roc_auc_vs_all", np.nan),
        errors="coerce",
    ).abs()
    best_row = family_only.sort_values("roc_auc", ascending=False).iloc[0]
    strongest_shift = family_only.sort_values("abs_delta_roc_auc", ascending=False).iloc[0]
    best_delta = best_row.get("delta_roc_auc_vs_all", np.nan)
    best_group = str(best_row.get("removed_feature_group", "unknown"))
    best_model = str(best_row.get("model", "unknown"))

    voice_proxy_rows = family_only[family_only["removed_feature_group"].astype(str).eq("voice_quality_proxies")].copy()
    voice_proxy_line = "Voice-quality proxy ablations were unavailable."
    if not voice_proxy_rows.empty:
        voice_proxy_rows["abs_delta_roc_auc"] = pd.to_numeric(
            voice_proxy_rows.get("delta_roc_auc_vs_all", np.nan),
            errors="coerce",
        ).abs()
        voice_row = voice_proxy_rows.sort_values("abs_delta_roc_auc", ascending=False).iloc[0]
        voice_proxy_line = (
            "Voice-quality proxy removal produced only a small maximum absolute ROC-AUC change "
            f"({fmt_metric(voice_row.get('delta_roc_auc_vs_all'))} for `{voice_row.get('model')}`), "
            "supporting the Phase 3 decision to keep these columns only with caution rather than presenting them as decisive."
        )

    takeaway = (
        f"The best-performing family-ablation row was `{best_model}` with `{best_group}` removed "
        f"(ROC-AUC={fmt_metric(best_row.get('roc_auc'))}, PR-AUC={fmt_metric(best_row.get('pr_auc'))}, "
        f"balanced accuracy={fmt_metric(best_row.get('balanced_accuracy'))}, "
        f"delta ROC-AUC vs that model's all-feature run={fmt_metric(best_delta)}). "
        "This should be read as a sensitivity diagnostic, not proof that the removed family is harmful: "
        "models were not retuned per ablation and the held-out set is small. The practical takeaway is that "
        "no single feature family should be overclaimed as the dementia signal; family-level results should be "
        "reported as robustness evidence for the classical baseline."
    )

    return {
        "feature_family_ablation_available": True,
        "best_ablation_model": best_model,
        "best_ablation_removed_group": best_group,
        "best_ablation_roc_auc": float(best_row.get("roc_auc", np.nan)),
        "best_ablation_pr_auc": float(best_row.get("pr_auc", np.nan)),
        "best_ablation_balanced_accuracy": float(best_row.get("balanced_accuracy", np.nan)),
        "best_ablation_delta_roc_auc": float(best_delta) if not pd.isna(best_delta) else np.nan,
        "largest_abs_ablation_delta": float(strongest_shift.get("abs_delta_roc_auc", np.nan)),
        "largest_abs_ablation_model": str(strongest_shift.get("model", "unknown")),
        "largest_abs_ablation_group": str(strongest_shift.get("removed_feature_group", "unknown")),
        "best_ablation_takeaway": takeaway,
        "voice_quality_proxy_ablation_line": voice_proxy_line,
    }


def write_feature_family_ablation_report(ablation_summary: Dict[str, object]) -> None:
    """Write a readable feature-family ablation report with an explicit takeaway."""
    ablation = read_csv(REPORTS_DIR / "classical_baseline_upgrade/feature_family_ablation_results.csv")
    summary_rows: List[str] = []
    if not ablation.empty and "ablation_type" in ablation.columns:
        family_only = ablation[ablation["ablation_type"].astype(str).eq("feature_family_ablation")].copy()
        if not family_only.empty:
            family_only["abs_delta_roc_auc"] = pd.to_numeric(
                family_only.get("delta_roc_auc_vs_all", np.nan),
                errors="coerce",
            ).abs()
            top = family_only.sort_values("roc_auc", ascending=False).head(5)
            summary_rows = [
                "| Model | Removed family | ROC-AUC | Delta ROC-AUC | PR-AUC | Balanced accuracy |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
            for _, row in top.iterrows():
                summary_rows.append(
                    f"| `{row.get('model')}` | `{row.get('removed_feature_group')}` | "
                    f"{fmt_metric(row.get('roc_auc'))} | {fmt_metric(row.get('delta_roc_auc_vs_all'))} | "
                    f"{fmt_metric(row.get('pr_auc'))} | {fmt_metric(row.get('balanced_accuracy'))} |"
                )

    lines = [
        "# Phase 3 Feature Family Ablation Report",
        "",
        "- Each ablation refits the already selected classical model configuration on the training split after removing one feature family.",
        "- Hyperparameters are not retuned per ablation; this keeps the comparison practical and stable, but it is a diagnostic rather than a definitive causal importance test.",
        "- Evaluation uses the same held-out split and the same 0.5 uncalibrated decision threshold used in the ablation diagnostics.",
        "- Families removed one at a time: MFCC, F0/prosody, pause/fluency, voice-quality proxies, and energy features.",
        "",
        "## Best-Performing Ablation Takeaway",
        "",
        str(ablation_summary.get("best_ablation_takeaway", "Ablation interpretation unavailable.")),
        "",
        "## Voice-Quality Proxy Check",
        "",
        str(ablation_summary.get("voice_quality_proxy_ablation_line", "Voice-quality proxy ablation interpretation unavailable.")),
        "",
        "## Top Ablation Rows By ROC-AUC",
        "",
        *(summary_rows if summary_rows else ["Ablation table was not available when this report was generated."]),
        "",
        "Detailed table: `reports/classical_baseline_upgrade/feature_family_ablation_results.csv`.",
    ]
    FEATURE_FAMILY_ABLATION_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_retained_feature_correlation_appendix(
    pruned: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    top_n: int = 25,
) -> Dict[str, object]:
    """Write a top-pairs appendix for retained final features after pruning."""
    final_features = (
        feature_manifest.loc[feature_manifest["final_baseline_feature"] == True, "feature"].astype(str).tolist()
        if not feature_manifest.empty and "final_baseline_feature" in feature_manifest.columns
        else []
    )
    final_features = [feature for feature in final_features if feature in pruned.columns]
    train_df = pruned[pruned["datasplit"].astype(str).eq("train")].copy() if "datasplit" in pruned.columns else pruned.copy()
    if train_df.empty:
        train_df = pruned.copy()

    manifest_lookup = feature_manifest.set_index("feature").to_dict("index") if "feature" in feature_manifest.columns else {}
    pair_rows: List[Dict[str, object]] = []
    if final_features and not train_df.empty:
        numeric_train = train_df[final_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        corr = numeric_train.corr(method="spearman", min_periods=20)
        columns = list(corr.columns)
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                value = corr.loc[left, right]
                if pd.isna(value):
                    continue
                pair_rows.append(
                    {
                        "feature_a": left,
                        "family_a": str(manifest_lookup.get(left, {}).get("family", feature_family(left))),
                        "feature_b": right,
                        "family_b": str(manifest_lookup.get(right, {}).get("family", feature_family(right))),
                        "spearman_correlation": float(value),
                        "abs_spearman_correlation": abs(float(value)),
                        "correlation_split": "train",
                        "pruning_threshold": REDUNDANCY_CORRELATION_THRESHOLD,
                        "note": "retained_after_redundancy_pruning",
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values("abs_spearman_correlation", ascending=False).reset_index(drop=True)
        pair_df.insert(0, "rank", np.arange(1, len(pair_df) + 1))
    else:
        pair_df = pd.DataFrame(
            columns=[
                "rank",
                "feature_a",
                "family_a",
                "feature_b",
                "family_b",
                "spearman_correlation",
                "abs_spearman_correlation",
                "correlation_split",
                "pruning_threshold",
                "note",
            ]
        )
    pair_df.head(top_n).to_csv(RETAINED_CORRELATION_APPENDIX_CSV, index=False)

    threshold_count = (
        int((pair_df["abs_spearman_correlation"] >= REDUNDANCY_CORRELATION_THRESHOLD).sum())
        if not pair_df.empty
        else 0
    )
    high_95_count = int((pair_df["abs_spearman_correlation"] >= 0.95).sum()) if not pair_df.empty else 0
    max_corr = float(pair_df["abs_spearman_correlation"].max()) if not pair_df.empty else np.nan
    top_pair_line = "- No retained feature-pair correlations were available to summarize."
    if not pair_df.empty:
        top = pair_df.iloc[0]
        top_pair_line = (
            f"- Highest retained-pair absolute Spearman correlation: **{fmt_corr(top['abs_spearman_correlation'])}** "
            f"between `{top['feature_a']}` and `{top['feature_b']}`."
        )

    table_lines: List[str] = []
    if not pair_df.empty:
        table_lines = [
            "| Rank | Feature A | Feature B | Family A | Family B | Abs Spearman |",
            "| ---: | --- | --- | --- | --- | ---: |",
        ]
        for _, row in pair_df.head(10).iterrows():
            table_lines.append(
                f"| {int(row['rank'])} | `{row['feature_a']}` | `{row['feature_b']}` | "
                f"`{row['family_a']}` | `{row['family_b']}` | {fmt_corr(row['abs_spearman_correlation'])} |"
            )

    if threshold_count == 0:
        conclusion = (
            f"No retained feature pairs meet the formal pruning threshold of "
            f"`abs(Spearman) >= {REDUNDANCY_CORRELATION_THRESHOLD:.2f}` on the training split. "
            "This supports the closeout claim that pruning did not leave an obvious high-correlation redundancy cluster."
        )
    else:
        conclusion = (
            f"{threshold_count} retained feature pairs still meet or exceed the formal pruning threshold. "
            "These pairs should be reviewed before treating the pruned set as fully redundancy-controlled."
        )

    lines = [
        "# Phase 3 Retained Feature Correlation Appendix",
        "",
        "Purpose: show the strongest remaining correlations among final retained features after train-split redundancy pruning.",
        "",
        "## Method",
        "",
        f"- Feature set: final retained features in `data/features/features_phase3_governed_pruned.csv` (**{len(final_features)}** numeric features).",
        "- Split used: `datasplit == train` only.",
        "- Correlation: Spearman rank correlation with `min_periods=20`.",
        f"- Formal pruning threshold: `abs(Spearman) >= {REDUNDANCY_CORRELATION_THRESHOLD:.2f}`.",
        "- Labels and held-out validation/test distributions are not used in this appendix.",
        "",
        "## Summary",
        "",
        top_pair_line,
        f"- Retained pairs at or above the formal pruning threshold: **{threshold_count}**.",
        f"- Retained pairs at or above `0.95`: **{high_95_count}**.",
        "",
        conclusion,
        "",
        "## Top Retained Correlated Pairs",
        "",
        *(table_lines if table_lines else ["No retained pairs were available."]),
        "",
        f"Machine-readable top {top_n} pairs: `reports/phase3_retained_feature_correlation_pairs.csv`.",
    ]
    RETAINED_CORRELATION_APPENDIX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "retained_correlation_pair_count": int(len(pair_df)),
        "retained_correlation_top_pairs_written": int(min(top_n, len(pair_df))),
        "retained_correlation_pairs_at_pruning_threshold": threshold_count,
        "retained_correlation_pairs_at_0_95": high_95_count,
        "max_retained_abs_spearman": max_corr,
    }


def best_clip_metric_summary() -> Dict[str, object]:
    """Return compact best clip-level metric summaries for the final takeaway."""
    metrics = read_csv(TABLES_DIR / "baseline_metrics_enhanced.csv")
    if metrics.empty:
        return {}

    rows: Dict[str, object] = {}
    uncal = metrics[metrics["variant"].astype(str).eq("uncalibrated")].copy() if "variant" in metrics.columns else metrics.copy()
    if not uncal.empty and "roc_auc" in uncal.columns:
        best_roc = uncal.sort_values("roc_auc", ascending=False).iloc[0]
        rows["best_uncalibrated_model"] = best_roc.get("model")
        rows["best_uncalibrated_roc_auc"] = best_roc.get("roc_auc")
        rows["best_uncalibrated_pr_auc"] = best_roc.get("pr_auc")
        rows["best_uncalibrated_balanced_accuracy"] = best_roc.get("balanced_accuracy")

    cal = metrics[metrics["variant"].astype(str).eq("calibrated")].copy() if "variant" in metrics.columns else pd.DataFrame()
    if not cal.empty:
        sort_cols = [col for col in ["balanced_accuracy", "recall", "specificity", "pr_auc"] if col in cal.columns]
        if sort_cols:
            cal_best = cal.sort_values(sort_cols, ascending=False).iloc[0]
        else:
            cal_best = cal.iloc[0]
        rows["best_calibrated_model"] = cal_best.get("model")
        rows["best_calibrated_recall"] = cal_best.get("recall")
        rows["best_calibrated_specificity"] = cal_best.get("specificity")
        rows["best_calibrated_balanced_accuracy"] = cal_best.get("balanced_accuracy")
        rows["best_calibrated_pr_auc"] = cal_best.get("pr_auc")
    return rows


def best_speaker_metric_summary() -> Dict[str, object]:
    """Return compact speaker-level metric summary for the final takeaway."""
    speaker_metrics = read_csv(REPORTS_DIR / "classical_baseline_upgrade/speaker_level_metrics.csv")
    if speaker_metrics.empty:
        return {}
    mean_cal = speaker_metrics[
        (speaker_metrics.get("variant", "") == "calibrated")
        & (speaker_metrics.get("aggregation_method", "") == "mean")
    ].copy()
    if mean_cal.empty:
        return {}
    sort_cols = [col for col in ["balanced_accuracy", "specificity", "pr_auc", "roc_auc", "recall"] if col in mean_cal.columns]
    row = mean_cal.sort_values(sort_cols, ascending=False).iloc[0] if sort_cols else mean_cal.iloc[0]
    return {
        "best_speaker_model": row.get("model"),
        "best_speaker_recall": row.get("recall"),
        "best_speaker_specificity": row.get("specificity"),
        "best_speaker_pr_auc": row.get("pr_auc"),
        "best_speaker_roc_auc": row.get("roc_auc"),
    }


def write_final_baseline_takeaway(
    counts: Dict[str, object],
    ablation_summary: Dict[str, object],
    correlation_summary: Dict[str, object],
) -> None:
    """Write a one-page final baseline takeaway for Phase 3."""
    clip = best_clip_metric_summary()
    speaker = best_speaker_metric_summary()

    lines = [
        "# Phase 3 Final Baseline Takeaway",
        "",
        "Scope: one-page closeout summary for the Phase 3 classical ML baseline only.",
        "",
        "## Most Important Findings",
        "",
        "1. **The official Phase 3 baseline is now governed and reproducible.** "
        f"The frozen model-input table is `data/features/features_phase3_governed_pruned.csv` with "
        f"**{counts.get('pruned_feature_rows')}** governed rows and **{counts.get('final_baseline_feature_count')}** final numeric features. "
        f"Phase 2 governance excluded **{counts.get('governance_excluded_rows')}** rows before baseline training/evaluation.",
        "",
        "2. **Preprocessing and feature extraction are stable enough for a classical baseline, but not clinically definitive.** "
        f"Current validation records **{counts.get('preprocessed_ok')}** usable processed outputs, "
        f"**{counts.get('preprocessing_failed_or_skipped')}** preprocessing failures/skips, and "
        f"**{counts.get('feature_extraction_failures_rows')}** feature-extraction failures. "
        "The pipeline standardizes files to 16 kHz mono WAV and keeps missing-value handling inside model pipelines.",
        "",
        "3. **Held-out clip-level model performance is modest and should be described as screening-baseline evidence.** "
        f"The best uncalibrated ROC-AUC row is `{clip.get('best_uncalibrated_model', 'n/a')}` "
        f"(ROC-AUC={fmt_metric(clip.get('best_uncalibrated_roc_auc'))}, PR-AUC={fmt_metric(clip.get('best_uncalibrated_pr_auc'))}, "
        f"balanced accuracy={fmt_metric(clip.get('best_uncalibrated_balanced_accuracy'))}). "
        f"The most balanced calibrated row is `{clip.get('best_calibrated_model', 'n/a')}` "
        f"(recall={fmt_metric(clip.get('best_calibrated_recall'))}, specificity={fmt_metric(clip.get('best_calibrated_specificity'))}, "
        f"balanced accuracy={fmt_metric(clip.get('best_calibrated_balanced_accuracy'))}).",
        "",
        "4. **Speaker-level reporting makes the baseline more defensible.** "
        f"Mean probability aggregation at speaker level reports `{speaker.get('best_speaker_model', 'n/a')}` as the best calibrated row "
        f"(recall={fmt_metric(speaker.get('best_speaker_recall'))}, specificity={fmt_metric(speaker.get('best_speaker_specificity'))}, "
        f"PR-AUC={fmt_metric(speaker.get('best_speaker_pr_auc'))}, ROC-AUC={fmt_metric(speaker.get('best_speaker_roc_auc'))}). "
        "This is still exploratory, but it better matches the speaker-level screening use case than isolated clip metrics alone.",
        "",
        "5. **Ablation and redundancy checks argue for caution, not feature overclaiming.** "
        + str(ablation_summary.get("best_ablation_takeaway", "Feature-family ablations were unavailable.")),
        "",
        "## Redundancy Check Takeaway",
        "",
        f"Train-split redundancy pruning removed **{counts.get('redundancy_pruned_feature_count')}** features before the official model-input table was frozen. "
        f"The retained-feature appendix found **{correlation_summary.get('retained_correlation_pairs_at_pruning_threshold')}** retained pairs at or above "
        f"`abs(Spearman) >= {REDUNDANCY_CORRELATION_THRESHOLD:.2f}`; maximum retained absolute Spearman correlation was "
        f"**{fmt_corr(correlation_summary.get('max_retained_abs_spearman'))}**. "
        "This supports using the pruned feature set as the official Phase 3 table while still acknowledging that moderate feature correlations remain.",
        "",
        "Official closeout note: `reports/phase3_closeout.md`.",
    ]
    FINAL_BASELINE_TAKEAWAY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bool_flag(value: object) -> bool:
    """Coerce bool-like manifest values safely."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def is_phase4_proxy_like(row: pd.Series) -> bool:
    """Return true for retained/omitted features that need proxy interpretation."""
    feature = str(row.get("feature", ""))
    family = str(row.get("family", ""))
    measure = str(row.get("direct_or_proxy", ""))
    if measure in {"proxy", "acoustic_estimate"}:
        return True
    if family in {"voice_quality_proxy", "pause_vad_fluency_proxy"}:
        return True
    if feature in ENERGY_MODULATION_PROXY_FEATURES:
        return True
    return False


def proxy_policy_decision(row: pd.Series) -> Dict[str, str]:
    """Classify a feature for Phase 4 proxy-readiness reporting."""
    feature = str(row.get("feature", ""))
    family = str(row.get("family", ""))
    status = str(row.get("status", ""))
    pruning_status = str(row.get("redundancy_pruning_status", ""))
    final_feature = bool_flag(row.get("final_baseline_feature", False))
    proxy_like = is_phase4_proxy_like(row)

    if status == "dropped":
        return {
            "phase4_proxy_policy_category": "should_be_excluded",
            "phase4_modeling_use": "excluded_from_official_final_table",
            "phase4_proxy_policy_reason": "Excluded by Phase 3 hard feature policy; not safe for the official baseline.",
            "phase4_interpretation_caution": "Do not use for Phase 4 unless the feature extraction/quality issue is explicitly fixed and the manifest is regenerated.",
        }

    if not final_feature:
        return {
            "phase4_proxy_policy_category": "exploratory_only",
            "phase4_modeling_use": "not_in_official_final_table",
            "phase4_proxy_policy_reason": (
                "Quality-accepted but removed from the official model-input table by redundancy pruning."
                if pruning_status == "dropped_redundant"
                else "Not part of the official final model-input table."
            ),
            "phase4_interpretation_caution": "May be inspected only as an exploratory or audit variable; do not treat as an official Phase 3 modeling feature.",
        }

    if proxy_like:
        if family == "voice_quality_proxy":
            reason = "Retained with caution as a frame-level pitch/RMS variability proxy, not clinical cycle-level jitter or shimmer."
            caution = "Interpret only as an acoustic voice-quality approximation; do not call it clinical jitter/shimmer."
        elif family == "pause_vad_fluency_proxy":
            reason = "Retained with caution as an energy-based pause/VAD/fluency proxy."
            caution = "Interpret as threshold-dependent acoustic segmentation, not transcript-derived fluency, hesitation, or words-per-minute."
        elif family == "pitch_prosody":
            reason = "Retained with caution as an F0 estimate from automatic pitch tracking."
            caution = "Interpret as an acoustic estimate on voiced frames, not a manually verified laryngeal or clinical measurement."
        elif feature in ENERGY_MODULATION_PROXY_FEATURES:
            reason = "Retained with caution as a short-term energy-modulation proxy for speech dynamics."
            caution = "Interpret as an acoustic dynamics approximation, not a transcript- or articulation-based tempo measure."
        else:
            reason = "Retained with caution as a documented acoustic proxy."
            caution = "Interpret as an approximation, not a clinical biomarker."
        return {
            "phase4_proxy_policy_category": "proxy_but_retained",
            "phase4_modeling_use": "included_in_official_final_table",
            "phase4_proxy_policy_reason": reason,
            "phase4_interpretation_caution": caution,
        }

    return {
        "phase4_proxy_policy_category": "direct",
        "phase4_modeling_use": "included_in_official_final_table",
        "phase4_proxy_policy_reason": "Retained as a direct acoustic descriptor or QC descriptor after Phase 3 quality and redundancy checks.",
        "phase4_interpretation_caution": "Still an acoustic model input, not a clinical biomarker.",
    }


def apply_phase4_proxy_policy(feature_manifest: pd.DataFrame) -> pd.DataFrame:
    """Add explicit Phase 4 proxy policy columns to the official feature manifest."""
    if feature_manifest.empty:
        feature_manifest.to_csv(FEATURE_MANIFEST_CSV, index=False)
        return feature_manifest

    updated = feature_manifest.copy()
    policy_rows = updated.apply(proxy_policy_decision, axis=1, result_type="expand")
    for col in policy_rows.columns:
        updated[col] = policy_rows[col]
    updated.to_csv(FEATURE_MANIFEST_CSV, index=False)
    return updated


def write_proxy_feature_status(feature_manifest: pd.DataFrame) -> Dict[str, object]:
    """Write a machine-readable direct/proxy/excluded status table for Phase 4."""
    columns = [
        "feature",
        "family",
        "direct_or_proxy",
        "phase4_proxy_policy_category",
        "phase4_modeling_use",
        "final_baseline_feature",
        "status",
        "redundancy_pruning_status",
        "missing_pct",
        "variance",
        "near_zero_variance",
        "phase4_proxy_policy_reason",
        "phase4_interpretation_caution",
    ]
    if feature_manifest.empty:
        pd.DataFrame(columns=columns).to_csv(PROXY_FEATURE_STATUS_CSV, index=False)
        return {}

    status = feature_manifest[[col for col in columns if col in feature_manifest.columns]].copy()
    status["stability_missingness_note"] = status.apply(
        lambda row: (
            f"missing_pct={float(row.get('missing_pct', np.nan)):.4f}; "
            f"near_zero_variance={row.get('near_zero_variance', '')}; "
            f"variance={float(row.get('variance', np.nan)):.6g}"
        ),
        axis=1,
    )
    status.to_csv(PROXY_FEATURE_STATUS_CSV, index=False)

    counts = status["phase4_proxy_policy_category"].value_counts(dropna=False).to_dict()
    retained_proxy = status[
        status["phase4_proxy_policy_category"].eq("proxy_but_retained")
        & status["final_baseline_feature"].map(bool_flag)
    ]
    return {
        "proxy_policy_counts": counts,
        "phase4_retained_proxy_feature_count": int(len(retained_proxy)),
        "phase4_exploratory_feature_count": int((status["phase4_proxy_policy_category"] == "exploratory_only").sum()),
        "phase4_should_be_excluded_feature_count": int((status["phase4_proxy_policy_category"] == "should_be_excluded").sum()),
    }


def table_metric(path: Path, metric: str) -> object:
    """Fetch a metric from a two-column Phase 3 summary table."""
    df = read_csv(path)
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return ""
    hit = df[df["metric"].astype(str) == metric]
    return hit["value"].iloc[0] if not hit.empty else ""


def git_note() -> Dict[str, str]:
    """Return a lightweight code/version note without requiring a clean git state."""
    def run_git(args: List[str]) -> Tuple[int, str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode, result.stdout.strip()
        except Exception:
            return 1, ""

    inside_code, inside_stdout = run_git(["rev-parse", "--is-inside-work-tree"])
    inside_work_tree = inside_code == 0 and inside_stdout.lower() == "true"

    rev_code, rev = run_git(["rev-parse", "HEAD"])
    short_code, short_rev = run_git(["rev-parse", "--short", "HEAD"])
    _, branch = run_git(["branch", "--show-current"])
    _, status = run_git(["status", "--short"])

    status_lines = [line for line in status.splitlines() if line.strip()]
    status_counts: Dict[str, int] = {}
    for line in status_lines:
        key = line[:2].strip() or "tracked_change"
        status_counts[key] = status_counts.get(key, 0) + 1

    if not inside_work_tree:
        workspace_state = "not_a_git_work_tree"
    elif status_lines:
        workspace_state = "uncommitted_or_untracked_changes_present"
    else:
        workspace_state = "clean"

    if rev_code != 0:
        rev = "not_available_no_commit_or_unreadable_head"
    if short_code != 0:
        short_rev = "not_available_no_commit_or_unreadable_head"

    return {
        "git_available": str(bool(inside_work_tree)),
        "git_revision": short_rev,
        "git_commit_hash": rev,
        "git_branch": branch or "not_available",
        "workspace_state": workspace_state,
        "working_tree_clean": str(bool(inside_work_tree and not status_lines)),
        "git_status_entry_count": str(len(status_lines)),
        "git_status_summary": json.dumps(status_counts, sort_keys=True),
    }


def legacy_git_note() -> Dict[str, str]:
    """Deprecated compatibility wrapper kept to avoid changing older imports."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        rev = ""
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        status = ""
    return {
        "git_revision": rev or "not_available",
        "workspace_state": "uncommitted_or_untracked_changes_present" if status else "clean_or_not_a_git_repo",
    }


def json_safe(value: object) -> object:
    """Convert pandas/numpy values to JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def split_counts(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Summarize counts by split for manifest reporting."""
    if df.empty or "datasplit" not in df.columns:
        return []
    rows: List[Dict[str, object]] = []
    for split, sub in df.groupby("datasplit", dropna=False):
        rows.append(
            {
                "split": str(split),
                "rows": int(len(sub)),
                "dementia_rows": int((sub.get("label", pd.Series(dtype=int)).astype(int) == 1).sum()) if "label" in sub else 0,
                "non_dementia_rows": int((sub.get("label", pd.Series(dtype=int)).astype(int) == 0).sum()) if "label" in sub else 0,
                "speakers": int(sub["speaker_id"].nunique()) if "speaker_id" in sub.columns else 0,
            }
        )
    return rows


def build_counts(
    raw_features: pd.DataFrame,
    governed: pd.DataFrame,
    pruned: pd.DataFrame,
    governance: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    pruning_manifest: pd.DataFrame,
    unmatched_decisions: pd.DataFrame,
) -> Dict[str, object]:
    """Collect exact closeout counts from current artifacts."""
    preprocess_failures = read_csv(TABLES_DIR / "phase3_preprocessing_failures.csv")
    feature_failures = read_csv(TABLES_DIR / "phase3_feature_extraction_failures.csv")
    preprocessing_summary_path = TABLES_DIR / "phase3_preprocessing_summary.csv"
    quality = read_csv(TABLES_DIR / "phase3_feature_quality.csv")
    row_missing = read_csv(TABLES_DIR / "phase3_row_missingness.csv")
    preprocess_log = read_csv(PREPROCESS_LOG_CSV)

    accepted_features = (
        feature_manifest[feature_manifest.get("accepted_by_quality_policy", feature_manifest.get("final_baseline_feature", False)) == True]
        if not feature_manifest.empty
        else pd.DataFrame()
    )
    final_features = feature_manifest[feature_manifest["final_baseline_feature"] == True] if not feature_manifest.empty else pd.DataFrame()
    dropped_features = feature_manifest[feature_manifest["status"] == "dropped"] if not feature_manifest.empty else pd.DataFrame()
    caution_features = feature_manifest[feature_manifest["status"] == "kept_with_caution"] if not feature_manifest.empty else pd.DataFrame()
    pruned_features = (
        pruning_manifest[pruning_manifest["redundancy_pruning_status"] == "dropped_redundant"]
        if not pruning_manifest.empty and "redundancy_pruning_status" in pruning_manifest.columns
        else pd.DataFrame()
    )
    ablation_results = read_csv(REPORTS_DIR / "classical_baseline_upgrade/feature_family_ablation_results.csv")
    speaker_metrics = read_csv(REPORTS_DIR / "classical_baseline_upgrade/speaker_level_metrics.csv")

    numeric_feature_count = int(len(final_features))
    missing_max = float(quality["missing_pct"].max()) if not quality.empty and "missing_pct" in quality.columns else np.nan
    row_missing_pct_col = "row_missing_pct" if "row_missing_pct" in row_missing.columns else "feature_missing_pct"
    row_missing_max = (
        float(row_missing[row_missing_pct_col].max())
        if not row_missing.empty and row_missing_pct_col in row_missing.columns
        else np.nan
    )

    counts = {
        "metadata_rows": int(len(read_csv(METADATA_CSV))),
        "raw_feature_rows": int(len(raw_features)),
        "governed_feature_rows": int(len(governed)),
        "pruned_feature_rows": int(len(pruned)),
        "governance_included_rows": int(governance["phase3_baseline_included"].sum()) if not governance.empty else 0,
        "governance_excluded_rows": int((~governance["phase3_baseline_included"]).sum()) if not governance.empty else 0,
        "governance_unmatched_decision_rows": int(len(unmatched_decisions)),
        "preprocessed_ok": table_metric(preprocessing_summary_path, "processed_ok"),
        "preprocessed_reused_existing_valid": table_metric(preprocessing_summary_path, "reused_existing_valid"),
        "preprocessing_failed_or_skipped": table_metric(preprocessing_summary_path, "failed_or_skipped"),
        "preprocessing_failures_rows": int(len(preprocess_failures)),
        "feature_extraction_failures_rows": int(len(feature_failures)),
        "candidate_feature_count": int(len(feature_manifest)),
        "quality_accepted_feature_count": int(len(accepted_features)),
        "final_baseline_feature_count": numeric_feature_count,
        "redundancy_pruned_feature_count": int(len(pruned_features)),
        "dropped_feature_count": int(len(dropped_features)),
        "kept_with_caution_feature_count": int(len(caution_features)),
        "feature_family_ablation_rows": int(len(ablation_results)),
        "speaker_level_metric_rows": int(len(speaker_metrics)),
        "max_feature_missing_pct": missing_max,
        "max_row_missing_pct": row_missing_max,
        "raw_split_counts": split_counts(raw_features),
        "governed_split_counts": split_counts(governed),
        "pruned_split_counts": split_counts(pruned),
        "preprocess_status_counts": preprocess_log["status"].value_counts(dropna=False).to_dict()
        if not preprocess_log.empty and "status" in preprocess_log
        else {},
        "governance_action_counts": governance["phase3_governance_action"].value_counts(dropna=False).to_dict()
        if not governance.empty
        else {},
        "governance_decision_counts": governance["phase2_decision"].value_counts(dropna=False).to_dict()
        if not governance.empty
        else {},
        "feature_status_counts": feature_manifest["status"].value_counts(dropna=False).to_dict()
        if not feature_manifest.empty
        else {},
        "proxy_policy_counts": feature_manifest["phase4_proxy_policy_category"].value_counts(dropna=False).to_dict()
        if not feature_manifest.empty and "phase4_proxy_policy_category" in feature_manifest.columns
        else {},
        "phase4_retained_proxy_feature_count": int(
            (
                (feature_manifest.get("phase4_proxy_policy_category", pd.Series(dtype=str)) == "proxy_but_retained")
                & (feature_manifest.get("final_baseline_feature", pd.Series(dtype=bool)) == True)
            ).sum()
        )
        if not feature_manifest.empty and "phase4_proxy_policy_category" in feature_manifest.columns
        else 0,
        "phase4_exploratory_feature_count": int(
            (feature_manifest.get("phase4_proxy_policy_category", pd.Series(dtype=str)) == "exploratory_only").sum()
        )
        if not feature_manifest.empty and "phase4_proxy_policy_category" in feature_manifest.columns
        else 0,
        "phase4_should_be_excluded_feature_count": int(
            (feature_manifest.get("phase4_proxy_policy_category", pd.Series(dtype=str)) == "should_be_excluded").sum()
        )
        if not feature_manifest.empty and "phase4_proxy_policy_category" in feature_manifest.columns
        else 0,
    }
    return counts


def write_pipeline_spec(counts: Dict[str, object]) -> None:
    """Write the authoritative Phase 3 pipeline specification."""
    lines = [
        "# Phase 3 Pipeline Spec",
        "",
        "Scope: this document is the authoritative Phase 3 specification for the classical ML baseline only. It excludes wav2vec, deep learning, UI, deployment, and presentation work.",
        "",
        "## Canonical Inputs",
        "",
        "- Raw audio root: `data/raw_audio/`.",
        "- Canonical metadata with speaker-grouped splits: `data/metadata/metadata_fixed_splits.csv`.",
        "- Phase 2 governance decisions: `reports/dataset_audit/pre_phase3_data_decisions.csv`.",
        "- Preprocessing policy: `docs/PHASE3_PREPROCESSING_POLICY.md`.",
        "",
        "## Canonical Entry Points",
        "",
        "| Stage | Command | Primary outputs |",
        "| --- | --- | --- |",
        "| Preprocessing | `make PYTHON=.venv/bin/python phase3-preprocess` | `data/processed_audio/`, `data/metadata/preprocess_log.csv`, `reports/phase3_preprocessing_report.md` |",
        "| Feature extraction | `make PYTHON=.venv/bin/python phase3-features` | `data/features/features_enhanced.csv`, `reports/tables/phase3_feature_quality.csv`, `reports/phase3_feature_engineering_report.md` |",
        "| Governance and redundancy pruning | `make PYTHON=.venv/bin/python phase3-governance` | `data/features/features_phase3_governed.csv`, `data/features/features_phase3_governed_pruned.csv`, `reports/phase3_governed_record_decisions.csv`, `reports/phase3_redundancy_pruning_manifest.csv` |",
        "| Pruning contract alias | `make PYTHON=.venv/bin/python phase3-prune` | Confirms the governed/pruned feature outputs generated by `phase3-governance` |",
        "| Classical baseline | `make PYTHON=.venv/bin/python phase3-baseline` | `models/baseline_phase3_governed/`, `models/calibrated_phase3_governed/`, `reports/tables/baseline_metrics_enhanced.csv` |",
        "| Closeout validation | `make PYTHON=.venv/bin/python phase3-validate` | `reports/phase3_gap_assessment.md`, `reports/phase3_run_manifest.md`, `reports/phase3_closeout_validation.md`, `reports/phase3_closeout.md`, proxy/provenance/readiness notes, ablation/speaker/pruning/correlation/takeaway notes |",
        "| Full Phase 3 rerun | `make PYTHON=.venv/bin/python phase3-closeout` | All outputs above, regenerated in order |",
        "",
        "## Data Flow",
        "",
        "1. `metadata_fixed_splits.csv` provides one row per raw recording, label, speaker grouping key, and split.",
        "2. `src.preprocess` reads the metadata rows and writes standardized WAV files under `data/processed_audio/`, mirroring the raw path layout.",
        "3. `src.features` reads the processed WAV paths implied by metadata and writes one feature row per usable recording to `data/features/features_enhanced.csv`.",
        "4. `scripts/phase3_closeout.py` joins feature rows to Phase 2 decisions by repository-relative raw audio path and writes the governed table `data/features/features_phase3_governed.csv`.",
        "5. The same closeout script applies train-split-only redundancy pruning and writes `data/features/features_phase3_governed_pruned.csv` plus `reports/phase3_redundancy_pruning_manifest.csv`.",
        "6. `src.train_baseline --enhanced` trains only on the pruned governed table, so final held-out metrics are not computed on records excluded by Phase 2 final-evaluation rules.",
        "",
        "## Frozen Phase 3 Decisions",
        "",
        "- Target audio format is 16 kHz mono WAV PCM_16.",
        "- Preprocessing keeps internal pauses; only leading/trailing silence is trimmed.",
        "- Denoising is conservative: DC removal plus 50 Hz high-pass filtering; no aggressive spectral denoising.",
        "- Governance exclusions are applied at baseline-ready feature table creation, while preprocessing and raw feature extraction remain complete for traceability/QC.",
        "- Final classical baseline input is `data/features/features_phase3_governed_pruned.csv`.",
        "- Feature acceptance is governed by `docs/PHASE3_FEATURE_ACCEPTANCE_POLICY.md` and recorded in `reports/phase3_feature_manifest.csv`.",
        "- Redundancy pruning uses only training-split feature correlations and is recorded in `reports/phase3_redundancy_pruning_manifest.csv`.",
        "- Proxy/acoustic-estimate features are retained only with explicit caution and are recorded in `reports/phase3_proxy_feature_status.csv`.",
        "",
        "## Rerun Contract",
        "",
        "- `phase3-preprocess` reuses existing processed WAVs only when header validation confirms target sample rate, mono channel count, and non-empty WAV frames; otherwise it regenerates the processed file.",
        "- `phase3-features` regenerates `data/features/features_enhanced.csv` and Phase 3 feature QC tables.",
        "- `phase3-governance` regenerates the governed feature table, governed-record decision file, pruned feature table, and redundancy-pruning manifest.",
        "- `phase3-baseline` trains and evaluates only from `data/features/features_phase3_governed_pruned.csv`.",
        "- `phase3-validate` refreshes the gap assessment, manifests, proxy-status table, closeout validation, pre-Phase-4 verification note, readiness note, and closeout note.",
        "",
        "## Official Frozen Output Set",
        "",
        "- Official preprocessing policy: `docs/PHASE3_PREPROCESSING_POLICY.md`.",
        "- Official governed record decisions: `reports/phase3_governed_record_decisions.csv`.",
        "- Official governed full feature table: `data/features/features_phase3_governed.csv`.",
        "- Official final model input feature table: `data/features/features_phase3_governed_pruned.csv`.",
        "- Official feature manifest: `reports/phase3_feature_manifest.csv`.",
        "- Official proxy feature status table: `reports/phase3_proxy_feature_status.csv`.",
        "- Official redundancy pruning manifest: `reports/phase3_redundancy_pruning_manifest.csv`.",
        "- Official retained-feature correlation appendix: `reports/phase3_retained_feature_correlation_appendix.md` and `reports/phase3_retained_feature_correlation_pairs.csv`.",
        "- Official feature-family ablations: `reports/classical_baseline_upgrade/feature_family_ablation_results.csv`.",
        "- Official feature dictionary: `docs/FEATURE_DICTIONARY.md`.",
        "- Official speaker-level evaluation note: `reports/phase3_speaker_level_evaluation.md`.",
        "- Official final baseline takeaway: `reports/phase3_final_baseline_takeaway.md`.",
        "- Official run manifest: `reports/phase3_run_manifest.md` and `reports/phase3_run_manifest.json`.",
        "- Official pre-Phase-4 gap check: `reports/phase3_pre_phase4_gap_check.md`.",
        "- Official pre-Phase-4 verification note: `reports/phase3_pre_phase4_verification.md`.",
        "- Official Phase 4 readiness note: `reports/phase3_phase4_readiness.md`.",
        "- Official closeout signoff: `reports/phase3_closeout.md`.",
        "",
        "## Superseded or Intermediate Outputs",
        "",
        "- `data/features/features_enhanced.csv` is retained as the raw enhanced feature extraction output, but Phase 4 should use the pruned governed table for official classical-baseline comparisons.",
        "- `data/features/features_phase3_governed.csv` is retained as the full governed feature table before redundancy pruning; model training uses the pruned version.",
        "- `reports/phase3_governance_manifest.csv` is retained as the verbose governance audit table; the official project-management record decision file is `reports/phase3_governed_record_decisions.csv`.",
        "- Legacy Phase 1/Phase 2 reports remain traceability artifacts and are not the official Phase 3 closeout output set.",
        "",
        "## Current Closeout Counts",
        "",
        f"- Raw feature rows before governance: **{counts.get('raw_feature_rows')}**.",
        f"- Governed baseline-ready feature rows: **{counts.get('governed_feature_rows')}**.",
        f"- Quality-accepted features before redundancy pruning: **{counts.get('quality_accepted_feature_count')}**.",
        f"- Redundant features pruned: **{counts.get('redundancy_pruned_feature_count')}**.",
        f"- Final numeric baseline features: **{counts.get('final_baseline_feature_count')}**.",
        f"- Dropped candidate features: **{counts.get('dropped_feature_count')}**.",
        f"- Preprocessing failures/skips: **{counts.get('preprocessing_failed_or_skipped')}**.",
        f"- Feature extraction failures: **{counts.get('feature_extraction_failures_rows')}**.",
        "",
        "## Known Limitations",
        "",
        "- Speaker IDs are metadata/path-derived and are not biometric speaker verification.",
        "- F0, jitter/shimmer, pause, VAD, and speaking-rate features are acoustic estimates or proxies, not clinical measurements.",
        "- Uncertain metadata/audio matches can be used for training with flags where Phase 2 allows it, but are excluded from final validation/test metrics until manual resolution.",
        "- Scaling and imputation are handled inside model pipelines; feature CSVs intentionally retain missing values for the model pipeline to impute on training data.",
    ]
    PIPELINE_SPEC_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_acceptance_policy() -> None:
    """Write the formal feature acceptance and rejection policy."""
    lines = [
        "# Phase 3 Feature Acceptance Policy",
        "",
        "Scope: candidate hand-crafted acoustic features for the classical ML baseline.",
        "",
        "## Acceptance Rules",
        "",
        f"- Drop a feature when missingness is greater than `{MAX_FEATURE_MISSING_PCT:.2f}` of rows.",
        "- Drop a feature when all values are missing, infinite, non-numeric, or effectively one unique non-null value.",
        "- Drop a feature when it is numerically unstable enough to produce non-finite values after feature cleaning.",
        "- Retain a feature with caution when it has low but non-zero missingness and passes the drop threshold.",
        f"- Retain a feature with caution when its IQR outlier rate is greater than `{IQR_OUTLIER_CAUTION_PCT:.2f}` unless the values are non-finite or clearly extraction failures.",
        f"- Retain a feature with caution when the missingness gap between classes is at least `{CLASS_MISSING_GAP_CAUTION:.2f}`; if the gap is caused by extraction failure concentrated in one class, drop or investigate before final reporting.",
        "- Proxy features are allowed only when they are deterministic, documented, and named as proxies.",
        "- No global winsorization or clipping is applied in Phase 3. Outliers are documented and retained unless they violate missingness/finite-value/variance rules.",
        "- Remaining missing values are not imputed globally in the feature table. They are imputed inside sklearn model pipelines fitted on training data only.",
        "",
        "## Final Status Labels",
        "",
        "- `kept`: passes acceptance checks and is a direct acoustic descriptor or QC feature.",
        "- `kept_with_caution`: passes hard checks but needs interpretation caution because it is a proxy, has low missingness, or has a notable outlier rate.",
        "- `exploratory_only`: not used in the official baseline feature table, but retained for future analysis notes if a feature is not defensible for the frozen baseline.",
        "- `dropped`: excluded from the final baseline feature set by the hard policy.",
        "",
        "## Applied Manifest",
        "",
        "The machine-readable application of this policy is saved to `reports/phase3_feature_manifest.csv`. That file records every candidate feature, its family, status, missingness, variance, outlier rate, proxy/direct label, and decision reason.",
        "",
        "## Redundancy Pruning",
        "",
        f"- After quality acceptance, train-split-only Spearman correlations are used to identify redundant feature pairs with absolute correlation at or above `{REDUNDANCY_CORRELATION_THRESHOLD:.2f}`.",
        "- One feature from each redundant pair is removed from the official model-input table, preferring direct acoustic descriptors over proxies, lower missingness, and deterministic tie-breaks.",
        "- Held-out validation/test feature distributions and labels are not used to choose redundant features.",
        "- Redundancy decisions are saved to `reports/phase3_redundancy_pruning_manifest.csv`; the official pruned model input is `data/features/features_phase3_governed_pruned.csv`.",
        "",
        "## Jitter/Shimmer Decision",
        "",
        "The current jitter and shimmer columns are kept with caution as frame-level proxies only. They are not clinical cycle-level jitter or shimmer because this repo does not use glottal-cycle pitch-marking tooling such as Praat/parselmouth. They may be useful as classical voice-quality descriptors, but they must be reported as proxy features.",
    ]
    FEATURE_POLICY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_dictionary() -> None:
    """Write the final readable feature dictionary."""
    lines = [
        "# Phase 3 Feature Dictionary",
        "",
        "Scope: final hand-crafted features for the Phase 3 classical ML baseline. These features support a screening baseline only; they are not clinical diagnostic measurements.",
        "",
        "## Metadata Columns",
        "",
        "| Column | Meaning | Modeling use |",
        "| --- | --- | --- |",
        "| `metadata_index` | Row identifier from the cleaned metadata table. | Traceability only. |",
        "| `speaker_id` | Metadata/path-derived speaker grouping key. | Grouped splitting, leakage checks, and speaker-level evaluation only. |",
        "| `audio_file` | Source raw audio path. | Traceability and Phase 2 governance matching only. |",
        "| `processed_file` | Standardized WAV used for feature extraction. | Traceability only. |",
        "| `label`, `label_name` | Binary target and readable class label. | Target/audit only, never input features. |",
        "| `datasplit` | Speaker-grouped split assignment. | Train/validation/test filtering only. |",
        "| `dementia_type`, `gender`, `ethnicity` | Source metadata fields when available. | Not model features in the Phase 3 baseline. |",
        "| `feature_extraction_status`, `error` | Extraction QC fields. | QC filtering only. |",
        "",
        "## Preprocessing Assumptions",
        "",
        "All features are extracted from standardized audio under `data/processed_audio/`: 16 kHz, mono, WAV, PCM_16. The preprocessing order is load/resample/mono, finite-value cleanup, DC removal, conservative high-pass filtering, leading/trailing silence trim, capped RMS normalization, WAV write, and QC logging.",
        "",
        "Aggressive spectral denoising is excluded because it can distort pitch, shimmer, jitter proxies, and spectral shape in a way that is hard to defend on a small heterogeneous speech dataset.",
        "",
        "## Feature Families",
        "",
        "| Feature family | Columns | What it measures | Why it may matter | Unit/scale | Direct or proxy | High-level computation | Cautions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        "| Duration | `duration_sec` | Length of the processed clip. | Useful QC and possible confound; clip length can affect feature stability. | Seconds | Direct QC | Samples divided by sample rate. | Interpret as QC/confound context, not dementia evidence. |",
        "| MFCC summaries | `mfcc_1_*` through `mfcc_13_*` | Vocal-tract/spectral-envelope shape. | Standard classical speech descriptors for articulation and spectral consistency. | MFCC coefficient scale | Direct acoustic descriptor | `librosa.feature.mfcc`, summarized by mean/std/min/max. | Sensitive to recording conditions. |",
        "| Delta MFCC summaries | `delta_mfcc_1_*` through `delta_mfcc_13_*` | First-order MFCC change over time. | Captures acoustic dynamics and articulation change. | MFCC change/frame | Direct acoustic descriptor | Safe-width `librosa.feature.delta`, summarized. | Less stable on very short clips. |",
        "| Delta-delta MFCC summaries | `delta2_mfcc_1_*` through `delta2_mfcc_13_*` | Second-order MFCC change. | Captures acceleration/instability in acoustic trajectories. | MFCC acceleration/frame | Direct acoustic descriptor | Second-order delta, summarized. | Some outlier rates are documented in the feature manifest. |",
        "| Zero crossing rate | `zcr_mean`, `zcr_std`, `zcr_min`, `zcr_max` | Rate of sign changes in the waveform. | Reflects noisiness, frication, and voicing balance. | Fraction/frame | Direct acoustic descriptor | `librosa.feature.zero_crossing_rate`, summarized. | Noise-sensitive. |",
        "| Spectral centroid | `spectral_centroid_*` | Frequency center of mass. | Captures spectral brightness/noisiness. | Hz | Direct acoustic descriptor | `librosa.feature.spectral_centroid`, summarized by distribution stats. | Recording and microphone sensitive. |",
        "| Spectral bandwidth | `spectral_bandwidth_*` | Spread of spectral energy. | May reflect articulation, breathiness, or noise. | Hz | Direct acoustic descriptor | `librosa.feature.spectral_bandwidth`, summarized. | Not disease-specific. |",
        "| Spectral rolloff | `spectral_rolloff_*` | Frequency below which 85 percent of energy lies. | Captures high-frequency energy distribution. | Hz | Direct acoustic descriptor | `librosa.feature.spectral_rolloff(roll_percent=0.85)`, summarized. | Noise-sensitive. |",
        "| Spectral flatness | `spectral_flatness_*` | Tone-like versus noise-like spectrum. | Helps characterize noisy/breathy regions and degraded recordings. | Unitless ratio | Direct acoustic descriptor | `librosa.feature.spectral_flatness`, summarized. | Low-level values can have outliers. |",
        "| F0 statistics | `f0_mean`, `f0_median`, `f0_p10`, `f0_p25`, `f0_p75`, `f0_p90`, `f0_std`, `f0_min`, `f0_max`, `f0_range`, `f0_iqr`, `f0_semitone_std` | Fundamental-frequency estimates on voiced frames. | Prosody and pitch variability may relate to speech planning, affect, and voice quality. Percentiles reduce dependence on extreme F0 estimates. | Hz or semitones | Acoustic estimate | `librosa.pyin` on a bounded analysis window; NaNs excluded and summarized. | Estimate only; not laryngoscopic measurement. |",
        "| F0 coverage | `f0_voiced_frame_count`, `voiced_ratio`, `f0_analysis_duration_sec` | Amount of usable voiced pitch evidence. | Low voiced coverage flags less stable pitch features. | Count, ratio, seconds | QC/proxy | pYIN voiced-frame count and ratio. | Coverage depends on audio quality and voicing. |",
        "| Jitter proxies | `jitter_local_proxy`, `jitter_rap_proxy`, `pitch_period_std_proxy` | Frame-to-frame pitch-period variability on voiced frames. | May capture voice instability as a classical descriptor. | Unitless ratio or seconds | Proxy | Convert voiced F0 estimates to periods and summarize local variation. | Final decision: kept with caution as proxy only, not clinical jitter. |",
        "| Shimmer proxies | `shimmer_local_proxy`, `shimmer_db_proxy` | Frame-to-frame RMS amplitude variability on voiced frames. | May capture amplitude instability as a classical descriptor. | Unitless ratio or dB change | Proxy | Align RMS to voiced F0 frames and summarize amplitude changes. | Final decision: kept with caution as proxy only, not clinical shimmer. |",
        "| RMS energy | `rms_mean`, `rms_std`, `rms_min`, `rms_p10`, `rms_p25`, `rms_median`, `rms_p75`, `rms_p90`, `rms_max` | Short-time waveform energy distribution. | Captures loudness dynamics and supports pause/low-energy analysis. | Linear RMS amplitude | Direct acoustic descriptor | `librosa.feature.rms`, summarized by distribution stats. | Microphone distance and normalization matter. |",
        "| Energy modulation | `energy_modulation_mean_abs`, `energy_modulation_std`, `log_energy_modulation_std` | Frame-to-frame RMS change. | Captures local loudness dynamics and speech-flow variation without transcripts. | Linear or log RMS change/frame | Proxy | First differences of RMS and log-RMS frame sequences. | Sensitive to background noise and gain changes. |",
        "| Low energy ratio | `low_energy_ratio` | Fraction of frames below the file-level median RMS. | Simple low-energy distribution marker. | Ratio | Proxy | Count RMS frames below median RMS. | File-relative threshold, not absolute loudness. |",
        "| Pause/silence ratios | `silence_ratio`, `pause_ratio`, `unvoiced_ratio`, `energy_voiced_ratio` | Fraction of clip classified as low-energy or speech-like. | Pause behavior is a common fluency marker in cognitive speech analysis. | Ratio | Energy-based proxy | Frame-RMS energy intervals define speech regions; silence is the complement. | Sensitive to background noise and threshold choice. |",
        "| Pause durations/counts | `pause_count`, `avg_pause_duration_sec`, `std_pause_duration_sec`, `median_pause_duration_sec`, pause duration percentiles, `max_pause_duration_sec`, `total_silence_duration_sec`, `pause_rate_per_min` | Number and duration distribution of pauses at least 0.20 seconds. | Approximates pausing and hesitations without transcripts; percentiles are more robust than max alone. | Count, seconds, rate/min | Energy-based proxy | Low-energy gaps between speech intervals are counted as pauses when above threshold and summarized. | Not transcript-based hesitation annotation. |",
        "| Silence event summaries | `silence_event_count`, `avg_silence_duration_sec`, `std_silence_duration_sec`, `median_silence_duration_sec`, silence duration percentiles, `max_silence_duration_sec` | All detected low-energy gaps, including brief gaps. | Separates brief gaps from longer pauses. | Count, seconds | Energy-based proxy | Durations between energy-based speech intervals. | May merge/split regions under noise. |",
        "| Speech segment summaries | `speech_segment_count`, `avg_speech_segment_duration_sec`, `std_speech_segment_duration_sec`, `median_speech_segment_duration_sec`, speech-segment duration percentiles, `max_speech_segment_duration_sec`, `speech_duration_sec`, `speech_segment_rate_per_min` | Number and duration distribution of detected speech-like regions. | Fragmented or highly variable speech regions can be a fluency proxy. | Count, seconds, rate/min | Energy-based proxy | Durations of frame-RMS speech intervals. | Energy-based, not phonetic segmentation. |",
        "| Transition proxies | `pause_to_speech_transition_count`, `speech_to_pause_transition_count`, `pause_speech_transition_rate_per_min` | Energy-state transitions around pauses. | Captures coarse flow/continuity without ASR or transcripts. | Count, rate/min | Energy-based proxy | Counts boundaries around pauses lasting at least 0.20 seconds. | Threshold-sensitive and not a phonetic turn-taking measure. |",
        "| Speaking-rate proxies | `speaking_rate_proxy`, `onset_count`, `onset_rate_per_speech_sec` | Acoustic onset density. | Approximates syllable/utterance activity when transcripts are unavailable. | Count or rate/sec | Proxy | `librosa.onset` events per total duration and per speech duration. | Not words per minute. |",
        "",
        "Distribution suffixes such as `_p10`, `_p25`, `_median`, `_p75`, and `_p90` summarize frame-level or event-duration feature distributions. Suffixes `_mean`, `_std`, `_min`, and `_max` summarize central tendency, variability, and range.",
        "",
        "## Final Proxy-Feature Policy For Phase 4",
        "",
        "Option 1 is selected for Phase 4 readiness: proxy/acoustic-estimate features already present in the official Phase 3 table are retained, but they must be labeled and interpreted cautiously. The official modeling table is not changed by this policy.",
        "",
        "- `proxy_but_retained`: F0/acoustic estimates, jitter/shimmer proxies, pause/VAD/fluency proxies, low-energy ratio, and energy-modulation speech-dynamics proxies that remain in `data/features/features_phase3_governed_pruned.csv`.",
        "- `exploratory_only`: quality-accepted features that were removed from the official table by redundancy pruning; these may be discussed only as exploratory/audit variables unless the manifest is regenerated.",
        "- `should_be_excluded`: features excluded by the hard Phase 3 feature policy, currently including `voiced_unvoiced_ratio`.",
        "- `direct`: retained acoustic or QC descriptors that are not proxy-labeled by the Phase 4 policy.",
        "",
        "Machine-readable status for every candidate feature is saved in `reports/phase3_proxy_feature_status.csv`. Proxy features are acoustic approximations, not clinical biomarkers, and should not be presented as direct measures of dementia.",
        "",
        "## Excluded Feature",
        "",
        "`voiced_unvoiced_ratio` is excluded from the final baseline feature set because it exceeded the Phase 3 missingness threshold when silence duration approached zero. The more stable component ratios and duration features are retained instead.",
        "",
        "## Scaling and Missing Values",
        "",
        "Infinite values are converted to missing during feature cleaning. Remaining missing values are not globally imputed in the CSV. Median imputation and scaling are fitted inside sklearn pipelines on training data only. Logistic Regression and SVM use imputation plus `StandardScaler`; Random Forest uses imputation without scaling. The official model input is the governed table after train-split redundancy pruning.",
    ]
    FEATURE_DICTIONARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vad_note(counts: Dict[str, object]) -> None:
    """Write the VAD/pause robustness note."""
    lines = [
        "# Phase 3 VAD and Pause-Feature Robustness Note",
        "",
        "Phase 3 uses conservative energy-based segmentation, not a learned speech/non-speech model.",
        "",
        "## Current Logic",
        "",
        "- Speech-like intervals are estimated from frame RMS energy with `top_db=30`.",
        "- Feature extraction uses `frame_length=2048`, `hop_length=512`, and `min_pause_sec=0.20`.",
        "- Leading/trailing silence trimming uses the same conservative energy threshold family.",
        "- Internal pauses are preserved; they are used for pause and fluency proxy features.",
        "",
        "## Features Supported by This Logic",
        "",
        "- Silence and pause ratios: `silence_ratio`, `pause_ratio`, `unvoiced_ratio`, `energy_voiced_ratio`.",
        "- Pause counts/durations: `pause_count`, mean/std/percentile pause duration summaries, `max_pause_duration_sec`, and `pause_rate_per_min`.",
        "- Speech-region summaries: `speech_segment_count`, `speech_duration_sec`, and mean/std/percentile speech segment duration summaries.",
        "- Coarse flow proxies: `pause_to_speech_transition_count`, `speech_to_pause_transition_count`, and `pause_speech_transition_rate_per_min`.",
        "- Speaking-rate proxies: `speaking_rate_proxy`, `onset_count`, and `onset_rate_per_speech_sec`.",
        "",
        "## Sensitivity and Edge Cases",
        "",
        "- Background noise can reduce detected silence and make pause counts too low.",
        "- Very quiet speech can be mistaken for silence, especially when clips have low signal-to-noise ratio.",
        "- Music, applause, interviewer speech, or other non-target audio can be counted as speech-like energy.",
        "- Continuous speech with few low-energy gaps can make ratios stable but makes `voiced_unvoiced_ratio` unstable when silence duration approaches zero; that feature is therefore dropped.",
        "- The features are proxies for pausing and fluency. They are not transcript-based hesitation, word-rate, or phonetic segmentation measures.",
        "",
        "## Phase 3 Decision",
        "",
        "Pause/VAD/speaking-rate features are kept with caution because they are interpretable, deterministic, and useful for a classical baseline. They must be described as energy-based proxies in reports. The feature manifest marks these columns as `kept_with_caution`.",
        "",
        f"Current governed baseline rows: **{counts.get('governed_feature_rows')}**. Final baseline features: **{counts.get('final_baseline_feature_count')}**.",
    ]
    VAD_NOTE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_leakage_note() -> None:
    """Write leakage-safe scaling/evaluation confirmation."""
    calibration_notes = read_csv(REPORTS_DIR / "classical_baseline_upgrade/threshold_selection_summary.csv")
    split_verification = read_csv(REPORTS_DIR / "classical_baseline_upgrade/split_verification.csv")
    fallback_used = ""
    if not calibration_notes.empty and "validation_split_fallback_used" in calibration_notes.columns:
        fallback_used = str(bool(calibration_notes["validation_split_fallback_used"].fillna(False).any()))
    split_lines = []
    if not split_verification.empty:
        for _, row in split_verification.iterrows():
            split_lines.append(
                f"- `{row.get('split')}`: {int(row.get('n_clips', 0))} clips, {int(row.get('n_unique_speakers', 0))} speakers, both classes: {row.get('has_both_classes')}."
            )

    lines = [
        "# Phase 3 Scaling and Leakage Note",
        "",
        "This note describes the actual classical baseline path used by Phase 3.",
        "",
        "## Scaling and Imputation",
        "",
        "- Logistic Regression uses `Pipeline(SimpleImputer(strategy=\"median\"), StandardScaler(), LogisticRegression(...))`.",
        "- SVM RBF uses `Pipeline(SimpleImputer(strategy=\"median\"), StandardScaler(), SVC(...))`.",
        "- Random Forest uses `Pipeline(SimpleImputer(strategy=\"median\"), RandomForestClassifier(...))`; no scaling is applied because tree splits do not require standardized feature scales.",
        "- Because imputation and scaling are inside sklearn pipelines, they are fit during `.fit()` on the training data provided to that estimator or CV fold.",
        "- Feature extraction does not globally impute or globally scale features before splitting.",
        "- The older simple baseline path uses the same model-building function, but the official Phase 3 path is the governed enhanced baseline target.",
        "",
        "## Split and Evaluation Behavior",
        "",
        "- The enhanced baseline loads `data/features/features_phase3_governed_pruned.csv` in the Phase 3 Make target.",
        "- Training, validation, and test splits come from `metadata_fixed_splits.csv` and are checked for speaker overlap.",
        "- Hyperparameter tuning uses `StratifiedGroupKFold` on training speakers.",
        "- External calibration is fit on a validation-speaker calibration subset.",
        "- High-recall threshold selection uses a disjoint validation-speaker threshold-selection subset when feasible.",
        "- The selected threshold and calibrated model are then evaluated on the selected held-out evaluation split.",
        "",
        "## Current Split Verification",
        "",
        *(split_lines if split_lines else ["- Split verification table was not available when this note was generated."]),
        "",
        "## Calibration/Threshold Limitation",
        "",
        f"- Validation fallback used in the latest threshold table: `{fallback_used or 'unknown'}`.",
        "- If fallback is ever `True`, calibration and threshold selection reused the same validation records because there were too few validation speakers per class; this must be reported as a limitation.",
        "- In the current closeout run, the expected behavior is no fallback and disjoint validation-speaker subsets.",
        "",
        "## Conclusion",
        "",
        "The Phase 3 scaling path is leakage-safe for the classical ML baseline because no imputer or scaler is fit globally before splitting. Remaining leakage risk is governed by split quality and metadata correctness, which are checked separately through speaker-overlap and Phase 2 governance manifests.",
    ]
    LEAKAGE_NOTE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_speaker_level_evaluation_note(counts: Dict[str, object]) -> None:
    """Write a concise note for the official speaker-level evaluation artifact."""
    speaker_metrics = read_csv(REPORTS_DIR / "classical_baseline_upgrade/speaker_level_metrics.csv")
    best_line = "- Speaker-level metrics were not available when this note was generated."
    if not speaker_metrics.empty:
        mean_cal = speaker_metrics[
            (speaker_metrics.get("variant", "") == "calibrated")
            & (speaker_metrics.get("aggregation_method", "") == "mean")
        ].copy()
        if not mean_cal.empty:
            sort_cols = [col for col in ["balanced_accuracy", "specificity", "pr_auc", "roc_auc", "recall"] if col in mean_cal.columns]
            if sort_cols:
                row = mean_cal.sort_values(sort_cols, ascending=False).iloc[0]
            else:
                row = mean_cal.iloc[0]
            best_line = (
                f"- Best calibrated speaker-mean row: `{row.get('model')}` with "
                f"recall={row.get('recall', np.nan):.3f}, specificity={row.get('specificity', np.nan):.3f}, "
                f"PR-AUC={row.get('pr_auc', np.nan):.3f}, ROC-AUC={row.get('roc_auc', np.nan):.3f}."
            )

    lines = [
        "# Phase 3 Speaker-Level Evaluation Note",
        "",
        "- Speaker-level evaluation is part of the official classical baseline reporting path.",
        "- Clip probabilities are aggregated per speaker using mean, median, and max aggregation.",
        "- Speaker-level labels are used only when all governed evaluation clips for that speaker share the same class label.",
        "- Mixed-label speakers are excluded from speaker-level metrics and counted in the metrics table.",
        "- The speaker-level artifact strengthens interpretation because the project concern is closer to speaker-level dementia-risk screening than isolated clip classification.",
        "",
        best_line,
        "",
        f"- Speaker-level metric rows available: **{counts.get('speaker_level_metric_rows')}**.",
        "- Machine-readable metrics: `reports/classical_baseline_upgrade/speaker_level_metrics.csv`.",
        "- Machine-readable predictions: `reports/classical_baseline_upgrade/speaker_level_predictions.csv`.",
    ]
    SPEAKER_LEVEL_EVALUATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gap_assessment(counts: Dict[str, object]) -> None:
    """Write the short inspection/gap assessment requested for Phase 3 closeout."""
    lines = [
        "# Phase 3 Gap Assessment",
        "",
        "Scope: closeout inspection for the classical ML baseline only.",
        "",
        "## What Already Exists",
        "",
        "- Preprocessing code: `src/preprocess.py` and shared audio helpers in `src/audio_utils.py`.",
        "- Feature extraction code: `src/features.py`.",
        "- Classical baseline code: `src/train_baseline.py`.",
        "- Configuration: `src/config.py`, with target sample rate fixed at 16 kHz.",
        "- Rerun entry points: `Makefile` Phase 3 targets.",
        "- Phase 2 governance inputs: `reports/dataset_audit/pre_phase3_data_decisions.md` and `.csv`.",
        "- Phase 3 outputs: preprocessing reports, feature QC tables, governed/pruned features, baseline metrics, speaker-level metrics, ablations, run manifest, and closeout reports.",
        "",
        "## Current Canonical Inputs",
        "",
        "- Metadata: `data/metadata/metadata_fixed_splits.csv`.",
        "- Raw audio: `data/raw_audio/`.",
        "- Phase 2 decisions: `reports/dataset_audit/pre_phase3_data_decisions.csv`.",
        "- Preprocessing policy: `docs/PHASE3_PREPROCESSING_POLICY.md`.",
        "",
        "## Gaps Found During Closeout Inspection",
        "",
        "- A single end-to-end pipeline spec was needed to centralize commands, data flow, and frozen outputs.",
        "- Phase 2 decisions needed an explicit governed-record output under the requested project-management name.",
        "- Feature acceptance decisions needed a single machine-readable manifest with keep/drop/caution status.",
        "- Jitter/shimmer and pause/VAD proxy features needed explicit final status notes.",
        "- Scaling/leakage behavior needed a dedicated note tied to the actual training path.",
        "- A formal closeout/signoff note and run manifest were needed to designate one official Phase 3 output set.",
        "- Additional closeout diagnostics were useful for academic defensibility: feature-family ablations, speaker-level evaluation notes, redundancy pruning, retained-feature correlation review, and a one-page final baseline takeaway.",
        "",
        "## Official Frozen Phase 3 Outputs",
        "",
        "- Pipeline spec: `docs/PHASE3_PIPELINE_SPEC.md`.",
        "- Feature acceptance policy: `docs/PHASE3_FEATURE_ACCEPTANCE_POLICY.md`.",
        "- Feature dictionary: `docs/FEATURE_DICTIONARY.md`.",
        "- Governed record decisions: `reports/phase3_governed_record_decisions.csv`.",
        "- Full governed feature table: `data/features/features_phase3_governed.csv`.",
        "- Final pruned model-input feature table: `data/features/features_phase3_governed_pruned.csv`.",
        "- Feature manifest: `reports/phase3_feature_manifest.csv`.",
        "- Proxy feature status table: `reports/phase3_proxy_feature_status.csv`.",
        "- Redundancy pruning manifest: `reports/phase3_redundancy_pruning_manifest.csv`.",
        "- Retained-feature correlation appendix: `reports/phase3_retained_feature_correlation_appendix.md` and `reports/phase3_retained_feature_correlation_pairs.csv`.",
        "- Final baseline takeaway: `reports/phase3_final_baseline_takeaway.md`.",
        "- Pre-Phase-4 gap check: `reports/phase3_pre_phase4_gap_check.md`.",
        "- Pre-Phase-4 verification note: `reports/phase3_pre_phase4_verification.md`.",
        "- Phase 4 readiness note: `reports/phase3_phase4_readiness.md`.",
        "- Run manifest: `reports/phase3_run_manifest.md` and `reports/phase3_run_manifest.json`.",
        "- Closeout validation: `reports/phase3_closeout_validation.md`.",
        "- Closeout signoff: `reports/phase3_closeout.md`.",
        "",
        "## Current Counts",
        "",
        f"- Raw feature rows before governance: **{counts.get('raw_feature_rows')}**.",
        f"- Governed baseline-ready rows: **{counts.get('governed_feature_rows')}**.",
        f"- Governance exclusions: **{counts.get('governance_excluded_rows')}**.",
        f"- Redundancy-pruned features: **{counts.get('redundancy_pruned_feature_count')}**.",
        f"- Final baseline features: **{counts.get('final_baseline_feature_count')}**.",
        f"- Dropped features: **{counts.get('dropped_feature_count')}**.",
        "",
        "## Verdict",
        "",
        "The remaining Phase 3 gaps were closeout/governance/documentation gaps, not a need to redesign preprocessing or move beyond classical ML. The canonical Phase 3 handoff to Phase 4 is the pruned governed feature table plus the official documents and manifests listed above.",
    ]
    GAP_ASSESSMENT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_status_line(label: str, path: Path) -> str:
    """Return a markdown line describing whether an artifact exists."""
    rel = path.relative_to(ROOT).as_posix()
    status = "present" if path.exists() else "missing"
    return f"- {label}: `{rel}` ({status})."


def write_pre_phase4_gap_check(counts: Dict[str, object]) -> None:
    """Write the concise pre-Phase-4 gap check before freezing Phase 3."""
    version = git_note()
    proxy_counts = counts.get("proxy_policy_counts", {})
    lines = [
        "# Phase 3 Pre-Phase-4 Gap Check",
        "",
        "Scope: final cleanup check for provenance and proxy-feature status before moving to Phase 4.",
        "",
        "## Current Official Phase 3 Outputs",
        "",
        artifact_status_line("Official final model-input feature table", PRUNED_FEATURE_CSV),
        artifact_status_line("Governed full feature table", GOVERNED_FEATURE_CSV),
        artifact_status_line("Governed record decisions", GOVERNED_RECORD_DECISIONS_CSV),
        artifact_status_line("Feature manifest", FEATURE_MANIFEST_CSV),
        artifact_status_line("Proxy feature status table", PROXY_FEATURE_STATUS_CSV),
        artifact_status_line("Feature dictionary", FEATURE_DICTIONARY_MD),
        artifact_status_line("Preprocessing policy", DOCS_DIR / "PHASE3_PREPROCESSING_POLICY.md"),
        artifact_status_line("Pipeline spec", PIPELINE_SPEC_MD),
        artifact_status_line("Run manifest", RUN_MANIFEST_MD),
        artifact_status_line("Closeout note", CLOSEOUT_MD),
        "",
        "## Provenance Already Recorded",
        "",
        f"- Git branch: `{version.get('git_branch')}`.",
        f"- Git commit hash: `{version.get('git_commit_hash')}`.",
        f"- Working tree state: `{version.get('workspace_state')}`.",
        f"- Working tree clean: `{version.get('working_tree_clean')}`.",
        f"- Git status entries: `{version.get('git_status_entry_count')}`.",
        "- Canonical full rerun command is recorded as `make PYTHON=.venv/bin/python phase3-closeout`.",
        "- Final verification command is recorded as `make PYTHON=.venv/bin/python phase3-validate`.",
        "",
        "## Proxy Feature Status Before Phase 4",
        "",
        f"- Direct retained or direct accepted features: **{proxy_counts.get('direct', 0)}**.",
        f"- Proxy/acoustic-estimate features retained in the official table: **{counts.get('phase4_retained_proxy_feature_count')}**.",
        f"- Exploratory-only features not in the official table: **{counts.get('phase4_exploratory_feature_count')}**.",
        f"- Features that should remain excluded: **{counts.get('phase4_should_be_excluded_feature_count')}**.",
        "- Proxy families include F0/acoustic estimates, jitter/shimmer proxies, pause/VAD/fluency proxies, low-energy ratio, and energy-modulation speech-dynamics proxies.",
        "",
        "## Remaining Clarification Needed",
        "",
        "- Provenance needed branch, commit availability, dirty/clean state, exact final verification command, and artifact-reuse language in the run manifest.",
        "- Proxy features needed one machine-readable status table and an explicit Phase 4 policy decision.",
        "- No new modeling phase or official feature-table redesign was needed.",
    ]
    PRE_PHASE4_GAP_CHECK_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pre_phase4_verification(counts: Dict[str, object]) -> None:
    """Write the final controlled verification pass note."""
    lines = [
        "# Phase 3 Pre-Phase-4 Verification",
        "",
        "## What Was Executed",
        "",
        "- Final controlled pass used the closeout validation path:",
        "",
        "```bash",
        "make PYTHON=.venv/bin/python phase3-validate",
        "```",
        "",
        "- This is a verification/closeout-artifact regeneration pass, not a full clean rebuild.",
        "- It reruns `scripts/phase3_closeout.py`, regenerating governance outputs, feature manifest, proxy-status table, run manifest, validation notes, and closeout documents from the current Phase 3 artifacts.",
        "- It reuses the already validated processed audio, enhanced feature table, and baseline metric/model artifacts.",
        "- A full rebuild remains available through `make PYTHON=.venv/bin/python phase3-closeout`.",
        "",
        "## What Was Verified",
        "",
        f"- Final pruned feature table exists with **{counts.get('pruned_feature_rows')}** rows.",
        f"- Final numeric baseline features: **{counts.get('final_baseline_feature_count')}**.",
        f"- Governed rows before pruning: **{counts.get('governed_feature_rows')}**.",
        f"- Governance exclusions carried from Phase 2: **{counts.get('governance_excluded_rows')}**.",
        f"- Preprocessed usable outputs recorded in Phase 3 summary: **{counts.get('preprocessed_ok')}**.",
        f"- Preprocessing failures/skips: **{counts.get('preprocessing_failed_or_skipped')}**.",
        f"- Feature-extraction failures: **{counts.get('feature_extraction_failures_rows')}**.",
        f"- Retained proxy/acoustic-estimate features in official table: **{counts.get('phase4_retained_proxy_feature_count')}**.",
        f"- Features marked exploratory-only or excluded in proxy policy: **{counts.get('phase4_exploratory_feature_count')}** exploratory-only, **{counts.get('phase4_should_be_excluded_feature_count')}** excluded.",
        "",
        "## Artifact Presence",
        "",
        artifact_status_line("Official final model-input feature table", PRUNED_FEATURE_CSV),
        artifact_status_line("Feature manifest with proxy policy columns", FEATURE_MANIFEST_CSV),
        artifact_status_line("Proxy feature status CSV", PROXY_FEATURE_STATUS_CSV),
        artifact_status_line("Run manifest", RUN_MANIFEST_MD),
        artifact_status_line("Closeout note", CLOSEOUT_MD),
        artifact_status_line("Phase 4 readiness note", PHASE4_READINESS_MD),
        "",
        "## Result",
        "",
        "The verification pass is consistent with the official Phase 3 pipeline definition. No mismatch was found that requires reopening preprocessing, feature extraction, or baseline model training before Phase 4.",
    ]
    PRE_PHASE4_VERIFICATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase4_readiness(counts: Dict[str, object]) -> None:
    """Write the final short readiness signoff for Phase 4."""
    version = git_note()
    lines = [
        "# Phase 3 to Phase 4 Readiness",
        "",
        "## Decision",
        "",
        "Phase 3 is safe to freeze for Phase 4 for the classical ML baseline.",
        "",
        "## Provenance",
        "",
        f"- Git branch: `{version.get('git_branch')}`.",
        f"- Git commit hash: `{version.get('git_commit_hash')}`.",
        f"- Working tree state: `{version.get('workspace_state')}`.",
        f"- Git status entries: `{version.get('git_status_entry_count')}`.",
        "- The repository currently records provenance even though the working tree is not clean/committed. This is acceptable for Phase 3 handoff only if the listed artifacts are treated as the frozen output set.",
        "",
        "## Official Dataset/Table",
        "",
        "- Official final Phase 3 modeling table: `data/features/features_phase3_governed_pruned.csv`.",
        f"- Rows: **{counts.get('pruned_feature_rows')}**.",
        f"- Final numeric features: **{counts.get('final_baseline_feature_count')}**.",
        "",
        "## Proxy-Feature Policy",
        "",
        "Option 1 is selected: keep proxy/acoustic-estimate features already present in the official Phase 3 table, but require explicit cautious interpretation in Phase 4.",
        f"- Retained proxy/acoustic-estimate features: **{counts.get('phase4_retained_proxy_feature_count')}**.",
        f"- Exploratory-only features not in the official table: **{counts.get('phase4_exploratory_feature_count')}**.",
        f"- Excluded features: **{counts.get('phase4_should_be_excluded_feature_count')}**.",
        "- Proxy features are acoustic approximations, not clinical biomarkers.",
        "",
        "## Non-Blocking Limitations",
        "",
        "- The working tree has uncommitted/untracked files; the run manifest records this explicitly.",
        "- Speaker IDs are metadata/path-derived rather than biometrically verified.",
        "- Proxy features remain threshold- or estimator-dependent and should not be overinterpreted.",
        "- The dataset remains small and heterogeneous.",
        "",
        "## Phase 4 Boundary",
        "",
        "Phase 4 may build on the official final table, governed-record decisions, feature manifest, proxy-status CSV, run manifest, and closeout note. Phase 4 should not casually reinterpret excluded or exploratory-only features as official modeling inputs without regenerating the Phase 3 manifests.",
    ]
    PHASE4_READINESS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_manifest(counts: Dict[str, object]) -> None:
    """Write markdown and JSON run manifests."""
    now = datetime.now(ZoneInfo("Asia/Beirut")).isoformat(timespec="seconds")
    version = git_note()
    manifest = {
        "generated_at": now,
        "timezone": "Asia/Beirut",
        "scope": "Phase 3 classical ML closeout",
        "final_pre_phase4_pass": {
            "actual_command": "make PYTHON=.venv/bin/python phase3-validate",
            "run_mode": "verification_and_closeout_artifact_regeneration",
            "reuse_statement": (
                "This pass reruns the closeout/governance/documentation script and reuses the already validated "
                "processed audio, enhanced feature table, baseline model artifacts, and baseline metric artifacts."
            ),
            "full_clean_rebuild_command": "make PYTHON=.venv/bin/python phase3-closeout",
        },
        "commands": [
            "make PYTHON=.venv/bin/python phase3-preprocess",
            "make PYTHON=.venv/bin/python phase3-features",
            "make PYTHON=.venv/bin/python phase3-governance",
            "make PYTHON=.venv/bin/python phase3-prune",
            "make PYTHON=.venv/bin/python phase3-baseline",
            "make PYTHON=.venv/bin/python phase3-validate",
            "make PYTHON=.venv/bin/python phase3-closeout",
        ],
        "inputs": {
            "metadata_csv": "data/metadata/metadata_fixed_splits.csv",
            "raw_audio_root": "data/raw_audio",
            "phase2_decisions_csv": "reports/dataset_audit/pre_phase3_data_decisions.csv",
        },
        "outputs": {
            "processed_audio_root": "data/processed_audio",
            "preprocess_log": "data/metadata/preprocess_log.csv",
            "raw_feature_table": "data/features/features_enhanced.csv",
            "governed_feature_table": "data/features/features_phase3_governed.csv",
            "pruned_governed_feature_table": "data/features/features_phase3_governed_pruned.csv",
            "governed_baseline_models": "models/baseline_phase3_governed",
            "governed_calibrated_models": "models/calibrated_phase3_governed",
            "governed_baseline_metrics": "reports/tables/baseline_metrics_enhanced.csv",
            "governed_record_decisions": "reports/phase3_governed_record_decisions.csv",
            "governance_manifest": "reports/phase3_governance_manifest.csv",
            "feature_manifest": "reports/phase3_feature_manifest.csv",
            "proxy_feature_status": "reports/phase3_proxy_feature_status.csv",
            "redundancy_pruning_manifest": "reports/phase3_redundancy_pruning_manifest.csv",
            "retained_feature_correlation_appendix": "reports/phase3_retained_feature_correlation_appendix.md",
            "retained_feature_correlation_pairs": "reports/phase3_retained_feature_correlation_pairs.csv",
            "feature_family_ablation_results": "reports/classical_baseline_upgrade/feature_family_ablation_results.csv",
            "feature_family_ablation_report": "reports/phase3_feature_family_ablation_report.md",
            "speaker_level_metrics": "reports/classical_baseline_upgrade/speaker_level_metrics.csv",
            "speaker_level_evaluation_note": "reports/phase3_speaker_level_evaluation.md",
            "final_baseline_takeaway": "reports/phase3_final_baseline_takeaway.md",
            "pre_phase4_gap_check": "reports/phase3_pre_phase4_gap_check.md",
            "pre_phase4_verification": "reports/phase3_pre_phase4_verification.md",
            "phase4_readiness": "reports/phase3_phase4_readiness.md",
            "gap_assessment": "reports/phase3_gap_assessment.md",
            "closeout_validation": "reports/phase3_closeout_validation.md",
        },
        "config": {
            "target_sample_rate_hz": 16000,
            "target_channels": "mono",
            "target_format": "WAV PCM_16",
            "trim_top_db": 30,
            "highpass_hz": 50.0,
            "normalization": "capped RMS target -20 dBFS, max gain 12 dB, peak limit 0.99",
            "feature_missingness_drop_threshold": MAX_FEATURE_MISSING_PCT,
            "redundancy_pruning_abs_spearman_threshold": REDUNDANCY_CORRELATION_THRESHOLD,
        },
        "counts": counts,
        "version": version,
    }
    RUN_MANIFEST_JSON.write_text(json.dumps(json_safe(manifest), indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 3 Run Manifest",
        "",
        f"- Generated at: `{now}` (`Asia/Beirut`).",
        "- Scope: Phase 3 classical ML preprocessing, feature engineering, governance, baseline-ready output, and closeout validation.",
        f"- Git revision note: `{version['git_revision']}`.",
        f"- Git commit hash: `{version['git_commit_hash']}`.",
        f"- Git branch: `{version['git_branch']}`.",
        f"- Workspace state: `{version['workspace_state']}`.",
        f"- Working tree clean: `{version['working_tree_clean']}`.",
        f"- Git status entry count: `{version['git_status_entry_count']}`.",
        f"- Git status summary: `{version['git_status_summary']}`.",
        "",
        "## Final Pre-Phase-4 Pass",
        "",
        "- Actual command used for the final controlled pre-Phase-4 verification pass:",
        "",
        "```bash",
        "make PYTHON=.venv/bin/python phase3-validate",
        "```",
        "",
        "- Run mode: verification and closeout-artifact regeneration.",
        "- Regenerated in this pass: governance outputs, governed/pruned feature tables, feature manifest, proxy-status CSV, run manifest, closeout documentation, and readiness notes.",
        "- Reused in this pass: previously validated processed WAVs, enhanced feature table, baseline model artifacts, and baseline metric artifacts.",
        "- Full clean Phase 3 rebuild command remains `make PYTHON=.venv/bin/python phase3-closeout`.",
        "",
        "## Commands",
        "",
        "Canonical full closeout command:",
        "",
        "```bash",
        "make PYTHON=.venv/bin/python phase3-closeout",
        "```",
        "",
        "Expanded command sequence:",
        "",
        *[f"- `{cmd}`" for cmd in manifest["commands"][:-1]],
        "",
        "## Inputs",
        "",
        "- Metadata: `data/metadata/metadata_fixed_splits.csv`.",
        "- Raw audio: `data/raw_audio/`.",
        "- Phase 2 decisions: `reports/dataset_audit/pre_phase3_data_decisions.csv`.",
        "",
        "## Outputs",
        "",
        "- Processed audio: `data/processed_audio/`.",
        "- Preprocessing log: `data/metadata/preprocess_log.csv`.",
        "- Raw enhanced features: `data/features/features_enhanced.csv`.",
        "- Full governed baseline-ready features: `data/features/features_phase3_governed.csv`.",
        "- Official pruned governed model-input features: `data/features/features_phase3_governed_pruned.csv`.",
        "- Governed baseline models: `models/baseline_phase3_governed/`.",
        "- Governed calibrated models: `models/calibrated_phase3_governed/`.",
        "- Governed baseline metrics: `reports/tables/baseline_metrics_enhanced.csv`.",
        "- Governed record decisions: `reports/phase3_governed_record_decisions.csv`.",
        "- Verbose governance manifest: `reports/phase3_governance_manifest.csv`.",
        "- Feature manifest: `reports/phase3_feature_manifest.csv`.",
        "- Proxy feature status: `reports/phase3_proxy_feature_status.csv`.",
        "- Redundancy pruning manifest: `reports/phase3_redundancy_pruning_manifest.csv`.",
        "- Retained-feature correlation appendix: `reports/phase3_retained_feature_correlation_appendix.md`.",
        "- Retained-feature top correlation pairs: `reports/phase3_retained_feature_correlation_pairs.csv`.",
        "- Feature-family ablations: `reports/classical_baseline_upgrade/feature_family_ablation_results.csv`.",
        "- Feature-family ablation report: `reports/phase3_feature_family_ablation_report.md`.",
        "- Speaker-level metrics: `reports/classical_baseline_upgrade/speaker_level_metrics.csv`.",
        "- Speaker-level evaluation note: `reports/phase3_speaker_level_evaluation.md`.",
        "- Final baseline takeaway: `reports/phase3_final_baseline_takeaway.md`.",
        "- Pre-Phase-4 gap check: `reports/phase3_pre_phase4_gap_check.md`.",
        "- Pre-Phase-4 verification note: `reports/phase3_pre_phase4_verification.md`.",
        "- Phase 4 readiness note: `reports/phase3_phase4_readiness.md`.",
        "- Gap assessment: `reports/phase3_gap_assessment.md`.",
        "- Closeout validation: `reports/phase3_closeout_validation.md`.",
        "- Machine-readable manifest: `reports/phase3_run_manifest.json`.",
        "",
        "## Key Counts",
        "",
        f"- Metadata rows: **{counts.get('metadata_rows')}**.",
        f"- Raw feature rows before governance: **{counts.get('raw_feature_rows')}**.",
        f"- Governed baseline-ready rows: **{counts.get('governed_feature_rows')}**.",
        f"- Pruned governed feature rows: **{counts.get('pruned_feature_rows')}**.",
        f"- Governance-excluded rows: **{counts.get('governance_excluded_rows')}**.",
        f"- Preprocessed OK: **{counts.get('preprocessed_ok')}**.",
        f"- Reused existing valid processed files: **{counts.get('preprocessed_reused_existing_valid')}**.",
        f"- Preprocessing failures/skips: **{counts.get('preprocessing_failed_or_skipped')}**.",
        f"- Feature extraction failures: **{counts.get('feature_extraction_failures_rows')}**.",
        f"- Candidate features: **{counts.get('candidate_feature_count')}**.",
        f"- Quality-accepted features before redundancy pruning: **{counts.get('quality_accepted_feature_count')}**.",
        f"- Redundancy-pruned features: **{counts.get('redundancy_pruned_feature_count')}**.",
        f"- Final numeric baseline features: **{counts.get('final_baseline_feature_count')}**.",
        f"- Features kept with caution: **{counts.get('kept_with_caution_feature_count')}**.",
        f"- Dropped features: **{counts.get('dropped_feature_count')}**.",
        f"- Retained proxy/acoustic-estimate features for Phase 4: **{counts.get('phase4_retained_proxy_feature_count')}**.",
        f"- Exploratory-only features for Phase 4: **{counts.get('phase4_exploratory_feature_count')}**.",
        f"- Features excluded by Phase 4 proxy policy: **{counts.get('phase4_should_be_excluded_feature_count')}**.",
        f"- Retained feature-pair correlations reviewed: **{counts.get('retained_correlation_pair_count')}**.",
        f"- Retained pairs at the formal pruning threshold: **{counts.get('retained_correlation_pairs_at_pruning_threshold')}**.",
        "",
        "## Governance Summary",
        "",
        "Phase 2 decisions are enforced in `data/features/features_phase3_governed.csv`. Records excluded from training or final validation/test use are listed with reasons in `reports/phase3_governed_record_decisions.csv`.",
    ]
    RUN_MANIFEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_closeout_validation(counts: Dict[str, object]) -> None:
    """Write the saved end-to-end closeout validation summary."""
    action_counts = counts.get("governance_action_counts", {})
    feature_status_counts = counts.get("feature_status_counts", {})
    governed_splits = counts.get("governed_split_counts", [])
    split_lines = [
        f"- `{row['split']}`: {row['rows']} rows, {row['dementia_rows']} dementia, {row['non_dementia_rows']} non_dementia, {row['speakers']} speakers."
        for row in governed_splits
    ]
    lines = [
        "# Phase 3 Closeout Validation",
        "",
        "This report records the saved proof that Phase 3 outputs are present, governed, and baseline-ready.",
        "",
        "## End-to-End Counts",
        "",
        f"- Files/metadata rows entering preprocessing: **{counts.get('metadata_rows')}**.",
        f"- Usable processed outputs: **{counts.get('preprocessed_ok')}**.",
        f"- Preprocessing failures/skips: **{counts.get('preprocessing_failed_or_skipped')}**.",
        f"- Feature rows before governance: **{counts.get('raw_feature_rows')}**.",
        f"- Governed baseline-ready feature rows before redundancy pruning: **{counts.get('governed_feature_rows')}**.",
        f"- Pruned governed model-input rows: **{counts.get('pruned_feature_rows')}**.",
        f"- Quality-accepted candidate features before redundancy pruning: **{counts.get('quality_accepted_feature_count')}**.",
        f"- Redundant features pruned: **{counts.get('redundancy_pruned_feature_count')}**.",
        f"- Final numeric baseline features: **{counts.get('final_baseline_feature_count')}**.",
        f"- Features kept with caution: **{counts.get('kept_with_caution_feature_count')}**.",
        f"- Candidate features dropped: **{counts.get('dropped_feature_count')}**.",
        f"- Feature extraction failures: **{counts.get('feature_extraction_failures_rows')}**.",
        f"- Unmatched Phase 2 decision rows: **{counts.get('governance_unmatched_decision_rows')}**.",
        "",
        "## Governed Split Composition",
        "",
        *(split_lines if split_lines else ["- Governed split counts unavailable."]),
        "",
        "## Governance Actions",
        "",
        *[f"- `{key}`: {value}" for key, value in action_counts.items()],
        "",
        "## Feature Acceptance Status",
        "",
        *[f"- `{key}`: {value}" for key, value in feature_status_counts.items()],
        "",
        "## Proxy Policy Status",
        "",
        *[f"- `{key}`: {value}" for key, value in counts.get("proxy_policy_counts", {}).items()],
        "",
        "## Ablation and Speaker-Level Artifacts",
        "",
        f"- Feature-family ablation result rows: **{counts.get('feature_family_ablation_rows')}**.",
        f"- Speaker-level metric rows: **{counts.get('speaker_level_metric_rows')}**.",
        f"- Retained feature-pair correlations reviewed: **{counts.get('retained_correlation_pair_count')}**.",
        f"- Retained pairs at the formal pruning threshold: **{counts.get('retained_correlation_pairs_at_pruning_threshold')}**.",
        f"- Maximum retained absolute Spearman correlation: **{fmt_corr(counts.get('max_retained_abs_spearman'))}**.",
        "",
        "## Missingness and Stability",
        "",
        f"- Maximum retained feature missingness before model-pipeline imputation: **{counts.get('max_feature_missing_pct'):.4f}**.",
        f"- Maximum row-level feature missingness: **{counts.get('max_row_missing_pct'):.4f}**.",
        "- Remaining missing values are acceptable under the Phase 3 policy and are imputed inside training pipelines.",
        "- No global imputation or scaling is performed on the feature table before splitting.",
        "",
        "## Validation Result",
        "",
        "Phase 3 closeout validation passes if the governed and pruned feature tables exist, no preprocessing/feature failures are present, Phase 2 exclusions are represented in `reports/phase3_governed_record_decisions.csv`, and the final baseline uses `features_phase3_governed_pruned.csv`.",
    ]
    CLOSEOUT_VALIDATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_closeout(counts: Dict[str, object]) -> None:
    """Write the formal Phase 3 closeout/signoff note."""
    lines = [
        "# Phase 3 Closeout",
        "",
        "Phase 3 is complete and frozen for the classical ML baseline.",
        "",
        "## Official Phase 3 Outputs",
        "",
        "- Preprocessing policy: `docs/PHASE3_PREPROCESSING_POLICY.md`.",
        "- Pipeline spec: `docs/PHASE3_PIPELINE_SPEC.md`.",
        "- Feature acceptance policy: `docs/PHASE3_FEATURE_ACCEPTANCE_POLICY.md`.",
        "- Feature dictionary: `docs/FEATURE_DICTIONARY.md`.",
        "- Preprocessing log: `data/metadata/preprocess_log.csv`.",
        "- Raw enhanced feature table: `data/features/features_enhanced.csv`.",
        "- Full governed baseline-ready feature table: `data/features/features_phase3_governed.csv`.",
        "- Official pruned governed model-input feature table: `data/features/features_phase3_governed_pruned.csv`.",
        "- Governed baseline model artifacts: `models/baseline_phase3_governed/` and `models/calibrated_phase3_governed/`.",
        "- Governed baseline metrics: `reports/tables/baseline_metrics_enhanced.csv`.",
        "- Official governed-data decisions file: `reports/phase3_governed_record_decisions.csv`.",
        "- Verbose governance audit manifest: `reports/phase3_governance_manifest.csv`.",
        "- Feature manifest: `reports/phase3_feature_manifest.csv`.",
        "- Proxy feature status table: `reports/phase3_proxy_feature_status.csv`.",
        "- Redundancy pruning manifest: `reports/phase3_redundancy_pruning_manifest.csv`.",
        "- Retained-feature correlation appendix: `reports/phase3_retained_feature_correlation_appendix.md` and `reports/phase3_retained_feature_correlation_pairs.csv`.",
        "- Feature-family ablations: `reports/classical_baseline_upgrade/feature_family_ablation_results.csv`.",
        "- Feature-family ablation report: `reports/phase3_feature_family_ablation_report.md`.",
        "- Speaker-level metrics: `reports/classical_baseline_upgrade/speaker_level_metrics.csv`.",
        "- Final baseline takeaway: `reports/phase3_final_baseline_takeaway.md`.",
        "- Run manifest: `reports/phase3_run_manifest.md` and `reports/phase3_run_manifest.json`.",
        "- Pre-Phase-4 gap check: `reports/phase3_pre_phase4_gap_check.md`.",
        "- Pre-Phase-4 verification note: `reports/phase3_pre_phase4_verification.md`.",
        "- Phase 4 readiness note: `reports/phase3_phase4_readiness.md`.",
        "- Gap assessment: `reports/phase3_gap_assessment.md`.",
        "- Closeout validation: `reports/phase3_closeout_validation.md`.",
        "",
        "## Official Rerun Command",
        "",
        "```bash",
        "make PYTHON=.venv/bin/python phase3-closeout",
        "```",
        "",
        "## Governance Carried From Phase 2",
        "",
        "- Duplicate-copy exclusions are removed from the governed baseline table.",
        "- The low-volume Karl Lagerfeld file is excluded until manual audio review.",
        "- Uncertain metadata/audio matches are allowed for training only when Phase 2 allows it and are excluded from final validation/test metrics.",
        "- Nested `nodementia/nodementia` path-layout files are retained with flags when speaker grouping is available.",
        "- The canonical Angela Lansbury duplicate copy is retained with a duplicate-pair flag; the non-canonical duplicate copy is excluded.",
        "",
        "## Feature Decisions",
        "",
        "- Quality-accepted features before redundancy pruning: "
        f"**{counts.get('quality_accepted_feature_count')}**.",
        "- Redundant features pruned: "
        f"**{counts.get('redundancy_pruned_feature_count')}**.",
        "- Final numeric baseline features after pruning: "
        f"**{counts.get('final_baseline_feature_count')}**.",
        "- Dropped feature count: "
        f"**{counts.get('dropped_feature_count')}**.",
        "- Features kept with caution: "
        f"**{counts.get('kept_with_caution_feature_count')}**.",
        "- `voiced_unvoiced_ratio` is dropped because missingness exceeded the Phase 3 threshold and the ratio is unstable when silence duration approaches zero.",
        "- Jitter/shimmer-like columns are kept with caution as frame-level proxies, not clinical jitter/shimmer.",
        "- Pause/VAD/speaking-rate columns are kept with caution as energy-based proxies.",
        "- F0 columns are retained as automatic acoustic estimates, not manually verified clinical measures.",
        "- Energy-modulation columns are retained as acoustic speech-dynamics proxies, not transcript-derived tempo/articulation measures.",
        "- Final proxy policy for Phase 4: keep retained proxy/acoustic-estimate features with explicit caution; do not remove any additional columns from the official Phase 3 table.",
        "- Machine-readable proxy categories are saved in `reports/phase3_proxy_feature_status.csv`.",
        "- Feature-family ablations are saved to compare MFCC, F0/prosody, pause/fluency, voice-quality proxy, and energy-family removal.",
        "- The ablation report includes a best-performing ablation takeaway paragraph so the diagnostic table is interpretable without overclaiming causality.",
        "- The retained-feature correlation appendix shows the strongest remaining retained-pair correlations after pruning.",
        "- Speaker-level evaluation by probability aggregation is part of official Phase 3 reporting.",
        "",
        "## Superseded or Intermediate Outputs",
        "",
        "- `data/features/features_enhanced.csv` is an intermediate raw enhanced feature table. Do not use it for official Phase 4 baseline comparisons without reapplying governance.",
        "- `data/features/features_phase3_governed.csv` is the full governed feature table before redundancy pruning; official model training uses `data/features/features_phase3_governed_pruned.csv`.",
        "- `data/features/file_features.csv` is an older/simple feature output and is not the official Phase 3 closeout table.",
        "- `reports/phase3_governance_manifest.csv` is retained as a verbose audit table; the official governed-record decision file is `reports/phase3_governed_record_decisions.csv`.",
        "- `models/baseline_enhanced_v2/` and `models/calibrated_v2/` are superseded by `models/baseline_phase3_governed/` and `models/calibrated_phase3_governed/` for Phase 3 closeout.",
        "- Any feature not marked `included_in_official_final_table` in `reports/phase3_proxy_feature_status.csv` is not an official Phase 4 modeling input unless the Phase 3 manifest is regenerated.",
        "",
        "## Scaling and Leakage",
        "",
        "Scaling and imputation remain inside sklearn model pipelines. Logistic Regression and SVM use training-fitted median imputation plus `StandardScaler`; Random Forest uses training-fitted median imputation without scaling. No global scaler or imputer is fit on the full dataset before splitting.",
        "",
        "## Remaining Limitations Before Phase 4",
        "",
        "- Speaker IDs are metadata/path-derived rather than biometrically verified.",
        "- Some training records retain metadata/audio uncertainty flags; final evaluation excludes those where required, but manual review would still strengthen the dataset.",
        "- F0 estimates, voice-quality proxies, pause/VAD/fluency proxies, speaking-rate proxies, and energy-modulation proxies should not be interpreted clinically.",
        "- Small dataset size and recording heterogeneity remain the main constraints on generalization.",
        "",
        "## Phase 4 Boundary",
        "",
        "Phase 4 may build on `data/features/features_phase3_governed_pruned.csv`, `reports/phase3_governed_record_decisions.csv`, `reports/phase3_proxy_feature_status.csv`, the governed baseline artifacts, feature-family ablations, retained-feature correlation appendix, speaker-level metrics, final baseline takeaway, and the Phase 3 documentation listed above. Phase 4 should not reinterpret excluded records, exploratory-only features, or proxy features as direct clinical measures without updating the governed-record and feature-policy manifests.",
    ]
    CLOSEOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_closeout() -> None:
    """Generate all Phase 3 closeout deliverables."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    raw_features = read_csv(FEATURE_CSV)
    decisions = read_csv(DECISIONS_CSV)
    decision_summary = summarize_decisions(decisions)
    governed, governance, unmatched_decisions = apply_governance(raw_features, decision_summary)
    feature_manifest = build_feature_manifest(governed)
    pruned, pruning_manifest, feature_manifest = apply_redundancy_pruning(governed, feature_manifest)
    feature_manifest = apply_phase4_proxy_policy(feature_manifest)
    proxy_summary = write_proxy_feature_status(feature_manifest)
    counts = build_counts(raw_features, governed, pruned, governance, feature_manifest, pruning_manifest, unmatched_decisions)
    correlation_summary = write_retained_feature_correlation_appendix(pruned, feature_manifest)
    ablation_summary = summarize_feature_family_ablation()
    counts.update(proxy_summary)
    counts.update(correlation_summary)
    counts.update(ablation_summary)

    write_gap_assessment(counts)
    write_pipeline_spec(counts)
    write_feature_acceptance_policy()
    write_feature_dictionary()
    write_vad_note(counts)
    write_leakage_note()
    write_speaker_level_evaluation_note(counts)
    write_feature_family_ablation_report(ablation_summary)
    write_final_baseline_takeaway(counts, ablation_summary, correlation_summary)
    write_pre_phase4_gap_check(counts)
    write_phase4_readiness(counts)
    write_pre_phase4_verification(counts)
    write_run_manifest(counts)
    write_closeout_validation(counts)
    write_closeout(counts)

    print("Phase 3 closeout artifacts generated.")
    print(f"Governed feature rows: {counts['governed_feature_rows']}")
    print(f"Final numeric features: {counts['final_baseline_feature_count']}")
    print(f"Redundancy-pruned features: {counts['redundancy_pruned_feature_count']}")
    print(f"Governance exclusions: {counts['governance_excluded_rows']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 3 closeout and governance artifacts.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    run_closeout()
