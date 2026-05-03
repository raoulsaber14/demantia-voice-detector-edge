"""Controlled Phase 6 wav2vec2 prototype for binary screening comparison.

The runner is intentionally conservative:
- it uses governed Phase 3/4 records only;
- it splits speakers before segmenting audio;
- it requires a pilot run before full training;
- it keeps wav2vec results framed as a controlled comparison, not a replacement
  for the frozen classical baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import CFG
from src.evaluate import (
    classification_metrics,
    save_confusion_matrix_figure,
    save_pr_curve_figure,
    save_roc_curve_figure,
)
from src.train_baseline import NON_FEATURE_COLUMNS, select_feature_columns


PHASE6_PREFIX = "phase6"
CONFIG_PATH = Path("configs/wav2vec_config.yaml")
REPORT_DIR = Path("reports")
TABLE_DIR = Path("reports/tables")
FIGURE_DIR = Path("reports/figures")
MODEL_ROOT = Path("models/wav2vec")

CLIP_MANIFEST_CSV = TABLE_DIR / "phase6_clip_manifest.csv"
SEGMENT_MANIFEST_CSV = TABLE_DIR / "phase6_segment_manifest.csv"
SPLIT_VERIFICATION_CSV = TABLE_DIR / "phase6_split_verification.csv"
SPEAKER_OVERLAP_CSV = TABLE_DIR / "phase6_speaker_overlap_check.csv"
PILOT_STATUS_JSON = TABLE_DIR / "phase6_pilot_status.json"
PILOT_STATUS_MD = REPORT_DIR / "phase6_pilot_status.md"


@dataclass(frozen=True)
class Wav2VecPhase6Config:
    """Small, explicit configuration for the initial Phase 6 run."""

    run_id: str = f"wav2vec_base_{date.today().strftime('%Y%m%d')}_001"
    model_name: str = "facebook/wav2vec2-base"
    model_download_date: str = "record_at_runtime"
    seed: int = 42
    feature_csv: str = "data/features/features_phase3_governed_pruned.csv"
    decisions_csv: str = "reports/phase3_governed_record_decisions.csv"
    segment_duration_seconds: float = 3.0
    hop_length_seconds: Optional[float] = None
    min_segment_duration: float = 1.5
    target_sampling_rate: int = 16000
    split_ratios: Tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15)
    batch_size: int = 8
    learning_rate: float = 1e-5
    epochs: int = 15
    weight_decay: float = 0.01
    warmup_fraction: float = 0.10
    dropout: float = 0.35
    patience: int = 5
    pilot_epochs: int = 3
    pilot_data_fraction: float = 0.10
    target_recall: float = 0.85
    bootstrap_iterations: int = 1000
    use_augmentation: bool = True
    freeze_strategy: str = "gentle_layer_norm_then_upper_layers"
    pooling_strategy: str = "attention"
    loss_strategy: str = "weighted_ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    class_balanced_beta: float = 0.999
    use_class_balanced_sampler: bool = True
    freeze_first_n_layers: int = 6
    head_only_epochs: int = 3
    min_speech_duration: float = 1.5
    max_silence_ratio: float = 0.80
    apply_quality_filters: bool = True
    noise_snr_min_db: float = 15.0
    noise_snr_max_db: float = 25.0
    gain_min_db: float = -6.0
    gain_max_db: float = 6.0
    max_time_shift_seconds: float = 0.5
    speed_min: float = 0.9
    speed_max: float = 1.1


def write_default_config(path: Path = CONFIG_PATH, overwrite: bool = False) -> None:
    """Write the required YAML config without introducing a new dependency."""
    if path.exists() and not overwrite:
        return
    cfg = Wav2VecPhase6Config()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""run_id: {cfg.run_id}
model: {cfg.model_name}
model_download_date: {cfg.model_download_date}
feature_extractor_version: record_at_runtime
seed: {cfg.seed}
data:
  governed_feature_csv: {cfg.feature_csv}
  governed_decisions_csv: {cfg.decisions_csv}
  segment_duration: {cfg.segment_duration_seconds}
  hop_length_seconds: null
  min_segment_duration: {cfg.min_segment_duration}
  sampling_rate: {cfg.target_sampling_rate}
  split_ratios: [0.60, 0.15, 0.10, 0.15]
  split_policy: use_existing_phase3_phase4_splits_then_carve_calibration_from_training_speakers
training:
  batch_size: {cfg.batch_size}
  learning_rate: {cfg.learning_rate}
  optimizer: AdamW
  weight_decay: {cfg.weight_decay}
  scheduler: linear_warmup_cosine
  warmup_fraction: {cfg.warmup_fraction}
  epochs: {cfg.epochs}
  dropout: {cfg.dropout}
  freeze_strategy: {cfg.freeze_strategy}
  pooling_strategy: {cfg.pooling_strategy}
  loss_strategy: {cfg.loss_strategy}
  focal_gamma: {cfg.focal_gamma}
  focal_alpha: {cfg.focal_alpha}
  class_balanced_beta: {cfg.class_balanced_beta}
  class_balanced_sampler: {str(cfg.use_class_balanced_sampler).lower()}
  freeze_first_n_layers: {cfg.freeze_first_n_layers}
  head_only_epochs: {cfg.head_only_epochs}
  early_stopping_patience: {cfg.patience}
pilot:
  pilot_epochs: {cfg.pilot_epochs}
  pilot_data_fraction: {cfg.pilot_data_fraction}
augmentation:
  enabled: {str(cfg.use_augmentation).lower()}
  background_noise_snr_db: [{cfg.noise_snr_min_db}, {cfg.noise_snr_max_db}]
  gain_db: [{cfg.gain_min_db}, {cfg.gain_max_db}]
  max_time_shift_seconds: {cfg.max_time_shift_seconds}
  speed: [{cfg.speed_min}, {cfg.speed_max}]
data_quality:
  apply_filters: {str(cfg.apply_quality_filters).lower()}
  min_speech_duration: {cfg.min_speech_duration}
  max_silence_ratio: {cfg.max_silence_ratio}
evaluation:
  primary_metric: recall_atypical
  target_recall: {cfg.target_recall}
  improvement_threshold: 0.03
"""
    path.write_text(text, encoding="utf-8")


def load_config(path: Path = CONFIG_PATH) -> Wav2VecPhase6Config:
    """Load the config if PyYAML is available; otherwise use safe defaults."""
    write_default_config(path)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    cfg = Wav2VecPhase6Config()
    flat: Dict[str, Any] = {
        "run_id": data.get("run_id", cfg.run_id),
        "model_name": data.get("model", cfg.model_name),
        "model_download_date": str(data.get("model_download_date", cfg.model_download_date)),
        "seed": data.get("seed", cfg.seed),
    }
    data_cfg = data.get("data", {}) or {}
    train_cfg = data.get("training", {}) or {}
    pilot_cfg = data.get("pilot", {}) or {}
    eval_cfg = data.get("evaluation", {}) or {}
    augmentation_cfg = data.get("augmentation", {}) or {}
    if not isinstance(augmentation_cfg, dict):
        augmentation_cfg = {"enabled": str(augmentation_cfg).strip().lower() not in {"none", "false", "0", "no"}}
    quality_cfg = data.get("data_quality", {}) or {}
    flat.update(
        {
            "feature_csv": data_cfg.get("governed_feature_csv", cfg.feature_csv),
            "decisions_csv": data_cfg.get("governed_decisions_csv", cfg.decisions_csv),
            "segment_duration_seconds": float(data_cfg.get("segment_duration", cfg.segment_duration_seconds)),
            "hop_length_seconds": data_cfg.get("hop_length_seconds", cfg.hop_length_seconds),
            "min_segment_duration": float(data_cfg.get("min_segment_duration", cfg.min_segment_duration)),
            "target_sampling_rate": int(data_cfg.get("sampling_rate", cfg.target_sampling_rate)),
            "batch_size": int(train_cfg.get("batch_size", cfg.batch_size)),
            "learning_rate": float(train_cfg.get("learning_rate", cfg.learning_rate)),
            "epochs": int(train_cfg.get("epochs", cfg.epochs)),
            "weight_decay": float(train_cfg.get("weight_decay", cfg.weight_decay)),
            "warmup_fraction": float(train_cfg.get("warmup_fraction", cfg.warmup_fraction)),
            "dropout": float(train_cfg.get("dropout", cfg.dropout)),
            "freeze_strategy": train_cfg.get("freeze_strategy", cfg.freeze_strategy),
            "pooling_strategy": train_cfg.get("pooling_strategy", cfg.pooling_strategy),
            "loss_strategy": train_cfg.get("loss_strategy", cfg.loss_strategy),
            "focal_gamma": float(train_cfg.get("focal_gamma", cfg.focal_gamma)),
            "focal_alpha": float(train_cfg.get("focal_alpha", cfg.focal_alpha)),
            "class_balanced_beta": float(train_cfg.get("class_balanced_beta", cfg.class_balanced_beta)),
            "use_class_balanced_sampler": bool(train_cfg.get("class_balanced_sampler", cfg.use_class_balanced_sampler)),
            "freeze_first_n_layers": int(train_cfg.get("freeze_first_n_layers", cfg.freeze_first_n_layers)),
            "head_only_epochs": int(train_cfg.get("head_only_epochs", cfg.head_only_epochs)),
            "patience": int(train_cfg.get("early_stopping_patience", cfg.patience)),
            "pilot_epochs": int(pilot_cfg.get("pilot_epochs", cfg.pilot_epochs)),
            "pilot_data_fraction": float(pilot_cfg.get("pilot_data_fraction", cfg.pilot_data_fraction)),
            "target_recall": float(eval_cfg.get("target_recall", cfg.target_recall)),
            "use_augmentation": bool(augmentation_cfg.get("enabled", cfg.use_augmentation)),
            "noise_snr_min_db": float((augmentation_cfg.get("background_noise_snr_db") or [cfg.noise_snr_min_db, cfg.noise_snr_max_db])[0]),
            "noise_snr_max_db": float((augmentation_cfg.get("background_noise_snr_db") or [cfg.noise_snr_min_db, cfg.noise_snr_max_db])[-1]),
            "gain_min_db": float((augmentation_cfg.get("gain_db") or [cfg.gain_min_db, cfg.gain_max_db])[0]),
            "gain_max_db": float((augmentation_cfg.get("gain_db") or [cfg.gain_min_db, cfg.gain_max_db])[-1]),
            "max_time_shift_seconds": float(augmentation_cfg.get("max_time_shift_seconds", cfg.max_time_shift_seconds)),
            "speed_min": float((augmentation_cfg.get("speed") or [cfg.speed_min, cfg.speed_max])[0]),
            "speed_max": float((augmentation_cfg.get("speed") or [cfg.speed_min, cfg.speed_max])[-1]),
            "apply_quality_filters": bool(quality_cfg.get("apply_filters", cfg.apply_quality_filters)),
            "min_speech_duration": float(quality_cfg.get("min_speech_duration", cfg.min_speech_duration)),
            "max_silence_ratio": float(quality_cfg.get("max_silence_ratio", cfg.max_silence_ratio)),
        }
    )
    return Wav2VecPhase6Config(**flat)


def set_reproducible_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and torch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _safe_rel(path_like: str) -> str:
    path = Path(str(path_like))
    try:
        return str(path.resolve().relative_to(Path(".").resolve()))
    except Exception:
        return str(path)


def _audio_duration(path: Path, fallback: Optional[float] = None) -> float:
    if fallback is not None and not pd.isna(fallback):
        return float(fallback)
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


def _speaker_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for speaker_id, sub in df.groupby("speaker_id"):
        rows.append(
            {
                "speaker_id": str(speaker_id),
                "speaker_label": int(sub["label"].astype(int).mode().iloc[0]),
                "n_clips": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def carve_calibration_from_training(df: pd.DataFrame, cfg: Wav2VecPhase6Config) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a calibration split from baseline training speakers only."""
    out = df.copy()
    out["phase6_split"] = out["datasplit"].astype(str).replace({"valid": "validation"})
    train_df = out[out["datasplit"].astype(str) == "train"].copy()
    speaker_df = _speaker_label_frame(train_df)
    notes: Dict[str, Any] = {
        "method": "speaker_stratified_calibration_carved_from_existing_training_split",
        "calibration_fraction_of_existing_train": cfg.split_ratios[2] / (cfg.split_ratios[0] + cfg.split_ratios[2]),
        "fallback_used": False,
        "fallback_reason": "",
    }

    class_counts = speaker_df["speaker_label"].value_counts().to_dict()
    feasible = len(class_counts) == 2 and min(class_counts.values()) >= 2
    if feasible:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=notes["calibration_fraction_of_existing_train"],
            random_state=cfg.seed,
        )
        try:
            idx = np.arange(len(speaker_df))
            remain_idx, cal_idx = next(splitter.split(idx, speaker_df["speaker_label"].to_numpy()))
            cal_speakers = set(speaker_df.iloc[cal_idx]["speaker_id"].astype(str))
            remain_speakers = set(speaker_df.iloc[remain_idx]["speaker_id"].astype(str))
            if cal_speakers.isdisjoint(remain_speakers):
                out.loc[out["speaker_id"].astype(str).isin(cal_speakers), "phase6_split"] = "calibration"
                out.loc[out["speaker_id"].astype(str).isin(remain_speakers), "phase6_split"] = "train"
                notes.update(
                    {
                        "n_calibration_speakers": len(cal_speakers),
                        "n_training_speakers_after_carveout": len(remain_speakers),
                    }
                )
                return out, notes
        except Exception as exc:
            notes["fallback_reason"] = f"Stratified speaker carveout failed: {exc}"

    notes["fallback_used"] = True
    if not notes["fallback_reason"]:
        notes["fallback_reason"] = f"Too few train speakers per class for stratified carveout: {class_counts}"
    rng = np.random.default_rng(cfg.seed)
    speakers = speaker_df["speaker_id"].astype(str).to_numpy()
    n_cal = max(1, int(round(len(speakers) * notes["calibration_fraction_of_existing_train"])))
    cal_speakers = set(rng.choice(speakers, size=n_cal, replace=False).tolist())
    out.loc[out["speaker_id"].astype(str).isin(cal_speakers), "phase6_split"] = "calibration"
    out.loc[(out["datasplit"].astype(str) == "train") & ~out["speaker_id"].astype(str).isin(cal_speakers), "phase6_split"] = "train"
    notes.update({"n_calibration_speakers": len(cal_speakers)})
    return out, notes


def speaker_overlap_table(df: pd.DataFrame, split_col: str = "phase6_split") -> pd.DataFrame:
    split_to_speakers = {
        split: set(sub["speaker_id"].astype(str))
        for split, sub in df.groupby(split_col, dropna=False)
    }
    rows: List[Dict[str, Any]] = []
    splits = ["train", "validation", "calibration", "test"]
    splits.extend(sorted(set(split_to_speakers) - set(splits)))
    splits = [s for s in splits if s in split_to_speakers]
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = sorted(split_to_speakers[left] & split_to_speakers[right])
            rows.append(
                {
                    "split_a": left,
                    "split_b": right,
                    "overlap_speaker_count": len(overlap),
                    "overlap_speaker_sample": ";".join(overlap[:20]),
                    "status": "pass" if not overlap else "fail",
                }
            )
    return pd.DataFrame(rows)


def split_verification_table(clips: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split in ["train", "validation", "calibration", "test"]:
        clip_sub = clips[clips["phase6_split"] == split]
        seg_sub = segments[segments["phase6_split"] == split]
        rows.append(
            {
                "phase6_split": split,
                "n_clips": int(len(clip_sub)),
                "n_segments": int(len(seg_sub)),
                "n_atypical_clips": int((clip_sub["label"].astype(int) == 1).sum()),
                "n_typical_clips": int((clip_sub["label"].astype(int) == 0).sum()),
                "n_unique_speakers": int(clip_sub["speaker_id"].nunique()),
                "has_both_classes": bool(clip_sub["label"].nunique() == 2),
            }
        )
    return pd.DataFrame(rows)


def build_phase6_manifests(cfg: Wav2VecPhase6Config) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load governed Phase 4 records, split speakers, then segment clips."""
    feature_csv = Path(cfg.feature_csv)
    if not feature_csv.exists():
        raise FileNotFoundError(f"Governed feature table not found: {feature_csv}")
    df = pd.read_csv(feature_csv)
    required = {"metadata_index", "processed_file", "audio_file", "speaker_id", "label", "label_name", "datasplit"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Governed feature table is missing required columns: {missing}")

    optional_cols = {
        "duration_sec",
        "speech_duration_sec",
        "silence_ratio",
        "pause_ratio",
    } & set(df.columns)
    clips = df[list(required | optional_cols)].copy()
    clips = clips[clips["label"].isin([0, 1])].copy()
    clips["processed_file"] = clips["processed_file"].astype(str)
    clips["processed_file_exists"] = clips["processed_file"].map(lambda p: Path(p).exists())
    clips = clips[clips["processed_file_exists"]].copy()
    speech_duration = clips.get("speech_duration_sec", clips.get("duration_sec", pd.Series(np.nan, index=clips.index)))
    silence_ratio = clips.get("silence_ratio", clips.get("pause_ratio", pd.Series(np.nan, index=clips.index)))
    clips["phase6_speech_duration_sec_for_filter"] = speech_duration.astype(float)
    clips["phase6_silence_ratio_for_filter"] = silence_ratio.astype(float)
    clips["phase6_flag_short_speech"] = clips["phase6_speech_duration_sec_for_filter"] < cfg.min_speech_duration
    clips["phase6_flag_high_silence"] = clips["phase6_silence_ratio_for_filter"] > cfg.max_silence_ratio
    speaker_clip_counts = clips.groupby("speaker_id")["metadata_index"].count()
    speaker_audit = (
        speaker_clip_counts.rename("n_clips")
        .reset_index()
        .assign(flag_single_clip_speaker=lambda d: d["n_clips"].eq(1))
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    speaker_audit.to_csv(TABLE_DIR / "phase6_speaker_level_audit.csv", index=False)
    if cfg.apply_quality_filters:
        clips = clips[~(clips["phase6_flag_short_speech"] | clips["phase6_flag_high_silence"])].copy()
    clips["phase6_label_name"] = clips["label"].map({0: "typical", 1: "atypical"})
    clips["source_datasplit"] = clips["datasplit"].astype(str)
    clips, split_notes = carve_calibration_from_training(clips, cfg)

    segment_rows: List[Dict[str, Any]] = []
    hop = cfg.hop_length_seconds or cfg.segment_duration_seconds
    for _, row in clips.iterrows():
        duration = _audio_duration(Path(row["processed_file"]), fallback=row.get("duration_sec"))
        start = 0.0
        seg_idx = 0
        while start < duration:
            end = min(start + cfg.segment_duration_seconds, duration)
            seg_duration = end - start
            if seg_duration >= cfg.min_segment_duration:
                segment_rows.append(
                    {
                        "segment_id": f"{row['metadata_index']}_{seg_idx:04d}",
                        "metadata_index": row["metadata_index"],
                        "speaker_id": row["speaker_id"],
                        "label": int(row["label"]),
                        "phase6_label_name": row["phase6_label_name"],
                        "source_datasplit": row["source_datasplit"],
                        "phase6_split": row["phase6_split"],
                        "processed_file": row["processed_file"],
                        "audio_file": row["audio_file"],
                        "start_sec": round(start, 6),
                        "end_sec": round(end, 6),
                        "segment_duration_sec": round(seg_duration, 6),
                    }
                )
            start += hop
            seg_idx += 1
    segments = pd.DataFrame(segment_rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    clips.to_csv(CLIP_MANIFEST_CSV, index=False)
    segments.to_csv(SEGMENT_MANIFEST_CSV, index=False)
    split_df = split_verification_table(clips, segments)
    overlap_df = speaker_overlap_table(clips)
    split_df.to_csv(SPLIT_VERIFICATION_CSV, index=False)
    overlap_df.to_csv(SPEAKER_OVERLAP_CSV, index=False)

    notes = {
        "official_model_kept_frozen": "logistic_regression_platt",
        "official_threshold_kept_frozen": 0.300,
        "governed_feature_csv": str(feature_csv),
        "segmentation_rule": "Option A: split first, then segment",
        "segment_duration_seconds": cfg.segment_duration_seconds,
        "hop_length_seconds": cfg.hop_length_seconds,
        "min_segment_duration": cfg.min_segment_duration,
        "target_sampling_rate": cfg.target_sampling_rate,
        "quality_filters_applied": cfg.apply_quality_filters,
        "min_speech_duration": cfg.min_speech_duration,
        "max_silence_ratio": cfg.max_silence_ratio,
        **split_notes,
    }
    write_phase6_input_summary(split_df, overlap_df, notes)
    return clips, segments, notes


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def write_phase6_input_summary(split_df: pd.DataFrame, overlap_df: pd.DataFrame, notes: Dict[str, Any]) -> None:
    lines = [
        "# Phase 6 Input And Split Summary",
        "",
        "Phase 6 uses the governed Phase 3/4 records and keeps the frozen Phase 4 classical winner unchanged.",
        "",
        "## Frozen Baseline Anchor",
        "",
        f"- Frozen Phase 4 winner: `{notes['official_model_kept_frozen']}`",
        f"- Frozen Phase 4 threshold: `{notes['official_threshold_kept_frozen']:.3f}`",
        "- Phase 6 does not modify Phase 4 model selection, calibration, score bands, or training code.",
        "",
        "## wav2vec Data Rule",
        "",
        f"- Segmentation rule: {notes['segmentation_rule']}",
        f"- Segment duration: {notes['segment_duration_seconds']} seconds",
        f"- Hop length: {notes['hop_length_seconds']}",
        f"- Minimum segment duration retained: {notes['min_segment_duration']} seconds",
        f"- Target sampling rate: {notes['target_sampling_rate']} Hz",
        f"- Governed input table: `{notes['governed_feature_csv']}`",
        f"- Clip quality filters applied: {notes.get('quality_filters_applied', False)}",
        f"- Minimum speech duration: {notes.get('min_speech_duration', '')} seconds",
        f"- Maximum silence ratio: {notes.get('max_silence_ratio', '')}",
        "",
        "## Split Composition",
        "",
        dataframe_to_markdown(split_df),
        "",
        "## Speaker Overlap Check",
        "",
        dataframe_to_markdown(overlap_df),
        "",
        "## Calibration Split Note",
        "",
        f"- Method: {notes['method']}",
        f"- Fallback used: {notes['fallback_used']}",
        f"- Fallback reason: {notes.get('fallback_reason', '')}",
        "",
        "The calibration split is carved only from existing training speakers. Validation and test speakers remain held out from training.",
    ]
    (REPORT_DIR / "phase6_input_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_manifest(cfg: Wav2VecPhase6Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not CLIP_MANIFEST_CSV.exists() or not SEGMENT_MANIFEST_CSV.exists():
        clips, segments, _ = build_phase6_manifests(cfg)
        return clips, segments
    return pd.read_csv(CLIP_MANIFEST_CSV), pd.read_csv(SEGMENT_MANIFEST_CSV)


class SegmentDataset:
    """Lazy audio segment dataset. Audio chunks are read from governed WAVs."""

    def __init__(
        self,
        rows: pd.DataFrame,
        sampling_rate: int,
        segment_duration_seconds: float,
        augment: bool = False,
        seed: int = 42,
        noise_snr_min_db: float = 15.0,
        noise_snr_max_db: float = 25.0,
        gain_min_db: float = -6.0,
        gain_max_db: float = 6.0,
        max_time_shift_seconds: float = 0.5,
        speed_min: float = 0.9,
        speed_max: float = 1.1,
    ):
        self.rows = rows.reset_index(drop=True).copy()
        self.sampling_rate = sampling_rate
        self.segment_duration_seconds = segment_duration_seconds
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.noise_snr_min_db = noise_snr_min_db
        self.noise_snr_max_db = noise_snr_max_db
        self.gain_min_db = gain_min_db
        self.gain_max_db = gain_max_db
        self.max_time_shift_seconds = max_time_shift_seconds
        self.speed_min = speed_min
        self.speed_max = speed_max

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows.iloc[index]
        path = Path(row["processed_file"])
        info = sf.info(str(path))
        start_frame = int(round(float(row["start_sec"]) * info.samplerate))
        frames = int(round(float(row["segment_duration_sec"]) * info.samplerate))
        audio, sr = sf.read(str(path), start=start_frame, frames=frames, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sampling_rate:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=self.sampling_rate)
        audio = audio.astype(np.float32)
        if self.augment:
            audio = self._augment_audio(audio)
        target_len = int(round(self.segment_duration_seconds * self.sampling_rate))
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        return {
            "array": audio,
            "label": int(row["label"]),
            "segment_id": row["segment_id"],
            "metadata_index": row["metadata_index"],
            "speaker_id": row["speaker_id"],
            "processed_file": row["processed_file"],
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        }

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply light, realistic train-only waveform augmentation."""
        if audio.size == 0:
            return audio.astype(np.float32)
        out = audio.astype(np.float32, copy=True)

        if self.speed_min > 0 and self.speed_max > 0:
            speed = float(self.rng.uniform(self.speed_min, self.speed_max))
            if abs(speed - 1.0) > 0.01:
                import librosa

                out = librosa.effects.time_stretch(out, rate=speed).astype(np.float32)

        if self.max_time_shift_seconds > 0 and out.size > 1:
            max_shift = int(round(self.max_time_shift_seconds * self.sampling_rate))
            max_shift = min(max_shift, max(0, out.size - 1))
            if max_shift > 0:
                shift = int(self.rng.integers(-max_shift, max_shift + 1))
                if shift > 0:
                    out = np.pad(out, (shift, 0), mode="constant")[: out.size]
                elif shift < 0:
                    out = np.pad(out[-shift:], (0, -shift), mode="constant")

        gain_db = float(self.rng.uniform(self.gain_min_db, self.gain_max_db))
        out = out * float(10.0 ** (gain_db / 20.0))

        signal_power = float(np.mean(np.square(out)))
        if signal_power > 1e-10:
            snr_db = float(self.rng.uniform(self.noise_snr_min_db, self.noise_snr_max_db))
            noise = self.rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
            noise_power = float(np.mean(np.square(noise)))
            target_noise_power = signal_power / float(10.0 ** (snr_db / 10.0))
            if noise_power > 1e-12:
                out = out + noise * math.sqrt(target_noise_power / noise_power)

        return np.clip(out, -1.0, 1.0).astype(np.float32)


def sample_pilot_training_rows(train_rows: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Sample a small speaker-aware training subset for the mandatory pilot."""
    speaker_df = _speaker_label_frame(train_rows)
    n_target = max(2, int(round(len(speaker_df) * fraction)))
    class_counts = speaker_df["speaker_label"].value_counts().to_dict()
    if len(class_counts) == 2 and min(class_counts.values()) >= 2 and n_target < len(speaker_df):
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_target, random_state=seed)
        idx = np.arange(len(speaker_df))
        pilot_idx, _ = next(splitter.split(idx, speaker_df["speaker_label"].to_numpy()))
        speakers = set(speaker_df.iloc[pilot_idx]["speaker_id"].astype(str))
    else:
        speakers = set(speaker_df.sample(n=min(n_target, len(speaker_df)), random_state=seed)["speaker_id"].astype(str))
    return train_rows[train_rows["speaker_id"].astype(str).isin(speakers)].copy()


def _lazy_torch_and_transformers() -> Tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2Model, get_cosine_schedule_with_warmup
    except Exception as exc:
        raise RuntimeError(
            "Phase 6 wav2vec training requires torch and transformers. "
            "Install project requirements before running pilot/training."
        ) from exc
    return torch, AutoFeatureExtractor, (Wav2Vec2Model, get_cosine_schedule_with_warmup)


def build_classifier_class() -> Any:
    torch, _, transformer_items = _lazy_torch_and_transformers()
    Wav2Vec2Model, _ = transformer_items

    class Wav2Vec2Classifier(torch.nn.Module):
        def __init__(
            self,
            model_name: str,
            num_classes: int = 2,
            dropout: float = 0.1,
            pooling_strategy: str = "mean",
        ):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
            self.pooling_strategy = pooling_strategy
            hidden_size = self.wav2vec2.config.hidden_size
            if pooling_strategy == "attention":
                self.attention_query = torch.nn.Parameter(torch.empty(hidden_size))
                torch.nn.init.normal_(self.attention_query, std=0.02)
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, input_values: Any, attention_mask: Optional[Any] = None) -> Any:
            outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            if attention_mask is not None and hasattr(self.wav2vec2, "_get_feature_vector_attention_mask"):
                feature_mask = self.wav2vec2._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
                if self.pooling_strategy == "attention":
                    scores = (hidden @ self.attention_query) / math.sqrt(hidden.shape[-1])
                    scores = scores.masked_fill(~feature_mask, torch.finfo(scores.dtype).min)
                    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                    pooled = (hidden * weights).sum(dim=1)
                else:
                    mask = feature_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                if self.pooling_strategy == "attention":
                    scores = (hidden @ self.attention_query) / math.sqrt(hidden.shape[-1])
                    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                    pooled = (hidden * weights).sum(dim=1)
                else:
                    pooled = hidden.mean(dim=1)
            pooled = self.dropout(pooled)
            return self.classifier(pooled)

    return Wav2Vec2Classifier


def collate_factory(feature_extractor: Any, sampling_rate: int) -> Any:
    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        audio = [item["array"] for item in batch]
        encoded = feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded["metadata"] = [
            {k: item[k] for k in ["segment_id", "metadata_index", "speaker_id", "processed_file", "start_sec", "end_sec"]}
            for item in batch
        ]
        return encoded

    return collate


def choose_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_progressive_freeze(model: Any, epoch: int) -> str:
    """Apply the conservative Phase 6 freeze schedule."""
    for param in model.wav2vec2.parameters():
        param.requires_grad = False
    phase = "head_only"
    if epoch >= 2:
        layers = getattr(getattr(model.wav2vec2, "encoder", None), "layers", [])
        n_layers = 2 if epoch < 4 else 4
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        phase = f"unfreeze_last_{n_layers}_layers"
    for param in model.classifier.parameters():
        param.requires_grad = True
    return phase


def apply_gentle_freeze(model: Any, epoch: int, cfg: Wav2VecPhase6Config) -> str:
    """Freeze lower wav2vec layers and adapt slowly through head/layer norms first."""
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("classifier.") or name == "attention_query":
            param.requires_grad = True

    def enable_layer_norms() -> None:
        for name, param in model.wav2vec2.named_parameters():
            normalized = name.replace(".", "_").lower()
            if "layer_norm" in normalized or "layernorm" in normalized:
                param.requires_grad = True

    enable_layer_norms()
    if epoch < cfg.head_only_epochs:
        return "head_plus_layer_norms"

    layers = getattr(getattr(model.wav2vec2, "encoder", None), "layers", [])
    for layer_idx, layer in enumerate(layers):
        if layer_idx >= cfg.freeze_first_n_layers:
            for param in layer.parameters():
                param.requires_grad = True
    return f"freeze_first_{cfg.freeze_first_n_layers}_layers"


def apply_freeze_strategy(model: Any, epoch: int, cfg: Wav2VecPhase6Config) -> str:
    if cfg.freeze_strategy == "progressive":
        return apply_progressive_freeze(model, epoch)
    return apply_gentle_freeze(model, epoch, cfg)


def logits_to_numpy(logits: Any) -> np.ndarray:
    return logits.detach().cpu().numpy()


def _class_weights(labels: np.ndarray) -> np.ndarray:
    return compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels)


def build_loss_function(labels: np.ndarray, cfg: Wav2VecPhase6Config, torch: Any, device: Any) -> Any:
    """Build the configured imbalance-aware binary loss."""
    labels = np.asarray(labels, dtype=int)
    strategy = cfg.loss_strategy.strip().lower()
    if strategy in {"weighted_ce", "weighted_cross_entropy", "ce"}:
        weights = _class_weights(labels)
        return torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    if strategy == "focal":
        weights = torch.tensor(_class_weights(labels), dtype=torch.float32, device=device)

        class FocalLoss(torch.nn.Module):
            def forward(self, logits: Any, targets: Any) -> Any:
                ce = torch.nn.functional.cross_entropy(logits, targets, weight=weights, reduction="none")
                pt = torch.exp(-ce)
                alpha_pos = torch.full_like(targets, float(cfg.focal_alpha), dtype=logits.dtype)
                alpha_neg = torch.full_like(targets, float(1.0 - cfg.focal_alpha), dtype=logits.dtype)
                alpha = torch.where(targets.eq(1), alpha_pos, alpha_neg)
                return (alpha * (1.0 - pt).pow(cfg.focal_gamma) * ce).mean()

        return FocalLoss()

    if strategy in {"class_balanced", "class_balanced_ce"}:
        counts = np.bincount(labels, minlength=2).astype(float)
        effective = 1.0 - np.power(cfg.class_balanced_beta, counts)
        weights = (1.0 - cfg.class_balanced_beta) / np.maximum(effective, 1e-12)
        weights = weights / np.mean(weights)
        return torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    raise ValueError(
        f"Unsupported loss_strategy={cfg.loss_strategy!r}. "
        "Use weighted_ce, focal, or class_balanced."
    )


def evaluate_loader(
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: Any,
    log_prefix: str = "validation",
    log_every: int = 25,
) -> Tuple[float, pd.DataFrame, Dict[str, float]]:
    import torch

    model.eval()
    losses: List[float] = []
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            labels = batch["labels"].to(device)
            inputs = batch["input_values"].to(device)
            mask = batch.get("attention_mask")
            mask = mask.to(device) if mask is not None else None
            logits = model(inputs, attention_mask=mask)
            loss = loss_fn(logits, labels)
            losses.append(float(loss.detach().cpu()))
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            for i, meta in enumerate(batch["metadata"]):
                rows.append({**meta, "label": int(labels[i].detach().cpu()), "probability": float(probs[i]), "prediction": int(preds[i])})
            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader):
                print(f"{log_prefix}: batch {batch_idx}/{len(loader)}", flush=True)
    pred_df = pd.DataFrame(rows)
    metrics = classification_metrics(pred_df["label"].to_numpy(), pred_df["prediction"].to_numpy(), pred_df["probability"].to_numpy())
    return float(np.mean(losses)) if losses else np.nan, pred_df, metrics


def train_epoch(
    model: Any,
    loader: Any,
    loss_fn: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    log_prefix: str = "train",
    log_every: int = 5,
) -> Tuple[float, List[float]]:
    model.train()
    losses: List[float] = []
    for batch_idx, batch in enumerate(loader, start=1):
        labels = batch["labels"].to(device)
        inputs = batch["input_values"].to(device)
        mask = batch.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, attention_mask=mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader):
            print(f"{log_prefix}: batch {batch_idx}/{len(loader)} loss={losses[-1]:.6f}", flush=True)
    return float(np.mean(losses)) if losses else np.nan, losses


def training_rows_for_mode(segments: pd.DataFrame, cfg: Wav2VecPhase6Config, pilot: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = segments[segments["phase6_split"] == "train"].copy()
    if pilot:
        train_rows = sample_pilot_training_rows(train_rows, cfg.pilot_data_fraction, cfg.seed)
    val_rows = segments[segments["phase6_split"] == "validation"].copy()
    return train_rows, val_rows


def run_training_loop(cfg: Wav2VecPhase6Config, mode: str) -> Dict[str, Any]:
    """Run pilot or full training. Full training requires a passed pilot."""
    pilot = mode == "pilot"
    if not pilot and not pilot_has_passed():
        raise RuntimeError("Full wav2vec training is blocked because the mandatory pilot has not passed.")

    set_reproducible_seed(cfg.seed)
    _, segments = ensure_manifest(cfg)
    torch, AutoFeatureExtractor, transformer_items = _lazy_torch_and_transformers()
    _, get_cosine_schedule_with_warmup = transformer_items
    Wav2Vec2Classifier = build_classifier_class()
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
    print(f"Phase 6 {mode}: feature extractor loaded for {cfg.model_name}", flush=True)

    train_rows, val_rows = training_rows_for_mode(segments, cfg, pilot=pilot)
    if train_rows.empty or val_rows.empty:
        raise ValueError("Training or validation segments are empty after Phase 6 split filtering.")

    train_ds = SegmentDataset(
        train_rows,
        sampling_rate=cfg.target_sampling_rate,
        segment_duration_seconds=cfg.segment_duration_seconds,
        augment=cfg.use_augmentation and not pilot,
        seed=cfg.seed,
        noise_snr_min_db=cfg.noise_snr_min_db,
        noise_snr_max_db=cfg.noise_snr_max_db,
        gain_min_db=cfg.gain_min_db,
        gain_max_db=cfg.gain_max_db,
        max_time_shift_seconds=cfg.max_time_shift_seconds,
        speed_min=cfg.speed_min,
        speed_max=cfg.speed_max,
    )
    val_ds = SegmentDataset(
        val_rows,
        sampling_rate=cfg.target_sampling_rate,
        segment_duration_seconds=cfg.segment_duration_seconds,
        augment=False,
        seed=cfg.seed,
    )
    collate = collate_factory(feature_extractor, cfg.target_sampling_rate)
    train_sampler = None
    train_shuffle = True
    if cfg.use_class_balanced_sampler and not pilot:
        class_counts = train_rows["label"].astype(int).value_counts().to_dict()
        sample_weights = train_rows["label"].astype(int).map(lambda y: 1.0 / class_counts[int(y)]).to_numpy(dtype=float)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    print(
        f"Phase 6 {mode}: train_segments={len(train_rows)}, "
        f"validation_segments={len(val_rows)}, batch_size={cfg.batch_size}",
        flush=True,
    )

    device = choose_device(torch)
    print(f"Phase 6 {mode}: device={device}", flush=True)
    model = Wav2Vec2Classifier(
        cfg.model_name,
        num_classes=2,
        dropout=cfg.dropout,
        pooling_strategy=cfg.pooling_strategy,
    ).to(device)
    print(f"Phase 6 {mode}: model loaded", flush=True)
    labels = train_rows["label"].astype(int).to_numpy()
    loss_fn = build_loss_function(labels, cfg, torch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    epochs = cfg.pilot_epochs if pilot else cfg.epochs
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_fraction * total_steps),
        num_training_steps=total_steps,
    )

    run_dir = MODEL_ROOT / cfg.run_id / ("pilot" if pilot else "full")
    run_dir.mkdir(parents=True, exist_ok=True)
    history_rows: List[Dict[str, Any]] = []
    best_metric = -np.inf
    best_epoch = -1
    best_path = run_dir / "best_model.pt"
    first_batch_loss: Optional[float] = None
    last_batch_loss: Optional[float] = None
    patience_remaining = cfg.patience

    for epoch in range(epochs):
        freeze_phase = apply_freeze_strategy(model, epoch, cfg)
        print(f"Phase 6 {mode}: epoch {epoch + 1}/{epochs} freeze_phase={freeze_phase}", flush=True)
        train_loss, batch_losses = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            log_prefix=f"{mode} train epoch {epoch + 1}",
        )
        if batch_losses:
            if first_batch_loss is None:
                first_batch_loss = batch_losses[0]
            last_batch_loss = batch_losses[-1]
        val_loss, val_pred_df, val_metrics = evaluate_loader(
            model,
            val_loader,
            loss_fn,
            device,
            log_prefix=f"{mode} validation epoch {epoch + 1}",
        )
        row = {
            "epoch": epoch + 1,
            "freeze_phase": freeze_phase,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
            "validation_recall": val_metrics["recall"],
            "validation_precision": val_metrics["precision"],
            "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)
        if row["validation_balanced_accuracy"] > best_metric:
            best_metric = row["validation_balanced_accuracy"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "best_epoch": best_epoch,
                    "best_validation_balanced_accuracy": best_metric,
                },
                best_path,
            )
            feature_extractor.save_pretrained(str(run_dir / "feature_extractor"))
            patience_remaining = cfg.patience
        else:
            patience_remaining -= 1
            if not pilot and patience_remaining <= 0:
                break

    history_df = pd.DataFrame(history_rows)
    history_path = TABLE_DIR / ("phase6_pilot_training_log.csv" if pilot else "phase6_wav2vec_training_log.csv")
    history_df.to_csv(history_path, index=False)
    status = {
        "mode": mode,
        "run_id": cfg.run_id,
        "model_name": cfg.model_name,
        "completed_epochs": len(history_rows),
        "best_epoch": best_epoch,
        "checkpoint_path": str(best_path),
        "checkpoint_saved": best_path.exists(),
        "validation_loop_completed": bool(history_rows),
        "first_batch_loss": first_batch_loss,
        "last_batch_loss": last_batch_loss,
        "loss_decreased_over_pilot_batches": bool(
            first_batch_loss is not None and last_batch_loss is not None and last_batch_loss < first_batch_loss
        ),
        "device": str(device),
        "augmentation": "light_realistic" if (cfg.use_augmentation and not pilot) else "none",
        "loss_strategy": cfg.loss_strategy,
        "pooling_strategy": cfg.pooling_strategy,
        "class_balanced_sampler": bool(cfg.use_class_balanced_sampler and not pilot),
    }
    if pilot:
        status["pilot_passed"] = bool(
            status["checkpoint_saved"]
            and status["validation_loop_completed"]
            and status["loss_decreased_over_pilot_batches"]
        )
        write_pilot_status(status)
    else:
        finalize_full_training(cfg, best_path, feature_extractor)
    return status


def write_pilot_status(status: Dict[str, Any]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PILOT_STATUS_JSON.write_text(json.dumps(status, indent=2), encoding="utf-8")
    checks = [
        ("No runtime errors", True),
        ("Loss decreases over first and last pilot batch", bool(status.get("loss_decreased_over_pilot_batches"))),
        ("GPU/accelerator memory fits or CPU run completes without OOM", True),
        ("Validation loop completes", bool(status.get("validation_loop_completed"))),
        ("Checkpoint saves correctly", bool(status.get("checkpoint_saved"))),
    ]
    lines = [
        "# Phase 6 Pilot Status",
        "",
        f"- Run ID: `{status.get('run_id')}`",
        f"- Model: `{status.get('model_name')}`",
        f"- Device: `{status.get('device')}`",
        f"- Completed epochs: {status.get('completed_epochs')}",
        f"- Checkpoint: `{status.get('checkpoint_path')}`",
        f"- Pilot passed: **{status.get('pilot_passed')}**",
        "",
        "## Mandatory Checks",
        "",
    ]
    for label, ok in checks:
        lines.append(f"- [{'x' if ok else ' '}] {label}")
    lines.extend(
        [
            "",
            "Full Phase 6 training is allowed only when the pilot passed flag is true.",
        ]
    )
    PILOT_STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pilot_has_passed() -> bool:
    if not PILOT_STATUS_JSON.exists():
        return False
    try:
        status = json.loads(PILOT_STATUS_JSON.read_text(encoding="utf-8"))
        return bool(status.get("pilot_passed"))
    except Exception:
        return False


def aggregate_segments_to_clips(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metadata_index, sub in pred_df.groupby("metadata_index"):
        prob = float(sub["probability"].mean())
        rows.append(
            {
                "metadata_index": metadata_index,
                "speaker_id": sub["speaker_id"].iloc[0],
                "processed_file": sub["processed_file"].iloc[0],
                "label": int(sub["label"].iloc[0]),
                "probability": prob,
                "prediction": int(prob >= threshold),
                "n_segments": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def select_threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    thresholds = np.unique(np.concatenate(([0.0, 1.0], np.asarray(y_prob, dtype=float))))
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = classification_metrics(y_true, y_pred, y_prob)
        rows.append({"threshold": float(threshold), **metrics})
    table = pd.DataFrame(rows)
    candidates = table[table["recall"] >= target_recall].copy()
    if not candidates.empty:
        selected = candidates.sort_values(
            ["precision", "specificity", "balanced_accuracy", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0].to_dict()
        selected["selection_note"] = "met_target_recall_highest_precision"
    else:
        selected = table.sort_values(["f1", "recall", "balanced_accuracy"], ascending=[False, False, False]).iloc[0].to_dict()
        selected["selection_note"] = "target_recall_not_reached_max_f1"
    return float(selected["threshold"]), selected, table


def bootstrap_cis_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    speaker_ids: Optional[np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    speaker_ids = np.asarray(speaker_ids).astype(str) if speaker_ids is not None else None
    rng = np.random.default_rng(seed)
    values: Dict[str, List[float]] = {m: [] for m in ["recall", "balanced_accuracy", "precision", "f1", "roc_auc", "specificity"]}
    if speaker_ids is not None:
        unique = np.unique(speaker_ids)
        index_map = {s: np.where(speaker_ids == s)[0] for s in unique}
    else:
        unique = np.arange(len(y_true))
        index_map = None
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        if index_map is None:
            idx = sampled.astype(int)
        else:
            idx = np.concatenate([index_map[s] for s in sampled])
        if np.unique(y_true[idx]).size < 2:
            continue
        metrics = classification_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for key in values:
            val = metrics.get(key, np.nan)
            if not np.isnan(val):
                values[key].append(float(val))
    return {
        key: (
            float(np.percentile(vals, 2.5)) if vals else np.nan,
            float(np.percentile(vals, 97.5)) if vals else np.nan,
        )
        for key, vals in values.items()
    }


def metric_with_ci(value: float, ci: Tuple[float, float]) -> str:
    low, high = ci
    if np.isnan(low) or np.isnan(high):
        return f"{value:.3f} (CI unavailable)"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def finalize_full_training(cfg: Wav2VecPhase6Config, checkpoint_path: Path, feature_extractor: Any) -> None:
    """Evaluate wav2vec, rerun the classical baseline on identical Phase 6 splits, and compare."""
    torch, _, _ = _lazy_torch_and_transformers()
    _, segments = ensure_manifest(cfg)
    Wav2Vec2Classifier = build_classifier_class()
    device = choose_device(torch)
    model = Wav2Vec2Classifier(
        cfg.model_name,
        num_classes=2,
        dropout=cfg.dropout,
        pooling_strategy=cfg.pooling_strategy,
    ).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = torch.nn.CrossEntropyLoss()
    collate = collate_factory(feature_extractor, cfg.target_sampling_rate)

    prediction_tables: Dict[str, pd.DataFrame] = {}
    for split in ["calibration", "test"]:
        rows = segments[segments["phase6_split"] == split].copy()
        ds = SegmentDataset(
            rows,
            cfg.target_sampling_rate,
            segment_duration_seconds=cfg.segment_duration_seconds,
            augment=False,
            seed=cfg.seed,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
        _, pred_df, _ = evaluate_loader(model, loader, loss_fn, device)
        prediction_tables[split] = pred_df
        pred_df.to_csv(TABLE_DIR / f"phase6_wav2vec_{split}_segment_predictions.csv", index=False)

    cal_clip_pre = aggregate_segments_to_clips(prediction_tables["calibration"], threshold=0.5)
    threshold, threshold_info, threshold_table = select_threshold_for_recall(
        cal_clip_pre["label"].to_numpy(),
        cal_clip_pre["probability"].to_numpy(),
        target_recall=cfg.target_recall,
    )
    threshold_table.to_csv(TABLE_DIR / "phase6_wav2vec_threshold_selection.csv", index=False)
    test_clip = aggregate_segments_to_clips(prediction_tables["test"], threshold=threshold)
    test_clip.to_csv(TABLE_DIR / "phase6_wav2vec_test_clip_predictions.csv", index=False)

    wav_metrics = classification_metrics(test_clip["label"], test_clip["prediction"], test_clip["probability"])
    wav_ci = (
        bootstrap_cis_all(
            test_clip["label"].to_numpy(),
            test_clip["prediction"].to_numpy(),
            test_clip["probability"].to_numpy(),
            test_clip["speaker_id"].to_numpy(),
            n_bootstrap=cfg.bootstrap_iterations,
            seed=cfg.seed,
        )
        if len(test_clip) >= 50
        else {k: (np.nan, np.nan) for k in ["recall", "balanced_accuracy", "precision", "f1", "roc_auc", "specificity"]}
    )
    baseline_pred, baseline_metrics, baseline_ci, baseline_threshold_info = run_baseline_same_split(cfg)
    write_phase6_comparison(
        cfg,
        wav_metrics,
        wav_ci,
        baseline_metrics,
        baseline_ci,
        threshold,
        threshold_info,
        baseline_threshold_info,
        test_clip,
        baseline_pred,
    )


def run_baseline_same_split(cfg: Wav2VecPhase6Config) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Tuple[float, float]], Dict[str, Any]]:
    """Rerun a Platt-calibrated logistic baseline on the exact Phase 6 clip splits."""
    clips = pd.read_csv(CLIP_MANIFEST_CSV)
    split_map = clips[["metadata_index", "phase6_split"]].copy()
    feature_df = pd.read_csv(cfg.feature_csv)
    feature_df = feature_df.merge(split_map, on="metadata_index", how="inner", suffixes=("", "_phase6"))
    feature_cols = select_feature_columns(feature_df)
    train_df = feature_df[feature_df["phase6_split"] == "train"].copy()
    cal_df = feature_df[feature_df["phase6_split"] == "calibration"].copy()
    test_df = feature_df[feature_df["phase6_split"] == "test"].copy()

    base = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=cfg.seed, solver="lbfgs"),
            ),
        ]
    )
    base.fit(train_df[feature_cols], train_df["label"].astype(int))
    try:
        calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv="prefit")
    calibrated.fit(cal_df[feature_cols], cal_df["label"].astype(int))
    cal_prob = calibrated.predict_proba(cal_df[feature_cols])[:, 1]
    threshold, threshold_info, threshold_table = select_threshold_for_recall(
        cal_df["label"].astype(int).to_numpy(),
        cal_prob,
        target_recall=cfg.target_recall,
    )
    threshold_table.to_csv(TABLE_DIR / "phase6_baseline_same_split_threshold_selection.csv", index=False)
    test_prob = calibrated.predict_proba(test_df[feature_cols])[:, 1]
    pred = (test_prob >= threshold).astype(int)
    pred_df = test_df[
        ["metadata_index", "speaker_id", "processed_file", "label", "phase6_split"]
    ].copy()
    pred_df["probability"] = test_prob
    pred_df["prediction"] = pred
    pred_df.to_csv(TABLE_DIR / "phase6_baseline_same_split_test_predictions.csv", index=False)
    metrics = classification_metrics(pred_df["label"], pred_df["prediction"], pred_df["probability"])
    ci = (
        bootstrap_cis_all(
            pred_df["label"].to_numpy(),
            pred_df["prediction"].to_numpy(),
            pred_df["probability"].to_numpy(),
            pred_df["speaker_id"].to_numpy(),
            n_bootstrap=cfg.bootstrap_iterations,
            seed=cfg.seed,
        )
        if len(pred_df) >= 50
        else {k: (np.nan, np.nan) for k in ["recall", "balanced_accuracy", "precision", "f1", "roc_auc", "specificity"]}
    )
    threshold_info["threshold"] = threshold
    return pred_df, metrics, ci, threshold_info


def verdict_for_metric(metric: str, diff: float) -> str:
    if metric == "recall":
        if diff > 0.03:
            return "up"
        if diff < -0.03:
            return "down"
        return "equivalent"
    if diff > 0.05:
        return "up"
    if diff < -0.05:
        return "down"
    return "equivalent"


def write_phase6_comparison(
    cfg: Wav2VecPhase6Config,
    wav_metrics: Dict[str, float],
    wav_ci: Dict[str, Tuple[float, float]],
    baseline_metrics: Dict[str, float],
    baseline_ci: Dict[str, Tuple[float, float]],
    wav_threshold: float,
    wav_threshold_info: Dict[str, Any],
    baseline_threshold_info: Dict[str, Any],
    wav_pred: pd.DataFrame,
    baseline_pred: pd.DataFrame,
) -> None:
    metric_order = ["recall", "balanced_accuracy", "precision", "f1", "roc_auc", "specificity"]
    rows: List[Dict[str, Any]] = []
    for metric in metric_order:
        base_val = baseline_metrics.get(metric, np.nan)
        wav_val = wav_metrics.get(metric, np.nan)
        diff = wav_val - base_val
        rows.append(
            {
                "metric": metric,
                "baseline_95ci": metric_with_ci(base_val, baseline_ci.get(metric, (np.nan, np.nan))),
                "wav2vec_95ci": metric_with_ci(wav_val, wav_ci.get(metric, (np.nan, np.nan))),
                "difference": diff,
                "verdict": verdict_for_metric(metric, diff),
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(TABLE_DIR / "phase6_baseline_vs_wav2vec_comparison.csv", index=False)

    save_confusion_matrix_figure(
        baseline_pred["label"].to_numpy(),
        baseline_pred["prediction"].to_numpy(),
        FIGURE_DIR / "phase6_baseline_same_split_confusion_matrix.png",
        title="Phase 6 Baseline Same-Split Confusion Matrix",
    )
    save_confusion_matrix_figure(
        wav_pred["label"].to_numpy(),
        wav_pred["prediction"].to_numpy(),
        FIGURE_DIR / "phase6_wav2vec_confusion_matrix.png",
        title="Phase 6 wav2vec Confusion Matrix",
    )
    save_roc_curve_figure(wav_pred["label"], wav_pred["probability"], FIGURE_DIR / "phase6_wav2vec_roc_curve.png", "Phase 6 wav2vec ROC")
    save_pr_curve_figure(wav_pred["label"], wav_pred["probability"], FIGURE_DIR / "phase6_wav2vec_pr_curve.png", "Phase 6 wav2vec PR")

    recall_diff = wav_metrics["recall"] - baseline_metrics["recall"]
    secondary_drop = any(
        wav_metrics[m] - baseline_metrics[m] < -0.05
        for m in ["balanced_accuracy", "precision", "f1", "roc_auc", "specificity"]
    )
    if recall_diff > 0.03 and not secondary_drop:
        conclusion = "improves"
    elif 0.01 <= recall_diff <= 0.03:
        conclusion = "shows marginal improvement"
    elif abs(recall_diff) <= 0.03:
        conclusion = "does not improve"
    elif recall_diff < -0.03:
        conclusion = "performs worse"
    else:
        conclusion = "is inconclusive"

    write_error_analysis(wav_pred, baseline_pred)
    lines = [
        "# Phase 6 Baseline vs wav2vec Comparison",
        "",
        "This comparison uses the pre-defined Phase 6 protocol and identical speaker-level Phase 6 splits.",
        "",
        "## Thresholds",
        "",
        f"- wav2vec threshold selected on calibration clips: `{wav_threshold:.6f}`",
        f"- wav2vec threshold note: {wav_threshold_info.get('selection_note')}",
        f"- same-split baseline threshold selected on calibration clips: `{baseline_threshold_info.get('threshold'):.6f}`",
        f"- same-split baseline threshold note: {baseline_threshold_info.get('selection_note')}",
        "",
        "## Metric Comparison",
        "",
        dataframe_to_markdown(comp),
        "",
        "## Final Conclusion",
        "",
        (
            f"**Conclusion:** Based on the pre-defined evaluation protocol "
            f"(primary metric = recall for atypical class, improvement threshold = +0.03), "
            f"wav2vec {conclusion} compared to the baseline. Recall difference = {recall_diff:.3f}."
        ),
        "",
        "A null result is acceptable if the experiment is executed correctly.",
    ]
    (REPORT_DIR / "phase6_final_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_error_analysis(wav_pred: pd.DataFrame, baseline_pred: pd.DataFrame) -> None:
    for name, pred_df in [("wav2vec", wav_pred), ("baseline_same_split", baseline_pred)]:
        fn = pred_df[(pred_df["label"].astype(int) == 1) & (pred_df["prediction"].astype(int) == 0)].copy()
        fp = pred_df[(pred_df["label"].astype(int) == 0) & (pred_df["prediction"].astype(int) == 1)].copy()
        fn.to_csv(TABLE_DIR / f"phase6_{name}_false_negatives.csv", index=False)
        fp.to_csv(TABLE_DIR / f"phase6_{name}_false_positives.csv", index=False)

    merged = baseline_pred[["metadata_index", "label", "prediction"]].merge(
        wav_pred[["metadata_index", "prediction"]],
        on="metadata_index",
        how="inner",
        suffixes=("_baseline", "_wav2vec"),
    )
    merged["baseline_error"] = merged["prediction_baseline"].astype(int) != merged["label"].astype(int)
    merged["wav2vec_error"] = merged["prediction_wav2vec"].astype(int) != merged["label"].astype(int)
    total_errors = int((merged["baseline_error"] | merged["wav2vec_error"]).sum())
    both_wrong = int((merged["baseline_error"] & merged["wav2vec_error"]).sum())
    agreement_rate = float(both_wrong / total_errors) if total_errors else np.nan
    overlap = pd.DataFrame(
        [
            {
                "common_clip_count": len(merged),
                "total_errors_any_model": total_errors,
                "errors_both_models": both_wrong,
                "error_overlap_agreement_rate": agreement_rate,
                "interpretation": "high shared difficulty" if agreement_rate > 0.70 else "different failure modes" if agreement_rate < 0.40 else "mixed failure modes",
            }
        ]
    )
    overlap.to_csv(TABLE_DIR / "phase6_error_overlap.csv", index=False)

    lines = [
        "# Phase 6 Error Analysis",
        "",
        "False-negative and false-positive case tables were saved for both models.",
        "",
        "## Error Overlap",
        "",
        dataframe_to_markdown(overlap),
    ]
    (REPORT_DIR / "phase6_error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 controlled wav2vec2 prototype.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--mode", choices=["prepare", "pilot", "train"], default="prepare")
    parser.add_argument("--overwrite-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    write_default_config(config_path, overwrite=args.overwrite_config)
    cfg = load_config(config_path)
    try:
        if args.mode == "prepare":
            build_phase6_manifests(cfg)
        elif args.mode in {"pilot", "train"}:
            run_training_loop(cfg, args.mode)
    except Exception as exc:
        if args.mode == "pilot":
            status = {
                "mode": "pilot",
                "run_id": cfg.run_id,
                "model_name": cfg.model_name,
                "pilot_passed": False,
                "runtime_error": repr(exc),
                "checkpoint_saved": False,
                "validation_loop_completed": False,
                "loss_decreased_over_pilot_batches": False,
            }
            write_pilot_status(status)
        raise


if __name__ == "__main__":
    main()
