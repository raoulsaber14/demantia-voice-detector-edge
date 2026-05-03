"""Hand-crafted acoustic feature extraction for classical ML baselines."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "dementia_voice_numba_cache"))

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio_utils import energy_intervals, load_audio_safe
from src.config import CFG


PAUSE_FEATURE_COLUMNS = [
    "silence_ratio",
    "energy_voiced_ratio",
    "unvoiced_ratio",
    "voiced_unvoiced_ratio",
    "silence_event_count",
    "avg_silence_duration_sec",
    "std_silence_duration_sec",
    "median_silence_duration_sec",
    "p10_silence_duration_sec",
    "p25_silence_duration_sec",
    "p75_silence_duration_sec",
    "p90_silence_duration_sec",
    "max_silence_duration_sec",
    "pause_count",
    "avg_pause_duration_sec",
    "std_pause_duration_sec",
    "median_pause_duration_sec",
    "p10_pause_duration_sec",
    "p25_pause_duration_sec",
    "p75_pause_duration_sec",
    "p90_pause_duration_sec",
    "max_pause_duration_sec",
    "total_silence_duration_sec",
    "speech_segment_count",
    "avg_speech_segment_duration_sec",
    "std_speech_segment_duration_sec",
    "median_speech_segment_duration_sec",
    "p10_speech_segment_duration_sec",
    "p25_speech_segment_duration_sec",
    "p75_speech_segment_duration_sec",
    "p90_speech_segment_duration_sec",
    "max_speech_segment_duration_sec",
    "speech_duration_sec",
    "pause_rate_per_min",
    "speech_segment_rate_per_min",
    "pause_to_speech_transition_count",
    "speech_to_pause_transition_count",
    "pause_speech_transition_rate_per_min",
]

PHASE3_FEATURE_REPORT_DIR = Path("reports")
PHASE3_TABLE_DIR = PHASE3_FEATURE_REPORT_DIR / "tables"

FEATURE_METADATA_COLUMNS = {
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
    "feature_extraction_status",
    "error",
}

MAX_FEATURE_MISSING_PCT = 0.25


def _stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    """Return mean/std/min/max stats for a vector."""
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def _distribution_stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    """Return robust distribution stats for non-empty frame-level features."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_max": float(np.max(values)),
    }


def _pitch_signal_window(y: np.ndarray, sr: int, max_duration_sec: float) -> np.ndarray:
    """Return a bounded signal window for expensive pitch estimation."""
    if max_duration_sec <= 0:
        return y
    max_samples = int(sr * max_duration_sec)
    if len(y) <= max_samples:
        return y
    return y[:max_samples]


def _duration_stats(prefix: str, durations: List[float]) -> Dict[str, float]:
    """Return stable duration stats, using 0 when no events are present."""
    if not durations:
        return {
            f"avg_{prefix}_duration_sec": 0.0,
            f"std_{prefix}_duration_sec": 0.0,
            f"median_{prefix}_duration_sec": 0.0,
            f"p10_{prefix}_duration_sec": 0.0,
            f"p25_{prefix}_duration_sec": 0.0,
            f"p75_{prefix}_duration_sec": 0.0,
            f"p90_{prefix}_duration_sec": 0.0,
            f"max_{prefix}_duration_sec": 0.0,
        }
    arr = np.asarray(durations, dtype=float)
    return {
        f"avg_{prefix}_duration_sec": float(np.mean(arr)),
        f"std_{prefix}_duration_sec": float(np.std(arr)),
        f"median_{prefix}_duration_sec": float(np.median(arr)),
        f"p10_{prefix}_duration_sec": float(np.percentile(arr, 10)),
        f"p25_{prefix}_duration_sec": float(np.percentile(arr, 25)),
        f"p75_{prefix}_duration_sec": float(np.percentile(arr, 75)),
        f"p90_{prefix}_duration_sec": float(np.percentile(arr, 90)),
        f"max_{prefix}_duration_sec": float(np.max(arr)),
    }


def _safe_delta(matrix: np.ndarray, order: int = 1) -> np.ndarray:
    """Compute deltas only when the time axis is long enough."""
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        return np.zeros_like(matrix)
    width = min(9, matrix.shape[1] if matrix.shape[1] % 2 == 1 else matrix.shape[1] - 1)
    width = max(3, width)
    return librosa.feature.delta(matrix, order=order, width=width)


def _f0_proxy_features(f0_clean: np.ndarray, voiced_flag: np.ndarray | None) -> Dict[str, float]:
    """Compute pitch/prosody statistics from voiced F0 frames."""
    f0_clean = np.asarray(f0_clean, dtype=float)
    f0_clean = f0_clean[np.isfinite(f0_clean) & (f0_clean > 0)]
    out = _stats("f0", f0_clean)
    out["f0_median"] = float(np.median(f0_clean)) if f0_clean.size else np.nan
    out["f0_p10"] = float(np.percentile(f0_clean, 10)) if f0_clean.size else np.nan
    out["f0_p25"] = float(np.percentile(f0_clean, 25)) if f0_clean.size else np.nan
    out["f0_p75"] = float(np.percentile(f0_clean, 75)) if f0_clean.size else np.nan
    out["f0_p90"] = float(np.percentile(f0_clean, 90)) if f0_clean.size else np.nan
    out["f0_range"] = float(np.max(f0_clean) - np.min(f0_clean)) if f0_clean.size else np.nan
    out["f0_iqr"] = float(out["f0_p75"] - out["f0_p25"]) if f0_clean.size else np.nan
    if f0_clean.size > 1:
        semitone = 12.0 * np.log2(f0_clean / np.nanmedian(f0_clean))
        out["f0_semitone_std"] = float(np.nanstd(semitone))
    else:
        out["f0_semitone_std"] = np.nan
    out["f0_voiced_frame_count"] = int(f0_clean.size)
    if voiced_flag is None:
        out["voiced_ratio"] = np.nan
    else:
        out["voiced_ratio"] = float(np.mean(np.asarray(voiced_flag).astype(float)))
    return out


def _energy_modulation_features(rms: np.ndarray) -> Dict[str, float]:
    """Summarize frame-to-frame energy modulation without transcript assumptions."""
    rms_arr = np.asarray(rms, dtype=float)
    rms_arr = rms_arr[np.isfinite(rms_arr)]
    out = {
        "energy_modulation_mean_abs": np.nan,
        "energy_modulation_std": np.nan,
        "log_energy_modulation_std": np.nan,
    }
    if rms_arr.size < 2:
        return out
    delta = np.diff(rms_arr)
    out["energy_modulation_mean_abs"] = float(np.mean(np.abs(delta)))
    out["energy_modulation_std"] = float(np.std(delta))
    log_rms = np.log(np.maximum(rms_arr, 1e-9))
    out["log_energy_modulation_std"] = float(np.std(np.diff(log_rms)))
    return out


def _jitter_shimmer_proxy_features(
    f0: np.ndarray | None,
    rms: np.ndarray,
    voiced_flag: np.ndarray | None,
) -> Dict[str, float]:
    """Frame-level jitter/shimmer proxies.

    These are not cycle-accurate clinical jitter/shimmer. They summarize frame-to-frame
    pitch-period and RMS variability on voiced frames, which is defensible with librosa
    and fixed frame features.
    """
    out = {
        "jitter_local_proxy": np.nan,
        "jitter_rap_proxy": np.nan,
        "pitch_period_std_proxy": np.nan,
        "shimmer_local_proxy": np.nan,
        "shimmer_db_proxy": np.nan,
    }
    if f0 is None or voiced_flag is None:
        return out
    f0_arr = np.asarray(f0, dtype=float)
    voiced = np.asarray(voiced_flag).astype(bool)
    voiced_idx = np.where(voiced & np.isfinite(f0_arr) & (f0_arr > 0))[0]
    if voiced_idx.size >= 3:
        f0_voiced = f0_arr[voiced_idx]
        periods = 1.0 / f0_voiced
        denom = max(float(np.mean(periods)), 1e-9)
        out["jitter_local_proxy"] = float(np.mean(np.abs(np.diff(periods))) / denom)
        rap = np.abs(periods[1:-1] - (periods[:-2] + periods[1:-1] + periods[2:]) / 3.0)
        out["jitter_rap_proxy"] = float(np.mean(rap) / denom) if rap.size else np.nan
        out["pitch_period_std_proxy"] = float(np.std(periods))

    rms_arr = np.asarray(rms, dtype=float)
    valid_idx = voiced_idx[voiced_idx < rms_arr.size]
    if valid_idx.size >= 2:
        amp = rms_arr[valid_idx]
        amp = amp[np.isfinite(amp) & (amp > 1e-9)]
        if amp.size >= 2:
            out["shimmer_local_proxy"] = float(np.mean(np.abs(np.diff(amp))) / max(float(np.mean(amp)), 1e-9))
            out["shimmer_db_proxy"] = float(np.mean(np.abs(20.0 * np.diff(np.log10(amp)))))
    return out


def extract_pause_fluency_features(
    y: np.ndarray,
    sr: int,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    min_pause_sec: float = 0.20,
) -> Dict[str, float]:
    """Extract simple energy-based pause and fluency features."""
    duration_sec = len(y) / sr if sr > 0 else 0.0
    if duration_sec <= 0:
        return {col: np.nan for col in PAUSE_FEATURE_COLUMNS}

    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-6:
        pause_stats = _duration_stats("pause", [duration_sec])
        speech_stats = _duration_stats("speech_segment", [])
        return {
            "silence_ratio": 1.0,
            "energy_voiced_ratio": 0.0,
            "unvoiced_ratio": 1.0,
            "voiced_unvoiced_ratio": 0.0,
            "silence_event_count": 1,
            "avg_silence_duration_sec": float(duration_sec),
            "median_silence_duration_sec": float(duration_sec),
            "max_silence_duration_sec": float(duration_sec),
            "pause_count": 1,
            **pause_stats,
            "total_silence_duration_sec": float(duration_sec),
            "speech_segment_count": 0,
            **speech_stats,
            "speech_duration_sec": 0.0,
            "pause_rate_per_min": float(60.0 / duration_sec) if duration_sec > 0 else np.nan,
            "speech_segment_rate_per_min": 0.0,
            "pause_to_speech_transition_count": 0,
            "speech_to_pause_transition_count": 0,
            "pause_speech_transition_rate_per_min": 0.0,
        }

    intervals = energy_intervals(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    speech_intervals = [(int(start), int(end)) for start, end in intervals]
    speech_durations = [max(0.0, (end - start) / sr) for start, end in speech_intervals]

    silence_durations: List[float] = []
    silence_events: List[Tuple[int, int, float]] = []
    cursor = 0
    for start, end in speech_intervals:
        if start > cursor:
            duration = (start - cursor) / sr
            silence_durations.append(duration)
            silence_events.append((cursor, start, duration))
        cursor = max(cursor, end)
    if cursor < len(y):
        duration = (len(y) - cursor) / sr
        silence_durations.append(duration)
        silence_events.append((cursor, len(y), duration))

    if not speech_intervals:
        silence_durations = [duration_sec]
        silence_events = [(0, len(y), duration_sec)]

    speech_duration_sec = float(np.sum(speech_durations))
    total_silence_duration_sec = float(max(0.0, duration_sec - speech_duration_sec))
    pause_durations = [d for d in silence_durations if d >= min_pause_sec]
    silence_ratio = total_silence_duration_sec / duration_sec
    speech_ratio = speech_duration_sec / duration_sec

    pause_stats = _duration_stats("pause", pause_durations)
    silence_stats = _duration_stats("silence", silence_durations)
    speech_stats = _duration_stats("speech_segment", speech_durations)
    voiced_unvoiced_ratio = speech_duration_sec / total_silence_duration_sec if total_silence_duration_sec > 0 else np.nan
    pause_to_speech = sum(1 for _, end, d in silence_events if d >= min_pause_sec and end < len(y))
    speech_to_pause = sum(1 for start, _, d in silence_events if d >= min_pause_sec and start > 0)
    transition_count = pause_to_speech + speech_to_pause
    return {
        "silence_ratio": float(np.clip(silence_ratio, 0.0, 1.0)),
        "energy_voiced_ratio": float(np.clip(speech_ratio, 0.0, 1.0)),
        "unvoiced_ratio": float(np.clip(silence_ratio, 0.0, 1.0)),
        "voiced_unvoiced_ratio": float(voiced_unvoiced_ratio) if np.isfinite(voiced_unvoiced_ratio) else np.nan,
        "silence_event_count": int(len(silence_durations)),
        **silence_stats,
        "pause_count": int(len(pause_durations)),
        **pause_stats,
        "total_silence_duration_sec": total_silence_duration_sec,
        "speech_segment_count": int(len(speech_durations)),
        **speech_stats,
        "speech_duration_sec": speech_duration_sec,
        "pause_rate_per_min": float(len(pause_durations) / duration_sec * 60.0) if duration_sec > 0 else np.nan,
        "speech_segment_rate_per_min": float(len(speech_durations) / duration_sec * 60.0) if duration_sec > 0 else np.nan,
        "pause_to_speech_transition_count": int(pause_to_speech),
        "speech_to_pause_transition_count": int(speech_to_pause),
        "pause_speech_transition_rate_per_min": float(transition_count / duration_sec * 60.0) if duration_sec > 0 else np.nan,
    }


def extract_features_from_audio(
    audio_path: Path,
    include_pitch: bool = True,
    pitch_max_duration_sec: float = 10.0,
) -> Dict[str, float]:
    """Extract per-file acoustic features."""
    y, sr, err = load_audio_safe(audio_path, target_sr=CFG.target_sample_rate)
    if err or y is None or sr is None:
        raise ValueError(err or "audio_load_failed")
    duration_sec = len(y) / sr if sr > 0 else np.nan

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = _safe_delta(mfcc, order=1)
    mfcc_delta2 = _safe_delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y).ravel()
    zcr = librosa.feature.zero_crossing_rate(y).ravel()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).ravel()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).ravel()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).ravel()
    spectral_flatness = librosa.feature.spectral_flatness(y=y).ravel()
    low_energy_threshold = float(np.median(rms)) if rms.size else np.nan
    low_energy_ratio = float(np.mean(rms < low_energy_threshold)) if rms.size and np.isfinite(low_energy_threshold) else np.nan

    if include_pitch:
        y_pitch = _pitch_signal_window(y, sr, pitch_max_duration_sec)
        f0_analysis_duration_sec = len(y_pitch) / sr if sr > 0 else np.nan
        f0, voiced_flag, _ = librosa.pyin(
            y_pitch,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    else:
        f0 = None
        f0_clean = np.array([])
        voiced_flag = None
        f0_analysis_duration_sec = 0.0

    pause_features = extract_pause_fluency_features(y, sr)

    # Speaking-rate proxy: onset density.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_count = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr).shape[0]
    speaking_rate_proxy = float(onset_count / duration_sec) if duration_sec and duration_sec > 0 else np.nan
    onset_rate_per_speech_sec = (
        float(onset_count / pause_features["speech_duration_sec"])
        if pause_features.get("speech_duration_sec", 0) and pause_features["speech_duration_sec"] > 0
        else np.nan
    )

    feat: Dict[str, float] = {"duration_sec": float(duration_sec)}
    for i in range(min(13, mfcc.shape[0])):
        feat.update(_stats(f"mfcc_{i+1}", mfcc[i]))
        feat.update(_stats(f"delta_mfcc_{i+1}", mfcc_delta[i]))
        feat.update(_stats(f"delta2_mfcc_{i+1}", mfcc_delta2[i]))
    feat.update(_distribution_stats("rms", rms))
    feat.update(_stats("zcr", zcr))
    feat.update(_distribution_stats("spectral_centroid", spectral_centroid))
    feat.update(_distribution_stats("spectral_bandwidth", spectral_bandwidth))
    feat.update(_distribution_stats("spectral_rolloff", spectral_rolloff))
    feat.update(_distribution_stats("spectral_flatness", spectral_flatness))
    feat["low_energy_ratio"] = low_energy_ratio
    feat.update(_energy_modulation_features(rms))
    feat["f0_analysis_duration_sec"] = float(f0_analysis_duration_sec)
    feat.update(_f0_proxy_features(f0_clean, voiced_flag))
    feat.update(_jitter_shimmer_proxy_features(f0, rms, voiced_flag))
    feat["pause_ratio"] = pause_features["silence_ratio"]
    feat["speaking_rate_proxy"] = speaking_rate_proxy
    feat["onset_count"] = int(onset_count)
    feat["onset_rate_per_speech_sec"] = onset_rate_per_speech_sec
    feat.update(pause_features)
    return feat


def feature_numeric_columns(feat_df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns, excluding labels and metadata."""
    numeric_cols = feat_df.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric_cols if col not in FEATURE_METADATA_COLUMNS]


def coerce_feature_columns_to_numeric(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce model feature columns to numeric while leaving metadata untouched."""
    out = feat_df.copy()
    for col in out.columns:
        if col not in FEATURE_METADATA_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_feature_table(
    feat_df: pd.DataFrame,
    output_csv: Path,
    max_feature_missing_pct: float = MAX_FEATURE_MISSING_PCT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replace infinities and drop indefensible all-missing, unstable, or constant feature columns."""
    tables_dir = PHASE3_TABLE_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    cleaned = coerce_feature_columns_to_numeric(feat_df).replace([np.inf, -np.inf], np.nan).copy()
    feature_cols = feature_numeric_columns(cleaned)
    drop_rows: List[Dict[str, object]] = []
    drop_cols: List[str] = []
    for col in feature_cols:
        series = cleaned[col]
        missing_pct = float(series.isna().mean())
        nunique = int(series.dropna().nunique())
        variance = float(series.dropna().var()) if series.dropna().shape[0] > 1 else 0.0
        if missing_pct >= 1.0:
            drop_cols.append(col)
            drop_rows.append(
                {
                    "feature": col,
                    "drop_reason": "all_values_missing",
                    "missing_pct": missing_pct,
                    "variance": variance,
                    "n_unique_non_null": nunique,
                }
            )
        elif missing_pct > max_feature_missing_pct:
            drop_cols.append(col)
            drop_rows.append(
                {
                    "feature": col,
                    "drop_reason": f"missing_pct_above_{max_feature_missing_pct:.2f}",
                    "missing_pct": missing_pct,
                    "variance": variance,
                    "n_unique_non_null": nunique,
                }
            )
        elif nunique <= 1:
            drop_cols.append(col)
            drop_rows.append(
                {
                    "feature": col,
                    "drop_reason": "near_zero_variance_single_value",
                    "missing_pct": missing_pct,
                    "variance": variance,
                    "n_unique_non_null": nunique,
                }
            )
    if drop_cols:
        cleaned = cleaned.drop(columns=drop_cols)
    drop_df = pd.DataFrame(
        drop_rows,
        columns=["feature", "drop_reason", "missing_pct", "variance", "n_unique_non_null"],
    )
    drop_df.to_csv(tables_dir / "phase3_dropped_features.csv", index=False)
    return cleaned, drop_df


def save_feature_quality_reports(
    feat_df: pd.DataFrame,
    output_csv: Path,
    dropped_features: pd.DataFrame | None = None,
    extraction_log: pd.DataFrame | None = None,
) -> None:
    """Save pause-feature summaries and Phase 3 feature quality artifacts."""
    reports_dir = PHASE3_FEATURE_REPORT_DIR
    tables_dir = PHASE3_TABLE_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    pause_cols = [col for col in PAUSE_FEATURE_COLUMNS if col in feat_df.columns]
    numeric_cols = feat_df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = feature_numeric_columns(feat_df)
    numeric_df = feat_df[numeric_cols].replace([np.inf, -np.inf], np.nan) if numeric_cols else pd.DataFrame()

    if pause_cols:
        pause_summary = (
            feat_df.groupby("label_name", dropna=False)[pause_cols]
            .agg(["count", "mean", "std", "median", "min", "max"])
        )
        pause_summary.columns = ["_".join(col).strip("_") for col in pause_summary.columns.to_flat_index()]
        pause_summary = pause_summary.reset_index()
    else:
        pause_summary = pd.DataFrame()
    pause_summary.to_csv(tables_dir / "pause_feature_summary.csv", index=False)

    quality_rows = []
    for col in feature_cols:
        series = feat_df[col]
        non_null = series.dropna()
        variance = float(non_null.var()) if len(non_null) > 1 else 0.0
        q1 = float(non_null.quantile(0.25)) if len(non_null) else np.nan
        q3 = float(non_null.quantile(0.75)) if len(non_null) else np.nan
        iqr = q3 - q1 if np.isfinite(q1) and np.isfinite(q3) else np.nan
        outlier_count = int(((series < (q1 - 3 * iqr)) | (series > (q3 + 3 * iqr))).sum()) if iqr and iqr > 0 else 0
        quality_rows.append(
            {
                "feature": col,
                "nan_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "inf_count": int(np.isinf(series.dropna()).sum()) if len(series.dropna()) else 0,
                "non_null_count": int(series.notna().sum()),
                "n_unique_non_null": int(non_null.nunique()),
                "variance": variance,
                "near_zero_variance": bool(non_null.nunique() <= 1 or np.isclose(variance, 0.0)),
                "iqr_outlier_count": outlier_count,
                "iqr_outlier_pct": float(outlier_count / len(series)) if len(series) else np.nan,
            }
        )
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(tables_dir / "feature_quality_enhanced.csv", index=False)
    quality_df.to_csv(tables_dir / "phase3_feature_quality.csv", index=False)
    if not quality_df.empty:
        quality_df[quality_df["near_zero_variance"]].to_csv(tables_dir / "phase3_near_constant_features.csv", index=False)
        quality_df[quality_df["iqr_outlier_pct"] > 0.05].to_csv(tables_dir / "phase3_feature_outliers.csv", index=False)
    else:
        quality_df.to_csv(tables_dir / "phase3_near_constant_features.csv", index=False)
        quality_df.to_csv(tables_dir / "phase3_feature_outliers.csv", index=False)

    distribution_rows: List[Dict[str, object]] = []
    if feature_cols:
        groups = [("all", feat_df)]
        if "label_name" in feat_df.columns:
            groups.extend((str(label_name), sub_df) for label_name, sub_df in feat_df.groupby("label_name", dropna=False))
        for label_name, group_df in groups:
            for col in feature_cols:
                non_null = group_df[col].dropna().astype(float)
                distribution_rows.append(
                    {
                        "label_name": label_name,
                        "feature": col,
                        "count": int(non_null.shape[0]),
                        "mean": float(non_null.mean()) if len(non_null) else np.nan,
                        "std": float(non_null.std()) if len(non_null) > 1 else 0.0 if len(non_null) == 1 else np.nan,
                        "min": float(non_null.min()) if len(non_null) else np.nan,
                        "p05": float(non_null.quantile(0.05)) if len(non_null) else np.nan,
                        "median": float(non_null.median()) if len(non_null) else np.nan,
                        "p95": float(non_null.quantile(0.95)) if len(non_null) else np.nan,
                        "max": float(non_null.max()) if len(non_null) else np.nan,
                    }
                )
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df.to_csv(tables_dir / "phase3_feature_distribution_summary.csv", index=False)

    missing_by_label_rows: List[Dict[str, object]] = []
    if feature_cols and "label_name" in feat_df.columns:
        for label_name, group_df in feat_df.groupby("label_name", dropna=False):
            for col in feature_cols:
                missing_by_label_rows.append(
                    {
                        "label_name": label_name,
                        "feature": col,
                        "n_rows": int(len(group_df)),
                        "missing_count": int(group_df[col].isna().sum()),
                        "missing_pct": float(group_df[col].isna().mean()),
                    }
                )
    missing_by_label_df = pd.DataFrame(missing_by_label_rows)
    missing_by_label_df.to_csv(tables_dir / "phase3_feature_missingness_by_label.csv", index=False)
    if not missing_by_label_df.empty:
        missing_gap = (
            missing_by_label_df.pivot_table(index="feature", columns="label_name", values="missing_pct", aggfunc="max")
            .reset_index()
        )
        label_cols = [col for col in missing_gap.columns if col != "feature"]
        if label_cols:
            missing_gap["max_label_missing_gap"] = missing_gap[label_cols].max(axis=1) - missing_gap[label_cols].min(axis=1)
            missing_gap["systematic_missingness_risk"] = missing_gap["max_label_missing_gap"] >= 0.10
        missing_gap.to_csv(tables_dir / "phase3_feature_missingness_class_gap.csv", index=False)
    else:
        pd.DataFrame(columns=["feature", "max_label_missing_gap", "systematic_missingness_risk"]).to_csv(
            tables_dir / "phase3_feature_missingness_class_gap.csv",
            index=False,
        )

    row_missing = feat_df.copy()
    if feature_cols:
        row_missing["feature_missing_count"] = row_missing[feature_cols].isna().sum(axis=1)
        row_missing["feature_missing_pct"] = row_missing["feature_missing_count"] / len(feature_cols)
        row_missing_cols = [
            col
            for col in [
                "metadata_index",
                "speaker_id",
                "audio_file",
                "label",
                "label_name",
                "datasplit",
                "feature_extraction_status",
                "feature_missing_count",
                "feature_missing_pct",
            ]
            if col in row_missing.columns
        ]
        row_missing[row_missing_cols].to_csv(tables_dir / "phase3_row_missingness.csv", index=False)

    if extraction_log is not None and "status" in extraction_log.columns:
        error_count = int(extraction_log["status"].isin(["failed", "skipped"]).sum())
    else:
        error_count = int(feat_df.get("feature_extraction_status", pd.Series(dtype=str)).eq("failed").sum())
    total_nan = int(numeric_df.isna().sum().sum()) if not numeric_df.empty else 0
    total_inf = int(np.isinf(feat_df[numeric_cols].select_dtypes(include=["number"])).sum().sum()) if numeric_cols else 0
    dropped_count = 0 if dropped_features is None else len(dropped_features)
    lines = [
        "# Phase 3 Feature Engineering Report",
        "",
        f"- Feature table: `{output_csv}`",
        f"- Rows: {len(feat_df)}",
        f"- Numeric model features: {len(feature_cols)}",
        f"- Pause/fluency features added: {len(pause_cols)}",
        "- Added delta and delta-delta MFCC summaries.",
        "- Added spectral centroid/bandwidth/rolloff/flatness summaries.",
        "- Added frame-level jitter/shimmer proxy features where voiced F0 frames are available.",
        "- Added percentile summaries for F0, RMS/spectral distributions, pause durations, silence events, and speech-segment durations.",
        "- Added energy-modulation and pause/speech transition-rate proxies for speech-flow dynamics.",
        f"- Extraction failures: {error_count}",
        f"- Total numeric NaN values: {total_nan}",
        f"- Total numeric infinite values: {total_inf}",
        f"- Dropped all-missing, high-missingness, or constant feature columns: {dropped_count}",
        "",
        "Pause and fluency features use deterministic frame-RMS energy segmentation.",
        "Speaking rate, jitter, and shimmer are audio-derived proxies, not transcript-based or clinical cycle-level measurements.",
        "Missing numeric values are left as NaN for train-split-only imputation inside the ML pipeline; no global scaler/imputer is fit during feature extraction.",
        "",
        "New feature outputs:",
        "",
        "- `reports/tables/pause_feature_summary.csv`",
        "- `reports/tables/feature_quality_enhanced.csv`",
        "- `reports/tables/phase3_feature_quality.csv`",
        "- `reports/tables/phase3_feature_distribution_summary.csv`",
        "- `reports/tables/phase3_feature_missingness_by_label.csv`",
        "- `reports/tables/phase3_feature_missingness_class_gap.csv`",
        "- `reports/tables/phase3_row_missingness.csv`",
        "- `reports/tables/phase3_near_constant_features.csv`",
        "- `reports/tables/phase3_feature_outliers.csv`",
        "- `reports/tables/phase3_dropped_features.csv`",
    ]
    (reports_dir / "phase1_feature_upgrade_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (reports_dir / "phase3_feature_engineering_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_features(
    metadata_csv: Path = CFG.metadata_clean_csv,
    processed_root: Path = CFG.processed_audio_dir,
    output_csv: Path = Path("data/features/file_features.csv"),
    include_pitch: bool = True,
    pitch_max_duration_sec: float = 10.0,
    max_feature_missing_pct: float = MAX_FEATURE_MISSING_PCT,
) -> pd.DataFrame:
    """Extract and save one feature row per original audio file."""
    meta = pd.read_csv(metadata_csv)
    rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Extracting features"):
        raw_path = Path(str(row.get("audio_file", "")))
        if not raw_path.exists():
            status_rows.append(
                {
                    "metadata_index": row.get("metadata_index"),
                    "audio_file": str(raw_path),
                    "processed_file": "",
                    "status": "skipped",
                    "error": "raw audio file missing",
                }
            )
            continue
        rel = raw_path.resolve().relative_to(CFG.raw_audio_dir.resolve())
        proc_path = processed_root.resolve() / rel.with_suffix(".wav")
        if not proc_path.exists():
            status_rows.append(
                {
                    "metadata_index": row.get("metadata_index"),
                    "audio_file": str(raw_path.resolve()),
                    "processed_file": str(proc_path.resolve()),
                    "status": "skipped",
                    "error": "processed audio file missing",
                }
            )
            continue
        try:
            feat = extract_features_from_audio(
                proc_path,
                include_pitch=include_pitch,
                pitch_max_duration_sec=pitch_max_duration_sec,
            )
            feat.update(
                {
                    "metadata_index": row.get("metadata_index"),
                    "speaker_id": row.get("speaker_id"),
                    "audio_file": str(raw_path.resolve()),
                    "processed_file": str(proc_path.resolve()),
                    "label": row.get("label"),
                    "label_name": row.get("label_name"),
                    "datasplit": row.get("datasplit"),
                    "dementia_type": row.get("dementia_type"),
                    "gender": row.get("gender"),
                    "ethnicity": row.get("ethnicity"),
                    "feature_extraction_status": "success",
                }
            )
            rows.append(feat)
            status_rows.append(
                {
                    "metadata_index": row.get("metadata_index"),
                    "audio_file": str(raw_path.resolve()),
                    "processed_file": str(proc_path.resolve()),
                    "status": "success",
                    "error": "",
                }
            )
        except Exception as exc:
            status_rows.append(
                {
                    "metadata_index": row.get("metadata_index"),
                    "audio_file": str(raw_path.resolve()),
                    "processed_file": str(proc_path.resolve()),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    feat_df = pd.DataFrame(rows)
    status_df = pd.DataFrame(status_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_rows_csv = output_csv.with_name(f"{output_csv.stem}_raw_feature_rows.csv")
    feat_df.to_csv(raw_rows_csv, index=False)
    cleaned_df, dropped_features = clean_feature_table(
        feat_df,
        output_csv,
        max_feature_missing_pct=max_feature_missing_pct,
    )
    cleaned_df.to_csv(output_csv, index=False)
    status_df.to_csv(output_csv.with_name(f"{output_csv.stem}_extraction_log.csv"), index=False)

    tables_dir = PHASE3_TABLE_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(tables_dir / "phase3_feature_extraction_log.csv", index=False)
    status_summary = (
        status_df.groupby("status", dropna=False)
        .size()
        .reset_index(name="file_count")
        if not status_df.empty
        else pd.DataFrame(columns=["status", "file_count"])
    )
    status_summary.to_csv(tables_dir / "phase3_feature_extraction_summary.csv", index=False)
    status_df[status_df["status"].isin(["failed", "skipped"])].to_csv(
        tables_dir / "phase3_feature_extraction_failures.csv",
        index=False,
    )
    save_feature_quality_reports(
        cleaned_df,
        output_csv,
        dropped_features=dropped_features,
        extraction_log=status_df,
    )
    return cleaned_df


def parse_args() -> argparse.Namespace:
    """CLI arguments for feature extraction."""
    parser = argparse.ArgumentParser(description="Extract hand-crafted acoustic features.")
    parser.add_argument("--metadata-csv", type=str, default=str(CFG.metadata_clean_csv))
    parser.add_argument("--processed-root", type=str, default=str(CFG.processed_audio_dir))
    parser.add_argument("--output-csv", type=str, default="data/features/file_features.csv")
    parser.add_argument(
        "--skip-pitch",
        action="store_true",
        help="Skip expensive pitch/F0 extraction for faster iteration.",
    )
    parser.add_argument(
        "--pitch-max-duration-sec",
        type=float,
        default=10.0,
        help="Maximum seconds used for expensive pitch/F0 extraction; use 0 for full file.",
    )
    parser.add_argument(
        "--max-feature-missing-pct",
        type=float,
        default=MAX_FEATURE_MISSING_PCT,
        help="Drop numeric feature columns with missingness above this fraction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = build_features(
        metadata_csv=Path(args.metadata_csv),
        processed_root=Path(args.processed_root),
        output_csv=Path(args.output_csv),
        include_pitch=not args.skip_pitch,
        pitch_max_duration_sec=args.pitch_max_duration_sec,
        max_feature_missing_pct=args.max_feature_missing_pct,
    )
    print(f"Feature rows: {len(df)}")
