"""Batch preprocessing for classical ML: mono, 16 kHz, light cleanup, trim, normalize."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio_utils import (
    energy_intervals,
    highpass_filter,
    load_audio_safe,
    normalize_loudness,
    remove_dc_offset,
    save_audio,
    trim_silence,
)
from src.config import CFG


PREPROCESS_POLICY = {
    "target_sample_rate_hz": CFG.target_sample_rate,
    "target_channels": "mono",
    "target_file_format": "wav",
    "target_subtype": "PCM_16",
    "dc_offset_removal": True,
    "highpass_filter_hz": 50.0,
    "silence_trim_top_db": 30,
    "loudness_normalization": "RMS target with capped gain and peak limiting",
    "target_rms_dbfs": -20.0,
    "max_gain_db": 12.0,
    "peak_limit": 0.99,
    "aggressive_spectral_denoising": False,
    "vad_for_qc": "frame RMS energy intervals, top_db=30",
}


def build_preprocess_policy(
    trim: bool,
    trim_top_db: int,
    highpass_cutoff_hz: float,
    target_rms_dbfs: float,
    max_gain_db: float,
    peak_limit: float,
) -> Dict[str, object]:
    """Return the effective policy for one preprocessing run."""
    policy = dict(PREPROCESS_POLICY)
    policy.update(
        {
            "target_sample_rate_hz": CFG.target_sample_rate,
            "silence_trim_enabled": bool(trim),
            "silence_trim_top_db": int(trim_top_db),
            "highpass_filter_hz": float(highpass_cutoff_hz),
            "target_rms_dbfs": float(target_rms_dbfs),
            "max_gain_db": float(max_gain_db),
            "peak_limit": float(peak_limit),
            "step_order": (
                "load/resample/mono -> finite-value cleanup -> DC removal -> "
                "conservative high-pass filtering -> optional leading/trailing "
                "silence trim -> capped RMS normalization -> PCM_16 WAV write -> QC logging"
            ),
        }
    )
    policy["vad_for_qc"] = f"frame RMS energy intervals, top_db={int(trim_top_db)}"
    return policy


def output_path_for_audio(raw_audio_path: Path, raw_audio_root: Path, processed_root: Path) -> Path:
    """Mirror input tree under processed root with WAV extension."""
    rel = raw_audio_path.resolve().relative_to(raw_audio_root.resolve())
    return processed_root.resolve() / rel.with_suffix(".wav")


def processed_audio_info(path: Path) -> Dict[str, object]:
    """Read processed WAV header info with the stdlib wave reader."""
    with wave.open(str(path), "rb") as wav:
        sample_rate = int(wav.getframerate())
        channels = int(wav.getnchannels())
        frames = int(wav.getnframes())
        sample_width_bits = int(wav.getsampwidth() * 8)
    return {
        "processed_sample_rate": sample_rate,
        "processed_channels": channels,
        "processed_subtype": f"PCM_{sample_width_bits}",
        "processed_format": "WAV",
        "processed_frames": frames,
        "duration_after_sec": float(frames / sample_rate) if sample_rate else np.nan,
    }


def is_valid_processed_audio(path: Path) -> Tuple[bool, Dict[str, object], str]:
    """Check whether an existing processed WAV already matches Phase 3 targets."""
    if not path.exists():
        return False, {}, "processed_file_missing"
    try:
        info = processed_audio_info(path)
    except Exception as exc:
        return False, {}, f"processed_header_error={exc}"
    if info["processed_sample_rate"] != CFG.target_sample_rate:
        return False, info, "processed_sample_rate_mismatch"
    if info["processed_channels"] != 1:
        return False, info, "processed_not_mono"
    if info["processed_frames"] <= 0:
        return False, info, "processed_empty"
    return True, info, ""


def source_audio_info(path: Path) -> Dict[str, object]:
    """Return non-blocking source file metadata for logging."""
    try:
        return {
            "source_sample_rate": np.nan,
            "source_channels": np.nan,
            "source_frames": np.nan,
            "source_duration_sec_header": np.nan,
            "source_format": path.suffix.lower().lstrip("."),
            "source_subtype": "",
            "source_file_size_bytes": path.stat().st_size,
        }
    except Exception as exc:
        return {
            "source_sample_rate": np.nan,
            "source_channels": np.nan,
            "source_frames": np.nan,
            "source_duration_sec_header": np.nan,
            "source_format": "",
            "source_subtype": "",
            "source_info_error": str(exc),
        }


def vad_summary(y: np.ndarray, sr: int, top_db: int = 30) -> Dict[str, float]:
    """Return lightweight VAD/silence summary for preprocessing QC."""
    duration = len(y) / sr if sr > 0 else 0.0
    if duration <= 0:
        return {
            "vad_speech_regions": 0,
            "vad_speech_duration_sec": 0.0,
            "vad_silence_ratio": np.nan,
        }
    try:
        intervals = energy_intervals(y, top_db=top_db)
        speech_duration = float(sum((end - start) / sr for start, end in intervals))
        return {
            "vad_speech_regions": int(len(intervals)),
            "vad_speech_duration_sec": speech_duration,
            "vad_silence_ratio": float(np.clip(1.0 - speech_duration / duration, 0.0, 1.0)),
        }
    except Exception:
        return {
            "vad_speech_regions": np.nan,
            "vad_speech_duration_sec": np.nan,
            "vad_silence_ratio": np.nan,
        }


def preprocess_waveform(
    y: np.ndarray,
    sr: int,
    trim: bool,
    trim_top_db: int,
    highpass_cutoff_hz: float,
    target_rms_dbfs: float,
    max_gain_db: float,
    peak_limit: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply Phase 3 preprocessing steps in a documented, deterministic order."""
    stats: Dict[str, object] = {
        "duration_loaded_sec": len(y) / sr if sr else np.nan,
        "trim_applied": bool(trim),
        "trim_top_db": trim_top_db if trim else np.nan,
        "trim_fallback_used": False,
        "highpass_cutoff_hz": highpass_cutoff_hz,
        "target_rms_dbfs": target_rms_dbfs,
        "max_gain_db": max_gain_db,
        "peak_limit": peak_limit,
    }
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y_proc = remove_dc_offset(y)
    y_proc = highpass_filter(y_proc, sr=sr, cutoff_hz=highpass_cutoff_hz)

    stats["duration_before_trim_sec"] = len(y_proc) / sr if sr else np.nan
    if trim:
        y_trimmed = trim_silence(y_proc, top_db=trim_top_db)
        if len(y_trimmed) > 0:
            y_proc = y_trimmed
        else:
            stats["trim_fallback_used"] = True
    stats["duration_after_trim_sec"] = len(y_proc) / sr if sr else np.nan
    if np.isfinite(stats["duration_before_trim_sec"]) and np.isfinite(stats["duration_after_trim_sec"]):
        stats["trim_removed_sec"] = max(0.0, stats["duration_before_trim_sec"] - stats["duration_after_trim_sec"])
    else:
        stats["trim_removed_sec"] = np.nan

    y_proc, norm_stats = normalize_loudness(
        y_proc,
        target_rms_dbfs=target_rms_dbfs,
        max_gain_db=max_gain_db,
        peak_limit=peak_limit,
    )
    stats.update(norm_stats)
    stats.update(vad_summary(y_proc, sr, top_db=trim_top_db))
    stats["peak_abs_after"] = float(np.max(np.abs(y_proc))) if len(y_proc) else 0.0
    stats["duration_after_sec"] = len(y_proc) / sr if sr else np.nan
    return y_proc.astype(np.float32), stats


def write_preprocessing_reports(log_df: pd.DataFrame, policy: Dict[str, object]) -> None:
    """Write Phase 3 preprocessing policy, summary tables, and validation notes."""
    reports_dir = Path("reports")
    tables_dir = reports_dir / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ok_statuses = {"ok", "ok_reused"}
    ok = log_df[log_df["status"].isin(ok_statuses)].copy()
    summary_rows = [
        {"metric": "total_metadata_rows", "value": int(len(log_df))},
        {"metric": "processed_ok", "value": int(log_df["status"].isin(ok_statuses).sum())},
        {"metric": "reprocessed_ok", "value": int((log_df["status"] == "ok").sum())},
        {"metric": "reused_existing_valid", "value": int((log_df["status"] == "ok_reused").sum())},
        {"metric": "failed_or_skipped", "value": int((~log_df["status"].isin(ok_statuses)).sum())},
        {"metric": "target_sample_rate_hz", "value": policy["target_sample_rate_hz"]},
        {
            "metric": "processed_sample_rates",
            "value": json.dumps(ok["processed_sample_rate"].value_counts().to_dict())
            if not ok.empty and "processed_sample_rate" in ok
            else "{}",
        },
        {
            "metric": "processed_channels",
            "value": json.dumps(ok["processed_channels"].value_counts().to_dict())
            if not ok.empty and "processed_channels" in ok
            else "{}",
        },
        {"metric": "processed_subtypes", "value": json.dumps(ok["processed_subtype"].value_counts().to_dict()) if "processed_subtype" in ok else "{}"},
        {"metric": "min_duration_after_sec", "value": float(ok["duration_after_sec"].min()) if not ok.empty else np.nan},
        {"metric": "median_duration_after_sec", "value": float(ok["duration_after_sec"].median()) if not ok.empty else np.nan},
        {"metric": "max_duration_after_sec", "value": float(ok["duration_after_sec"].max()) if not ok.empty else np.nan},
        {"metric": "median_trim_removed_sec", "value": float(ok["trim_removed_sec"].median()) if not ok.empty else np.nan},
        {
            "metric": "files_with_trim_fallback",
            "value": int(ok["trim_fallback_used"].fillna(False).sum())
            if not ok.empty and "trim_fallback_used" in ok
            else 0,
        },
        {"metric": "max_normalization_gain_db", "value": float(ok["normalization_gain_db"].max()) if not ok.empty else np.nan},
        {"metric": "files_with_peak_limiting", "value": int(ok["normalization_clipped"].fillna(False).sum()) if not ok.empty else 0},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "phase3_preprocessing_summary.csv", index=False)

    failed = log_df[~log_df["status"].isin(ok_statuses)].copy()
    failed.to_csv(tables_dir / "phase3_preprocessing_failures.csv", index=False)

    policy_df = pd.DataFrame([{"parameter": k, "value": v} for k, v in policy.items()])
    policy_df.to_csv(tables_dir / "phase3_preprocessing_policy.csv", index=False)

    lines = [
        "# Phase 3 Preprocessing Report",
        "",
        "Scope: classical-ML preprocessing only. Raw audio files were not modified.",
        "",
        "## Standardization Policy",
        "",
        f"- Target sample rate: **{policy['target_sample_rate_hz']} Hz**",
        "- Target channels: **mono**",
        "- Target output: **WAV PCM_16**",
        f"- Step order: {policy['step_order']}.",
        f"- High-pass filter: **{policy['highpass_filter_hz']} Hz**, used as light denoising for DC/rumble only.",
        "- Aggressive spectral denoising was intentionally not applied because it can distort pitch, shimmer, and spectral features on a small heterogeneous speech dataset.",
        f"- Silence trim enabled: **{policy['silence_trim_enabled']}**.",
        f"- Silence trim: frame RMS energy trim with `top_db={policy['silence_trim_top_db']}`; used only for leading/trailing silence, not internal pause removal.",
        f"- Normalization: target RMS **{policy['target_rms_dbfs']} dBFS**, capped at **±{policy['max_gain_db']} dB**, peak-limited to **{policy['peak_limit']}**.",
        "",
        "## Validation Summary",
        "",
        summary_df.to_string(index=False),
        "",
        "Failures/skips are saved to `reports/tables/phase3_preprocessing_failures.csv`.",
        "Full per-file preprocessing log is saved to `data/metadata/preprocess_log.csv`.",
        "Rows marked `ok_reused` passed processed-WAV header checks and were not reloaded from raw audio.",
    ]
    (reports_dir / "phase3_preprocessing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    policy_table = ["| parameter | value |", "| --- | --- |"]
    for _, policy_row in policy_df.iterrows():
        value = str(policy_row["value"]).replace("|", "\\|")
        policy_table.append(f"| `{policy_row['parameter']}` | {value} |")
    doc_lines = [
        "# Phase 3 Preprocessing Policy",
        "",
        "This policy applies to the classical ML baseline only. Raw audio under `data/raw_audio/` is never modified.",
        "",
        "## Targets",
        "",
        f"- Sample rate: `{policy['target_sample_rate_hz']} Hz`",
        "- Channels: `mono`",
        "- Output format: `WAV`, subtype `PCM_16`",
        "- Output location: mirrored paths under `data/processed_audio/`",
        "",
        "## Step Order",
        "",
        f"1. {policy['step_order']}.",
        "",
        "## Rationale",
        "",
        "- Resampling and mono conversion make downstream hand-crafted features comparable across files.",
        "- DC removal and conservative high-pass filtering reduce electrical offset and low-frequency rumble without broad speech enhancement.",
        "- Leading/trailing silence trimming removes obvious padding while preserving internal pauses needed for fluency features.",
        "- RMS normalization uses capped gain and peak limiting, so very quiet clips are not amplified without bound.",
        "- Energy-based VAD is used for quality-control summaries and pause features, not as a hard deletion step for internal non-speech regions.",
        "- Aggressive spectral denoising is excluded because it can alter F0, shimmer, jitter proxies, and spectral descriptors in ways that are hard to defend on this dataset.",
        "",
        "## Effective Parameters",
        "",
        *policy_table,
    ]
    (docs_dir / "PHASE3_PREPROCESSING_POLICY.md").write_text("\n".join(doc_lines) + "\n", encoding="utf-8")


def preprocess_dataset(
    metadata_csv: Path = CFG.metadata_clean_csv,
    raw_audio_root: Path = CFG.raw_audio_dir,
    processed_root: Path = CFG.processed_audio_dir,
    trim: bool = False,
    trim_top_db: int = int(PREPROCESS_POLICY["silence_trim_top_db"]),
    highpass_cutoff_hz: float = float(PREPROCESS_POLICY["highpass_filter_hz"]),
    target_rms_dbfs: float = float(PREPROCESS_POLICY["target_rms_dbfs"]),
    max_gain_db: float = float(PREPROCESS_POLICY["max_gain_db"]),
    peak_limit: float = float(PREPROCESS_POLICY["peak_limit"]),
    reuse_existing_valid: bool = False,
) -> pd.DataFrame:
    """Preprocess all matched files and export a reproducible run log."""
    df = pd.read_csv(metadata_csv)
    policy = build_preprocess_policy(
        trim=trim,
        trim_top_db=trim_top_db,
        highpass_cutoff_hz=highpass_cutoff_hz,
        target_rms_dbfs=target_rms_dbfs,
        max_gain_db=max_gain_db,
        peak_limit=peak_limit,
    )
    policy["reuse_existing_valid_processed_audio"] = bool(reuse_existing_valid)
    log_rows: List[Dict[str, object]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing audio"):
        audio_file = Path(str(row.get("audio_file", "")))
        base_log = {
            "metadata_index": row.get("metadata_index"),
            "speaker_id": row.get("speaker_id"),
            "label": row.get("label"),
            "label_name": row.get("label_name"),
            "datasplit": row.get("datasplit"),
            "audio_file": str(audio_file),
        }
        if not audio_file.exists():
            log_rows.append({**base_log, "status": "skipped_missing", "error": "audio_file_missing"})
            continue

        out_path = output_path_for_audio(audio_file, raw_audio_root, processed_root)
        source_info = {
            "source_sample_rate": row.get("sample_rate"),
            "source_channels": np.nan,
            "source_frames": np.nan,
            "source_duration_sec_header": row.get("duration_sec"),
            "source_format": audio_file.suffix.lower().lstrip("."),
            "source_subtype": "",
        }

        if reuse_existing_valid:
            valid_existing, existing_info, existing_error = is_valid_processed_audio(out_path)
            if valid_existing:
                existing_stats = {
                    "duration_loaded_sec": np.nan,
                    "duration_before_trim_sec": np.nan,
                    "duration_after_trim_sec": np.nan,
                    "trim_removed_sec": np.nan,
                    "trim_applied": bool(trim),
                    "trim_top_db": trim_top_db if trim else np.nan,
                    "trim_fallback_used": False,
                    "highpass_cutoff_hz": highpass_cutoff_hz,
                    "target_rms_dbfs": target_rms_dbfs,
                    "max_gain_db": max_gain_db,
                    "peak_limit": peak_limit,
                    "rms_before_dbfs": np.nan,
                    "rms_after_dbfs": np.nan,
                    "normalization_gain_db": np.nan,
                    "normalization_clipped": np.nan,
                    "peak_abs_after": np.nan,
                    "vad_speech_regions": np.nan,
                    "vad_speech_duration_sec": np.nan,
                    "vad_silence_ratio": np.nan,
                    "reused_existing_valid": True,
                }
                log_rows.append(
                    {
                        **base_log,
                        **source_info,
                        "processed_file": str(out_path),
                        "status": "ok_reused",
                        "error": "",
                        "duration_before_sec": row.get("duration_sec"),
                        "sample_rate": existing_info["processed_sample_rate"],
                        **existing_info,
                        **existing_stats,
                    }
                )
                continue
            if existing_error and out_path.exists():
                base_log["existing_processed_error"] = existing_error

        y, sr, err = load_audio_safe(audio_file, target_sr=CFG.target_sample_rate)
        if err or y is None or sr is None:
            log_rows.append(
                {
                    **base_log,
                    **source_info,
                    "status": "failed",
                    "error": err or "unknown_load_error",
                }
            )
            continue

        y_proc, proc_stats = preprocess_waveform(
            y,
            sr,
            trim=trim,
            trim_top_db=trim_top_db,
            highpass_cutoff_hz=highpass_cutoff_hz,
            target_rms_dbfs=target_rms_dbfs,
            max_gain_db=max_gain_db,
            peak_limit=peak_limit,
        )

        try:
            save_audio(out_path, y_proc, sr, subtype=str(PREPROCESS_POLICY["target_subtype"]))
            written_info = processed_audio_info(out_path)
            log_rows.append(
                {
                    **base_log,
                    **source_info,
                    "processed_file": str(out_path),
                    "status": "ok",
                    "error": "",
                    "duration_before_sec": row.get("duration_sec"),
                    "sample_rate": sr,
                    **written_info,
                    "reused_existing_valid": False,
                    **proc_stats,
                }
            )
        except Exception as exc:
            log_rows.append(
                {
                    **base_log,
                    **source_info,
                    "processed_file": str(out_path),
                    "status": "failed",
                    "error": str(exc),
                    **proc_stats,
                }
            )

    log_df = pd.DataFrame(log_rows)
    out_log = Path("data/metadata/preprocess_log.csv")
    out_log.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(out_log, index=False)
    write_preprocessing_reports(log_df, policy)
    return log_df


def parse_args() -> argparse.Namespace:
    """CLI arguments for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess raw audio to standardized classical-ML WAV.")
    parser.add_argument("--metadata-csv", type=str, default=str(CFG.metadata_clean_csv))
    parser.add_argument("--raw-audio-root", type=str, default=str(CFG.raw_audio_dir))
    parser.add_argument("--processed-root", type=str, default=str(CFG.processed_audio_dir))
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--trim-top-db", type=int, default=int(PREPROCESS_POLICY["silence_trim_top_db"]))
    parser.add_argument("--highpass-cutoff-hz", type=float, default=float(PREPROCESS_POLICY["highpass_filter_hz"]))
    parser.add_argument("--target-rms-dbfs", type=float, default=float(PREPROCESS_POLICY["target_rms_dbfs"]))
    parser.add_argument("--max-gain-db", type=float, default=float(PREPROCESS_POLICY["max_gain_db"]))
    parser.add_argument("--peak-limit", type=float, default=float(PREPROCESS_POLICY["peak_limit"]))
    parser.add_argument(
        "--reuse-existing-valid",
        action="store_true",
        help="Reuse existing processed WAVs that already match target sample rate, mono, and non-empty checks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log = preprocess_dataset(
        metadata_csv=Path(args.metadata_csv),
        raw_audio_root=Path(args.raw_audio_root),
        processed_root=Path(args.processed_root),
        trim=args.trim_silence,
        trim_top_db=args.trim_top_db,
        highpass_cutoff_hz=args.highpass_cutoff_hz,
        target_rms_dbfs=args.target_rms_dbfs,
        max_gain_db=args.max_gain_db,
        peak_limit=args.peak_limit,
        reuse_existing_valid=args.reuse_existing_valid,
    )
    print("Preprocessing complete.")
    print(log["status"].value_counts(dropna=False))
