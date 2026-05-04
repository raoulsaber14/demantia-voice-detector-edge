#!/usr/bin/env python3
"""
Phase 2: Dataset quality audit for raw_audio (no raw file deletion).

Scans data/raw_audio, joins Phase 1 metadata when available, emits CSVs and a markdown report.

Heuristics are documented in reports/dataset_audit/dataset_audit_report.md (thresholds section).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project imports: run as `python scripts/dataset_audit.py` from repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

from src.metadata_utils import clean_text, normalize_key

# --- Thresholds (document in report) ---
VERY_SHORT_SEC = 1.0
VERY_LONG_SEC = 300.0
NEAR_ZERO_SEC = 0.05
SILENCE_RATIO_HIGH = 0.85
RMS_DB_LOW = -48.0
CLIP_FRAC_WARN = 1e-3
FLATNESS_NOISY = 0.32
ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac"}
MAX_HASH_BYTES = 80 * 1024 * 1024  # hash full file up to 80MB streamed


def effective_label_from_folder(folder_name: str) -> Tuple[str, int]:
    """Map physical raw_audio child folder to pipeline label."""
    f = clean_text(folder_name).lower()
    if f == "dementia":
        return "dementia", 1
    if f == "nodementia":
        return "non_dementia", 0
    return f"unknown_{folder_name}", -1


def list_audio_files(raw_root: Path) -> List[Path]:
    files: List[Path] = []
    if not raw_root.is_dir():
        return files
    for p in raw_root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix.lower() not in ALLOWED_EXT:
            continue
        files.append(p.resolve())
    return sorted(files)


def count_raw_tree_artifacts(raw_root: Path) -> Dict[str, int]:
    """Count physical files, real audio files, sidecars, and non-audio artifacts under raw_root."""
    counts = {
        "physical_files_total": 0,
        "appledouble_sidecar_files_skipped": 0,
        "appledouble_audio_sidecars_skipped": 0,
        "non_sidecar_audio_files": 0,
        "non_sidecar_non_audio_artifacts": 0,
    }
    if not raw_root.is_dir():
        return counts
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        counts["physical_files_total"] += 1
        is_sidecar = path.name.startswith("._")
        is_audio_ext = path.suffix.lower() in ALLOWED_EXT
        if is_sidecar:
            counts["appledouble_sidecar_files_skipped"] += 1
            if is_audio_ext:
                counts["appledouble_audio_sidecars_skipped"] += 1
        elif is_audio_ext:
            counts["non_sidecar_audio_files"] += 1
        else:
            counts["non_sidecar_non_audio_artifacts"] += 1
    return counts


def load_metadata_index(meta_path: Path) -> pd.DataFrame:
    """Load metadata CSV; normalize paths for joining."""
    if not meta_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(meta_path)
    if "audio_file" not in df.columns:
        return df
    norm_paths = []
    for x in df["audio_file"]:
        try:
            norm_paths.append(str(Path(str(x)).resolve()))
        except Exception:
            norm_paths.append(str(x))
    df = df.copy()
    df["_audio_path_resolved"] = norm_paths
    return df


def merge_metadata_row(path_resolved: str, meta: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "metadata_speaker_id": "",
        "metadata_language": "",
        "metadata_gender": "",
        "metadata_ethnicity": "",
        "metadata_label_name": "",
        "metadata_datasplit": "",
        "metadata_match_needs_review": "",
        "metadata_row_found": False,
    }
    if meta.empty or "_audio_path_resolved" not in meta.columns:
        return out
    m = meta[meta["_audio_path_resolved"] == path_resolved]
    if m.empty:
        return out
    row = m.iloc[0]
    out["metadata_row_found"] = True
    out["metadata_speaker_id"] = clean_text(row.get("speaker_id", ""))
    out["metadata_language"] = clean_text(row.get("language", ""))
    out["metadata_gender"] = clean_text(row.get("gender", ""))
    out["metadata_ethnicity"] = clean_text(row.get("ethnicity", ""))
    out["metadata_label_name"] = clean_text(row.get("label_name", ""))
    out["metadata_datasplit"] = clean_text(row.get("datasplit", ""))
    mr = row.get("match_needs_review", "")
    out["metadata_match_needs_review"] = (
        str(mr).strip().lower() in ("true", "1", "yes") if pd.notna(mr) else False
    )
    return out


def speaker_subfolder_from_rel_parts(parts: Tuple[str, ...]) -> Tuple[str, bool]:
    """
    Return (speaker_folder_name, redundant_nested_nodementia).

    Some trees duplicate the class folder: raw_audio/nodementia/nodementia/<Speaker>/...
    """
    if len(parts) >= 3 and parts[0] == "nodementia" and parts[1] == "nodementia":
        return parts[2], True
    if len(parts) >= 2:
        return parts[1], False
    return "", False


def infer_speaker_from_rel_parts(parts: Tuple[str, ...]) -> str:
    """Normalized speaker id from relative path parts under raw_audio."""
    name, _ = speaker_subfolder_from_rel_parts(parts)
    return normalize_key(name) or "unknown_speaker"


def file_sha256_partial(path: Path, max_bytes: int = MAX_HASH_BYTES) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def analyze_waveform(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute silence ratio, clipping fraction, RMS dB, spectral flatness mean."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if n == 0:
        return {
            "silence_ratio": 1.0,
            "clip_fraction": 0.0,
            "rms_db": -np.inf,
            "spectral_flatness_mean": np.nan,
        }
    eps = 1e-12
    rms = float(np.sqrt(np.mean(y**2)))
    rms_db = 20.0 * math.log10(rms + eps)

    max_abs = float(np.max(np.abs(y))) if n else 0.0
    ceiling = 0.999 if max_abs <= 1.01 else float(np.iinfo(np.int16).max)
    if max_abs > 1.01:
        # integer paths already converted to float by soundfile; treat as normalized
        clip_fraction = float(np.mean(np.abs(y) >= (ceiling * 0.99)))
    else:
        clip_fraction = float(np.mean(np.abs(y) >= 0.999))

    try:
        nonsilent = librosa.effects.split(y, top_db=30)
        speech = sum(int(e - s) for s, e in nonsilent)
        silence_ratio = float(1.0 - (speech / n)) if n else 1.0
    except Exception:
        silence_ratio = float(np.nan)

    try:
        flat = librosa.feature.spectral_flatness(y=y, hop_length=512)
        flat_mean = float(np.nanmean(flat))
    except Exception:
        flat_mean = float("nan")

    return {
        "silence_ratio": silence_ratio,
        "clip_fraction": clip_fraction,
        "rms_db": rms_db,
        "spectral_flatness_mean": flat_mean,
    }


def read_audio_for_analysis(path: Path) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str], Dict[str, Any]]:
    """Load audio as mono float for analysis."""
    info: Dict[str, Any] = {}
    try:
        info_obj = sf.info(str(path))
        info["subtype"] = str(info_obj.subtype)
        info["format"] = str(info_obj.format)
        info["channels"] = int(info_obj.channels)
        info["frames"] = int(info_obj.frames)
        info["duration_sec_header"] = float(info_obj.duration)
        info["sample_rate_header"] = int(info_obj.samplerate)
    except Exception as exc:
        return None, None, f"sf.info_failed:{exc}", info
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        original_channels = y.shape[1] if y.ndim == 2 else 1
        info.setdefault("channels", int(original_channels))
        if y.ndim == 2 and y.shape[1] > 1:
            y = np.mean(y, axis=1)
        else:
            y = y.ravel()
        return y, int(sr), None, info
    except Exception as exc:
        return None, None, f"sf.read_failed:{exc}", info


def build_inventory(
    raw_root: Path,
    meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    raw_root = raw_root.resolve()
    files = list_audio_files(raw_root)
    flagged_accum: List[Dict[str, Any]] = []

    for path in tqdm(files, desc="Auditing audio files"):
        rel = path.relative_to(raw_root)
        parts = rel.parts
        raw_folder = parts[0] if parts else ""
        eff_name, eff_label_int = effective_label_from_folder(raw_folder)
        speaker_folder, redundant_nested = speaker_subfolder_from_rel_parts(parts)

        md = merge_metadata_row(str(path), meta)
        speaker_inferred = infer_speaker_from_rel_parts(parts)
        speaker_effective = md["metadata_speaker_id"] or speaker_inferred

        row: Dict[str, Any] = {
            "repo_relative_path": str(Path("data/raw_audio") / rel).replace("\\", "/"),
            "absolute_path": str(path),
            "file_name": path.name,
            "raw_class_folder": raw_folder,
            "effective_label_name": eff_name,
            "effective_label_int": eff_label_int,
            "speaker_folder_name": speaker_folder,
            "redundant_nested_nodementia_folder": redundant_nested,
            "speaker_id_inferred_from_path": speaker_inferred,
            "speaker_id_effective": speaker_effective,
            "file_extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
        }
        row.update(md)
        row["metadata_row_found"] = bool(md.get("metadata_row_found", False))

        if bool(row.get("metadata_match_needs_review", False)):
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "metadata_audio_match_needs_review",
                    "severity": "medium",
                    "metric_or_threshold": "metadata_match_needs_review=True",
                    "recommended_action": "review_metadata_audio_match_before_training",
                }
            )

        if row["file_size_bytes"] == 0:
            row["readable"] = False
            row["read_error"] = "zero_byte_file"
            row["duration_sec"] = np.nan
            row["sample_rate"] = np.nan
            row["channels"] = np.nan
            row["frames"] = np.nan
            row["duration_sec_header"] = np.nan
            row["analysis_mixed_to_mono"] = False
            row["subtype"] = ""
            row["format"] = ""
            if redundant_nested:
                flagged_accum.append(
                    {
                        "file_path": row["repo_relative_path"],
                        "class": row["effective_label_name"],
                        "issue_type": "redundant_nested_nodementia_folder",
                        "severity": "low",
                        "metric_or_threshold": "path under nodementia/nodementia/",
                        "recommended_action": "consolidate_paths_for_consistency",
                    }
                )
            rows.append(row)
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "empty_or_zero_byte",
                    "severity": "high",
                    "metric_or_threshold": "file_size_bytes==0",
                    "recommended_action": "review_manually_or_exclude",
                }
            )
            continue

        y, sr, err, finfo = read_audio_for_analysis(path)
        row["subtype"] = finfo.get("subtype", "")
        row["format"] = finfo.get("format", "")
        if err:
            row["readable"] = False
            row["read_error"] = err
            row["duration_sec"] = np.nan
            row["sample_rate"] = np.nan
            row["channels"] = np.nan
            row["frames"] = np.nan
            row["duration_sec_header"] = np.nan
            row["analysis_mixed_to_mono"] = False
            if redundant_nested:
                flagged_accum.append(
                    {
                        "file_path": row["repo_relative_path"],
                        "class": row["effective_label_name"],
                        "issue_type": "redundant_nested_nodementia_folder",
                        "severity": "low",
                        "metric_or_threshold": "path under nodementia/nodementia/",
                        "recommended_action": "consolidate_paths_for_consistency",
                    }
                )
            rows.append(row)
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "unreadable_or_corrupt",
                    "severity": "high",
                    "metric_or_threshold": err[:200],
                    "recommended_action": "exclude_from_training_or_re_export",
                }
            )
            continue

        row["readable"] = True
        row["read_error"] = ""
        row["duration_sec"] = float(len(y)) / float(sr) if sr else 0.0
        row["sample_rate"] = int(sr)
        row["channels"] = int(finfo.get("channels", 1))
        row["frames"] = int(finfo.get("frames", len(y)))
        row["duration_sec_header"] = float(finfo.get("duration_sec_header", row["duration_sec"]))
        row["analysis_mixed_to_mono"] = bool(row["channels"] > 1)

        stats = analyze_waveform(y, sr)
        row["silence_ratio_energy_vad"] = stats["silence_ratio"]
        row["clip_fraction"] = stats["clip_fraction"]
        row["rms_db"] = stats["rms_db"]
        row["spectral_flatness_mean"] = stats["spectral_flatness_mean"]

        # Flags
        d = row["duration_sec"]
        if d <= NEAR_ZERO_SEC:
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "likely_near_zero_duration",
                    "severity": "high",
                    "metric_or_threshold": f"duration_sec={d:.4f} (threshold <={NEAR_ZERO_SEC})",
                    "recommended_action": "exclude_from_training",
                }
            )
        elif d < VERY_SHORT_SEC:
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "likely_too_short_for_screening",
                    "severity": "medium",
                    "metric_or_threshold": f"duration_sec={d:.3f} (<{VERY_SHORT_SEC}s)",
                    "recommended_action": "review_manually_or_exclude",
                }
            )
        if d > VERY_LONG_SEC:
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "unusually_long_may_need_segmentation",
                    "severity": "low",
                    "metric_or_threshold": f"duration_sec={d:.1f} (>{VERY_LONG_SEC}s)",
                    "recommended_action": "keep_but_segment_or_window_for_training",
                }
            )

        if not math.isnan(stats["silence_ratio"]) and stats["silence_ratio"] >= SILENCE_RATIO_HIGH:
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "likely_mostly_silent",
                    "severity": "medium",
                    "metric_or_threshold": f"silence_ratio>={SILENCE_RATIO_HIGH} (observed {stats['silence_ratio']:.3f})",
                    "recommended_action": "review_manually",
                }
            )
        if stats["rms_db"] < RMS_DB_LOW and not math.isinf(stats["rms_db"]):
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "likely_extremely_low_volume",
                    "severity": "medium",
                    "metric_or_threshold": f"rms_db={stats['rms_db']:.1f} (<{RMS_DB_LOW})",
                    "recommended_action": "review_manually",
                }
            )
        if stats["clip_fraction"] >= CLIP_FRAC_WARN:
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "likely_clipped_or_heavily_saturated",
                    "severity": "medium",
                    "metric_or_threshold": f"clip_fraction={stats['clip_fraction']:.5f} (>={CLIP_FRAC_WARN})",
                    "recommended_action": "review_manually",
                }
            )
        if not math.isnan(stats["spectral_flatness_mean"]) and stats["spectral_flatness_mean"] >= FLATNESS_NOISY:
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "likely_noisy_or_non_speech_like_heuristic",
                    "severity": "low",
                    "metric_or_threshold": f"flatness_mean={stats['spectral_flatness_mean']:.3f} (>={FLATNESS_NOISY})",
                    "recommended_action": "review_manually_if_model_fails",
                }
            )
        if redundant_nested:
            flagged_accum.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": "redundant_nested_nodementia_folder",
                    "severity": "low",
                    "metric_or_threshold": "path matches data/raw_audio/nodementia/nodementia/<speaker>/",
                    "recommended_action": "consolidate_paths_or_verify_not_duplicate_of_flat_nodementia_tree",
                }
            )

        rows.append(row)

    inv = pd.DataFrame(rows)
    return inv, flagged_accum


def add_duplicate_flags(
    inv: pd.DataFrame,
    flagged: List[Dict[str, Any]],
    collision_rows: List[Dict[str, Any]],
) -> None:
    """
    Flag only content duplicates: same (size, duration, SR) AND matching partial SHA256
    of file bytes (first 80MB). Many clips share common durations (e.g. 90s) without being
    duplicates — those are recorded in collision_rows, not as training issues.
    """
    if inv.empty:
        return
    sub = inv[inv["readable"] == True].copy()  # noqa: E712
    if sub.empty:
        return
    sub["dur_key"] = sub["duration_sec"].round(3)
    sub["gkey"] = (
        sub["file_size_bytes"].astype(str)
        + "|"
        + sub["dur_key"].astype(str)
        + "|"
        + sub["sample_rate"].astype(str)
    )
    groups = sub.groupby("gkey").groups
    for gkey, idxs in groups.items():
        if len(idxs) < 2:
            continue
        paths = sub.loc[list(idxs), "absolute_path"].astype(str).tolist()
        hashes = [file_sha256_partial(Path(p)) for p in paths]
        if len(set(hashes)) == 1:
            for p in paths:
                m = inv["absolute_path"].astype(str) == p
                rel = inv.loc[m, "repo_relative_path"].iloc[0]
                flagged.append(
                    {
                        "file_path": rel,
                        "class": inv.loc[m, "effective_label_name"].iloc[0],
                        "issue_type": "probable_byte_duplicate_candidate",
                        "severity": "medium",
                        "metric_or_threshold": "partial_sha256_match_same_size_duration_sr",
                        "recommended_action": "review_manually_deduplicate_one_copy",
                    }
                )
        else:
            collision_rows.append(
                {
                    "group_key": gkey,
                    "n_files_sharing_stat_signature": len(paths),
                    "note": "Same file_size+duration+SR but different partial hash; common for shared clip lengths, not necessarily duplicates.",
                }
            )


def add_dataset_level_flags(inv: pd.DataFrame, flagged: List[Dict[str, Any]]) -> None:
    """Flag non-dominant technical formats after the full inventory is available."""
    if inv.empty:
        return
    readable = inv[inv["readable"] == True].copy()  # noqa: E712
    if readable.empty:
        return

    def flag_non_dominant(column: str, issue_type: str, severity: str, action: str) -> None:
        if column not in readable.columns:
            return
        values = readable[column].dropna()
        if values.empty:
            return
        dominant = values.value_counts().idxmax()
        for _, row in readable[readable[column] != dominant].iterrows():
            flagged.append(
                {
                    "file_path": row["repo_relative_path"],
                    "class": row["effective_label_name"],
                    "issue_type": issue_type,
                    "severity": severity,
                    "metric_or_threshold": f"{column}={row[column]} differs from dominant {dominant}",
                    "recommended_action": action,
                }
            )

    flag_non_dominant(
        "sample_rate",
        "non_dominant_sample_rate",
        "low",
        "keep_if_preprocessing_resamples_but_verify_raw_consistency",
    )
    flag_non_dominant(
        "subtype",
        "non_dominant_audio_subtype",
        "low",
        "keep_if_readable_but_document_format_difference",
    )
    flag_non_dominant(
        "channels",
        "non_dominant_channel_count",
        "low",
        "keep_if_readable_but_document_channel_difference",
    )


def simulate_random_vs_speaker_leakage(inv: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """Simulate naive random clip split vs speaker-grouped split leakage."""
    rng = np.random.default_rng(seed)
    df = inv[(inv["readable"] == True) & (inv["effective_label_int"].isin([0, 1]))].copy()  # noqa: E712
    if df.empty:
        return {}
    n = len(df)
    perm = rng.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    splits = np.array(["test"] * n, dtype=object)
    splits[perm[:n_train]] = "train"
    splits[perm[n_train : n_train + n_val]] = "valid"
    splits[perm[n_train + n_val :]] = "test"
    df["_rand_split"] = splits

    # Speakers spanning multiple splits under random clip assignment
    sp = df.groupby("speaker_id_effective")["_rand_split"].nunique()
    multi_split_speakers = int((sp > 1).sum())
    total_speakers = int(df["speaker_id_effective"].nunique())

    # Speaker-level split simulation: assign each speaker to one split
    speakers = df["speaker_id_effective"].unique()
    rng2 = np.random.default_rng(seed + 1)
    sp_order = rng2.permutation(speakers)
    n_sp = len(speakers)
    train_n = max(1, int(0.70 * n_sp))
    valid_n = max(1, int(0.15 * n_sp))
    assign = {}
    for i, s in enumerate(sp_order):
        if i < train_n:
            assign[s] = "train"
        elif i < train_n + valid_n:
            assign[s] = "valid"
        else:
            assign[s] = "test"
    df["_sp_split"] = df["speaker_id_effective"].map(assign)
    sp_leak = df.groupby("speaker_id_effective")["_sp_split"].nunique()
    leak_after_speaker = int((sp_leak > 1).sum())

    return {
        "n_clips": n,
        "n_unique_speakers_effective": total_speakers,
        "random_clip_split_speakers_in_multiple_splits": multi_split_speakers,
        "speaker_level_split_speakers_in_multiple_splits": leak_after_speaker,
        "interpretation": "Naive random splitting assigns clips independently; many speakers have multiple clips, so the same speaker can appear in train and test. Speaker-grouped assignment forces 0 speaker leakage.",
    }


def plot_duration_histogram(inv: pd.DataFrame, out_dir: Path) -> None:
    df = inv[inv["readable"] == True].copy()  # noqa: E712
    if df.empty or "duration_sec" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(df["duration_sec"].clip(0, 180), bins=40, color="steelblue", edgecolor="black", alpha=0.85)
    plt.xlabel("Duration (s), clipped to 180s for display")
    plt.ylabel("Count")
    plt.title("Raw audio duration distribution (readable files)")
    plt.tight_layout()
    plt.savefig(out_dir / "duration_histogram_all.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label, color in zip(["dementia", "non_dementia"], ["coral", "seagreen"]):
        sub = df[df["effective_label_name"] == label]["duration_sec"].clip(0, 180)
        if len(sub) == 0:
            continue
        plt.hist(sub, bins=30, alpha=0.5, label=label, color=color, edgecolor="black")
    plt.xlabel("Duration (s), clipped to 180s for display")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Duration by effective label")
    plt.tight_layout()
    plt.savefig(out_dir / "duration_histogram_by_class.png", dpi=120)
    plt.close()


def write_markdown_report(
    out_dir: Path,
    inv: pd.DataFrame,
    class_csv: Path,
    duration_csv: Path,
    tech_csv: Path,
    flagged_csv: Path,
    speaker_csv: Path,
    summary_csv: Path,
    leakage: Dict[str, Any],
    artifact_counts: Dict[str, int],
) -> None:
    readable = inv[inv["readable"] == True]  # noqa: E712
    total = len(inv)
    n_read = int(readable.shape[0])
    n_bad = total - n_read

    def qseries(s: pd.Series) -> str:
        s = s.dropna()
        if s.empty:
            return "n/a"
        return (
            f"min={s.min():.3f}, p25={s.quantile(0.25):.3f}, median={s.median():.3f}, "
            f"p75={s.quantile(0.75):.3f}, max={s.max():.3f}, mean={s.mean():.3f}, std={s.std():.3f}"
        )

    # Imbalance
    c = readable["effective_label_name"].value_counts()
    n_dem = int(c.get("dementia", 0))
    n_nd = int(c.get("non_dementia", 0))
    minor = min(n_dem, n_nd)
    major = max(n_dem, n_nd)
    ratio = (major / minor) if minor > 0 else float("inf")
    if ratio < 1.5:
        imb = "mild"
    elif ratio < 3.0:
        imb = "moderate"
    else:
        imb = "severe"

    lines = [
        "# Phase 2 Dataset Audit Report",
        "",
        "**Scope:** `data/raw_audio` only (Phase 2 data audit; no modeling changes).",
        "",
        "## Label mapping (physical folder → effective ML label)",
        "",
        "| Physical folder under `data/raw_audio/` | Effective label for ML | Notes |",
        "|---|---|---|",
        "| `dementia` | `dementia` (positive class) | Matches pipeline `label_name` for dementia |",
        "| `nodementia` | `non_dementia` | Folder name `nodementia` is the project synonym for `non_dementia` (see `docs/DATA_ORGANIZATION.md`) |",
        "",
        "The inventory columns `raw_class_folder` and `effective_label_name` make this explicit for every file.",
        "",
        "## What was checked",
        "",
        "- File counts per physical folder and per effective label",
        "- Duration distribution (overall and per-class)",
        "- Sample rate, format, channel/subtype (via `soundfile`)",
        "- Raw tree sidecars and non-audio artifacts",
        "- Readability and zero-byte files",
        "- Heuristic flags: very short/long, mostly silent, low RMS, clipping, noise-like spectral flatness",
        "- Duplicate candidates (same size + duration + SR; partial SHA256 for confirmation)",
        "- Speaker grouping using `speaker_id` from Phase 1 metadata when matched, else normalized **speaker folder name** under the class folder (probabilistic identity — not ground-truth biometrics)",
        "- Language field from metadata when row exists",
        "",
        "## Heuristic thresholds (documented)",
        "",
        f"- Very short clip: duration < **{VERY_SHORT_SEC}s**",
        f"- Near-zero: duration ≤ **{NEAR_ZERO_SEC}s**",
        f"- Unusually long: duration > **{VERY_LONG_SEC}s** (segmentation may be needed)",
        f"- Mostly silent: `librosa.effects.split` non-speech ratio ≥ **{SILENCE_RATIO_HIGH}**",
        f"- Extremely low volume: RMS dB < **{RMS_DB_LOW}**",
        f"- Likely clipping: fraction of samples with |x|≥0.999 ≥ **{CLIP_FRAC_WARN}**",
        f"- Likely noise-like (heuristic): spectral flatness mean ≥ **{FLATNESS_NOISY}**",
        "",
        "## Key counts",
        "",
        f"- Total audio files scanned: **{total}**",
        f"- Readable: **{n_read}**; unreadable/broken: **{n_bad}**",
        f"- Physical files under raw tree: **{artifact_counts.get('physical_files_total', 0)}**",
        f"- AppleDouble sidecar files skipped: **{artifact_counts.get('appledouble_sidecar_files_skipped', 0)}**",
        f"- Non-audio non-sidecar artifacts skipped: **{artifact_counts.get('non_sidecar_non_audio_artifacts', 0)}**",
        f"- Per effective label (readable): dementia **{n_dem}**, non_dementia **{n_nd}**",
        f"- Class imbalance ratio (majority/minority): **{ratio:.2f}** ({imb})",
        "",
        "### Duration (readable files)",
        "",
        qseries(readable["duration_sec"]),
        "",
        "## Outputs (machine-readable)",
        "",
        f"- `{class_csv.relative_to(PROJECT_ROOT)}`",
        f"- `{duration_csv.relative_to(PROJECT_ROOT)}`",
        f"- `{tech_csv.relative_to(PROJECT_ROOT)}`",
        f"- `{flagged_csv.relative_to(PROJECT_ROOT)}`",
        f"- `{speaker_csv.relative_to(PROJECT_ROOT)}`",
        f"- `{summary_csv.relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'flagged_issue_counts.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'statistical_signature_collisions.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'language_distribution.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'channel_distribution.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'metadata_field_summary.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'manual_review_priority.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'raw_folder_label_mapping.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'raw_tree_artifact_summary.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'duration_outliers_flagged.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'master_inventory.csv').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'duration_histogram_all.png').resolve().relative_to(PROJECT_ROOT)}`",
        f"- `{Path(out_dir / 'duration_histogram_by_class.png').resolve().relative_to(PROJECT_ROOT)}`",
        "",
        "## Speaker identity and leakage risk",
        "",
        "- **Confirmed speaker key:** When a Phase 1 metadata row matches the file path, `metadata_speaker_id` is used.",
        "- **Inferred key:** Otherwise `speaker_id_inferred_from_path` = normalized folder name under the class directory (one folder per public figure / subject in this dataset layout).",
        "- **Uncertainty:** Multiple clips in the same folder are **same inferred speaker**; this is **not** independent verification of identity.",
        "",
    ]
    if leakage:
        lines.extend(
            [
                "### Simulation (seed=42, 70/15/15 clip partitions)",
                "",
                f"- Unique speakers (effective): **{leakage.get('n_unique_speakers_effective', 'n/a')}**",
                f"- Speakers appearing in **more than one** split under **random clip** assignment: **{leakage.get('random_clip_split_speakers_in_multiple_splits', 'n/a')}**",
                f"- Under **speaker-level** assignment: speakers in multiple splits: **{leakage.get('speaker_level_split_speakers_in_multiple_splits', 'n/a')}** (should be 0).",
                "",
            ]
        )

    lines.extend(
        [
            "## Final decision: train / validation / test strategy",
            "",
            "**Decision:** Use **speaker-level stratified** splitting (not i.i.d. random clips). Stratify by binary class (`dementia` vs `non_dementia`) so each split retains both classes.",
            "",
            "**Official Phase 3 rule:** All Phase 3 modeling must use speaker-level stratified splitting; naive random clip splitting is not permitted because the Phase 2 audit showed speaker leakage risk.",
            "",
            "**Why:** Many speakers contribute **multiple recordings**. Random clip splitting causes **speaker leakage**: the model can learn subject-specific traits and **inflate** validation/test scores. The current pipeline already uses corrected speaker-level split metadata (`metadata_fixed_splits.csv`).",
            "",
            "**Recommended procedure:**",
            "",
            "1. Define the speaker key as **`speaker_id_effective`** from this audit (metadata when present, else normalized speaker folder).",
            "2. Assign **entire speakers** to train/valid/test (e.g. 70/15/15) with **stratification** on class label derived from speaker majority label; if a speaker has mixed labels, **hold out for manual review** (rare).",
            "3. For **baselines** and **wav2vec**: use the **same speaker-grouped splits**; optionally add **group k-fold** by speaker for small-N stability instead of a single static split.",
            "",
            "**Naive random splitting:** Leakage is **likely** when multiple clips per speaker exist — see simulation above.",
            "",
            "**If speaker IDs were unavailable:** Safest practical alternative is **grouping by coarse identity proxies** (here: parent folder name under each class) and splitting by **group**, never by clip.",
            "",
            "## Language / accent metadata",
            "",
        ]
    )

    lines.append("- Full counts: see `language_distribution.csv` (empty string counted as `(empty)`).")
    lang_nonempty = readable["metadata_language"].fillna("").str.strip()
    lang_nonempty = lang_nonempty[lang_nonempty != ""]
    if lang_nonempty.empty:
        lines.append(
            "- No non-empty `language` values were found in joined metadata for readable files. **Limitation:** accent/language fairness analysis is not supported from current fields."
        )
    else:
        lines.append(lang_nonempty.value_counts().head(20).to_string())
    n_nested = int(inv["redundant_nested_nodementia_folder"].sum()) if "redundant_nested_nodementia_folder" in inv.columns else 0
    lines.extend(
        [
            "",
            "## Top data risks",
            "",
            "1. Speaker leakage if splitting by clip.",
            "2. Label noise from ambiguous metadata/audio matches (see `metadata_match_needs_review` and `metadata_audio_match_needs_review`).",
            "3. Duration and recording-condition mismatch across clips.",
            "4. Heuristic flags require manual spot-checks before exclusion.",
            f"5. **Parallel directory layout:** `{n_nested}` files live under `data/raw_audio/nodementia/nodementia/...` (duplicate class folder). Verify they are not unintended duplicates of files under `data/raw_audio/nodementia/<speaker>/`.",
            "",
        ]
    )

    lines.append("## Manual review priority")
    lines.append("")
    lines.append("1. Files flagged `unreadable_or_corrupt` or `empty_or_zero_byte`.")
    lines.append("2. Files flagged `likely_mostly_silent`, `likely_extremely_low_volume`, or `metadata_audio_match_needs_review`.")
    lines.append("3. Files flagged `probable_byte_duplicate_candidate` in `flagged_files.csv`.")
    lines.append("4. Low-severity format/path issues such as nested `nodementia/nodementia` and non-dominant sample rate/subtype.")
    lines.append("")
    lines.append("See `manual_review_priority.csv` for the sorted file-level review queue.")
    lines.append("")
    lines.append("## Pre-Phase-3 Data Decisions")
    lines.append("")
    lines.append(
        "Explicit handling decisions for metadata/audio review cases, low-volume files, duplicate candidates, nested `nodementia/nodementia` paths, and raw-format outliers are saved in `pre_phase3_data_decisions.md` and `pre_phase3_data_decisions.csv`."
    )
    lines.append("")
    lines.append("*Generated by `scripts/dataset_audit.py`.*")

    report_text = "\n".join(lines)
    (out_dir / "dataset_audit_report.md").write_text(report_text, encoding="utf-8")
    (out_dir / "phase2_dataset_audit_report.md").write_text(report_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 raw_audio dataset audit.")
    parser.add_argument(
        "--raw-audio-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw_audio",
        help="Root folder containing dementia/ and nodementia/",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "metadata" / "metadata_fixed_splits.csv",
        help="Prefer fixed splits metadata; fallback to metadata_clean.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset_audit",
        help="Output directory for CSVs and report",
    )
    args = parser.parse_args()

    raw_root = args.raw_audio_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = args.metadata_csv
    if not meta_path.exists():
        alt = PROJECT_ROOT / "data" / "metadata" / "metadata_clean.csv"
        meta_path = alt if alt.exists() else meta_path

    meta = load_metadata_index(meta_path)

    artifact_counts = count_raw_tree_artifacts(raw_root)
    inv, flagged = build_inventory(raw_root, meta)
    stat_collision_rows: List[Dict[str, Any]] = []
    add_duplicate_flags(inv, flagged, stat_collision_rows)
    add_dataset_level_flags(inv, flagged)

    inv_path = out_dir / "master_inventory.csv"
    inv.to_csv(inv_path, index=False)

    readable = inv[inv["readable"] == True].copy()  # noqa: E712

    # Class distribution
    total_files = len(inv)
    n_read = int(readable.shape[0])
    n_unread = total_files - n_read
    phys = inv["raw_class_folder"].value_counts().to_dict()
    eff = readable["effective_label_name"].value_counts().to_dict()
    c_dem = int(eff.get("dementia", 0))
    c_nd = int(eff.get("non_dementia", 0))
    minor = min(c_dem, c_nd) if c_dem and c_nd else 0
    major = max(c_dem, c_nd) if (c_dem or c_nd) else 0
    ratio = (major / minor) if minor else float("nan")
    if ratio < 1.5:
        imb = "mild"
    elif ratio < 3.0:
        imb = "moderate"
    else:
        imb = "severe"

    class_df = pd.DataFrame(
        [
            {"metric": "total_files", "value": total_files},
            {"metric": "readable_files", "value": n_read},
            {"metric": "unreadable_or_failed_files", "value": n_unread},
            {"metric": "physical_folder_dementia", "value": phys.get("dementia", 0)},
            {"metric": "physical_folder_nodementia", "value": phys.get("nodementia", 0)},
            {"metric": "effective_label_dementia", "value": c_dem},
            {"metric": "effective_label_non_dementia", "value": c_nd},
            {"metric": "pct_dementia", "value": round(100.0 * c_dem / n_read, 4) if n_read else 0},
            {"metric": "pct_non_dementia", "value": round(100.0 * c_nd / n_read, 4) if n_read else 0},
            {"metric": "imbalance_ratio_majority_over_minority", "value": round(ratio, 4)},
            {"metric": "imbalance_severity_label", "value": imb},
        ]
    )
    class_df.to_csv(out_dir / "class_distribution.csv", index=False)
    pd.DataFrame(
        [
            {"raw_class_folder": "dementia", "effective_label_name": "dementia", "effective_label_int": 1},
            {"raw_class_folder": "nodementia", "effective_label_name": "non_dementia", "effective_label_int": 0},
        ]
    ).to_csv(out_dir / "raw_folder_label_mapping.csv", index=False)
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in artifact_counts.items()]
    ).to_csv(out_dir / "raw_tree_artifact_summary.csv", index=False)

    # Duration summary
    dur_rows = []
    for name, sub in [("all", readable), ("dementia", readable[readable["effective_label_name"] == "dementia"]), ("non_dementia", readable[readable["effective_label_name"] == "non_dementia"])]:
        s = sub["duration_sec"].dropna()
        if s.empty:
            dur_rows.append({"subset": name, "n": 0})
            continue
        dur_rows.append(
            {
                "subset": name,
                "n": int(len(s)),
                "min_sec": float(s.min()),
                "p25_sec": float(s.quantile(0.25)),
                "median_sec": float(s.median()),
                "p75_sec": float(s.quantile(0.75)),
                "max_sec": float(s.max()),
                "mean_sec": float(s.mean()),
                "std_sec": float(s.std()),
            }
        )
    pd.DataFrame(dur_rows).to_csv(out_dir / "duration_summary.csv", index=False)

    # Duration outliers list
    outlier_rows = []
    for _, r in readable.iterrows():
        d = r["duration_sec"]
        reasons = []
        if d <= NEAR_ZERO_SEC:
            reasons.append("near_zero_duration")
        elif d < VERY_SHORT_SEC:
            reasons.append("very_short")
        if d > VERY_LONG_SEC:
            reasons.append("very_long")
        if reasons:
            outlier_rows.append(
                {
                    "repo_relative_path": r["repo_relative_path"],
                    "effective_label_name": r["effective_label_name"],
                    "duration_sec": d,
                    "outlier_flags": ";".join(reasons),
                }
            )
    pd.DataFrame(outlier_rows).to_csv(out_dir / "duration_outliers_flagged.csv", index=False)

    # Technical summary
    tech_rows = []
    for label, sub in [("all", readable), ("dementia", readable[readable["effective_label_name"] == "dementia"]), ("non_dementia", readable[readable["effective_label_name"] == "non_dementia"])]:
        if sub.empty:
            continue
        tech_rows.append(
            {
                "subset": label,
                "distinct_sample_rates": json.dumps(sub["sample_rate"].value_counts().to_dict()),
                "distinct_extensions": json.dumps(sub["file_extension"].value_counts().to_dict()),
                "distinct_formats": json.dumps(sub["format"].fillna("").value_counts().head(15).to_dict()),
                "distinct_subtypes": json.dumps(sub["subtype"].fillna("").value_counts().head(15).to_dict()),
                "distinct_channels": json.dumps(sub["channels"].fillna("").value_counts().head(15).to_dict()),
            }
        )
    pd.DataFrame(tech_rows).to_csv(out_dir / "technical_summary.csv", index=False)

    # Per-sample-rate counts (global)
    sr_counts = readable["sample_rate"].value_counts().sort_index()
    sr_df = pd.DataFrame({"sample_rate_hz": sr_counts.index.astype(int), "file_count": sr_counts.values})
    sr_df.to_csv(out_dir / "sample_rate_distribution.csv", index=False)

    ext_counts = readable["file_extension"].value_counts()
    pd.DataFrame({"extension": ext_counts.index, "file_count": ext_counts.values}).to_csv(
        out_dir / "format_distribution.csv", index=False
    )
    channel_counts = readable["channels"].value_counts().sort_index()
    pd.DataFrame({"channels": channel_counts.index.astype(int), "file_count": channel_counts.values}).to_csv(
        out_dir / "channel_distribution.csv", index=False
    )

    flagged_df = pd.DataFrame(flagged)
    if not flagged_df.empty:
        flagged_df = flagged_df.drop_duplicates()
    flagged_df.to_csv(out_dir / "flagged_files.csv", index=False)
    if not flagged_df.empty:
        flagged_df.groupby("issue_type").size().reset_index(name="count").to_csv(
            out_dir / "flagged_issue_counts.csv", index=False
        )
    else:
        pd.DataFrame(columns=["issue_type", "count"]).to_csv(out_dir / "flagged_issue_counts.csv", index=False)

    pd.DataFrame(stat_collision_rows).to_csv(out_dir / "statistical_signature_collisions.csv", index=False)

    if not flagged_df.empty:
        severity_rank = {"high": 1, "medium": 2, "low": 3}
        issue_rank = {
            "unreadable_or_corrupt": 1,
            "empty_or_zero_byte": 1,
            "metadata_audio_match_needs_review": 2,
            "likely_mostly_silent": 2,
            "likely_extremely_low_volume": 2,
            "probable_byte_duplicate_candidate": 3,
        }
        manual_df = flagged_df.copy()
        manual_df["severity_rank"] = manual_df["severity"].map(severity_rank).fillna(9).astype(int)
        manual_df["issue_rank"] = manual_df["issue_type"].map(issue_rank).fillna(5).astype(int)
        manual_df = manual_df.sort_values(["severity_rank", "issue_rank", "file_path", "issue_type"])
        manual_df.to_csv(out_dir / "manual_review_priority.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "file_path",
                "class",
                "issue_type",
                "severity",
                "metric_or_threshold",
                "recommended_action",
                "severity_rank",
                "issue_rank",
            ]
        ).to_csv(out_dir / "manual_review_priority.csv", index=False)

    # Speaker analysis
    sp = (
        readable.groupby("speaker_id_effective")
        .agg(
            n_files=("repo_relative_path", "count"),
            n_dementia=("effective_label_name", lambda x: int((x == "dementia").sum())),
            n_non_dementia=("effective_label_name", lambda x: int((x == "non_dementia").sum())),
        )
        .reset_index()
    )
    sp = sp.sort_values("n_files", ascending=False)
    sp.to_csv(out_dir / "speaker_analysis.csv", index=False)

    # Per-speaker class purity
    sp["mixed_label_speaker"] = (sp["n_dementia"] > 0) & (sp["n_non_dementia"] > 0)
    mixed_speakers = int(sp["mixed_label_speaker"].sum())

    leakage = simulate_random_vs_speaker_leakage(inv)

    # Dataset summary sheet (key-value)
    summary_kv = [
        ("physical_files_under_raw_audio", artifact_counts["physical_files_total"]),
        ("appledouble_sidecar_files_skipped", artifact_counts["appledouble_sidecar_files_skipped"]),
        ("appledouble_audio_sidecars_skipped", artifact_counts["appledouble_audio_sidecars_skipped"]),
        ("non_sidecar_non_audio_artifacts_skipped", artifact_counts["non_sidecar_non_audio_artifacts"]),
        ("total_files_scanned", total_files),
        ("readable_files", n_read),
        ("unreadable_files", n_unread),
        ("effective_dementia_count", c_dem),
        ("effective_non_dementia_count", c_nd),
        ("imbalance_ratio", round(ratio, 4)),
        ("imbalance_severity", imb),
        ("unique_speakers_effective", int(readable["speaker_id_effective"].nunique())),
        ("speakers_with_mixed_labels", mixed_speakers),
        ("metadata_rows_matched", int(inv["metadata_row_found"].sum()) if "metadata_row_found" in inv.columns else 0),
        ("flagged_issue_rows_total", len(flagged_df)),
        ("random_clip_split_speakers_spanning_splits", leakage.get("random_clip_split_speakers_in_multiple_splits", "")),
        ("speaker_split_speakers_spanning_splits", leakage.get("speaker_level_split_speakers_in_multiple_splits", "")),
        (
            "files_under_nodementia_nodementia_nested_path",
            int(readable["redundant_nested_nodementia_folder"].sum())
            if "redundant_nested_nodementia_folder" in readable.columns
            else 0,
        ),
    ]
    pd.DataFrame(summary_kv, columns=["metric", "value"]).to_csv(out_dir / "dataset_summary.csv", index=False)

    lang_series = readable["metadata_language"].fillna("").astype(str).str.strip()
    lang_series = lang_series.replace("", "(empty)")
    lang_series.value_counts().rename_axis("language_field").reset_index(name="file_count").to_csv(
        out_dir / "language_distribution.csv", index=False
    )
    metadata_summary_rows = []
    for field in ["metadata_language", "metadata_gender", "metadata_ethnicity"]:
        if field not in readable.columns:
            continue
        counts = readable[field].fillna("").astype(str).str.strip().replace("", "(empty)").value_counts()
        for value, count in counts.items():
            metadata_summary_rows.append({"metadata_field": field, "value": value, "file_count": int(count)})
    pd.DataFrame(metadata_summary_rows).to_csv(out_dir / "metadata_field_summary.csv", index=False)

    plot_duration_histogram(inv, out_dir)

    write_markdown_report(
        out_dir,
        inv,
        out_dir / "class_distribution.csv",
        out_dir / "duration_summary.csv",
        out_dir / "technical_summary.csv",
        out_dir / "flagged_files.csv",
        out_dir / "speaker_analysis.csv",
        out_dir / "dataset_summary.csv",
        leakage,
        artifact_counts,
    )

    split_doc = [
        "# Train / validation / test split strategy (Phase 2 decision)",
        "",
        "## Decision",
        "",
        "**Use speaker-level stratified splitting.** Do **not** use IID random clip splitting for final evaluation.",
        "",
        "**Official Phase 3 rule:** All Phase 3 modeling must use speaker-level stratified splitting; naive random clip splitting is not permitted because the Phase 2 audit showed speaker leakage risk.",
        "",
        "## Evidence from this audit",
        "",
        f"- Unique speakers (`speaker_id_effective`): **{int(readable['speaker_id_effective'].nunique())}**",
        f"- Clips: **{n_read}**",
        f"- Under simulated random **clip** assignment (70/15/15, seed=42), speakers appearing in **more than one** split: **{leakage.get('random_clip_split_speakers_in_multiple_splits', 'n/a')}**",
        f"- Under **speaker-grouped** assignment: speakers spanning splits: **{leakage.get('speaker_level_split_speakers_in_multiple_splits', 'n/a')}**",
        "",
        "## Concrete procedure",
        "",
        "1. Define groups by `speaker_id_effective` from `master_inventory.csv` (metadata `speaker_id` when the path matches Phase 1 metadata; else normalized speaker subfolder name under `dementia/` or `nodementia/`).",
        "2. Assign **entire speakers** to train/valid/test (e.g. 70/15/15) with **stratification** on the speaker's class (here: all speakers are single-class; `speakers_with_mixed_labels=0`).",
        "3. Use the **same** speaker splits for **classical baselines** and **wav2vec**.",
        "4. Optionally use **GroupKFold** or repeated speaker-stratified splits **by speaker** instead of a single static split when sample size is small.",
        "",
        "## If speaker IDs were unknown",
        "",
        "Split by **stable folder proxy** (e.g. one folder per subject under each class), never by individual clips.",
        "",
        f"Full narrative: `{Path('reports/dataset_audit/phase2_dataset_audit_report.md')}`",
        "",
        f"Pre-Phase-3 data handling decisions: `{Path('reports/dataset_audit/pre_phase3_data_decisions.md')}`",
        "",
        "*Generated by `scripts/dataset_audit.py`.*",
        "",
    ]
    (out_dir / "split_strategy_recommendation.md").write_text("\n".join(split_doc), encoding="utf-8")

    print(f"Wrote audit outputs to: {out_dir}")


if __name__ == "__main__":
    main()
