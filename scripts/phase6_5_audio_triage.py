"""Automatic audio triage for Phase 6.5 unresolved metadata/audio rows.

This script produces evidence for manual review. It does not change
adjudication decisions, manifests, thresholds, or model artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
DEFAULT_ADJUDICATION_CSV = ROOT / "reports/tables/phase6_5_uncertain_row_adjudication.csv"
DEFAULT_RESOLUTION_LOG_CSV = ROOT / "reports/tables/ambiguous_match_resolution_log.csv"
DEFAULT_PAIRWISE_CSV = ROOT / "reports/tables/phase6_5_audio_triage_pairwise.csv"
DEFAULT_ROW_SUMMARY_CSV = ROOT / "reports/tables/phase6_5_audio_triage_row_summary.csv"
DEFAULT_REPORT_MD = ROOT / "reports/phase6_5_audio_triage_summary.md"
DEFAULT_MANUAL_OBSERVATIONS_CSV = ROOT / "reports/tables/phase6_5_audio_triage_manual_observations.csv"

ANALYSIS_SAMPLE_RATE = 16000
# Triage must be fast enough to run repeatedly. Header duration is still read
# from the full file; waveform-derived usefulness/similarity uses a bounded
# deterministic leading window.
MAX_ANALYSIS_DURATION_SEC = 30.0
AUDIO_EXTENSIONS = {".wav", ".wave", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif"}


@dataclass(frozen=True)
class AudioStats:
    original_path: str
    resolved_path: str
    basename: str
    is_sidecar: bool
    exists: bool
    extension: str
    file_size_bytes: float | None
    duration_sec: float | None
    sample_rate: float | None
    channels: float | None
    load_error: str
    rms: float | None
    rms_db: float | None
    silence_ratio: float | None
    speech_ratio: float | None
    peak_abs: float | None
    mfcc_vector: tuple[float, ...] | None
    spectral_vector: tuple[float, ...] | None
    waveform_preview: tuple[float, ...] | None
    sha256_16mb: str
    usefulness: str
    usefulness_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 6.5 unresolved-row audio triage artifacts."
    )
    parser.add_argument("--adjudication-csv", type=Path, default=DEFAULT_ADJUDICATION_CSV)
    parser.add_argument("--resolution-log-csv", type=Path, default=DEFAULT_RESOLUTION_LOG_CSV)
    parser.add_argument("--pairwise-csv", type=Path, default=DEFAULT_PAIRWISE_CSV)
    parser.add_argument("--row-summary-csv", type=Path, default=DEFAULT_ROW_SUMMARY_CSV)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--manual-observations-csv", type=Path, default=DEFAULT_MANUAL_OBSERVATIONS_CSV)
    return parser.parse_args()


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def is_sidecar_path(path_text: str) -> bool:
    return Path(str(path_text)).name.startswith("._")


def resolve_audio_path(path_text: str) -> Path:
    """Resolve absolute, repo-relative, and current-directory-relative paths."""
    text = str(path_text).strip()
    if not text:
        return ROOT / "__missing_audio_path__"
    path = Path(text)
    if path.is_absolute():
        return path
    direct = ROOT / path
    if direct.exists():
        return direct
    # Some historic logs may omit the project folder but keep data/... suffixes.
    data_pos = text.find("data/")
    if data_pos >= 0:
        candidate = ROOT / text[data_pos:]
        if candidate.exists():
            return candidate
    return direct


def split_candidate_list(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split("|")]
    return [p for p in parts if p and not is_sidecar_path(p)]


def extract_note_value(notes: Any, key: str, next_keys: list[str]) -> str:
    if notes is None or pd.isna(notes):
        return ""
    text = str(notes)
    start = text.find(f"{key}=")
    if start < 0:
        return ""
    start += len(key) + 1
    end = len(text)
    for next_key in next_keys:
        marker = f"; {next_key}="
        pos = text.find(marker, start)
        if pos >= 0:
            end = min(end, pos)
    return text[start:end].strip()


def parse_score_summary_rank(score_summary: Any, candidate_path: str) -> int | None:
    if score_summary is None or pd.isna(score_summary):
        return None
    target = Path(str(candidate_path)).name
    entries = [p.strip() for p in str(score_summary).split(";") if p.strip()]
    for rank, entry in enumerate(entries, start=1):
        if entry.endswith(target) or target in entry:
            return rank
    return None


def load_optional_audio_libs() -> tuple[Any | None, Any | None, str]:
    messages: list[str] = []
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        sf = None
        messages.append(f"soundfile unavailable: {exc}")
    try:
        import librosa  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        librosa = None
        messages.append(f"librosa unavailable: {exc}")
    return sf, librosa, "; ".join(messages)


def sha256_prefix(path: Path, max_bytes: int = 16 * 1024 * 1024) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    remaining = max_bytes
    try:
        with path.open("rb") as f:
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def cosine_similarity(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float | None:
    if a is None or b is None:
        return None
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    if va.shape != vb.shape or va.size == 0:
        return None
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0 or not np.isfinite(denom):
        return None
    return float(np.dot(va, vb) / denom)


def waveform_correlation(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float | None:
    if a is None or b is None:
        return None
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    n = min(len(va), len(vb))
    if n < 100:
        return None
    va = va[:n] - float(np.mean(va[:n]))
    vb = vb[:n] - float(np.mean(vb[:n]))
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0 or not np.isfinite(denom):
        return None
    return float(np.dot(va, vb) / denom)


def usefulness_label(
    exists: bool,
    is_sidecar: bool,
    load_error: str,
    duration_sec: float | None,
    rms: float | None,
    rms_db: float | None,
    silence_ratio: float | None,
) -> tuple[str, float]:
    if is_sidecar:
        return "ignored_sidecar", 0.0
    if not exists:
        return "missing", 0.0
    if load_error:
        return "corrupt_or_unreadable", 0.2
    if duration_sec is not None and duration_sec < 2.0:
        return "too_short", 0.5
    if rms is not None and rms <= 1e-5:
        return "likely_unusable_too_quiet", 0.6
    if rms_db is not None and rms_db <= -70.0:
        return "likely_unusable_too_quiet", 0.6
    if silence_ratio is not None and silence_ratio >= 0.98:
        return "likely_unusable_mostly_silent", 0.6
    score = 2.0
    if silence_ratio is not None:
        score += max(0.0, min(1.0, 1.0 - silence_ratio))
    if duration_sec is not None:
        score += min(0.6, max(0.0, duration_sec / 120.0) * 0.6)
    if rms_db is not None:
        score += min(0.4, max(0.0, (rms_db + 55.0) / 35.0) * 0.4)
    if (silence_ratio is not None and silence_ratio >= 0.85) or (rms_db is not None and rms_db <= -55.0):
        return "low_usefulness", score
    return "usable", score + 0.5


def analyze_audio(path_text: str, sf: Any | None, librosa: Any | None) -> AudioStats:
    resolved = resolve_audio_path(path_text)
    is_sidecar = is_sidecar_path(path_text)
    exists = resolved.exists() and resolved.is_file() and not is_sidecar
    ext = resolved.suffix.lower()
    file_size = float(resolved.stat().st_size) if exists else None
    duration = sample_rate = channels = None
    load_error = ""
    rms = rms_db = silence_ratio = speech_ratio = peak_abs = None
    mfcc_vector: tuple[float, ...] | None = None
    spectral_vector: tuple[float, ...] | None = None
    waveform_preview: tuple[float, ...] | None = None

    if not exists:
        load_error = "sidecar_ignored" if is_sidecar else "missing_file"
    elif ext not in AUDIO_EXTENSIONS:
        load_error = f"unsupported_extension:{ext}"
    elif sf is None or librosa is None:
        load_error = "audio_libraries_unavailable"
    else:
        try:
            info = sf.info(str(resolved))
            duration = float(info.duration)
            sample_rate = float(info.samplerate)
            channels = float(info.channels)
        except Exception as exc:
            load_error = f"header_read_failed:{exc}"
        try:
            y, sr = librosa.load(
                str(resolved),
                sr=ANALYSIS_SAMPLE_RATE,
                mono=True,
                duration=MAX_ANALYSIS_DURATION_SEC,
            )
            y = np.asarray(y, dtype=np.float32)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if y.size == 0:
                load_error = "empty_audio"
            else:
                rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))
                rms_db = float(20.0 * math.log10(max(rms, 1e-12)))
                peak_abs = float(np.max(np.abs(y)))
                try:
                    nonsilent = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=1024)
                    nonsilent_samples = sum(int(end - start) for start, end in nonsilent)
                    silence_ratio = float(1.0 - (nonsilent_samples / len(y)))
                    speech_ratio = float(1.0 - silence_ratio)
                except Exception:
                    silence_ratio = None
                    speech_ratio = None
                try:
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=1024)
                    mfcc_vector = tuple(
                        np.concatenate([np.nanmean(mfcc, axis=1), np.nanstd(mfcc, axis=1)]).astype(float)
                    )
                except Exception:
                    mfcc_vector = None
                try:
                    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=1024).ravel()
                    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=1024).ravel()
                    rolloff = librosa.feature.spectral_rolloff(
                        y=y, sr=sr, roll_percent=0.85, hop_length=1024
                    ).ravel()
                    flatness = librosa.feature.spectral_flatness(y=y, hop_length=1024).ravel()
                    spectral_vector = tuple(
                        np.asarray(
                            [
                                np.log1p(np.nanmean(centroid)),
                                np.log1p(np.nanstd(centroid)),
                                np.log1p(np.nanmean(bandwidth)),
                                np.log1p(np.nanstd(bandwidth)),
                                np.log1p(np.nanmean(rolloff)),
                                np.log1p(np.nanstd(rolloff)),
                                np.nanmean(flatness),
                                np.nanstd(flatness),
                            ],
                            dtype=float,
                        )
                    )
                except Exception:
                    spectral_vector = None
                # A coarse preview is enough to catch exact/near-exact overlaps.
                preview_count = min(4000, len(y))
                if preview_count >= 100:
                    idx = np.linspace(0, len(y) - 1, preview_count).astype(int)
                    waveform_preview = tuple(y[idx].astype(float))
        except Exception as exc:
            if not load_error:
                load_error = f"audio_load_failed:{exc}"

    usefulness, usefulness_score = usefulness_label(
        exists=exists,
        is_sidecar=is_sidecar,
        load_error=load_error,
        duration_sec=duration,
        rms=rms,
        rms_db=rms_db,
        silence_ratio=silence_ratio,
    )
    return AudioStats(
        original_path=str(path_text),
        resolved_path=str(resolved),
        basename=resolved.name,
        is_sidecar=is_sidecar,
        exists=exists,
        extension=ext,
        file_size_bytes=file_size,
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=channels,
        load_error=load_error,
        rms=rms,
        rms_db=rms_db,
        silence_ratio=silence_ratio,
        speech_ratio=speech_ratio,
        peak_abs=peak_abs,
        mfcc_vector=mfcc_vector,
        spectral_vector=spectral_vector,
        waveform_preview=waveform_preview,
        sha256_16mb=sha256_prefix(resolved) if exists else "",
        usefulness=usefulness,
        usefulness_score=usefulness_score,
    )


def normalized_abs_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return None
    denom = max(abs(a), abs(b), 1.0)
    return float(abs(a - b) / denom)


def pair_triage(selected: AudioStats, candidate: AudioStats) -> tuple[str, str, str]:
    dur_diff = None
    if selected.duration_sec is not None and candidate.duration_sec is not None:
        dur_diff = abs(selected.duration_sec - candidate.duration_sec)
    size_diff_norm = normalized_abs_diff(selected.file_size_bytes, candidate.file_size_bytes)
    mfcc_sim = cosine_similarity(selected.mfcc_vector, candidate.mfcc_vector)
    spectral_sim = cosine_similarity(selected.spectral_vector, candidate.spectral_vector)
    wav_corr = waveform_correlation(selected.waveform_preview, candidate.waveform_preview)

    selected_bad = selected.usefulness.startswith(("missing", "corrupt", "too_short", "likely_unusable"))
    candidate_bad = candidate.usefulness.startswith(("missing", "corrupt", "too_short", "likely_unusable"))
    if selected_bad or candidate_bad:
        return (
            "likely_unusable",
            "medium" if selected.exists or candidate.exists else "high",
            f"usefulness selected={selected.usefulness}, candidate={candidate.usefulness}",
        )
    if selected.resolved_path == candidate.resolved_path or (
        selected.sha256_16mb and selected.sha256_16mb == candidate.sha256_16mb
    ):
        return ("likely_exact_duplicate", "high", "same resolved path or matching content hash prefix")
    if (
        dur_diff is not None
        and dur_diff <= 0.50
        and (size_diff_norm is not None and size_diff_norm <= 0.03)
        and (
            (wav_corr is not None and wav_corr >= 0.995)
            or (
                mfcc_sim is not None
                and spectral_sim is not None
                and mfcc_sim >= 0.995
                and spectral_sim >= 0.995
            )
        )
    ):
        return ("likely_near_duplicate", "high", "near-identical duration/size with very high audio similarity")
    if selected.resolved_path and candidate.resolved_path:
        same_folder = Path(selected.resolved_path).parent == Path(candidate.resolved_path).parent
    else:
        same_folder = False
    if same_folder and dur_diff is not None and dur_diff >= 1.0 and (
        (mfcc_sim is not None and mfcc_sim >= 0.90) or (spectral_sim is not None and spectral_sim >= 0.95)
    ):
        return ("likely_same_source_different_segment", "medium", "same folder/speaker with usable but non-identical audio")
    if dur_diff is not None and dur_diff >= 5.0 and (
        (mfcc_sim is not None and mfcc_sim < 0.95) or (spectral_sim is not None and spectral_sim < 0.98)
    ):
        return ("likely_distinct", "medium", "duration and audio-summary signals differ")
    return ("ambiguous_manual_review_required", "low", "signals are weak or conflicting")


def candidate_records_for_row(row: pd.Series, resolution_by_id: dict[str, pd.Series]) -> dict[str, Any]:
    row_id = str(row.get("row_id", "")).strip()
    log_row = resolution_by_id.get(row_id)
    selected = ""
    candidates: list[str] = []
    candidate_count = None
    rule_used = ""
    confidence = ""
    match_status = ""
    score_summary = ""

    if log_row is not None:
        selected = str(log_row.get("selected_audio_file", "")).strip()
        candidates = split_candidate_list(log_row.get("candidate_audio_files", ""))
        candidate_count = log_row.get("candidate_count", None)
        rule_used = str(log_row.get("rule_used", "")).strip()
        confidence = str(log_row.get("confidence", "")).strip()
        match_status = str(log_row.get("match_status", "")).strip()
        score_summary = str(log_row.get("score_summary", "")).strip()

    notes = row.get("notes", "")
    if not selected:
        selected = extract_note_value(notes, "selected_audio_file", ["candidate_audio_files", "score_summary"])
    if not candidates:
        candidate_text = extract_note_value(notes, "candidate_audio_files", ["score_summary", "required_manual_action"])
        candidates = split_candidate_list(candidate_text)
    if candidate_count is None or pd.isna(candidate_count):
        count_text = extract_note_value(notes, "candidate_count", ["rule_used"])
        try:
            candidate_count = int(count_text)
        except Exception:
            candidate_count = len(candidates)
    if not rule_used:
        rule_used = extract_note_value(notes, "rule_used", ["confidence"])
    if not confidence:
        confidence = extract_note_value(notes, "confidence", ["needs_review_in_match_log"])
    if not match_status:
        match_status = extract_note_value(notes, "match_status", ["match_rank_by_score_summary"])
    if not score_summary:
        score_summary = extract_note_value(notes, "score_summary", ["required_manual_action"])
    if not selected:
        selected = str(row.get("audio_path", "")).strip()

    # Keep order while removing sidecars and exact duplicate candidate strings.
    seen: set[str] = set()
    real_candidates: list[str] = []
    sidecar_ignored = 0
    for candidate in candidates:
        if is_sidecar_path(candidate):
            sidecar_ignored += 1
            continue
        key = str(resolve_audio_path(candidate))
        if key not in seen:
            seen.add(key)
            real_candidates.append(candidate)

    return {
        "row_id": row_id,
        "selected_audio_file": selected,
        "candidate_audio_files": real_candidates,
        "candidate_count_reported": candidate_count,
        "candidate_count_real": len(real_candidates),
        "sidecar_ignored_count": sidecar_ignored,
        "rule_used": rule_used,
        "confidence": confidence,
        "match_status": match_status,
        "score_summary": score_summary,
    }


def build_row_summary(row_id: str, selected_path: str, candidates: list[str], pair_rows: list[dict[str, Any]], stats_cache: dict[str, AudioStats]) -> dict[str, Any]:
    selected_stats = stats_cache[str(resolve_audio_path(selected_path))]
    nonself_pairs = [
        p for p in pair_rows
        if p["row_id"] == row_id and p["selected_resolved_path"] != p["candidate_resolved_path"]
    ]
    candidate_stats = [stats_cache[str(resolve_audio_path(c))] for c in candidates if str(resolve_audio_path(c)) in stats_cache]
    best = max(candidate_stats, key=lambda s: s.usefulness_score, default=selected_stats)
    exact_or_near = [
        p for p in nonself_pairs
        if p["triage_flag"] in {"likely_exact_duplicate", "likely_near_duplicate"}
    ]
    ambiguous_pairs = [
        p for p in nonself_pairs
        if p["triage_flag"] == "ambiguous_manual_review_required"
    ]
    unusable_pairs = [p for p in pair_rows if p["row_id"] == row_id and p["triage_flag"] == "likely_unusable"]
    distinct_pairs = [
        p for p in nonself_pairs
        if p["triage_flag"] in {"likely_distinct", "likely_same_source_different_segment"}
    ]

    selected_unusable = selected_stats.usefulness.startswith(("missing", "corrupt", "too_short", "likely_unusable"))
    usable_nonself = [
        p for p in nonself_pairs
        if not str(p.get("candidate_usefulness", "")).startswith(("missing", "corrupt", "too_short", "likely_unusable"))
    ]

    if selected_unusable and not usable_nonself:
        status = "likely_unusable_case"
        action = "exclude_if_no_new_evidence"
        reason = "selected file and available candidates are missing, unreadable, too short, or likely unusable"
    elif selected_unusable and usable_nonself:
        status = "candidate_preferred"
        action = "listen_selected_and_top_candidate"
        reason = "selected file has low usefulness, while at least one candidate appears more usable"
    elif exact_or_near and not distinct_pairs and not ambiguous_pairs:
        status = "likely_duplicate_case"
        action = "listen_selected_only"
        reason = "non-selected candidates appear exact or near duplicates of selected audio"
    elif best.resolved_path != selected_stats.resolved_path and best.usefulness_score >= selected_stats.usefulness_score + 0.75:
        status = "candidate_preferred"
        action = "listen_selected_and_top_candidate"
        reason = "a candidate has a materially stronger automated usefulness score"
    elif (
        not nonself_pairs
        and not distinct_pairs
        and not ambiguous_pairs
        and not unusable_pairs
        and selected_stats.usefulness in {"usable", "low_usefulness"}
    ):
        status = "selected_likely_ok"
        action = "listen_selected_only"
        reason = "selected file is usable and no non-selected candidate produced a strong contrary signal"
    else:
        status = "ambiguous_manual_review_required"
        action = "listen_all_candidates" if len(candidates) > 2 else "listen_selected_and_top_candidate"
        reason = "multiple usable/distinct candidates or weak/conflicting automated signals"

    notes = [
        f"selected_usefulness={selected_stats.usefulness}",
        f"candidate_count_real={len(candidates)}",
        f"exact_or_near_nonself_pairs={len(exact_or_near)}",
        f"distinct_nonself_pairs={len(distinct_pairs)}",
        f"ambiguous_nonself_pairs={len(ambiguous_pairs)}",
    ]
    return {
        "row_id": row_id,
        "selected_audio_file": selected_path,
        "candidate_count_real": len(candidates),
        "best_candidate_audio_file": best.original_path,
        "best_candidate_reason": reason,
        "row_triage_status": status,
        "selected_file_usefulness": selected_stats.usefulness,
        "best_candidate_usefulness": best.usefulness,
        "recommended_manual_action": action,
        "notes": "; ".join(notes),
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    show = df.where(pd.notna(df), "")
    lines = [
        "| " + " | ".join(str(c) for c in show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in show.columns) + " |")
    return "\n".join(lines)


def append_note(existing: Any, addition: str) -> str:
    text = "" if existing is None or pd.isna(existing) else str(existing).strip()
    return f"{text}; {addition}" if text else addition


def apply_manual_observations(
    pair_df: pd.DataFrame,
    row_df: pd.DataFrame,
    observation_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Apply user-provided listening observations as triage overrides only."""
    if not observation_path.exists():
        return pair_df, row_df, 0
    obs = pd.read_csv(observation_path)
    if obs.empty or "row_id" not in obs.columns or "observation" not in obs.columns:
        return pair_df, row_df, 0

    applied = 0
    pair_df = pair_df.copy()
    row_df = row_df.copy()
    for _, obs_row in obs.iterrows():
        row_id = str(obs_row.get("row_id", "")).strip()
        observation = str(obs_row.get("observation", "")).strip()
        source = str(obs_row.get("source", "")).strip()
        observation_date = str(obs_row.get("observation_date", "")).strip()
        if not row_id or not observation:
            continue
        addition = (
            f"manual_listening_observation={observation}; "
            f"manual_observation_source={source}; manual_observation_date={observation_date}"
        )

        pair_mask = pair_df["row_id"].astype(str).eq(row_id) & ~pair_df["candidate_is_selected_file"].astype(bool)
        if pair_mask.any() and "different_audio_not_identical" in observation:
            pair_df.loc[pair_mask, "triage_flag"] = "likely_distinct"
            pair_df.loc[pair_mask, "triage_confidence"] = "high"
            pair_df.loc[pair_mask, "notes"] = pair_df.loc[pair_mask, "notes"].map(
                lambda value: append_note(value, addition)
            )

        row_mask = row_df["row_id"].astype(str).eq(row_id)
        if row_mask.any() and "different_audio_not_identical" in observation:
            row_df.loc[row_mask, "row_triage_status"] = "ambiguous_manual_review_required"
            row_df.loc[row_mask, "recommended_manual_action"] = "manual_review_required"
            if "all_candidate_audio_files_are_real" in observation:
                reason = (
                    "Project owner reports candidate audio files are real and non-duplicate; "
                    "source-reference/governance adjudication is still required before final dataset decisions."
                )
            else:
                reason = (
                    "Manual evidence reports candidate audio differs from selected and is not an exact duplicate; "
                    "source-reference/governance adjudication is still required before final dataset decisions."
                )
            row_df.loc[row_mask, "best_candidate_reason"] = reason
            row_df.loc[row_mask, "notes"] = row_df.loc[row_mask, "notes"].map(
                lambda value: append_note(value, addition)
            )
        applied += 1
    return pair_df, row_df, applied


def write_report(
    report_path: Path,
    adjudication_path: Path,
    resolution_path: Path,
    unresolved_count: int,
    pair_df: pd.DataFrame,
    row_df: pd.DataFrame,
    library_note: str,
    parsed_candidate_rows: int,
    sidecar_ignored: int,
    manual_observation_path: Path,
    manual_observation_count: int,
) -> None:
    status_counts = row_df["row_triage_status"].value_counts().rename_axis("row_triage_status").reset_index(name="rows")
    action_counts = row_df["recommended_manual_action"].value_counts().rename_axis("recommended_manual_action").reset_index(name="rows")
    pair_counts = pair_df["triage_flag"].value_counts().rename_axis("triage_flag").reset_index(name="pairs")
    still_manual = row_df[
        row_df["recommended_manual_action"].isin(
            ["listen_all_candidates", "manual_review_required", "listen_selected_and_top_candidate"]
        )
    ].copy()
    easy = row_df[
        row_df["row_triage_status"].isin(["likely_duplicate_case", "selected_likely_ok", "likely_unusable_case"])
    ].copy()
    easy = easy[["row_id", "row_triage_status", "recommended_manual_action", "best_candidate_reason"]].head(12)
    manual_show = still_manual[["row_id", "row_triage_status", "recommended_manual_action", "best_candidate_reason"]].head(50)

    lines = [
        "# Phase 6.5 Audio Triage Summary",
        "",
        "## Inputs used",
        "",
        f"- Adjudication CSV: `{adjudication_path.relative_to(ROOT)}`.",
        f"- Ambiguous-match resolution log: `{resolution_path.relative_to(ROOT)}`.",
        f"- Unresolved adjudication rows processed: **{unresolved_count}**.",
        f"- Rows with parseable candidate lists: **{parsed_candidate_rows}**.",
        f"- Mac sidecar candidate files ignored: **{sidecar_ignored}**.",
        f"- Audio library status: `{library_note or 'soundfile and librosa available'}`.",
        f"- Manual observation CSV: `{manual_observation_path.relative_to(ROOT)}`.",
        f"- Manual observations applied: **{manual_observation_count}**.",
        "",
        "## Heuristics used",
        "",
        "- Sidecar files whose basename starts with `._` are ignored and never treated as real audio candidates.",
        "- File checks use existence, extension, byte size, duration, sample rate, channel count, and a SHA-256 prefix hash.",
        "- Usefulness checks use RMS, RMS dB, peak amplitude, and `librosa.effects.split` silence ratio.",
        "- Similarity checks use duration difference, normalized file-size difference, MFCC summary cosine similarity, spectral-summary cosine similarity, and coarse waveform-preview correlation.",
        f"- Full-file duration is read from audio headers; waveform-derived usefulness and similarity features use a bounded leading **{MAX_ANALYSIS_DURATION_SEC:.0f}-second** analysis window for reproducibility and runtime stability.",
        "- Pair flags are conservative: exact/near duplicate requires matching path/hash or very close duration/size with very high similarity; weak or conflicting signals become `ambiguous_manual_review_required`.",
        "",
        "## Intentionally not automated",
        "",
        "- No final `keep`, `exclude`, `relabel`, or `speaker_id_fix` decisions are written.",
        "- No manifests are changed.",
        "- No Phase 4 models, thresholds, calibration, or demo behavior are touched.",
        "- Speaker identity and exact metadata-to-audio correctness still require source-reference/governance adjudication before final dataset decisions.",
        "- Manual observations in `phase6_5_audio_triage_manual_observations.csv` may override duplicate-style automated triage labels, but they still do not create final keep/exclude decisions.",
        "",
        "## Row-level triage counts",
        "",
        markdown_table(status_counts),
        "",
        "## Pair-level triage counts",
        "",
        markdown_table(pair_counts),
        "",
        "## Recommended manual-action counts",
        "",
        markdown_table(action_counts),
        "",
        "## Rows still requiring governance/source-reference adjudication",
        "",
        markdown_table(manual_show),
        "",
        "## Easiest rows to resolve first",
        "",
        "Rows in this table have the clearest automated triage signal. They still require human confirmation before adjudication changes.",
        "",
        markdown_table(easy),
        "",
        "## Output tables",
        "",
        "- `reports/tables/phase6_5_audio_triage_pairwise.csv` contains one selected-vs-candidate comparison per real candidate.",
        "- `reports/tables/phase6_5_audio_triage_row_summary.csv` contains one triage summary per unresolved adjudication row.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.adjudication_csv.exists():
        raise FileNotFoundError(f"Adjudication CSV not found: {args.adjudication_csv}")
    adj = pd.read_csv(args.adjudication_csv)
    if "review_status" not in adj.columns or "decision" not in adj.columns:
        raise ValueError("Adjudication CSV must include review_status and decision columns.")
    unresolved = adj[
        adj["decision"].astype(str).eq("needs_more_review")
        | adj["review_status"].astype(str).isin(["needs_more_info", "cannot_resolve"])
    ].copy()
    if unresolved.empty:
        raise ValueError("No unresolved rows found in adjudication CSV.")

    resolution_by_id: dict[str, pd.Series] = {}
    if args.resolution_log_csv.exists():
        log = pd.read_csv(args.resolution_log_csv)
        if "metadata_index" in log.columns:
            for _, log_row in log.iterrows():
                resolution_by_id[str(log_row["metadata_index"]).strip()] = log_row

    sf, librosa, library_note = load_optional_audio_libs()
    stats_cache: dict[str, AudioStats] = {}
    pair_rows: list[dict[str, Any]] = []
    row_inputs: list[dict[str, Any]] = []
    parsed_candidate_rows = 0
    sidecar_ignored = 0

    for _, row in unresolved.iterrows():
        record = candidate_records_for_row(row, resolution_by_id)
        row_inputs.append(record)
        selected = record["selected_audio_file"]
        candidates = record["candidate_audio_files"]
        if candidates:
            parsed_candidate_rows += 1
        sidecar_ignored += int(record["sidecar_ignored_count"])
        all_paths = [selected] + candidates
        for audio_path in all_paths:
            resolved_key = str(resolve_audio_path(audio_path))
            if resolved_key not in stats_cache:
                stats_cache[resolved_key] = analyze_audio(audio_path, sf, librosa)

        selected_stats = stats_cache[str(resolve_audio_path(selected))]
        for candidate in candidates:
            candidate_stats = stats_cache[str(resolve_audio_path(candidate))]
            flag, confidence, flag_note = pair_triage(selected_stats, candidate_stats)
            duration_abs_diff = None
            if selected_stats.duration_sec is not None and candidate_stats.duration_sec is not None:
                duration_abs_diff = abs(selected_stats.duration_sec - candidate_stats.duration_sec)
            selected_size = selected_stats.file_size_bytes
            candidate_size = candidate_stats.file_size_bytes
            size_ratio = None
            if selected_size and selected_size > 0 and candidate_size is not None:
                size_ratio = float(candidate_size / selected_size)
            row_notes = [
                flag_note,
                f"rule_used={record['rule_used']}",
                f"match_confidence={record['confidence']}",
                f"match_status={record['match_status']}",
            ]
            rank = parse_score_summary_rank(record.get("score_summary", ""), candidate)
            if rank is not None:
                row_notes.append(f"candidate_rank_in_score_summary={rank}")
            pair_rows.append(
                {
                    "row_id": record["row_id"],
                    "selected_audio_file": selected,
                    "candidate_audio_file": candidate,
                    "selected_resolved_path": selected_stats.resolved_path,
                    "candidate_resolved_path": candidate_stats.resolved_path,
                    "candidate_is_selected_file": selected_stats.resolved_path == candidate_stats.resolved_path,
                    "selected_exists": selected_stats.exists,
                    "candidate_exists": candidate_stats.exists,
                    "selected_extension": selected_stats.extension,
                    "candidate_extension": candidate_stats.extension,
                    "selected_duration_sec": selected_stats.duration_sec,
                    "candidate_duration_sec": candidate_stats.duration_sec,
                    "duration_abs_diff_sec": duration_abs_diff,
                    "selected_sample_rate": selected_stats.sample_rate,
                    "candidate_sample_rate": candidate_stats.sample_rate,
                    "selected_channels": selected_stats.channels,
                    "candidate_channels": candidate_stats.channels,
                    "selected_file_size_bytes": selected_stats.file_size_bytes,
                    "candidate_file_size_bytes": candidate_stats.file_size_bytes,
                    "size_ratio": size_ratio,
                    "selected_silence_ratio": selected_stats.silence_ratio,
                    "candidate_silence_ratio": candidate_stats.silence_ratio,
                    "selected_speech_ratio": selected_stats.speech_ratio,
                    "candidate_speech_ratio": candidate_stats.speech_ratio,
                    "selected_rms": selected_stats.rms,
                    "candidate_rms": candidate_stats.rms,
                    "selected_rms_db": selected_stats.rms_db,
                    "candidate_rms_db": candidate_stats.rms_db,
                    "selected_usefulness": selected_stats.usefulness,
                    "candidate_usefulness": candidate_stats.usefulness,
                    "mfcc_similarity": cosine_similarity(selected_stats.mfcc_vector, candidate_stats.mfcc_vector),
                    "spectral_similarity": cosine_similarity(selected_stats.spectral_vector, candidate_stats.spectral_vector),
                    "waveform_correlation": waveform_correlation(selected_stats.waveform_preview, candidate_stats.waveform_preview),
                    "content_hash_prefix_match": bool(
                        selected_stats.sha256_16mb
                        and selected_stats.sha256_16mb == candidate_stats.sha256_16mb
                    ),
                    "triage_flag": flag,
                    "triage_confidence": confidence,
                    "notes": "; ".join(row_notes),
                }
            )

    row_summary_rows = [
        build_row_summary(
            row_id=record["row_id"],
            selected_path=record["selected_audio_file"],
            candidates=record["candidate_audio_files"],
            pair_rows=pair_rows,
            stats_cache=stats_cache,
        )
        for record in row_inputs
    ]
    pair_df = pd.DataFrame(pair_rows)
    row_df = pd.DataFrame(row_summary_rows)
    pair_df, row_df, manual_observation_count = apply_manual_observations(
        pair_df=pair_df,
        row_df=row_df,
        observation_path=args.manual_observations_csv,
    )
    args.pairwise_csv.parent.mkdir(parents=True, exist_ok=True)
    args.row_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(args.pairwise_csv, index=False)
    row_df.to_csv(args.row_summary_csv, index=False)
    write_report(
        report_path=args.report_md,
        adjudication_path=args.adjudication_csv,
        resolution_path=args.resolution_log_csv,
        unresolved_count=len(unresolved),
        pair_df=pair_df,
        row_df=row_df,
        library_note=library_note,
        parsed_candidate_rows=parsed_candidate_rows,
        sidecar_ignored=sidecar_ignored,
        manual_observation_path=args.manual_observations_csv,
        manual_observation_count=manual_observation_count,
    )
    print(f"Processed unresolved rows: {len(unresolved)}")
    print(f"Pairwise comparisons: {len(pair_df)}")
    print(f"Row summary counts: {row_df['row_triage_status'].value_counts().to_dict()}")
    print(f"Wrote: {args.pairwise_csv}")
    print(f"Wrote: {args.row_summary_csv}")
    print(f"Wrote: {args.report_md}")


if __name__ == "__main__":
    main()
