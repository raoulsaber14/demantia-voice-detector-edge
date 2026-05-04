"""Audio loading, preprocessing, normalization, and save utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.io import wavfile
import soundfile as sf

from src.config import CFG


def _pcm_to_float32(data: np.ndarray) -> np.ndarray:
    """Convert common PCM WAV arrays to float32 in approximately [-1, 1]."""
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    if data.dtype == np.uint8:
        return ((data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        max_abs = float(max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max))
        return (data.astype(np.float32) / max_abs).astype(np.float32)
    return data.astype(np.float32)


def load_audio_safe(path: Path, target_sr: int = CFG.target_sample_rate) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """Safely load an audio file and return waveform, sample-rate, error."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        y = data.mean(axis=1) if getattr(data, "ndim", 1) == 2 else data
        if y is None or len(y) == 0:
            return None, None, "empty_audio"
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if target_sr and sr != target_sr:
            factor = gcd(int(sr), int(target_sr))
            y = resample_poly(y, int(target_sr) // factor, int(sr) // factor)
            sr = target_sr
        return y.astype(np.float32), sr, None
    except Exception as sf_exc:
        try:
            sr, data = wavfile.read(str(path))
            y = _pcm_to_float32(np.asarray(data))
            y = y.mean(axis=1) if getattr(y, "ndim", 1) == 2 else y
            if y is None or len(y) == 0:
                return None, None, "empty_audio"
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if target_sr and sr != target_sr:
                factor = gcd(int(sr), int(target_sr))
                y = resample_poly(y, int(target_sr) // factor, int(sr) // factor)
                sr = target_sr
            return y.astype(np.float32), sr, None
        except Exception as wavfile_exc:
            try:
                import librosa

                y, sr = librosa.load(path, sr=target_sr, mono=True)
                if y is None or len(y) == 0:
                    return None, None, "empty_audio"
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                return y.astype(np.float32), sr, None
            except Exception as librosa_exc:
                return None, None, f"soundfile={sf_exc}; wavfile={wavfile_exc}; librosa={librosa_exc}"


def remove_dc_offset(y: np.ndarray) -> np.ndarray:
    """Remove constant DC offset without changing dynamic shape."""
    if len(y) == 0:
        return y
    return (y - float(np.mean(y))).astype(np.float32)


def highpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 50.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a conservative high-pass filter to reduce DC/low-frequency rumble."""
    if len(y) == 0 or sr <= 0 or cutoff_hz <= 0 or cutoff_hz >= sr / 2:
        return y.astype(np.float32)
    try:
        sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
        return sosfiltfilt(sos, y).astype(np.float32)
    except Exception:
        return y.astype(np.float32)


def rms_dbfs(y: np.ndarray, eps: float = 1e-9) -> float:
    """Return RMS level in dBFS for float audio where full-scale is 1.0."""
    if len(y) == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))
    return float(20.0 * np.log10(max(rms, eps)))


def normalize_amplitude(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Peak-normalize audio while preserving shape."""
    max_abs = np.max(np.abs(y)) if len(y) else 0.0
    if max_abs <= 0:
        return y
    return (y / max_abs * peak).astype(np.float32)


def normalize_loudness(
    y: np.ndarray,
    target_rms_dbfs: float = -20.0,
    max_gain_db: float = 12.0,
    peak_limit: float = 0.99,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """RMS-normalize with a gain cap so very quiet/noisy clips are not over-amplified."""
    if len(y) == 0:
        return y.astype(np.float32), {
            "rms_before_dbfs": float("nan"),
            "rms_after_dbfs": float("nan"),
            "normalization_gain_db": 0.0,
            "normalization_clipped": False,
        }

    before = rms_dbfs(y)
    if not np.isfinite(before):
        return y.astype(np.float32), {
            "rms_before_dbfs": before,
            "rms_after_dbfs": before,
            "normalization_gain_db": 0.0,
            "normalization_clipped": False,
        }

    desired_gain_db = target_rms_dbfs - before
    gain_db = float(np.clip(desired_gain_db, -max_gain_db, max_gain_db))
    gain = 10.0 ** (gain_db / 20.0)
    y_norm = y.astype(np.float32) * gain
    peak = float(np.max(np.abs(y_norm))) if len(y_norm) else 0.0
    clipped = peak > peak_limit
    if clipped and peak > 0:
        y_norm = y_norm / peak * peak_limit
    return y_norm.astype(np.float32), {
        "rms_before_dbfs": before,
        "rms_after_dbfs": rms_dbfs(y_norm),
        "normalization_gain_db": gain_db,
        "normalization_clipped": bool(clipped),
    }


def energy_intervals(
    y: np.ndarray,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Return sample intervals whose frame RMS is within top_db of the file peak RMS."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return np.empty((0, 2), dtype=int)

    if len(y) <= frame_length:
        starts = np.array([0], dtype=int)
    else:
        starts = np.arange(0, len(y) - frame_length + 1, hop_length, dtype=int)
        last_start = max(0, len(y) - frame_length)
        if starts[-1] != last_start:
            starts = np.append(starts, last_start)

    rms_values = np.empty(len(starts), dtype=np.float32)
    for idx, start in enumerate(starts):
        frame = y[start : min(start + frame_length, len(y))]
        rms_values[idx] = np.sqrt(np.mean(np.square(frame, dtype=np.float64))) if len(frame) else 0.0

    peak_rms = float(np.max(rms_values)) if len(rms_values) else 0.0
    if peak_rms <= 1e-9:
        return np.empty((0, 2), dtype=int)

    threshold = peak_rms * (10.0 ** (-float(top_db) / 20.0))
    active_idx = np.flatnonzero(rms_values >= threshold)
    if active_idx.size == 0:
        return np.empty((0, 2), dtype=int)

    groups = np.split(active_idx, np.where(np.diff(active_idx) > 1)[0] + 1)
    intervals = []
    for group in groups:
        start = int(starts[group[0]])
        end = int(min(len(y), starts[group[-1]] + frame_length))
        if end > start:
            intervals.append((start, end))
    return np.asarray(intervals, dtype=int)


def trim_silence(y: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading/trailing silence. Returns original on failure."""
    try:
        intervals = energy_intervals(y, top_db=top_db)
        if intervals.size == 0:
            return y
        start = int(intervals[0, 0])
        end = int(intervals[-1, 1])
        return y[start:end] if end > start else y
    except Exception:
        return y


def save_audio(path: Path, y: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Save waveform as standardized WAV to target path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)
