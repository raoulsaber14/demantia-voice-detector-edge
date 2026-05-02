"""Audio preprocessing for edge inference.

Bridges your signal-processing pipeline output (cleaned 16 kHz mono float32
ndarray) and the model's input tensor. This module:

  * resamples if (and only if) input sample rate is wrong
  * pads or truncates to the model's expected length
  * casts to the configured dtype and adds a batch dim if needed

It does NOT do trimming, denoising, normalization, or VAD — those happen
upstream in audio_pipeline.py. Doing them again here would double-process and
potentially conflict with what the model was trained on.

Resource notes (Pi 4, 4 GB):
  * scipy.signal.resample_poly: ~1–3 ms for 10 s of audio at 16 kHz
  * memory: a 10 s float32 buffer is 640 KB; trivial
"""

from __future__ import annotations

from math import gcd
from typing import Optional, Tuple

import numpy as np


# Only used if the caller actually hands us audio at a non-16k rate.
# Imported lazily because scipy import is ~50 ms on a Pi 4.
def _resample(y: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return y
    from scipy.signal import resample_poly  # local import

    factor = gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr) // factor
    down = int(src_sr) // factor
    return resample_poly(y, up, down).astype(np.float32, copy=False)


def _validate_audio(audio: np.ndarray) -> np.ndarray:
    """Coerce to 1-D float32 and reject obviously bad input."""
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"audio must be np.ndarray, got {type(audio).__name__}")

    if audio.size == 0:
        raise ValueError("audio is empty")

    # Stereo -> mono by averaging. Your pipeline should already deliver mono;
    # this is a defensive fallback so we don't silently feed garbage to the model.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {audio.shape}")

    if audio.dtype != np.float32:
        # int16 mic capture is the most likely incoming dtype.
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            scale = float(max(abs(info.min), info.max))
            audio = (audio.astype(np.float32) / scale)
        else:
            audio = audio.astype(np.float32, copy=False)

    if not np.all(np.isfinite(audio)):
        # Replace NaN/Inf with zero rather than failing — matches teammate's
        # load_audio_safe behavior in src/audio_utils.py.
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Hard cap on amplitude. If your pipeline normalized correctly this is a no-op.
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak

    return audio


def _pad_or_truncate(
    audio: np.ndarray,
    target_len: int,
    pad_mode: str,
    truncate_mode: str,
) -> np.ndarray:
    """Bring 1-D audio to exactly target_len samples."""
    n = audio.shape[0]
    if n == target_len:
        return audio

    if n < target_len:
        deficit = target_len - n
        if pad_mode == "zero":
            return np.pad(audio, (0, deficit), mode="constant", constant_values=0.0)
        if pad_mode == "reflect":
            # np.pad reflect can't extend beyond the signal length; fall back to zero
            # for very short inputs.
            if deficit > n - 1 and n > 1:
                return np.pad(audio, (0, deficit), mode="constant", constant_values=0.0)
            return np.pad(audio, (0, deficit), mode="reflect")
        raise ValueError(f"unknown pad_mode {pad_mode!r}")

    # n > target_len -> truncate
    if truncate_mode == "head":
        return audio[:target_len]
    if truncate_mode == "center":
        start = (n - target_len) // 2
        return audio[start : start + target_len]
    if truncate_mode == "loudest":
        return _loudest_window(audio, target_len)
    raise ValueError(f"unknown truncate_mode {truncate_mode!r}")


def _loudest_window(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Pick the contiguous window of target_len with the highest RMS.

    Uses a 1 s coarse stride to keep this cheap on a Pi 4 — exact maximum
    isn't needed; we just want to avoid silent regions.
    """
    n = audio.shape[0]
    stride = max(1, target_len // 10)  # 10 candidate windows
    best_start = 0
    best_energy = -1.0
    for start in range(0, n - target_len + 1, stride):
        window = audio[start : start + target_len]
        energy = float(np.mean(window * window))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return audio[best_start : best_start + target_len]


def prepare_input(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    target_samples: Optional[int],
    pad_mode: str = "zero",
    truncate_mode: str = "center",
    dtype: str = "float32",
    add_batch_dim: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Convert a waveform to a model-ready tensor.

    Parameters
    ----------
    audio:
        1-D or 2-D ndarray. Range roughly [-1, 1] for float, or any int range.
    sample_rate:
        Sample rate of `audio`. If it differs from target_sample_rate the
        signal is resampled with scipy.signal.resample_poly.
    target_sample_rate:
        Model's required sample rate (16000).
    target_samples:
        Number of samples the model expects per inference.
    pad_mode, truncate_mode:
        See EdgeConfig.
    dtype:
        Output tensor dtype.
    add_batch_dim:
        If True, prepend a leading axis of size 1.

    Returns
    -------
    tensor:
        ndarray ready to hand to the model backend.
    info:
        Dict with diagnostics (input_len_samples, was_resampled, was_padded,
        was_truncated). Useful for logging on edge.
    """
    info: dict = {
        "input_len_samples": int(audio.shape[-1]) if audio.ndim >= 1 else 0,
        "input_sample_rate": int(sample_rate),
        "was_resampled": False,
        "was_padded": False,
        "was_truncated": False,
    }

    audio = _validate_audio(audio)

    if sample_rate != target_sample_rate:
        audio = _resample(audio, sample_rate, target_sample_rate)
        info["was_resampled"] = True

    if target_samples is not None:
        n = audio.shape[0]
        if n < target_samples:
            info["was_padded"] = True
        elif n > target_samples:
            info["was_truncated"] = True
        audio = _pad_or_truncate(audio, target_samples, pad_mode, truncate_mode)

    if dtype == "float32":
        tensor = audio.astype(np.float32, copy=False)
    elif dtype == "int8":
        tensor = np.clip(audio * 127.0, -128, 127).astype(np.int8)
    elif dtype == "uint8":
        tensor = np.clip((audio + 1.0) * 127.5, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"unsupported dtype {dtype!r}")

    if add_batch_dim:
        tensor = tensor[np.newaxis, :]

    return tensor, info
