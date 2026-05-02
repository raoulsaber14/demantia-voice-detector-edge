"""
Audio Capture and Signal Processing Pipeline for Dementia Voice Screening
Target: Raspberry Pi 4 (1-8GB RAM variants)
Output: Cleaned 16kHz mono WAV ready for model inference

Resource estimates (Pi 4, 4GB):
  - Audio buffer (30s @ 16kHz float32): ~1.9 MB
  - FFT workspace (1024-point): ~8 KB
  - Spectral subtraction overhead: ~3 MB peak
  - Total pipeline peak memory: ~8-10 MB
  - Processing time for 30s audio: ~1.5-3s (all four stages)

IMPORTANT: This system produces screening indicators, NOT medical diagnoses.
All outputs are intended to support early pre-assessment and clinician referral.
"""

import logging
import time
import struct
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, medfilt
from scipy.io import wavfile

# Lazy import: sounddevice requires PortAudio which may not be present
# during testing or on headless systems. Only capture_audio() needs it.
sd = None

def _get_sounddevice():
    global sd
    if sd is None:
        import sounddevice as _sd
        sd = _sd
    return sd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("audio_pipeline")


class PipelineStage(Enum):
    RAW = "raw"
    FILTERED = "filtered"
    DENOISED = "denoised"
    VAD_TRIMMED = "vad_trimmed"
    NORMALIZED = "normalized"


@dataclass
class PipelineConfig:
    """All tunables in one place.  Change here, not in the code."""

    # -- Audio capture --
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"  # sounddevice dtype
    record_duration_s: float = 30.0
    device_index: Optional[int] = None  # None = system default

    # -- Bandpass filter --
    bp_low_hz: float = 80.0
    bp_high_hz: float = 8000.0
    bp_order: int = 5  # 5th-order Butterworth; good rolloff without ringing

    # -- Spectral subtraction noise reduction --
    noise_estimate_duration_s: float = 0.5  # first N seconds treated as noise profile
    n_fft: int = 1024  # 64 ms @ 16 kHz — good frequency resolution for speech
    hop_length: int = 256  # 16 ms hop — 75% overlap
    oversubtraction: float = 2.0  # alpha: aggressiveness (1.0-4.0)
    spectral_floor: float = 0.02  # beta: prevents musical noise artifacts

    # -- VAD (energy-based, enhanced) --
    vad_frame_ms: int = 30  # frame size in ms
    vad_energy_threshold_db: float = -35.0  # relative to peak; tune per environment
    vad_zcr_threshold: float = 0.3  # zero-crossing rate upper bound (speech < noise)
    vad_min_speech_ms: int = 200  # discard speech chunks shorter than this
    vad_padding_ms: int = 150  # keep this much context around speech segments
    vad_hangover_frames: int = 8  # keep N frames after energy drops (catches word tails)

    # -- Normalization --
    target_peak_db: float = -3.0  # peak normalize to this level
    target_rms_db: float = -20.0  # optional RMS normalization (set None to skip)

    # -- Clipping detection --
    clipping_threshold: float = 0.99  # samples above this fraction of max are clipped
    clipping_warn_ratio: float = 0.001  # warn if >0.1% of samples are clipped

    # -- Output --
    output_dir: str = "."
    output_prefix: str = "recording"


# ---------------------------------------------------------------------------
# Audio Capture
# ---------------------------------------------------------------------------

class MicDisconnectedError(Exception):
    """Raised when the microphone is disconnected during recording."""
    pass


class SilenceOnlyError(Exception):
    """Raised when the recording contains no detectable speech."""
    pass


def list_audio_devices() -> str:
    """Print available audio devices for debugging mic issues."""
    return _get_sounddevice().query_devices()


def capture_audio(config: PipelineConfig) -> np.ndarray:
    """
    Record audio from USB microphone.

    Returns:
        np.ndarray: shape (num_samples,) float32 in [-1, 1]

    Raises:
        MicDisconnectedError: if device disappears mid-recording
        PortAudioError: on hardware-level failures
    """
    _sd = _get_sounddevice()
    num_samples = int(config.sample_rate * config.record_duration_s)
    logger.info(
        f"Starting capture: {config.record_duration_s}s @ {config.sample_rate}Hz, "
        f"device={config.device_index or 'default'}"
    )

    try:
        # Verify device exists before recording
        if config.device_index is not None:
            dev_info = _sd.query_devices(config.device_index)
            if dev_info["max_input_channels"] < 1:
                raise MicDisconnectedError(
                    f"Device {config.device_index} has no input channels"
                )

        audio = _sd.rec(
            frames=num_samples,
            samplerate=config.sample_rate,
            channels=config.channels,
            dtype=config.dtype,
            device=config.device_index,
            blocking=True,
        )
    except _sd.PortAudioError as e:
        error_str = str(e).lower()
        if "device" in error_str or "unavailable" in error_str:
            raise MicDisconnectedError(f"Microphone error during capture: {e}") from e
        raise

    # Flatten to 1D
    audio = audio.flatten()

    # Sanity check: all zeros means mic is probably muted or broken
    if np.max(np.abs(audio)) < 1e-7:
        logger.warning("Captured audio is near-silent (max amplitude < 1e-7)")

    logger.info(
        f"Capture complete: {len(audio)} samples, "
        f"peak={np.max(np.abs(audio)):.4f}, "
        f"RMS={np.sqrt(np.mean(audio**2)):.4f}"
    )
    return audio


# ---------------------------------------------------------------------------
# Signal Processing: Bandpass Filter
# ---------------------------------------------------------------------------

def bandpass_filter(audio: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """
    5th-order Butterworth bandpass: keeps 80 Hz – 8 kHz.

    Uses sosfiltfilt (zero-phase) so it doesn't shift speech timing.
    CPU cost on Pi 4: ~50ms for 30s audio.
    """
    nyquist = config.sample_rate / 2.0
    low = config.bp_low_hz / nyquist
    high = config.bp_high_hz / nyquist

    # Clamp to valid range
    low = max(low, 1e-5)
    high = min(high, 1.0 - 1e-5)

    if low >= high:
        logger.warning(
            f"Invalid bandpass range ({config.bp_low_hz}-{config.bp_high_hz} Hz), "
            f"skipping filter"
        )
        return audio.copy()

    sos = butter(config.bp_order, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, audio).astype(np.float32)
    logger.info(
        f"Bandpass filter applied: {config.bp_low_hz}-{config.bp_high_hz} Hz, "
        f"order={config.bp_order}"
    )
    return filtered


# ---------------------------------------------------------------------------
# Signal Processing: Spectral Subtraction Noise Reduction
# ---------------------------------------------------------------------------

def estimate_snr_db(audio: np.ndarray, config: PipelineConfig) -> float:
    """
    Estimate SNR by comparing RMS of the noise segment vs. full signal.
    Rough but useful for before/after comparison.
    """
    noise_samples = int(config.noise_estimate_duration_s * config.sample_rate)
    noise_samples = min(noise_samples, len(audio))

    noise_rms = np.sqrt(np.mean(audio[:noise_samples] ** 2) + 1e-10)
    signal_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)

    if noise_rms < 1e-10:
        return 60.0  # effectively clean

    snr = 20.0 * np.log10(signal_rms / noise_rms)
    return float(snr)


def spectral_subtraction(audio: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """
    Spectral subtraction with oversubtraction factor and spectral floor.

    Algorithm:
        1. Estimate noise spectrum from first noise_estimate_duration_s
        2. For each frame: |Y| - alpha * |N|, floored at beta * |Y|
        3. Reconstruct with original phase

    CPU cost on Pi 4: ~200-400ms for 30s audio.
    Memory: ~3 MB peak (FFT buffers).
    """
    n_fft = config.n_fft
    hop = config.hop_length
    alpha = config.oversubtraction
    beta = config.spectral_floor

    # Estimate noise spectrum from initial segment
    noise_samples = int(config.noise_estimate_duration_s * config.sample_rate)
    noise_samples = max(noise_samples, n_fft)  # need at least one full frame
    noise_samples = min(noise_samples, len(audio))
    noise_segment = audio[:noise_samples]

    # Windowed noise frames
    window = np.hanning(n_fft).astype(np.float32)
    num_noise_frames = max(1, (len(noise_segment) - n_fft) // hop + 1)

    noise_power = np.zeros(n_fft // 2 + 1, dtype=np.float32)
    for i in range(num_noise_frames):
        start = i * hop
        frame = noise_segment[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spectrum = np.fft.rfft(frame * window)
        noise_power += np.abs(spectrum) ** 2
    noise_power /= num_noise_frames
    noise_mag = np.sqrt(noise_power)

    # Process full signal frame by frame
    num_frames = max(1, (len(audio) - n_fft) // hop + 1)
    output_length = (num_frames - 1) * hop + n_fft
    output = np.zeros(output_length, dtype=np.float32)
    window_sum = np.zeros(output_length, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop
        frame = audio[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))

        spectrum = np.fft.rfft(frame * window)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Subtract noise with floor
        subtracted_mag = mag - alpha * noise_mag
        floor_mag = beta * mag
        clean_mag = np.maximum(subtracted_mag, floor_mag)

        # Reconstruct
        clean_spectrum = clean_mag * np.exp(1j * phase)
        clean_frame = np.fft.irfft(clean_spectrum, n=n_fft).astype(np.float32)

        output[start : start + n_fft] += clean_frame * window
        window_sum[start : start + n_fft] += window ** 2

    # Normalize by window overlap
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    # Trim to original length
    output = output[: len(audio)]

    logger.info(
        f"Spectral subtraction applied: n_fft={n_fft}, alpha={alpha}, beta={beta}, "
        f"noise_ref={config.noise_estimate_duration_s}s"
    )
    return output


# ---------------------------------------------------------------------------
# Signal Processing: Voice Activity Detection (Energy + ZCR)
# ---------------------------------------------------------------------------

def _compute_rms_db(frame: np.ndarray) -> float:
    """RMS in dB relative to full scale."""
    rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
    return 20.0 * np.log10(rms + 1e-10)


def _compute_zcr(frame: np.ndarray) -> float:
    """Zero-crossing rate: fraction of adjacent samples that cross zero."""
    signs = np.sign(frame)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return crossings / (len(frame) - 1) if len(frame) > 1 else 0.0


def voice_activity_detection(
    audio: np.ndarray, config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhanced energy-based VAD with zero-crossing rate and hangover logic.

    Returns:
        speech_audio: np.ndarray — concatenated speech segments
        vad_mask: np.ndarray — boolean mask at frame level (for diagnostics)

    CPU cost on Pi 4: ~30-60ms for 30s audio.
    """
    frame_size = int(config.sample_rate * config.vad_frame_ms / 1000)
    num_frames = len(audio) // frame_size

    if num_frames == 0:
        logger.warning("Audio too short for VAD framing")
        return audio.copy(), np.array([True])

    # Compute per-frame features
    peak_db = 20.0 * np.log10(np.max(np.abs(audio)) + 1e-10)
    energy_threshold = peak_db + config.vad_energy_threshold_db

    frame_is_speech = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        start = i * frame_size
        frame = audio[start : start + frame_size]

        rms_db = _compute_rms_db(frame)
        zcr = _compute_zcr(frame)

        # Speech: above energy threshold AND not too high ZCR (noise is high-ZCR)
        if rms_db > energy_threshold and zcr < config.vad_zcr_threshold:
            frame_is_speech[i] = True

    # Hangover: keep N frames after speech drops (catches word-final consonants)
    hangover_mask = frame_is_speech.copy()
    hangover_counter = 0
    for i in range(num_frames):
        if frame_is_speech[i]:
            hangover_counter = config.vad_hangover_frames
        elif hangover_counter > 0:
            hangover_mask[i] = True
            hangover_counter -= 1

    # Median filter to smooth isolated frames (remove 1-frame spikes/gaps)
    if num_frames >= 3:
        smoothed = medfilt(hangover_mask.astype(np.float64), kernel_size=5) > 0.5
    else:
        smoothed = hangover_mask

    # Add padding around speech regions
    pad_frames = int(config.vad_padding_ms / config.vad_frame_ms)
    padded = smoothed.copy()
    for i in range(num_frames):
        if smoothed[i]:
            start_pad = max(0, i - pad_frames)
            end_pad = min(num_frames, i + pad_frames + 1)
            padded[start_pad:end_pad] = True

    # Remove short speech chunks (likely noise bursts)
    min_frames = int(config.vad_min_speech_ms / config.vad_frame_ms)
    final_mask = np.zeros(num_frames, dtype=bool)
    in_speech = False
    speech_start = 0

    for i in range(num_frames):
        if padded[i] and not in_speech:
            speech_start = i
            in_speech = True
        elif not padded[i] and in_speech:
            if (i - speech_start) >= min_frames:
                final_mask[speech_start:i] = True
            in_speech = False

    # Handle trailing speech
    if in_speech and (num_frames - speech_start) >= min_frames:
        final_mask[speech_start:num_frames] = True

    # Extract speech samples
    speech_segments = []
    for i in range(num_frames):
        if final_mask[i]:
            start = i * frame_size
            end = start + frame_size
            speech_segments.append(audio[start:end])

    if len(speech_segments) == 0:
        speech_frames = sum(final_mask)
        logger.warning(
            f"VAD found no speech (0/{num_frames} frames passed). "
            f"Peak={peak_db:.1f} dB, threshold={energy_threshold:.1f} dB"
        )
        raise SilenceOnlyError(
            "No speech detected in recording. Check microphone placement and "
            "ensure the subject is speaking during the recording window."
        )

    speech_audio = np.concatenate(speech_segments)

    total_s = len(audio) / config.sample_rate
    speech_s = len(speech_audio) / config.sample_rate
    logger.info(
        f"VAD: {speech_s:.1f}s speech from {total_s:.1f}s total "
        f"({100*speech_s/total_s:.0f}%), "
        f"{sum(final_mask)}/{num_frames} frames"
    )

    return speech_audio, final_mask


# ---------------------------------------------------------------------------
# Signal Processing: Normalization
# ---------------------------------------------------------------------------

def normalize_audio(audio: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """
    Peak normalization to target_peak_db.

    Does NOT apply RMS normalization by default to avoid altering the
    dynamic range that the model may rely on. If your teammate's model
    expects RMS-normalized input, set config.target_rms_db.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        logger.warning("Audio is near-silent, skipping normalization")
        return audio.copy()

    target_peak = 10.0 ** (config.target_peak_db / 20.0)
    normalized = audio * (target_peak / peak)

    # Clip safety (should not be needed after peak norm, but defensive)
    normalized = np.clip(normalized, -1.0, 1.0)

    logger.info(
        f"Normalized: peak {20*np.log10(peak+1e-10):.1f} dB -> "
        f"{config.target_peak_db} dB"
    )
    return normalized


# ---------------------------------------------------------------------------
# Clipping / Distortion Detection
# ---------------------------------------------------------------------------

def check_clipping(audio: np.ndarray, config: PipelineConfig) -> dict:
    """
    Detect clipping/distortion in raw audio.

    Returns dict with clipping stats. Should be called on RAW audio
    before processing, so you know if the input itself is bad.
    """
    abs_audio = np.abs(audio)
    clipped_samples = np.sum(abs_audio >= config.clipping_threshold)
    clipped_ratio = clipped_samples / len(audio) if len(audio) > 0 else 0.0
    is_clipped = clipped_ratio > config.clipping_warn_ratio

    result = {
        "clipped_samples": int(clipped_samples),
        "clipped_ratio": float(clipped_ratio),
        "is_clipped": is_clipped,
        "peak_amplitude": float(np.max(abs_audio)),
    }

    if is_clipped:
        logger.warning(
            f"CLIPPING DETECTED: {clipped_samples} samples "
            f"({100*clipped_ratio:.2f}%) above {config.clipping_threshold}. "
            f"Consider reducing microphone gain."
        )
    return result


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Everything downstream code needs."""

    cleaned_audio: np.ndarray
    sample_rate: int
    raw_duration_s: float
    speech_duration_s: float
    snr_before_db: float
    snr_after_db: float
    clipping_info: dict
    vad_mask: np.ndarray
    processing_time_s: float
    stages_applied: list = field(default_factory=list)


def run_pipeline(
    audio: np.ndarray,
    config: PipelineConfig,
    skip_stages: Optional[set[PipelineStage]] = None,
) -> PipelineResult:
    """
    Run the full signal processing chain.

    Args:
        audio: raw float32 mono audio in [-1, 1]
        config: pipeline configuration
        skip_stages: set of PipelineStage values to skip (for debugging)

    Returns:
        PipelineResult with cleaned audio and diagnostics

    MODEL INPUT CONTRACT:
        - Output shape: (num_samples,) float32
        - Sample rate: 16000 Hz
        - Range: [-1.0, 1.0], peak at -3 dB
        - Content: full audio including silence and pauses by default
        - VAD trimming is available but not applied unless explicitly enabled upstream
        Confirm with your teammate that their model expects this format.
    """
    skip = set(skip_stages) if skip_stages is not None else {PipelineStage.VAD_TRIMMED}
    t_start = time.monotonic()
    stages = []

    raw_duration = len(audio) / config.sample_rate

    # Clipping check (always runs on raw)
    clipping_info = check_clipping(audio, config)

    # SNR before processing
    snr_before = estimate_snr_db(audio, config)

    # Stage 1: Bandpass filter
    if PipelineStage.FILTERED not in skip:
        audio = bandpass_filter(audio, config)
        stages.append(PipelineStage.FILTERED)

    # Stage 2: Noise reduction
    if PipelineStage.DENOISED not in skip:
        audio = spectral_subtraction(audio, config)
        stages.append(PipelineStage.DENOISED)

    # Stage 3: VAD
    vad_mask = np.array([True])
    if PipelineStage.VAD_TRIMMED not in skip:
        audio, vad_mask = voice_activity_detection(audio, config)
        stages.append(PipelineStage.VAD_TRIMMED)

    # Stage 4: Normalization
    if PipelineStage.NORMALIZED not in skip:
        audio = normalize_audio(audio, config)
        stages.append(PipelineStage.NORMALIZED)

    # SNR after processing
    snr_after = estimate_snr_db(audio, config)

    speech_duration = len(audio) / config.sample_rate
    processing_time = time.monotonic() - t_start

    logger.info(
        f"Pipeline complete in {processing_time:.2f}s: "
        f"{raw_duration:.1f}s raw -> {speech_duration:.1f}s cleaned, "
        f"SNR {snr_before:.1f} -> {snr_after:.1f} dB"
    )

    return PipelineResult(
        cleaned_audio=audio,
        sample_rate=config.sample_rate,
        raw_duration_s=raw_duration,
        speech_duration_s=speech_duration,
        snr_before_db=snr_before,
        snr_after_db=snr_after,
        clipping_info=clipping_info,
        vad_mask=vad_mask,
        processing_time_s=processing_time,
        stages_applied=stages,
    )


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------

def save_wav(audio: np.ndarray, path: str, sample_rate: int = 16000) -> Path:
    """Save float32 audio as 16-bit PCM WAV (universally compatible)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert float32 [-1, 1] to int16
    audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Saved WAV: {out_path} ({size_kb:.1f} KB)")
    return out_path


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load WAV file and return (float32 audio, sample_rate)."""
    sr, data = wavfile.read(path)

    # Convert to float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        pass
    else:
        data = data.astype(np.float32)

    # Force mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    return data, sr
