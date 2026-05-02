#!/usr/bin/env python3
"""
Test script for the audio capture and signal processing pipeline.

Usage:
    # Default: record 30s, process, save both raw and cleaned
    python test_pipeline.py

    # Custom duration
    python test_pipeline.py --duration 10

    # Process an existing WAV file instead of recording
    python test_pipeline.py --input existing_recording.wav

    # List available audio devices (useful for finding USB mic index)
    python test_pipeline.py --list-devices

    # Use a specific device index
    python test_pipeline.py --device 2

    # Adjust noise reduction aggressiveness (1.0 = mild, 4.0 = aggressive)
    python test_pipeline.py --noise-alpha 3.0

    # Skip specific stages for debugging; VAD is skipped by default to preserve long pauses
    python test_pipeline.py --enable-vad  # enable VAD trimming for diagnostics
    python test_pipeline.py --skip-denoise

Output files (saved to --output-dir, default ./pipeline_output/):
    {prefix}_raw.wav          — unprocessed recording
    {prefix}_cleaned.wav      — fully processed, ready for model
    {prefix}_summary.txt      — processing stats and diagnostics

IMPORTANT: This is a screening support tool component, NOT a diagnostic device.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from audio_pipeline import (
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    MicDisconnectedError,
    SilenceOnlyError,
    capture_audio,
    run_pipeline,
    save_wav,
    load_wav,
    list_audio_devices,
    estimate_snr_db,
    check_clipping,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def print_summary(result: PipelineResult, raw_audio: np.ndarray, config: PipelineConfig) -> str:
    """Generate and print a human-readable summary. Returns the text."""
    lines = [
        "=" * 60,
        "  AUDIO PIPELINE PROCESSING SUMMARY",
        "=" * 60,
        f"  Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Sample rate:          {config.sample_rate} Hz",
        "",
        "  DURATIONS",
        f"  Raw recording:        {result.raw_duration_s:.2f} s",
        f"  After VAD (speech):   {result.speech_duration_s:.2f} s",
        f"  Speech ratio:         {100 * result.speech_duration_s / max(result.raw_duration_s, 0.001):.1f}%",
        "",
        "  SIGNAL QUALITY",
        f"  Est. SNR before:      {result.snr_before_db:.1f} dB",
        f"  Est. SNR after:       {result.snr_after_db:.1f} dB",
        f"  SNR improvement:      {result.snr_after_db - result.snr_before_db:+.1f} dB",
        "",
        "  CLIPPING",
        f"  Clipped samples:      {result.clipping_info['clipped_samples']}",
        f"  Clipped ratio:        {100 * result.clipping_info['clipped_ratio']:.3f}%",
        f"  Peak amplitude:       {result.clipping_info['peak_amplitude']:.4f}",
        f"  Clipping detected:    {'YES — consider reducing mic gain' if result.clipping_info['is_clipped'] else 'No'}",
        "",
        "  PROCESSING",
        f"  Total processing time: {result.processing_time_s:.3f} s",
        f"  Stages applied:       {', '.join(s.value for s in result.stages_applied)}",
        f"  Output samples:       {len(result.cleaned_audio)}",
        f"  Output size (16-bit): {len(result.cleaned_audio) * 2 / 1024:.1f} KB",
        "",
        "  PIPELINE CONFIG",
        f"  Bandpass:             {config.bp_low_hz}-{config.bp_high_hz} Hz (order {config.bp_order})",
        f"  Noise reduction:      spectral subtraction (alpha={config.oversubtraction}, beta={config.spectral_floor})",
        f"  Noise estimate:       first {config.noise_estimate_duration_s}s of recording",
        f"  VAD frame:            {config.vad_frame_ms} ms",
        f"  VAD energy threshold: {config.vad_energy_threshold_db} dB (relative to peak)",
        f"  Normalization:        peak at {config.target_peak_db} dB",
        "",
        "  NOTE: All outputs are screening indicators, not medical diagnoses.",
        "=" * 60,
    ]

    summary = "\n".join(lines)
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Test the audio capture and signal processing pipeline"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Recording duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to an existing WAV file (skips recording)"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (run --list-devices to see options)"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./pipeline_output",
        help="Directory for output files (default: ./pipeline_output/)"
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Output filename prefix (default: timestamp-based)"
    )
    parser.add_argument(
        "--noise-alpha", type=float, default=2.0,
        help="Noise reduction aggressiveness: 1.0=mild, 4.0=aggressive (default: 2.0)"
    )
    parser.add_argument(
        "--noise-estimate-s", type=float, default=0.5,
        help="Seconds of initial audio to use as noise profile (default: 0.5)"
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=-35.0,
        help="VAD energy threshold in dB relative to peak (default: -35)"
    )
    parser.add_argument(
        "--enable-vad", "--skip-vad",
        dest="enable_vad",
        action="store_true",
        help=(
            "Enable VAD trimming stage; by default VAD is skipped so silences and pauses are preserved. "
            "Use --enable-vad for clarity; --skip-vad is accepted as a legacy alias."
        )
    )
    parser.add_argument(
        "--skip-denoise", action="store_true",
        help="Skip noise reduction stage"
    )
    parser.add_argument(
        "--skip-filter", action="store_true",
        help="Skip bandpass filter stage"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("test_pipeline")

    # List devices and exit
    if args.list_devices:
        print("Available audio devices:")
        print(list_audio_devices())
        print("\nUse --device N to select a device by index.")
        sys.exit(0)

    # Config
    config = PipelineConfig(
        record_duration_s=args.duration,
        device_index=args.device,
        output_dir=args.output_dir,
        oversubtraction=args.noise_alpha,
        noise_estimate_duration_s=args.noise_estimate_s,
        vad_energy_threshold_db=args.vad_threshold,
    )

    # Output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = output_dir / f"{prefix}_raw.wav"
    cleaned_path = output_dir / f"{prefix}_cleaned.wav"
    summary_path = output_dir / f"{prefix}_summary.txt"

    # Skip stages (VAD is skipped by default to preserve silences)
    skip = {PipelineStage.VAD_TRIMMED}
    if args.enable_vad:
        skip.discard(PipelineStage.VAD_TRIMMED)
    if args.skip_denoise:
        skip.add(PipelineStage.DENOISED)
    if args.skip_filter:
        skip.add(PipelineStage.FILTERED)

    # -----------------------------------------------------------------------
    # Step 1: Get audio (record or load)
    # -----------------------------------------------------------------------
    if args.input:
        logger.info(f"Loading audio from: {args.input}")
        try:
            raw_audio, sr = load_wav(args.input)
        except FileNotFoundError:
            logger.error(f"File not found: {args.input}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load WAV: {e}")
            sys.exit(1)

        if sr != config.sample_rate:
            logger.warning(
                f"Input sample rate is {sr} Hz, expected {config.sample_rate} Hz. "
                f"Resampling is NOT implemented — ask your teammate to export "
                f"test files at {config.sample_rate} Hz, or add scipy.signal.resample."
            )
            # Basic resample using scipy
            from scipy.signal import resample
            target_samples = int(len(raw_audio) * config.sample_rate / sr)
            raw_audio = resample(raw_audio, target_samples).astype(np.float32)
            logger.info(f"Resampled {sr} -> {config.sample_rate} Hz")

        config.record_duration_s = len(raw_audio) / config.sample_rate
    else:
        print(f"\nRecording {config.record_duration_s}s from microphone...")
        print("Speak when ready. First 0.5s is captured as noise profile —")
        print("leave that silent if possible.\n")

        try:
            raw_audio = capture_audio(config)
        except MicDisconnectedError as e:
            logger.error(f"Microphone error: {e}")
            logger.error(
                "Troubleshooting:\n"
                "  1. Check USB mic is plugged in: lsusb | grep -i audio\n"
                "  2. List ALSA devices: arecord -l\n"
                "  3. Try a specific device: --device N (run --list-devices first)"
            )
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected capture error: {e}")
            sys.exit(1)

    # Save raw audio for comparison
    save_wav(raw_audio, str(raw_path), config.sample_rate)
    logger.info(f"Raw audio saved: {raw_path}")

    # -----------------------------------------------------------------------
    # Step 2: Run pipeline
    # -----------------------------------------------------------------------
    try:
        result = run_pipeline(raw_audio, config, skip_stages=skip)
    except SilenceOnlyError as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(
            "Possible causes:\n"
            "  - Microphone gain too low\n"
            "  - Subject did not speak during recording window\n"
            "  - Wrong device selected (ambient mic instead of USB)\n"
            "  - VAD threshold too aggressive (try --vad-threshold -45)"
        )
        # Still save what we have for debugging
        summary = (
            f"PIPELINE FAILED: No speech detected\n"
            f"Raw audio saved at: {raw_path}\n"
            f"Raw duration: {len(raw_audio)/config.sample_rate:.1f}s\n"
            f"Raw peak: {np.max(np.abs(raw_audio)):.4f}\n"
            f"Raw RMS: {np.sqrt(np.mean(raw_audio**2)):.4f}\n"
        )
        summary_path.write_text(summary)
        print(f"\nDiagnostics saved to: {summary_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Save cleaned audio
    # -----------------------------------------------------------------------
    save_wav(result.cleaned_audio, str(cleaned_path), config.sample_rate)
    logger.info(f"Cleaned audio saved: {cleaned_path}")

    # -----------------------------------------------------------------------
    # Step 4: Summary
    # -----------------------------------------------------------------------
    summary_text = print_summary(result, raw_audio, config)
    summary_path.write_text(summary_text)

    print(f"\nOutput files:")
    print(f"  Raw:     {raw_path}")
    print(f"  Cleaned: {cleaned_path}")
    print(f"  Summary: {summary_path}")
    print(f"\nCompare them: play both WAV files and listen for noise reduction")
    print(f"and silence trimming. The cleaned file should be shorter and cleaner.")
    print()
    print(f"Next step: pass {cleaned_path} to the inference wrapper.")
    print(f"Expected model input: float32 array, shape ({len(result.cleaned_audio)},), 16kHz mono")


if __name__ == "__main__":
    main()
