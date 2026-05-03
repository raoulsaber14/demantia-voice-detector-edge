#!/usr/bin/env python3
"""
Record audio from the mic, run it through the cleaning pipeline, and pass it
to the dementia screening model.

Usage:
    # Default: record 30s with the dummy model (validates the loop)
    python -m scripts.record_and_screen

    # Real ONNX model (after adding external artifacts under models/):
    python -m scripts.record_and_screen --config configs/edge_inference.onnx.example.yaml

    # Shorter recording:
    python -m scripts.record_and_screen --duration 10

    # Use a specific USB mic (run --list-devices to find it):
    python -m scripts.record_and_screen --device 2

    # Process an existing WAV instead of recording:
    python -m scripts.record_and_screen --input some_recording.wav

    # Run a quick latency benchmark after the inference:
    python -m scripts.record_and_screen --benchmark

What this script does, in order:
    1. Prompts you to press Enter, then records from the mic.
    2. Hands the raw audio to your audio_pipeline (denoise + bandpass + normalize).
    3. Hands the cleaned audio to the DementiaScreener wrapper.
    4. Prints the screening result.
    5. Optionally benchmarks inference latency.

Reminder: this is screening support, not a medical diagnosis.
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Your pipeline
from src.audio_pipeline import (
    PipelineConfig,
    MicDisconnectedError,
    SilenceOnlyError,
    capture_audio,
    run_pipeline,
    save_wav,
    load_wav,
    list_audio_devices,
)

# The inference wrapper
from edge_inference import DementiaScreener, load_config
from edge_inference.config import EdgeConfig


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_result(result, audio_seconds: float) -> None:
    """Print the screening result in a clear, screen-friendly format."""
    band = result.risk_band.upper()
    band_color = {
        "LOW": "\033[32m",      # green
        "REVIEW": "\033[33m",   # yellow
        "HIGH": "\033[31m",     # red
    }.get(band, "")
    reset = "\033[0m" if band_color else ""

    print()
    print("=" * 60)
    print("  DEMENTIA VOICE SCREENING — RESULT")
    print("=" * 60)
    print(f"  Risk score:         {result.risk_score} / 100")
    print(f"  Risk band:          {band_color}{band}{reset}")
    print(f"  Predicted label:    {result.predicted_label}")
    print(f"  Confidence:         {result.confidence:.2f}")
    print()
    print("  Per-class probabilities:")
    for label, prob in result.class_probabilities.items():
        bar = "█" * int(prob * 30)
        print(f"    {label:18s} {prob:.3f}  {bar}")
    print()
    if result.markers:
        print("  Markers:")
        for name, value in result.markers.items():
            print(f"    {name:18s} {value:.3f}")
        print()
    print(f"  Audio length:       {audio_seconds:.1f} s")
    print(f"  Preprocessing:      {result.preprocess_ms:.1f} ms")
    print(f"  Inference:          {result.inference_ms:.1f} ms")
    print()
    print(f"  {result.disclaimer}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Server integration helpers
# ---------------------------------------------------------------------------

def _server_status(url: str, state: str, message: str = "", logger=None) -> None:
    """POST a SessionStatus update to the local server (non-fatal on failure)."""
    try:
        import requests
        requests.post(
            f"{url}/status",
            json={"state": state, "message": message},
            timeout=2,
        )
    except Exception as exc:
        if logger:
            logger.debug("Could not update server status: %s", exc)


def _server_post_result(url: str, result, audio, sr, edge_cfg, logger) -> None:
    """POST the finished ScreeningPayload to the local server."""
    try:
        import requests
        payload = result.to_dict()
        payload["session_id"] = str(uuid.uuid4())
        payload["timestamp"]  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["audio_seconds"] = len(audio) / sr
        payload["model_info"] = {
            "backend":         edge_cfg.resolved_backend(),
            "model_path":      str(edge_cfg.model_path),
            "head_model_path": str(edge_cfg.head_model_path) if edge_cfg.head_model_path else None,
        }
        resp = requests.post(f"{url}/results", json=payload, timeout=5)
        resp.raise_for_status()
        logger.info("Result posted to server  session=%s", payload["session_id"])
    except Exception as exc:
        logger.warning("Could not post result to server (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Record from mic and run the dementia screener.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/edge_inference.yaml"),
        help="Path to the inference config YAML (default: configs/edge_inference.yaml)",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--input", type=str, default=None, help="WAV file to process instead of recording")
    parser.add_argument("--save-audio", action="store_true", help="Save raw + cleaned WAV to ./pipeline_output/")
    parser.add_argument("--benchmark", action="store_true", help="Run a 30-iter latency benchmark after inference")
    parser.add_argument("--threads", type=int, default=None, help="Cap inference threads (Pi 4 has 4 cores)")
    parser.add_argument("--save-results", action="store_true", help="Save full result JSON to ./pipeline_output/")
    parser.add_argument(
        "--no-clean", action="store_true",
        help=(
            "Skip bandpass filter, spectral subtraction, and normalization. "
            "Passes raw (resampled) audio directly to the model, matching the "
            "training pipeline exactly (load → resample → zero non-finite)."
        ),
    )
    parser.add_argument(
        "--post-result", action="store_true",
        help="POST result to the local server (start server first with: python -m server.run)",
    )
    parser.add_argument(
        "--server-url", default="http://localhost:8765",
        help="Server URL for --post-result (default: http://localhost:8765)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("record_and_screen")

    # --- list devices and exit ---
    if args.list_devices:
        print("Available audio devices:")
        print(list_audio_devices())
        print("\nUse --device N to select a device by index.")
        return 0

    # --- load inference config ---
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        logger.error("Hint: from the project root, default is configs/edge_inference.yaml")
        return 1
    edge_cfg: EdgeConfig = load_config(args.config)

    # --- pipeline config: tie sample rate to what the model expects ---
    pipeline_cfg = PipelineConfig(
        sample_rate=edge_cfg.sample_rate,        # both should be 16000
        record_duration_s=args.duration,
        device_index=args.device,
    )

    # --- step 1: get raw audio (record or load) ---
    if args.input:
        logger.info("Loading audio from %s", args.input)
        try:
            raw_audio, sr = load_wav(args.input)
        except FileNotFoundError:
            logger.error("File not found: %s", args.input)
            return 1
        if sr != pipeline_cfg.sample_rate:
            logger.warning("Input is %d Hz; the wrapper will resample to %d Hz.",
                           sr, pipeline_cfg.sample_rate)
            # Let the wrapper handle resampling rather than doing it here, so
            # the pipeline sees the original-rate audio. But the pipeline
            # config assumes 16 kHz, so if rates differ we just skip the
            # cleaning step and feed raw audio straight to the wrapper.
            logger.warning("Skipping signal-processing pipeline because of sample-rate mismatch.")
            cleaned_audio = raw_audio
            cleaned_sr = sr
        else:
            cleaned_sr = sr
            cleaned_audio = None  # filled below by run_pipeline
    else:
        print(f"\nRecording {pipeline_cfg.record_duration_s:.0f} seconds from microphone...")
        print("Speak when ready. The first ~0.5s is captured as a noise profile —")
        print("try to keep that part silent.\n")
        input("Press Enter to start recording...")
        if args.post_result:
            _server_status(args.server_url, "recording", "Recording in progress…", logger)
        try:
            raw_audio = capture_audio(pipeline_cfg)
        except MicDisconnectedError as e:
            logger.error("Microphone error: %s", e)
            logger.error("Hints: check `arecord -l`, try --list-devices, then --device N")
            if args.post_result:
                _server_status(args.server_url, "error", str(e), logger)
            return 1
        cleaned_audio = None
        cleaned_sr = pipeline_cfg.sample_rate

    # --- step 2: clean (skip if bypassed for sr mismatch or --no-clean) ---
    if cleaned_audio is None:
        if args.no_clean:
            logger.info(
                "--no-clean: skipping signal processing (matches training pipeline). "
                "Raw audio: %d samples (%.2f s)",
                len(raw_audio), len(raw_audio) / pipeline_cfg.sample_rate,
            )
            cleaned_audio = raw_audio
            if args.save_audio:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("./pipeline_output")
                out_dir.mkdir(parents=True, exist_ok=True)
                save_wav(raw_audio, str(out_dir / f"{stamp}_raw.wav"), pipeline_cfg.sample_rate)
        else:
            try:
                result = run_pipeline(raw_audio, pipeline_cfg)
            except SilenceOnlyError as e:
                logger.error("Pipeline failed: %s", e)
                logger.error("Try a quieter environment, louder speech, or longer --duration.")
                return 1

            cleaned_audio = result.cleaned_audio
            logger.info(
                "Cleaned audio: %d samples (%.2f s), SNR %.1f -> %.1f dB",
                len(cleaned_audio),
                len(cleaned_audio) / pipeline_cfg.sample_rate,
                result.snr_before_db,
                result.snr_after_db,
            )
            if args.save_audio:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("./pipeline_output")
                out_dir.mkdir(parents=True, exist_ok=True)
                save_wav(raw_audio, str(out_dir / f"{stamp}_raw.wav"), pipeline_cfg.sample_rate)
                save_wav(cleaned_audio, str(out_dir / f"{stamp}_cleaned.wav"), pipeline_cfg.sample_rate)

    # --- step 3: inference ---
    if args.post_result:
        _server_status(args.server_url, "processing", "Running inference…", logger)
    logger.info("Loading model: %s (backend=%s)", edge_cfg.model_path, edge_cfg.resolved_backend())
    with DementiaScreener(edge_cfg, num_threads=args.threads) as screener:
        screener.warmup(n=2)
        result = screener.predict(cleaned_audio, sample_rate=cleaned_sr)
        print_result(result, audio_seconds=len(cleaned_audio) / cleaned_sr)
        if args.post_result:
            _server_post_result(args.server_url, result, cleaned_audio, cleaned_sr, edge_cfg, logger)

        # --- step 4: optional benchmark ---
        if args.save_results:
            out_dir = Path("./pipeline_output")
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = out_dir / f"{stamp}_result.json"
            payload = result.to_dict()
            payload["timestamp"] = stamp
            payload["audio_seconds"] = len(cleaned_audio) / cleaned_sr
            payload["model_path"] = str(edge_cfg.model_path)
            payload["head_model_path"] = str(edge_cfg.head_model_path)
            payload["backend"] = edge_cfg.resolved_backend()
            payload["threads"] = args.threads
            result_path.write_text(json.dumps(payload, indent=2, default=str))
            print(f"\n  Results saved to: {result_path}")

        if args.benchmark:
            from edge_inference.benchmark import benchmark_screener
            print("\nRunning benchmark (30 iterations)...")
            bench = benchmark_screener(screener, audio=cleaned_audio, n_runs=30)
            print(bench.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
