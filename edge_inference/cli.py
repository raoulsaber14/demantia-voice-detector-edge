"""Command-line entry point for one-off testing.

Examples:

    # Run with the dummy backend on a real WAV file (no ML deps required):
    python -m edge_inference.cli --config edge_inference/model_config.yaml --audio test.wav

    # Run with ONNX model:
    #   first edit model_config.yaml: model_path: models/dementia_int8.onnx
    python -m edge_inference.cli --config edge_inference/model_config.yaml --audio test.wav

    # Benchmark mode:
    python -m edge_inference.cli --config edge_inference/model_config.yaml \
        --audio test.wav --benchmark --runs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from edge_inference.benchmark import benchmark_screener
from edge_inference.config import load_config
from edge_inference.wrapper import DementiaScreener


def _load_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Load WAV via soundfile (preferred) or stdlib wave as fallback."""
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.astype(np.float32, copy=False), int(sr)
    except ImportError:
        import wave

        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            sw = w.getsampwidth()
            ch = w.getnchannels()
            raw = w.readframes(n)
        if sw == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2_147_483_648.0
        elif sw == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sw}")
        if ch == 2:
            arr = arr.reshape(-1, 2).mean(axis=1)
        return arr, int(sr)


def main() -> int:
    p = argparse.ArgumentParser(description="Edge inference CLI for the dementia screener.")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    p.add_argument("--audio", type=Path, help="Path to WAV file (or omit with --benchmark to use synthetic)")
    p.add_argument("--benchmark", action="store_true", help="Run latency/memory benchmark")
    p.add_argument("--runs", type=int, default=30, help="Number of benchmark runs")
    p.add_argument("--threads", type=int, default=None, help="Optional: cap inference threads (Pi 4 has 4 cores)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)

    # Capture pre-load RSS so the benchmark can report model memory cost.
    from edge_inference.benchmark import _rss_mb
    rss_pre = _rss_mb()

    with DementiaScreener(cfg, num_threads=args.threads) as screener:
        if args.benchmark:
            audio = None
            if args.audio is not None:
                audio, sr = _load_wav(args.audio)
                if sr != cfg.sample_rate:
                    logging.warning(
                        "Audio is at %d Hz; will be resampled to %d Hz on every benchmark iteration. "
                        "Pre-resample for clean numbers.", sr, cfg.sample_rate,
                    )
            result = benchmark_screener(
                screener,
                audio=audio,
                n_runs=args.runs,
                rss_before_load_mb=rss_pre,
            )
            print(result.summary())
            print(json.dumps(result.to_dict(), indent=2, default=str))
            return 0

        if args.audio is None:
            print("--audio required when not in --benchmark mode", file=sys.stderr)
            return 2

        audio, sr = _load_wav(args.audio)
        result = screener.predict(audio, sample_rate=sr)
        print(json.dumps(result.to_dict(), indent=2))
        return 0


if __name__ == "__main__":
    sys.exit(main())
