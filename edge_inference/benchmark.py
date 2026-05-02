"""Latency and memory benchmarking on Pi 4.

Measures:
  * cold-start (first call after load) vs steady-state inference latency
  * preprocessing latency, separately
  * RSS memory before model load, after load, peak during inference
  * CPU temperature if /sys/class/thermal/thermal_zone0/temp is readable
    (Raspberry Pi exposes this)

Outputs a dict and optionally writes JSONL to disk for longitudinal logging.

Why both ms and a percentile bucket: on Pi 4 you'll see latency variance from
thermal throttling and other processes. Median + p95 from a sample of 30+ runs
is a more honest number than a single timing.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from edge_inference.wrapper import DementiaScreener

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    n_runs: int
    audio_seconds: float
    preprocess_ms_median: float
    preprocess_ms_p95: float
    inference_ms_cold: float
    inference_ms_median: float
    inference_ms_p95: float
    inference_ms_min: float
    inference_ms_max: float
    realtime_factor_median: float       # audio_seconds / (inference seconds). >1 means faster than realtime
    rss_mb_before_load: float
    rss_mb_after_load: float
    rss_mb_peak_during_inference: float
    cpu_temp_c_start: Optional[float]
    cpu_temp_c_end: Optional[float]
    backend: str
    model_path: str
    raw_inference_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Benchmark: backend={self.backend}, n={self.n_runs}, "
            f"audio={self.audio_seconds:.1f}s | "
            f"infer median={self.inference_ms_median:.1f} ms (p95 {self.inference_ms_p95:.1f}, "
            f"cold {self.inference_ms_cold:.1f}) | "
            f"realtime x{self.realtime_factor_median:.2f} | "
            f"RSS load+={self.rss_mb_after_load - self.rss_mb_before_load:.1f} MB, "
            f"peak {self.rss_mb_peak_during_inference:.1f} MB"
        )


def _rss_mb() -> float:
    """Return current process RSS in MiB. Linux-specific (Pi)."""
    try:
        with open(f"/proc/{os.getpid()}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0  # kB -> MiB
    except OSError:
        pass
    # Fallback for non-Linux: psutil if available, else NaN.
    try:
        import psutil  # type: ignore
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except ImportError:
        return float("nan")


def _cpu_temp_c() -> Optional[float]:
    """Read Pi CPU temp from sysfs. Returns None if unavailable."""
    path = "/sys/class/thermal/thermal_zone0/temp"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def benchmark_screener(
    screener: DementiaScreener,
    audio: Optional[np.ndarray] = None,
    n_runs: int = 30,
    warmup_runs: int = 2,
    rss_before_load_mb: Optional[float] = None,
    output_jsonl: Optional[Path] = None,
) -> BenchmarkResult:
    """Run a timing + memory benchmark.

    Parameters
    ----------
    screener:
        An already-loaded DementiaScreener.
    audio:
        If None, uses a sine-wave-shaped synthetic clip at the model's
        expected duration. Pass real audio when measuring representative
        latency (resampling cost depends on input sample rate).
    n_runs:
        Steady-state runs after warmup. 30 is enough to get a stable p95.
    warmup_runs:
        Discarded runs to settle ORT/TFLite caches and CPU caches.
    rss_before_load_mb:
        If known (recorded by the caller before constructing the screener),
        pass it here. Otherwise we report a placeholder.
    output_jsonl:
        Optional path. Appends one JSON line per benchmark run for offline analysis.
    """
    sr = screener.config.sample_rate
    target_samples = screener.config.expected_input_samples
    duration_s = target_samples / sr

    if audio is None:
        # Mild synthetic signal; not silence (which softmax-flattens to ~0.5).
        t = np.arange(target_samples, dtype=np.float32) / sr
        audio = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    rss_before = float(rss_before_load_mb) if rss_before_load_mb is not None else float("nan")
    rss_after_load = _rss_mb()
    cpu_start = _cpu_temp_c()

    # Warmup
    for _ in range(max(0, warmup_runs)):
        screener.predict(audio, sample_rate=sr)

    # Cold-equivalent timing: force a GC and re-measure once. This isn't a
    # true cold start (the session is already loaded) but reflects "first call
    # after idle" which is what the user actually sees.
    gc.collect()
    cold_t0 = time.perf_counter()
    screener.predict(audio, sample_rate=sr)
    cold_ms = (time.perf_counter() - cold_t0) * 1000.0

    inference_ms: List[float] = []
    preprocess_ms: List[float] = []
    rss_peak = rss_after_load

    for i in range(int(n_runs)):
        result = screener.predict(audio, sample_rate=sr)
        inference_ms.append(result.inference_ms)
        preprocess_ms.append(result.preprocess_ms)
        # Cheap RSS poll. Don't poll every iter on a huge run, but n=30 is fine.
        cur = _rss_mb()
        if cur > rss_peak:
            rss_peak = cur

        if output_jsonl is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "iter": i,
                    "inference_ms": result.inference_ms,
                    "preprocess_ms": result.preprocess_ms,
                    "rss_mb": cur,
                    "ts": time.time(),
                }) + "\n")

    cpu_end = _cpu_temp_c()
    median_inf = statistics.median(inference_ms)
    rt_factor = (duration_s * 1000.0) / median_inf if median_inf > 0 else float("inf")

    return BenchmarkResult(
        n_runs=int(n_runs),
        audio_seconds=duration_s,
        preprocess_ms_median=statistics.median(preprocess_ms),
        preprocess_ms_p95=_percentile(preprocess_ms, 95),
        inference_ms_cold=cold_ms,
        inference_ms_median=median_inf,
        inference_ms_p95=_percentile(inference_ms, 95),
        inference_ms_min=min(inference_ms),
        inference_ms_max=max(inference_ms),
        realtime_factor_median=rt_factor,
        rss_mb_before_load=rss_before,
        rss_mb_after_load=rss_after_load,
        rss_mb_peak_during_inference=rss_peak,
        cpu_temp_c_start=cpu_start,
        cpu_temp_c_end=cpu_end,
        backend=screener.config.resolved_backend(),
        model_path=str(screener.config.model_path),
        raw_inference_ms=inference_ms,
    )


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac
