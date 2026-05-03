"""Smoke test: verifies the package works end-to-end with the dummy backend
and (if onnx/onnxruntime are available) the ONNX backend."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_inference import DementiaScreener, load_config
from edge_inference.config import EdgeConfig
from edge_inference.benchmark import benchmark_screener


def test_dummy_backend_basic():
    cfg = EdgeConfig(
        model_path=Path("dummy"),
        backend="dummy",
        expected_input_seconds=2.0,  # short for fast test
    )
    cfg.validate()

    rng = np.random.default_rng(0)
    audio = rng.standard_normal(int(2 * 16_000)).astype(np.float32) * 0.1

    with DementiaScreener(cfg) as s:
        result = s.predict(audio, sample_rate=16_000)
        assert 0 <= result.risk_score <= 100, f"risk_score out of range: {result.risk_score}"
        assert 0.0 <= result.prob_dementia <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.risk_band in {"low", "review", "high"}
        assert result.predicted_label in cfg.label_names
        assert set(result.class_probabilities) == set(cfg.label_names)
        assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-4
        assert result.inference_ms >= 0
        assert result.preprocess_ms >= 0
        assert result.disclaimer
        print(f"  [dummy] risk={result.risk_score} band={result.risk_band} "
              f"infer={result.inference_ms:.2f}ms pre={result.preprocess_ms:.2f}ms")


def test_dummy_pad_and_truncate():
    cfg = EdgeConfig(model_path=Path("dummy"), backend="dummy", expected_input_seconds=2.0)

    with DementiaScreener(cfg) as s:
        # Short audio -> padded
        short = np.zeros(int(0.5 * 16_000), dtype=np.float32)
        r = s.predict(short, sample_rate=16_000)
        assert r.info["was_padded"] is True
        assert r.info["was_truncated"] is False

        # Long audio -> truncated
        long = np.zeros(int(5.0 * 16_000), dtype=np.float32)
        r = s.predict(long, sample_rate=16_000)
        assert r.info["was_padded"] is False
        assert r.info["was_truncated"] is True

        # Wrong sample rate -> resampled
        wrong = np.zeros(int(2.0 * 8_000), dtype=np.float32)
        r = s.predict(wrong, sample_rate=8_000)
        assert r.info["was_resampled"] is True
        print("  [pad/truncate/resample] all paths exercised")


def test_dummy_int_audio_input():
    """Mic input is often int16. Verify we handle it without misinterpreting."""
    cfg = EdgeConfig(model_path=Path("dummy"), backend="dummy", expected_input_seconds=1.0)

    with DementiaScreener(cfg) as s:
        rng = np.random.default_rng(0)
        audio_i16 = rng.integers(-1000, 1000, size=16_000, dtype=np.int16)
        r = s.predict(audio_i16, sample_rate=16_000)
        assert 0 <= r.risk_score <= 100
        print(f"  [int16 input] risk={r.risk_score}")


def test_yaml_loading():
    yaml_text = """
model_path: dummy
backend: dummy
sample_rate: 16000
expected_input_seconds: 1.5
pad_mode: zero
truncate_mode: center
label_names: [non_dementia, dementia]
positive_class_index: 1
risk_band_thresholds: [0.30, 0.70]
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        fpath = Path(f.name)

    try:
        cfg = load_config(fpath)
        assert cfg.expected_input_seconds == 1.5
        assert cfg.risk_band_thresholds == (0.30, 0.70)
        assert cfg.label_names == ("non_dementia", "dementia")
        print(f"  [yaml] loaded: {cfg.expected_input_samples} samples, bands={cfg.risk_band_thresholds}")
    finally:
        fpath.unlink()


def test_benchmark_runs():
    cfg = EdgeConfig(model_path=Path("dummy"), backend="dummy", expected_input_seconds=1.0)

    with DementiaScreener(cfg) as s:
        result = benchmark_screener(s, n_runs=10, warmup_runs=1)
        assert result.n_runs == 10
        assert result.inference_ms_median > 0
        assert result.realtime_factor_median > 0
        print(f"  [benchmark] {result.summary()}")


def test_real_onnx_if_available():
    """If onnx + onnxruntime are installed, build a real ONNX model and run it."""
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        print("  [onnx] skipped (onnx or onnxruntime not installed)")
        return

    from edge_inference.make_dummy_model import make_onnx

    n_samples = 16_000  # 1 second
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tiny.onnx"
        make_onnx(path, n_samples=n_samples, n_classes=2)
        cfg = EdgeConfig(model_path=path, backend="onnx", expected_input_seconds=1.0)

        with DementiaScreener(cfg) as s:
            audio = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32) * 0.1
            r = s.predict(audio, sample_rate=16_000)
            assert 0 <= r.risk_score <= 100
            print(f"  [onnx real] risk={r.risk_score} band={r.risk_band} "
                  f"infer={r.inference_ms:.2f}ms input_shape={s._backend.input_shape}")


def test_two_stage_onnx_if_available():
    """End-to-end test of Wav2Vec2TwoStageBackend with the real model files."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("  [two_stage] skipped (onnxruntime not installed)")
        return

    root = Path(__file__).resolve().parents[1]
    models_root = root / "models" / "phaseD_frozen_ensemble_deploy"
    backbone = models_root / "wav2vec2_base" / "wav2vec2_base_chunk8_accumulator_fp32_external" / "model.onnx"
    head     = models_root / "wav2vec2_component_head_fp32.onnx"

    if not backbone.exists() or not head.exists():
        print(f"  [two_stage] skipped (model files not found at expected paths)")
        return

    cfg = EdgeConfig(
        model_path=backbone,
        head_model_path=head,
        backend="wav2vec2_two_stage",
        expected_input_seconds=8.0,
    )
    cfg.validate()

    rng = np.random.default_rng(0)

    with DementiaScreener(cfg) as s:
        assert s._backend.__class__.__name__ == "Wav2Vec2TwoStageBackend", \
            f"Expected Wav2Vec2TwoStageBackend, got {s._backend.__class__.__name__}"

        audio = rng.standard_normal(int(8.0 * 16_000)).astype(np.float32) * 0.05
        r = s.predict(audio, sample_rate=16_000)

        assert 0 <= r.risk_score <= 100, f"risk_score out of range: {r.risk_score}"
        assert 0.0 <= r.prob_dementia <= 1.0
        assert r.risk_band in {"low", "review", "high"}
        assert set(r.class_probabilities) == set(cfg.label_names)
        assert abs(sum(r.class_probabilities.values()) - 1.0) < 1e-4

        # Also verify multi-chunk path (audio > 8 s = 2 chunks)
        long_audio = rng.standard_normal(int(12.0 * 16_000)).astype(np.float32) * 0.05
        r2 = s.predict(long_audio, sample_rate=16_000)
        assert 0 <= r2.risk_score <= 100

        print(f"  [two_stage] single-chunk risk={r.risk_score} band={r.risk_band} "
              f"infer={r.inference_ms:.1f}ms")
        print(f"  [two_stage] multi-chunk  risk={r2.risk_score} band={r2.risk_band} "
              f"infer={r2.inference_ms:.1f}ms")


def test_validation_rejects_bad_config():
    # Wrong sample rate
    try:
        EdgeConfig(model_path=Path("dummy"), backend="dummy", sample_rate=8000).validate()
    except ValueError:
        print("  [validation] caught wrong sample_rate")
    else:
        raise AssertionError("Expected ValueError for sample_rate=8000")

    # Bad band thresholds
    try:
        EdgeConfig(model_path=Path("dummy"), backend="dummy",
                   risk_band_thresholds=(0.7, 0.3)).validate()
    except ValueError:
        print("  [validation] caught bad band thresholds")
    else:
        raise AssertionError("Expected ValueError for bad bands")


class EdgeInferenceSmokeTests(unittest.TestCase):
    def test_dummy_backend_basic(self):
        test_dummy_backend_basic()

    def test_dummy_pad_and_truncate(self):
        test_dummy_pad_and_truncate()

    def test_dummy_int_audio_input(self):
        test_dummy_int_audio_input()

    def test_yaml_loading(self):
        test_yaml_loading()

    def test_benchmark_runs(self):
        test_benchmark_runs()

    def test_validation_rejects_bad_config(self):
        test_validation_rejects_bad_config()

    def test_real_onnx_if_available(self):
        test_real_onnx_if_available()

    def test_two_stage_onnx_if_available(self):
        test_two_stage_onnx_if_available()


def main():
    tests = [
        test_dummy_backend_basic,
        test_dummy_pad_and_truncate,
        test_dummy_int_audio_input,
        test_yaml_loading,
        test_benchmark_runs,
        test_validation_rejects_bad_config,
        test_real_onnx_if_available,
        test_two_stage_onnx_if_available,
    ]
    failed = 0
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 50}")
    if failed:
        print(f"FAILED: {failed}/{len(tests)} tests")
        sys.exit(1)
    print(f"PASSED: {len(tests)}/{len(tests)} tests")


if __name__ == "__main__":
    unittest.main()
