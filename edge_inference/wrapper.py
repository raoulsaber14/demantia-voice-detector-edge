"""Public wrapper: DementiaScreener.

Single class your audio_pipeline.py talks to. Hides backend choice, model
loading, preprocessing, and post-processing behind one .predict() call.

Threading:
    Construction loads the model and is NOT thread-safe.
    .predict() is also NOT thread-safe — onnxruntime sessions and tflite
    interpreters mutate internal state per call. If you need concurrent
    inference, build one DementiaScreener per thread.

On Pi 4, model load is a one-shot cost at app startup (typically 200–800 ms
for a quantized wav2vec2). Inference is the steady-state cost — see benchmark.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from edge_inference.backends import InferenceBackend, Wav2Vec2TwoStageBackend, build_backend
from edge_inference.config import EdgeConfig
from edge_inference.preprocessing import prepare_input

logger = logging.getLogger(__name__)


DISCLAIMER = (
    "This output is a screening risk indicator only and is not a medical diagnosis. "
    "Refer to a clinician for any health decision."
)


@dataclass
class ScreeningResult:
    """Structured output of one inference call.

    Attributes
    ----------
    risk_score:
        Integer in [0, 100] — P(dementia) * 100, rounded. The "risk score"
        framing avoids implying a calibrated medical probability.
    prob_dementia:
        Raw P(dementia) in [0, 1].
    confidence:
        P of the predicted class in [0, 1]. Convenience for UI.
    risk_band:
        "low" | "review" | "high" — driven by config.risk_band_thresholds.
    predicted_label:
        Decoded label string (e.g. "non_dementia").
    class_probabilities:
        Per-class probabilities keyed by label name.
    markers:
        Per-marker contributions if the model exposes them. Empty dict when
        the model does not. Currently no marker output is wired — placeholder
        until the teammate confirms her output schema.
    inference_ms:
        Wall-clock ms for the model.run() call only (not preprocessing).
    preprocess_ms:
        Wall-clock ms for waveform → tensor conversion.
    info:
        Diagnostics from preprocessing (resampled? padded? truncated?).
    disclaimer:
        Fixed legal/safety string. Do not strip when displaying or transmitting.
    """

    risk_score: int
    prob_dementia: float
    confidence: float
    risk_band: str
    predicted_label: str
    class_probabilities: Dict[str, float]
    markers: Dict[str, float] = field(default_factory=dict)
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    info: Dict[str, object] = field(default_factory=dict)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


class DementiaScreener:
    """End-to-end edge inference wrapper.

    Lifecycle:
        screener = DementiaScreener(cfg)
        result = screener.predict(audio, sample_rate=16000)
        screener.close()    # optional; releases backend native handles
    """

    def __init__(
        self,
        config: EdgeConfig,
        backend: Optional[InferenceBackend] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        config.validate()
        self.config = config
        self._owns_backend = backend is None
        self._backend: InferenceBackend = backend or self._build_backend(num_threads)
        self._log_model_info()

    # --- construction helpers ----------------------------------------------

    def _build_backend(self, num_threads: Optional[int]) -> InferenceBackend:
        kind = self.config.resolved_backend()
        dummy_shape = (
            (1, self.config.expected_input_samples)
            if self.config.add_batch_dim
            else (self.config.expected_input_samples,)
        )
        return build_backend(
            backend_kind=kind,
            model_path=self.config.model_path,
            input_name=self.config.input_name,
            num_threads=num_threads,
            dummy_input_shape=dummy_shape,
            dummy_output_classes=len(self.config.label_names),
            head_path=self.config.head_model_path,
        )

    def _log_model_info(self) -> None:
        logger.info(
            "DementiaScreener loaded: backend=%s input_name=%s input_shape=%s outputs=%s",
            self.config.resolved_backend(),
            self._backend.input_name,
            self._backend.input_shape,
            self._backend.output_names,
        )

    # --- public API ---------------------------------------------------------

    def predict(self, audio: np.ndarray, sample_rate: int) -> ScreeningResult:
        """Run one inference on a cleaned audio buffer.

        Parameters
        ----------
        audio:
            1-D float32 ndarray, roughly in [-1, 1]. 2-D stereo is averaged
            to mono. int16 etc. is accepted and rescaled.
        sample_rate:
            Sample rate of `audio`. Should be 16000 to skip resampling.

        Returns
        -------
        ScreeningResult
        """
        t0 = time.perf_counter()
        # The two-stage accumulator backend handles variable-length audio by
        # chunking internally. Padding to a fixed length here would inject silent
        # frames that dilute the mean+std embedding, so we skip pad/truncate.
        target_samples = (
            None
            if isinstance(self._backend, Wav2Vec2TwoStageBackend)
            else self.config.expected_input_samples
        )
        tensor, info = prepare_input(
            audio=audio,
            sample_rate=sample_rate,
            target_sample_rate=self.config.sample_rate,
            target_samples=target_samples,
            pad_mode=self.config.pad_mode,
            truncate_mode=self.config.truncate_mode,
            dtype=self.config.input_dtype,
            add_batch_dim=self.config.add_batch_dim,
        )
        t1 = time.perf_counter()
        outputs = self._backend.run(tensor)
        t2 = time.perf_counter()

        probs = self._extract_class_probabilities(outputs)
        markers = self._extract_markers(outputs)

        return self._build_result(
            probs=probs,
            markers=markers,
            preprocess_ms=(t1 - t0) * 1000.0,
            inference_ms=(t2 - t1) * 1000.0,
            info=info,
        )

    def warmup(self, n: int = 2) -> None:
        """Run a few throwaway inferences to settle ORT/TFLite caches.

        First inference is always 2–5x slower than steady state. Call this
        once at app startup before showing any latency claim to the user.
        """
        dummy = np.zeros(self.config.expected_input_samples, dtype=np.float32)
        for _ in range(max(1, int(n))):
            self.predict(dummy, sample_rate=self.config.sample_rate)

    def close(self) -> None:
        if self._owns_backend and self._backend is not None:
            self._backend.close()

    def __enter__(self) -> "DementiaScreener":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- output decoding ----------------------------------------------------

    def _extract_class_probabilities(self, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Pick the classification head from the model outputs and softmax it.

        Heuristics, in priority order:
          1. An output named "probabilities" / "probs" / "softmax" — used as-is.
          2. An output named "logits" — softmax applied.
          3. Single output — assumed to be logits if any value is outside [0,1]
             or doesn't sum to 1, otherwise treated as probabilities.

        Squeezes leading batch dim. Validates length matches label_names.
        """
        n_classes = len(self.config.label_names)

        candidate_keys_probs = ("probabilities", "probs", "softmax")
        candidate_keys_logits = ("logits", "output", "logit")

        arr: Optional[np.ndarray] = None
        is_logits: bool = False

        for k in candidate_keys_probs:
            for name, value in outputs.items():
                if name.lower() == k:
                    arr = value
                    is_logits = False
                    break
            if arr is not None:
                break
        if arr is None:
            for k in candidate_keys_logits:
                for name, value in outputs.items():
                    if name.lower() == k:
                        arr = value
                        is_logits = True
                        break
                if arr is not None:
                    break
        if arr is None:
            # Single-output fallback.
            classification_outputs = [v for v in outputs.values() if v.ndim >= 1]
            if not classification_outputs:
                raise RuntimeError(f"Model produced no usable outputs. Got: {list(outputs)}")
            arr = classification_outputs[0]
            is_logits = self._looks_like_logits(arr)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim > 2:
            raise RuntimeError(f"Unexpected output shape {arr.shape}; cannot decode")

        if arr.shape[-1] != n_classes:
            raise RuntimeError(
                f"Model output has {arr.shape[-1]} classes but config.label_names has {n_classes}. "
                f"Update label_names in config to match."
            )

        if is_logits:
            arr = _softmax(arr)
        else:
            # If the values don't sum to ~1, force-normalize.
            s = float(arr.sum())
            if s > 0 and abs(s - 1.0) > 1e-3:
                arr = arr / s

        return arr

    @staticmethod
    def _looks_like_logits(arr: np.ndarray) -> bool:
        flat = np.asarray(arr).ravel()
        if flat.size == 0:
            return True
        if flat.min() < -0.01 or flat.max() > 1.01:
            return True
        # Non-negative but doesn't sum to 1 per row -> probably logits passed
        # through some non-softmax activation; safer to softmax.
        if abs(float(flat.sum()) - 1.0) > 0.05:
            return True
        return False

    def _extract_markers(self, outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Pull per-marker contributions if the model declares them.

        Until the teammate ships a model with a marker head, this returns {}.
        Wired here so the wrapper doesn't need a code change later — only a
        config update (`marker_names: [pause_burden, prosody, ...]`) and a
        matching output tensor in the model.
        """
        if not self.config.marker_names:
            return {}
        for name in ("markers", "marker_scores", "explanations"):
            for out_name, value in outputs.items():
                if out_name.lower() == name:
                    flat = np.asarray(value, dtype=np.float32).ravel()
                    if flat.shape[0] != len(self.config.marker_names):
                        logger.warning(
                            "Marker output length %d != config marker_names length %d; skipping",
                            flat.shape[0],
                            len(self.config.marker_names),
                        )
                        return {}
                    return {n: float(v) for n, v in zip(self.config.marker_names, flat)}
        return {}

    def _build_result(
        self,
        probs: np.ndarray,
        markers: Dict[str, float],
        preprocess_ms: float,
        inference_ms: float,
        info: Dict[str, object],
    ) -> ScreeningResult:
        pos_idx = self.config.positive_class_index
        prob_pos = float(probs[pos_idx])
        pred_idx = int(np.argmax(probs))
        pred_label = self.config.label_names[pred_idx]
        confidence = float(probs[pred_idx])

        low_thr, high_thr = self.config.risk_band_thresholds
        if prob_pos < low_thr:
            band = "low"
        elif prob_pos < high_thr:
            band = "review"
        else:
            band = "high"

        class_probs = {
            self.config.label_names[i]: float(probs[i]) for i in range(probs.shape[0])
        }

        return ScreeningResult(
            risk_score=int(round(prob_pos * 100)),
            prob_dementia=prob_pos,
            confidence=confidence,
            risk_band=band,
            predicted_label=pred_label,
            class_probabilities=class_probs,
            markers=markers,
            inference_ms=inference_ms,
            preprocess_ms=preprocess_ms,
            info=info,
        )


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    # 1-D numerically stable softmax. Avoids scipy.special.softmax dependency.
    x = x - float(np.max(x))
    e = np.exp(x)
    return e / float(np.sum(e))
