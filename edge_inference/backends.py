"""Model backends for edge inference.

Three backends are supported:
  * onnx                : onnxruntime, CPU provider only on Pi 4.
  * tflite              : tflite-runtime (lightweight) or full tensorflow as fallback.
  * wav2vec2_two_stage  : two-stage ONNX pipeline — chunk accumulator backbone
                          followed by a classification head.

The backend interface is intentionally tiny:

    class Backend:
        input_name: str
        input_shape: tuple[int, ...]   # full shape including batch
        output_names: list[str]
        def run(self, tensor: np.ndarray) -> dict[str, np.ndarray]: ...
        def close(self) -> None: ...

Each backend lazy-imports its dependency so users only pay the import cost
for the runtime they actually use. On a Pi 4, onnxruntime cold-import is
~150 ms; tflite-runtime is ~30 ms.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class InferenceBackend(ABC):
    """Common surface for all model backends."""

    input_name: str
    input_shape: Tuple[int, ...]
    output_names: List[str]

    @abstractmethod
    def run(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """Run forward pass. Returns a dict keyed by output tensor name."""

    def close(self) -> None:
        """Release native resources. Default is a no-op."""
        return None

    def __enter__(self) -> "InferenceBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ----------------------------------------------------------------------------
# ONNX Runtime backend
# ----------------------------------------------------------------------------

class OnnxBackend(InferenceBackend):
    """Wrap an onnxruntime InferenceSession.

    On Pi 4 we explicitly request CPUExecutionProvider — no GPU, no NNAPI.
    Quantized (INT8) ONNX models still expose a float input in most cases;
    the int8 quantization is internal to the graph.
    """

    def __init__(
        self,
        model_path: Path,
        input_name_override: Optional[str] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime not installed. On Pi 4: pip install onnxruntime"
            ) from exc

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        sess_options = ort.SessionOptions()
        if num_threads is not None:
            sess_options.intra_op_num_threads = int(num_threads)
            sess_options.inter_op_num_threads = 1
        # Disable graph optimizations only if we hit issues; default ALL is fine.
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Pi 4 has no GPU; pin to CPU explicitly to avoid provider warnings.
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._inputs = self._session.get_inputs()
        self._outputs = self._session.get_outputs()

        if input_name_override is not None:
            if input_name_override not in {i.name for i in self._inputs}:
                raise ValueError(
                    f"input_name {input_name_override!r} not in model inputs "
                    f"{[i.name for i in self._inputs]}"
                )
            self.input_name = input_name_override
        else:
            if len(self._inputs) != 1:
                raise ValueError(
                    f"Model has {len(self._inputs)} inputs; please set input_name "
                    f"in config. Available: {[i.name for i in self._inputs]}"
                )
            self.input_name = self._inputs[0].name

        # Shape may contain None / strings for dynamic axes; keep as-is for diagnostics.
        in_meta = next(i for i in self._inputs if i.name == self.input_name)
        self.input_shape = tuple(in_meta.shape)
        self.output_names = [o.name for o in self._outputs]

    def run(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        outputs = self._session.run(self.output_names, {self.input_name: tensor})
        return {name: arr for name, arr in zip(self.output_names, outputs)}

    def close(self) -> None:
        # onnxruntime session has no explicit close; rely on GC. Drop reference
        # so a long-lived process can release memory if backend is replaced.
        self._session = None  # type: ignore[assignment]


# ----------------------------------------------------------------------------
# TFLite backend
# ----------------------------------------------------------------------------

class TFLiteBackend(InferenceBackend):
    """Wrap a TFLite interpreter.

    Prefers `tflite_runtime.interpreter` (small, ARM-friendly) and falls back
    to `tensorflow.lite.Interpreter`. On Pi 4 install:
        pip install tflite-runtime
    """

    def __init__(
        self,
        model_path: Path,
        input_name_override: Optional[str] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        Interpreter = self._import_interpreter()

        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        kwargs = {"model_path": str(model_path)}
        if num_threads is not None:
            kwargs["num_threads"] = int(num_threads)

        self._interpreter = Interpreter(**kwargs)
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        if len(self._input_details) != 1 and input_name_override is None:
            raise ValueError(
                f"Model has {len(self._input_details)} inputs; please set input_name. "
                f"Available: {[d['name'] for d in self._input_details]}"
            )

        if input_name_override is not None:
            matched = [d for d in self._input_details if d["name"] == input_name_override]
            if not matched:
                raise ValueError(
                    f"input_name {input_name_override!r} not in model inputs "
                    f"{[d['name'] for d in self._input_details]}"
                )
            self._input_detail = matched[0]
        else:
            self._input_detail = self._input_details[0]

        self.input_name = self._input_detail["name"]
        self.input_shape = tuple(int(d) for d in self._input_detail["shape"])
        self.output_names = [d["name"] for d in self._output_details]

    @staticmethod
    def _import_interpreter():
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
            return Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter  # type: ignore
                return Interpreter
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Neither tflite_runtime nor tensorflow installed. "
                    "On Pi 4: pip install tflite-runtime"
                ) from exc

    def run(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        # TFLite often has fixed shapes; resize input if our tensor differs.
        expected = self._input_detail["shape"]
        if tuple(int(d) for d in expected) != tuple(tensor.shape):
            self._interpreter.resize_tensor_input(self._input_detail["index"], list(tensor.shape))
            self._interpreter.allocate_tensors()

        self._interpreter.set_tensor(self._input_detail["index"], tensor)
        self._interpreter.invoke()
        return {
            d["name"]: self._interpreter.get_tensor(d["index"])
            for d in self._output_details
        }

    def close(self) -> None:
        self._interpreter = None  # type: ignore[assignment]


# ----------------------------------------------------------------------------
# Two-stage wav2vec2 accumulator + classification head
# ----------------------------------------------------------------------------

class Wav2Vec2TwoStageBackend(InferenceBackend):
    """Two-stage ONNX inference pipeline for wav2vec2 dementia screening.

    Stage 1 — backbone (accumulator):
        Inputs:  input_audio  [1, 128000]  float32
                 valid_samples [1]          int64   (real samples in this chunk)
        Outputs: sum_vec      [1, 768]     float32  (frame-level hidden-state sum)
                 sumsq_vec    [1, 768]     float32  (frame-level sum of squares)
                 frame_count  scalar/[1]   int64

    For audio longer than one chunk the backbone is called once per chunk and
    the statistics are accumulated externally before computing the embedding.

    Stage 2 — head:
        Input:  input_features [1, 1536]   float32  (concat of mean and std)
        Output: logits         [1, 2]      float32  (Platt-calibrated; softmax → P)
    """

    CHUNK_SIZE: int = 128_000             # 8 s at 16 kHz, must match the exported model
    MIN_RETAINED_SAMPLES: int = 8_000    # 0.5 s — from accumulator contract clip_level_rule

    def __init__(
        self,
        backbone_path: Path,
        head_path: Path,
        num_threads: Optional[int] = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime not installed. On Pi 4: pip install onnxruntime"
            ) from exc

        for p in (backbone_path, head_path):
            if not p.exists():
                raise FileNotFoundError(f"ONNX model not found: {p}")

        def _make_session(path: Path) -> "ort.InferenceSession":
            opts = ort.SessionOptions()
            if num_threads is not None:
                opts.intra_op_num_threads = int(num_threads)
                opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )

        self._backbone = _make_session(backbone_path)
        self._head = _make_session(head_path)

        # Cache output name lists to avoid repeated get_outputs() calls per chunk.
        self._backbone_output_names: List[str] = [
            o.name for o in self._backbone.get_outputs()
        ]
        self._head_output_names: List[str] = [
            o.name for o in self._head.get_outputs()
        ]

        self.input_name = "input_audio"
        self.input_shape = (1, self.CHUNK_SIZE)
        self.output_names = ["logits"]

    def run(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        audio: np.ndarray = np.asarray(tensor, dtype=np.float32).ravel()
        n_samples = len(audio)
        is_multi_chunk = n_samples > self.CHUNK_SIZE
        n_chunks_expected = (n_samples + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
        chunk_idx = 0
        logger.debug(
            "run(): audio_samples=%d (%.2f s), expected_chunks=%d",
            n_samples, n_samples / 16_000, n_chunks_expected,
        )

        acc_sum: Optional[np.ndarray] = None
        acc_sumsq: Optional[np.ndarray] = None
        acc_frames: int = 0

        offset = 0
        while offset < n_samples:
            raw_chunk = audio[offset : offset + self.CHUNK_SIZE]
            valid_len = len(raw_chunk)
            is_partial = valid_len < self.CHUNK_SIZE

            # Contract drop rule: in multi-chunk mode, discard a final partial
            # chunk that is shorter than 0.5 s — it doesn't contribute enough
            # frames to be meaningful and wasn't used during training.
            if is_partial and is_multi_chunk and valid_len < self.MIN_RETAINED_SAMPLES:
                break

            # Contract single-chunk rule: if the only chunk is shorter than 0.5 s
            # treat valid_samples as 0.5 s (minimum retained) before passing to model.
            if is_partial and not is_multi_chunk and valid_len < self.MIN_RETAINED_SAMPLES:
                valid_len = self.MIN_RETAINED_SAMPLES

            if is_partial:
                raw_chunk = np.pad(raw_chunk, (0, self.CHUNK_SIZE - len(raw_chunk)))
            chunk_in = raw_chunk.reshape(1, self.CHUNK_SIZE)
            valid_in = np.array([valid_len], dtype=np.int64)

            outs = self._backbone.run(
                self._backbone_output_names,
                {"input_audio": chunk_in, "valid_samples": valid_in},
            )
            d = dict(zip(self._backbone_output_names, outs))

            fc = int(np.squeeze(d["frame_count"]))
            logger.debug(
                "  chunk %d: offset=%d valid_samples=%d frame_count=%d",
                chunk_idx, offset, valid_len, fc,
            )
            chunk_idx += 1

            if fc == 0:
                offset += self.CHUNK_SIZE
                continue

            if acc_sum is None:
                acc_sum = d["sum_vec"].copy()
                acc_sumsq = d["sumsq_vec"].copy()
            else:
                acc_sum = acc_sum + d["sum_vec"]
                acc_sumsq = acc_sumsq + d["sumsq_vec"]
            acc_frames += fc
            offset += self.CHUNK_SIZE

        if acc_sum is None or acc_frames == 0:
            logger.warning("No valid frames accumulated — returning neutral logits.")
            return {"logits": np.zeros((1, 2), dtype=np.float32)}

        mean = acc_sum / float(acc_frames)
        var = acc_sumsq / float(acc_frames) - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0) + 1e-8)
        embedding = np.concatenate([mean, std], axis=-1).reshape(1, -1).astype(np.float32)

        logger.debug(
            "Embedding built: total_frames=%d  mean_norm=%.4f  std_norm=%.4f  "
            "embedding_shape=%s",
            acc_frames,
            float(np.linalg.norm(mean)),
            float(np.linalg.norm(std)),
            embedding.shape,
        )

        head_outs = self._head.run(
            self._head_output_names, {"input_features": embedding}
        )
        head_d = dict(zip(self._head_output_names, head_outs))
        logits = head_d["logits"]
        logger.debug(
            "Head output: raw_logits=%s", logits.ravel().tolist()
        )
        return {"logits": logits}

    def close(self) -> None:
        self._backbone = None  # type: ignore[assignment]
        self._head = None  # type: ignore[assignment]


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------

def build_backend(
    backend_kind: str,
    model_path: Path,
    input_name: Optional[str] = None,
    num_threads: Optional[int] = None,
    head_path: Optional[Path] = None,
) -> InferenceBackend:
    """Create the backend selected by config.

    Centralized so wrapper.py never imports onnxruntime / tflite-runtime
    speculatively.
    """
    if backend_kind == "onnx":
        return OnnxBackend(model_path, input_name_override=input_name, num_threads=num_threads)
    if backend_kind == "tflite":
        return TFLiteBackend(model_path, input_name_override=input_name, num_threads=num_threads)
    if backend_kind == "wav2vec2_two_stage":
        if head_path is None:
            raise ValueError(
                "head_path is required for the wav2vec2_two_stage backend. "
                "Set head_model_path in your EdgeConfig."
            )
        return Wav2Vec2TwoStageBackend(model_path, head_path, num_threads=num_threads)
    raise ValueError(f"Unknown backend {backend_kind!r}")
