"""Create a real (random-weight) ONNX or TFLite model for end-to-end testing.

Generates a tiny graph that takes a (1, N) float32 audio tensor and emits a
(1, num_classes) logits tensor — same I/O contract a real wav2vec2 export
would have, just much smaller. Lets you exercise the full ONNX / TFLite code
paths on the Pi without waiting for the teammate's real model.

Usage:
    python -m edge_inference.make_dummy_model \
        --format onnx --samples 160000 --classes 2 \
        --out /tmp/dummy_dementia.onnx

Note: this creates a real on-disk model file, distinct from DummyBackend
(which doesn't need a file at all). Use DummyBackend when you don't want any
ML runtime installed; use this script when you specifically want to verify
the ONNX / TFLite code paths work on your Pi setup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple


def make_onnx(out_path: Path, n_samples: int, n_classes: int) -> Path:
    """Build a tiny ONNX model: input (1,N) -> linear -> logits (1,C)."""
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError(
            "onnx package required. Install on dev machine with: pip install onnx"
        ) from exc

    rng = np.random.default_rng(0)
    # Linear layer: matmul (1,N) x (N, C) -> (1, C). Tiny weights to keep sane outputs.
    weights = (rng.standard_normal((n_samples, n_classes)) * 0.001).astype("float32")
    bias = np.zeros((n_classes,), dtype="float32")

    input_tensor = helper.make_tensor_value_info("input_values", TensorProto.FLOAT, [1, n_samples])
    output_tensor = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n_classes])

    weights_init = numpy_helper.from_array(weights, name="W")
    bias_init = numpy_helper.from_array(bias, name="B")

    matmul = helper.make_node("MatMul", ["input_values", "W"], ["matmul_out"])
    add = helper.make_node("Add", ["matmul_out", "B"], ["logits"])

    graph = helper.make_graph(
        nodes=[matmul, add],
        name="dummy_dementia",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weights_init, bias_init],
    )

    # opset 13 is widely supported by onnxruntime versions on Pi (1.14+).
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="edge_inference.make_dummy_model",
    )
    # ir_version 7 matches opset 13; newer onnxruntime accepts higher.
    model.ir_version = 7
    onnx.checker.check_model(model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path


def make_tflite(out_path: Path, n_samples: int, n_classes: int) -> Path:
    """Build a tiny TFLite model with the same I/O.

    Requires tensorflow on the dev machine to author. The Pi only needs
    tflite-runtime to *consume* the model.
    """
    try:
        import numpy as np
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow required to author TFLite. On dev machine: "
            "pip install tensorflow. (Pi only needs tflite-runtime.)"
        ) from exc

    rng = np.random.default_rng(0)

    class Tiny(tf.keras.Model):
        def __init__(self):
            super().__init__()
            init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=0)
            self.dense = tf.keras.layers.Dense(n_classes, kernel_initializer=init, use_bias=True)

        def call(self, x):
            return self.dense(x)

    model = Tiny()
    model.build(input_shape=(1, n_samples))

    @tf.function(input_signature=[tf.TensorSpec([1, n_samples], tf.float32, name="input_values")])
    def serve(x):
        return {"logits": model(x)}

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()])
    tflite_bytes = converter.convert()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Create a dummy ONNX/TFLite model for testing.")
    p.add_argument("--format", choices=["onnx", "tflite"], required=True)
    p.add_argument("--samples", type=int, default=160_000, help="Audio samples per inference (default 10s @ 16kHz)")
    p.add_argument("--classes", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.format == "onnx":
        path = make_onnx(args.out, args.samples, args.classes)
    else:
        path = make_tflite(args.out, args.samples, args.classes)

    print(f"Wrote {args.format} dummy model: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
