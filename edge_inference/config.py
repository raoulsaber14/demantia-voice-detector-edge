"""Configuration for the edge inference wrapper.

Single source of truth for: which model file to load, expected input shape,
label mapping, risk-band thresholds. Edit `model_config.yaml` when the teammate
delivers a real model — code does not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EdgeConfig:
    """Runtime configuration for on-device inference.

    Attributes
    ----------
    model_path:
        Filesystem path to the model file. Suffix determines backend
        (.onnx, .tflite, or "dummy" for test mode).
    backend:
        One of {"auto", "onnx", "tflite", "dummy"}. "auto" infers from file
        extension; "dummy" returns random scores so the rest of the pipeline
        can be validated without a real model.
    sample_rate:
        Hard contract with the model. Must be 16000 to match teammate's training.
    expected_input_seconds:
        Audio duration the model was trained on. Wav2vec2 training used 3.0 s
        segments; the inference path used max 20 s. We pick a value that fits
        on a Pi 4 (≤10 s recommended for latency).
    pad_mode:
        "zero" pads short clips with silence; "reflect" mirrors. Zero is safer
        for VAD-cleaned input.
    truncate_mode:
        "head" keeps the first window; "center" keeps the middle; "loudest"
        picks the highest-RMS window. Center is a defensible default.
    input_name:
        ONNX/TFLite input tensor name. Auto-detected if None.
    input_dtype:
        "float32" for raw waveform models (wav2vec2). "int8" only if the
        teammate ships a fully-quantized model with an integer input — rare;
        usually quantized models still take float input and quantize internally.
    add_batch_dim:
        Most exported audio models expect shape (1, samples). Some classical
        feature-based models expect (1, n_features). Set False for the latter.
    label_names:
        Index → label string. Order must match the model's output classes.
    positive_class_index:
        Which output index represents "dementia". Confirmed as 1 in
        teammate's inference.py (`probs[1]`).
    risk_band_thresholds:
        Cutoffs on P(positive_class) for low/medium/high banding. Per
        final_model_decision.md the model is screening-support only — three
        bands ("low", "review", "high") are appropriate, not a binary verdict.
    """

    model_path: Path
    backend: str = "auto"
    sample_rate: int = 16_000
    expected_input_seconds: float = 10.0
    pad_mode: str = "zero"
    truncate_mode: str = "center"
    input_name: Optional[str] = None
    input_dtype: str = "float32"
    add_batch_dim: bool = True
    label_names: Tuple[str, ...] = ("non_dementia", "dementia")
    positive_class_index: int = 1
    risk_band_thresholds: Tuple[float, float] = (0.33, 0.66)
    marker_names: Tuple[str, ...] = ()
    # Two-stage pipeline: set this to the head model path to activate
    # wav2vec2_two_stage backend (backbone path goes in model_path as usual).
    head_model_path: Optional[Path] = None

    @property
    def expected_input_samples(self) -> int:
        return int(round(self.sample_rate * self.expected_input_seconds))

    def resolved_backend(self) -> str:
        """Return concrete backend after resolving 'auto'."""
        if self.backend != "auto":
            return self.backend
        # Two-stage pipeline detected when a head path is provided alongside
        # an ONNX backbone.
        if self.head_model_path is not None:
            return "wav2vec2_two_stage"
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        if suffix == ".tflite":
            return "tflite"
        if str(self.model_path).lower() == "dummy" or suffix == ".dummy":
            return "dummy"
        raise ValueError(
            f"Cannot infer backend from path {self.model_path!r}. "
            f"Set backend explicitly in config (one of: onnx, tflite, dummy, wav2vec2_two_stage)."
        )

    def validate(self) -> None:
        """Fail fast on misconfiguration."""
        if self.sample_rate != 16_000:
            raise ValueError(
                f"sample_rate={self.sample_rate}; teammate's model is trained at 16 kHz. "
                "Resample upstream rather than changing this."
            )
        if self.expected_input_seconds <= 0:
            raise ValueError("expected_input_seconds must be > 0")
        if self.pad_mode not in {"zero", "reflect"}:
            raise ValueError(f"pad_mode={self.pad_mode!r} not in {{'zero', 'reflect'}}")
        if self.truncate_mode not in {"head", "center", "loudest"}:
            raise ValueError(f"truncate_mode={self.truncate_mode!r} not in {{'head','center','loudest'}}")
        if self.input_dtype not in {"float32", "int8", "uint8"}:
            raise ValueError(f"input_dtype={self.input_dtype!r} unsupported")
        if not (0 <= self.positive_class_index < len(self.label_names)):
            raise ValueError("positive_class_index out of range for label_names")
        low, high = self.risk_band_thresholds
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"risk_band_thresholds must be 0<=low<high<=1, got ({low},{high})")


def load_config(path: str | Path) -> EdgeConfig:
    """Load an EdgeConfig from a YAML file.

    YAML is optional; if PyYAML isn't installed, we parse a minimal subset
    ourselves (key: value, lists with [a, b, c]). Keeps the Pi install lean.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
    except ImportError:
        data = _parse_minimal_yaml(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")

    # Coerce types; tolerate missing optional keys.
    model_path_raw = data.get("model_path")
    if not model_path_raw:
        raise ValueError("Config missing required field: model_path")

    head_model_path_raw = data.get("head_model_path")

    cfg = EdgeConfig(
        model_path=Path(str(model_path_raw)),
        backend=str(data.get("backend", "auto")),
        sample_rate=int(data.get("sample_rate", 16_000)),
        expected_input_seconds=float(data.get("expected_input_seconds", 10.0)),
        pad_mode=str(data.get("pad_mode", "zero")),
        truncate_mode=str(data.get("truncate_mode", "center")),
        input_name=(str(data["input_name"]) if data.get("input_name") else None),
        input_dtype=str(data.get("input_dtype", "float32")),
        add_batch_dim=bool(data.get("add_batch_dim", True)),
        label_names=tuple(data.get("label_names", ("non_dementia", "dementia"))),
        positive_class_index=int(data.get("positive_class_index", 1)),
        risk_band_thresholds=tuple(data.get("risk_band_thresholds", (0.33, 0.66))),
        marker_names=tuple(data.get("marker_names", ())),
        head_model_path=Path(str(head_model_path_raw)) if head_model_path_raw else None,
    )
    cfg.validate()
    return cfg


def _parse_minimal_yaml(text: str) -> Dict[str, object]:
    """Tiny YAML subset parser so PyYAML isn't a hard dep on the Pi.

    Supports:
        key: value                # scalars (str, int, float, bool, null)
        key: [a, b, c]            # inline lists
        # comments
    Does not support nested mappings, multi-line strings, or anchors.
    """
    out: Dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not value:
            continue
        out[key] = _coerce_yaml_value(value)
    return out


def _coerce_yaml_value(value: str) -> object:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce_yaml_value(item.strip()) for item in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    low = value.lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    if low in {"null", "none", "~"}:
        return None
    try:
        if "." in value or "e" in low:
            return float(value)
        return int(value)
    except ValueError:
        return value
