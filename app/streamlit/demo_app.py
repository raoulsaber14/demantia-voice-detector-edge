"""Simple Streamlit demo for local inference."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import DISCLAIMER, predict_with_baseline, predict_with_wav2vec
from src.phaseD_ensemble_runtime import predict_with_phaseD_ensemble


st.set_page_config(page_title="Dementia Voice Screening Demo", layout="centered")
st.title("Dementia Voice Screening Demo")
st.warning(DISCLAIMER)

model_type = st.selectbox("Model type", ["baseline", "wav2vec", "phaseD_ensemble"])
if model_type == "baseline":
    default_path = "models/calibrated_phase3_governed/logistic_regression_calibrated.joblib"
    default_threshold = 0.300
elif model_type == "wav2vec":
    default_path = "models/wav2vec/wav2vec_base_20260418_001/full/best_model.pt"
    default_threshold = 0.454939
else:
    default_path = "models/phaseD_frozen_ensemble_deploy/max_probability_ensemble_manifest.json"
    default_threshold = 0.500
model_path = st.text_input("Model path", value=default_path)
threshold = st.number_input("Decision threshold", min_value=0.0, max_value=1.0, value=float(default_threshold), step=0.01)
uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac"])

if uploaded is not None and st.button("Run prediction"):
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    try:
        if model_type == "baseline":
            result = predict_with_baseline(tmp_path, Path(model_path), threshold=float(threshold))
        elif model_type == "wav2vec":
            result = predict_with_wav2vec(tmp_path, Path(model_path), threshold=float(threshold))
        else:
            result = predict_with_phaseD_ensemble(tmp_path, Path(model_path), threshold_override=float(threshold))
            result["disclaimer"] = DISCLAIMER
        st.success(f"Predicted label: {result['predicted_label']}")
        st.write(f"Confidence: {result['confidence_score']:.4f}")
        st.write(f"Dementia probability: {result['probability_dementia']:.4f}")
        st.write(f"Threshold: {result['threshold']:.6f}")
        if "max_component_name" in result:
            st.write(f"Max component: {result['max_component_name']}")
        if "n_segments" in result:
            st.write(f"Segments scored: {result['n_segments']}")
        if "component_scores" in result:
            st.json(result["component_scores"])
        st.info(result["explanation"])
        st.caption(result["disclaimer"])
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
