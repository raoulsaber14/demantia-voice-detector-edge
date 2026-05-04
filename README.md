# Edge AI Dementia Voice Early Detector

**Course:** EECE 490/690 — Final Year Project
**Team:** Celine Sadaka · Zeina Hammoud · Raoul Saber
**Institution:** American University of Beirut

---

## Project Overview

Early dementia risk signs are often missed when clinical follow-up is delayed. Speech is a low-cost, accessible signal: dementia affects language fluency, word retrieval, and prosody in measurable ways. This project builds a voice-based dementia-risk screening-support system that runs fully on a Raspberry Pi 4 — no cloud, no external API, no internet connection required.

The system records a speech sample from a USB microphone (or accepts an uploaded WAV file), cleans the audio through a signal processing pipeline, runs a two-stage ONNX inference pipeline, and displays a risk band to the clinician through a browser-based UI.

**This is a screening-support tool, not a medical diagnosis system.** All outputs are risk indicators intended to support referral decisions, not to confirm or rule out dementia.

---

## Problem Statement

- Dementia affects millions globally and is severely under-detected in early stages.
- Clinical assessment is expensive, time-consuming, and access-limited.
- Speech patterns — pause frequency, word retrieval latency, prosody, fluency — degrade measurably with dementia onset.
- A low-cost, deployable voice screening tool could support earlier referrals in under-resourced settings.

**Goal:** Build a recall-prioritized voice-based dementia-risk screening model that can run on edge hardware (Raspberry Pi 4) without an internet connection.

---

## Dataset

The dataset is a curated, trusted subset of the DementiaBank corpus (Pitt Corpus), cleaned and governed through multiple audit passes.

| Split | Speakers | Clips |
|---|---|---|
| Dementia-positive | 84 | 94 |
| Non-dementia (control) | 107 | 254 |
| **Total** | **191** | **348** |

Key characteristics:
- Positive-speaker evidence is sparse: 74 of 84 dementia speakers have only one clip; 10 have two clips.
- All evaluation uses speaker-grouped cross-validation (`StratifiedGroupKFold` by `speaker_id`) to prevent speaker leakage across folds.
- 5 random seeds (42–46) are used for stability analysis.
- Audio is resampled to 16 kHz mono and preprocessed through bandpass filtering, spectral subtraction, and VAD before feature extraction.

---

## System Architecture

The project has two coupled components:

```
┌─────────────────────────────────────────────────────┐
│  ML Training Pipeline  (ml project repo)            │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Features │→ │Baseline  │→ │Deep Embedding     │  │
│  │ (256     │  │LR Model  │  │Ensemble (Phase D) │  │
│  │ acoustic)│  │          │  │HuBERT + wav2vec2  │  │
│  └──────────┘  └──────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────┘
                                        │ exported as ONNX
┌─────────────────────────────────────────────────────┐
│  Edge Deployment  (this repo)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │ USB Mic /    │→ │ Audio        │→ │ ONNX     │  │
│  │ WAV upload   │  │ Pipeline     │  │ Inference│  │
│  └──────────────┘  │ (16kHz, VAD, │  │ (2-stage)│  │
│                    │ bandpass,    │  └──────────┘  │
│                    │ denoise)     │        │        │
│                    └──────────────┘        ↓        │
│                                    ┌──────────────┐  │
│                                    │ Flask UI     │  │
│                                    │ Risk band +  │  │
│                                    │ score output │  │
│                                    └──────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Repository Layout

```text
.
├── app/
│   └── flask/
│       ├── web_server.py        # Flask server (entry point)
│       └── templates/
│           └── index.html       # Browser UI
├── configs/
│   └── edge_inference.yaml      # Model paths, labels, risk thresholds
├── edge_inference/
│   ├── backends.py              # ONNX / TFLite / two-stage backends
│   ├── config.py                # EdgeConfig dataclass + YAML loader
│   ├── preprocessing.py         # Waveform → tensor (resample, pad, truncate)
│   └── wrapper.py               # DementiaScreener public API
├── models/
│   ├── baseline_phase3_governed/
│   │   └── logistic_regression.joblib          # Raw LR baseline
│   ├── calibrated_phase3_governed/
│   │   └── logistic_regression_calibrated.joblib  # Isotonic-calibrated LR
│   └── phaseD_frozen_ensemble_deploy/          # Final model (used by Flask server)
│       └── (wav2vec2 + HuBERT ONNX weights, see Models section)
├── src/
│   ├── audio_pipeline.py        # Mic capture, VAD cleaning, WAV loading
│   ├── audio_utils.py           # Low-level audio I/O and signal helpers
│   ├── config.py                # Central CFG dataclass (paths, constants)
│   ├── evaluate.py              # Metrics, calibration curves, result figures
│   ├── features.py              # Hand-crafted acoustic feature extraction
│   ├── metadata_utils.py        # Dataset metadata loading and validation
│   ├── preprocess.py            # Audio preprocessing pipeline
│   ├── train_baseline.py        # Logistic regression / SVM / RF baseline training
│   ├── train_wav2vec.py         # wav2vec2 fine-tuning pipeline
│   └── vad_segment.py           # Voice activity detection segmentation
├── scripts/
│   ├── dataset_audit.py         # Dataset integrity checks
│   ├── phase3_closeout.py       # Phase 3 governance and pruning
│   └── phase4_baseline_experiments.py  # Extended baseline experiment suite
├── tests/
│   └── test_metadata_utils.py   # Unit tests for metadata utilities
├── requirements.txt
├── Makefile
└── Dockerfile
```

---

## Models

### Baseline Model — Logistic Regression (isotonic calibrated)

A classical ML baseline trained on 256 hand-crafted acoustic and prosodic features extracted from the governed-pruned feature table.

**Feature families:** MFCC (mean/variance per coefficient), spectral centroid/bandwidth/rolloff, energy/RMS, pitch (F0) statistics, pause and speech timing, zero-crossing rate, voice quality indicators.

**Training:** L2 logistic regression, median imputation, standard scaling, balanced class weights, speaker-grouped 5-fold CV, isotonic calibration. Threshold frozen at 0.250.

**Clip-level results (seed 42, threshold 0.250):**

| Metric | Value |
|---|---|
| Recall (sensitivity) | **0.404** |
| Specificity | 0.693 |
| Precision | 0.328 |
| F1 | 0.362 |
| Balanced Accuracy | 0.549 |
| PR-AUC | 0.342 |
| Brier Score | 0.195 |
| TN / FP / FN / TP | 176 / 78 / 56 / 38 |

The baseline catches 38 of 94 dementia clips and misses 56. Recall is intentionally prioritized over specificity given the screening framing (false negatives are more harmful than false positives in a referral-support context).

Saved at: `models/calibrated_phase3_governed/logistic_regression_calibrated.joblib`
Training code: [`src/train_baseline.py`](src/train_baseline.py)

---

### Final Model — Max Probability Ensemble (Phase D)

The final deployed model is a three-component ensemble of frozen deep speech embedding models, selected after a controlled recall-recovery benchmark (Phase D).

**Components:**
1. `facebook/hubert-base-ls960` — HuBERT embeddings only (1536-dim mean+std pooling)
2. `facebook/hubert-base-ls960` + 256 handcrafted features fused — HuBERT + acoustic fusion
3. `facebook/wav2vec2-base` — wav2vec2 embeddings only (1536-dim mean+std pooling)

**Ensemble rule:** `max_probability` — a speaker is flagged positive if any component exceeds the threshold. This maximizes recall at the cost of some specificity.

**Why wav2vec2 matters:** HuBERT-only and HuBERT+handcrafted were highly redundant with each other (FN overlap 0.815, FP overlap 0.728). wav2vec2 provided the main complementary recovery signal, contributing 51 of 70 unique true-positive recoveries across seeds.

**Speaker-level results (mean across seeds 42–46):**

| Metric | Baseline LR | Final Ensemble | Change |
|---|---|---|---|
| Recall | 0.404 | **0.660** | +0.256 |
| Specificity | 0.693 | 0.680 | −0.013 |
| F1 | 0.362 | **0.638** | +0.276 |

The ensemble improves speaker recall by +25.6 percentage points while preserving specificity above the minimum acceptable floor (0.50). The target zone (recall ≥ 0.70, specificity ≥ 0.50) was not reached — this result is held as the best candidate, not a fully validated endpoint.

**False negative / false positive breakdown across seeds 42–46:**

| Seed | False Negatives | False Positives |
|---|---|---|
| 42 | 25 | 36 |
| 43 | 27 | 31 |
| 44 | 31 | 37 |
| 45 | 29 | 33 |
| 46 | 31 | 34 |
| **Mean** | **28.6** | **34.2** |

Exported to ONNX and deployed at: `models/phaseD_frozen_ensemble_deploy/`

---

### Model Progression Summary

| Phase | Model | Speaker Recall | Speaker Specificity |
|---|---|---|---|
| Baseline (Phase 4) | Logistic Regression (isotonic) | 0.404 | 0.693 |
| Phase C | HuBERT frozen embedding + LR | 0.476 | 0.867 |
| Phase D — HuBERT only | HuBERT aggregation (majority vote) | 0.495 | 0.757 |
| Phase D — HuBERT + handcrafted | Fusion LR | 0.476 | 0.768 |
| **Phase D — Ensemble** | **Max probability (HuBERT + wav2vec2)** | **0.660** | **0.680** |

---

### Deployed Model Files

```
models/
├── baseline_phase3_governed/
│   └── logistic_regression.joblib                      ← raw baseline
├── calibrated_phase3_governed/
│   └── logistic_regression_calibrated.joblib           ← isotonic-calibrated baseline
├── phaseD_frozen_ensemble_deploy/                       ← used by the Flask web server
│   ├── max_probability_ensemble_manifest.json
│   ├── wav2vec2_component.json
│   ├── wav2vec2_component_head_fp32.onnx               ← wav2vec2 classification head
│   ├── wav2vec2_component_head_contract.json
│   ├── wav2vec2_base/
│   │   ├── wav2vec2_base_chunk8_accumulator_fp32_external/
│   │   │   └── model.onnx                             ← wav2vec2 backbone (fp32)
│   │   └── wav2vec2_base_chunk8_accumulator_int8_external/
│   │       └── model.onnx                             ← wav2vec2 backbone (int8, quantized)
│   ├── hubert_only_component.json
│   ├── hubert_only_component_head_fp32.onnx
│   ├── hubert_plus_handcrafted_component.json
│   ├── hubert_plus_handcrafted_component_head_fp32.onnx
│   └── hubert_base_ls960/
│       ├── hubert_base_ls960_chunk8_accumulator_fp32_external/
│       │   └── model.onnx                             ← HuBERT backbone (fp32)
│       └── hubert_base_ls960_chunk8_accumulator_int8_external/
│           └── model.onnx                             ← HuBERT backbone (int8, quantized)
└── wav2vec/
    └── wav2vec_base_20260418_001/
        └── full/
            └── config.json
```

The Flask web server uses the **wav2vec2 fp32 two-stage pipeline** by default (`configs/edge_inference.yaml`): backbone at `phaseD_frozen_ensemble_deploy/wav2vec2_base/wav2vec2_base_chunk8_accumulator_fp32_external/model.onnx` and head at `phaseD_frozen_ensemble_deploy/wav2vec2_component_head_fp32.onnx`.

---

## Audio Processing Pipeline

All audio (microphone or uploaded WAV) passes through a four-stage signal processing chain before inference:

1. **Bandpass filter** — 5th-order Butterworth, 80 Hz – 8 kHz. Removes sub-bass rumble and high-frequency noise outside the speech band.
2. **Spectral subtraction** — estimates the noise profile from the first 0.5 s of the recording and subtracts it frame by frame (n_fft=1024, hop=256, oversubtraction α=2.0, spectral floor β=0.02).
3. **Voice Activity Detection (VAD)** — energy + zero-crossing rate per 30 ms frame, with hangover smoothing and minimum speech chunk filtering. Skipped by default to preserve the full audio length for the accumulator backbone.
4. **Peak normalization** — normalizes peak amplitude to −3 dB.

Output: 16 kHz mono float32 WAV ready for ONNX inference.

Pipeline code: [`src/audio_pipeline.py`](src/audio_pipeline.py)

---

## Edge Inference

The edge inference module (`edge_inference/`) wraps the ONNX models behind a clean API:

- **`DementiaScreener`** (`wrapper.py`) — public API: `.predict(audio, sample_rate)` → `ScreeningResult`
- **`backends.py`** — supports ONNX (single-stage), TFLite, and the two-stage wav2vec2 accumulator pipeline
- **`preprocessing.py`** — resamples and pads/truncates audio to match the model's input contract
- **`config.py`** — loads `configs/edge_inference.yaml`, resolves backend, labels, and risk thresholds

**Risk banding** (configured in `configs/edge_inference.yaml`):

| P(dementia) | Risk Band |
|---|---|
| < 0.33 | LOW RISK |
| 0.33 – 0.66 | REQUIRES REVIEW |
| > 0.66 | HIGH RISK INDICATOR |

The output is always presented as a screening indicator, never as a diagnosis.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires PortAudio for microphone capture:
```bash
sudo apt install portaudio19-dev   # Raspberry Pi OS / Debian
brew install portaudio             # macOS
```

---

## Running

```bash
python3 -m app.flask.web_server
```

Then open `http://<pi-ip>:5000` in any browser on the same network.

**Important:** run as a module (`python3 -m app.flask.web_server`) from the project root, not as a script (`python3 app/flask/web_server.py`).

### Makefile targets

| Target | Description |
|---|---|
| `make setup` | Create venv and install requirements |
| `make baseline` | Train logistic regression baseline |
| `make features` | Extract hand-crafted acoustic features |
| `make preprocess` | Run audio preprocessing pipeline |
| `make dataset-audit` | Run dataset integrity checks |
| `make compile` | Syntax-check all Python source files |
| `make test` | Run unit tests |
| `make clean-generated` | Remove `__pycache__` and `.pytest_cache` |

---

## How It Works

The Pi acts as a local web server. The browser UI lets a clinician either record directly from the Pi's USB microphone or upload a WAV file. The audio passes through the cleaning pipeline, then through the two-stage ONNX inference:

1. **Backbone** — the wav2vec2 accumulator processes audio in 8-second chunks and accumulates frame-level mean+std statistics across chunks.
2. **Head** — a classification head takes the pooled embedding and outputs P(dementia).

Results are displayed as a risk band (LOW / REQUIRES REVIEW / HIGH RISK INDICATOR) with a 0–100 risk score. A plain-text report can be downloaded per session.

---

## Limitations

- **Small dataset.** 191 speakers and 348 clips. Positive-speaker evidence is sparse (74/84 dementia speakers have only one clip). Results may not generalize to broader populations.
- **Not clinically validated.** This system has not been evaluated by clinicians or tested on an external dataset. It must not be used for diagnosis, treatment, or any unsupervised patient-facing decision.
- **False negatives remain substantial.** Under the final ensemble, a mean of 28.6 dementia-positive speakers per seed are still missed.
- **False positives increased under max aggregation.** A mean of 34.2 non-dementia speakers per seed are incorrectly flagged. The false-positive burden is real and acknowledged.
- **Target recall not reached.** The project target zone was recall ≥ 0.70 with specificity ≥ 0.50. The best held result (recall 0.660) did not reach the target. The final model is the best candidate found, not a validated endpoint.
- **Single-user, single-mic design.** One active session at a time. Browser mic is not supported; audio must come from the Pi's USB microphone or a WAV upload.

---

## Future Work

- Collect more dementia-positive speakers and more clips per speaker to address the sparse evidence problem.
- Preserve per-speaker and per-clip ensemble prediction artifacts for future case-level analysis.
- Improve probability calibration for more reliable risk scores.
- Validate with clinicians and evaluate on an external dataset.
- Explore review-zone outputs and uncertainty quantification.

---

## Repository Quick-Start for Graders

To verify the code compiles and imports correctly without needing the dataset:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
make compile          # syntax-check all source files — should complete with no errors
make test             # run unit tests
python3 -c "from src.train_baseline import MODEL_ORDER; from src.config import CFG; print('imports OK')"
```

To verify the edge inference module loads (requires the model files, which are committed to this repo):

```bash
python3 -c "from edge_inference import DementiaScreener, load_config; cfg = load_config('configs/edge_inference.yaml'); print('edge config OK:', cfg.resolved_backend())"
```
