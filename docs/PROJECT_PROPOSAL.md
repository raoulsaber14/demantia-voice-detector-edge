# Edge AI Dementia Voice Early Detector

Historical proposal document.

This file captures the original project pitch. It is not the final source of truth for the submission outcome. For the final packaged position, use:

- `reports/final/FINAL_BASELINE_SUMMARY.md`
- `reports/final/final_results.md`
- `reports/final/final_model_decision.md`

AI-powered speech analysis and processing system for early dementia risk screening using voice interpretation and deployable edge inference.

## Team

- Zeina Hammoud
- Celine Sadaka
- Raoul Saber

## Selection

- Option selected: A
- Deployment category: Edge-based Deep Learning using wav2vec
- Validation contact: Dr. Khaled Sidani, Resident Neurosurgeon at AUBMC

## Problem Definition and Real-World Relevance

Early dementia symptoms can appear in speech before severe functional decline, but routine specialist assessment is time-consuming, expensive, and not always accessible.

This project proposes a non-invasive voice-based screening support tool that flags patterns consistent with early cognitive decline. It is not a diagnosis system; it is a triage and monitoring aid for earlier referral.

Defendable impact:

- Low-cost, repeatable, remote-friendly screening for home, clinic, and telehealth contexts.
- Edge deployment for privacy, low latency, and standalone use.
- Clinically motivated speech markers with an explainable risk score.

## Why Voice Can Indicate Early Dementia

Dementia and mild cognitive impairment can affect language retrieval, processing speed, working memory, motor speech coordination, and conversational fluency. These changes may manifest as measurable acoustic and temporal anomalies even when symptoms are subtle.

Target marker families:

- Longer pauses, silences, and increased response gap duration.
- Hesitations, fillers, repetitions, restarts, and reduced fluency.
- Speech rate, articulation consistency, prosody, pitch variability, and energy variability.
- Lexical retrieval difficulty and reduced linguistic richness where transcripts are available.

## Baseline Strategy

The baseline should be simple and defendable before deep learning is introduced.

- Extract hand-crafted features: MFCCs, jitter/shimmer proxies, formants where feasible, F0 statistics, energy, speaking rate, pause ratio, silence duration histogram, gap count, and voiced/unvoiced ratio.
- Train classical models: logistic regression, SVM, random forest, and optionally XGBoost.
- Use interpretable heuristics such as pause burden, speech rate, and prosody instability as a non-deep-learning benchmark.

## Deep Learning Approach

Primary model family: wav2vec-based speech representation model fine-tuned for binary or multi-class dementia-risk classification.

Why this is appropriate:

- Self-supervised speech representations capture rich acoustic-temporal structure.
- Fine-tuning can work with limited labeled datasets when splits and leakage controls are handled carefully.

Training pipeline scope:

- Preprocessing, augmentation, segmentation, feature extraction, label handling, class balancing, cross-validation, and experiment tracking.

Inference pipeline scope:

- On-device audio capture.
- Denoising and voice activity detection.
- Segmentation.
- wav2vec embedding and classifier.
- Risk score, confidence, and marker summary.

## Edge Deployment Plan

Target deployment: edge-first inference on a Raspberry Pi-based system with optional cloud training and update workflow.

Edge runtime:

- Optimized and quantized model using ONNX, TensorFlow Lite, or PyTorch Mobile.
- Short inference windows and fully offline-capable scoring.

Edge device role:

- Dedicated microphone for controlled audio capture.
- On-device DSP pipeline using 16 kHz sampling, filtering, noise reduction, framing, and feature extraction.
- Real-time local inference.
- Optional simple UI for displaying the screening recommendation.

Cloud role:

- GPU-enabled training and validation.
- Periodic model optimization and updates.
- No raw audio transmission required for inference.

Technical constraints:

- Limited CPU/RAM on Raspberry Pi.
- Need for model quantization and efficient DSP.
- Robustness to environmental noise and microphone variability.
- Fairness across accents, languages, ages, and genders.
- Careful control of false positives and false negatives.

## Medical Framing

The system provides a cognitive risk screening recommendation, not a medical diagnosis. It is intended to support early pre-assessment and longitudinal monitoring.
