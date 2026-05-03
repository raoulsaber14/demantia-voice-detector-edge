# Project Roadmap

Historical planning document.

This roadmap was written before the final submission freeze. The final packaged repository differs from this plan in two important ways:

- the official final academic baseline remained the documented classical logistic-regression system
- the root-level edge app remained a prototype engineering layer rather than a validated deployment artifact

## Phase 1: Project Foundation

- Keep the research pipeline organized as a single project root.
- Lock down data organization and leakage controls.
- Verify metadata-to-audio matching.
- Establish a reproducible baseline pipeline.

## Phase 2: Classical ML Baseline

- Preprocess raw audio into normalized 16 kHz mono files.
- Extract interpretable acoustic and temporal features.
- Train logistic regression, SVM, and random forest baselines.
- Report accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and feature importance.

## Phase 3: wav2vec Fine-Tuning

- Build train/validation datasets from processed audio.
- Fine-tune wav2vec2 for binary risk classification.
- Compare against classical baselines.
- Add class imbalance handling and speaker-grouped cross-validation.

## Phase 4: Explainability and Clinical Framing

- Summarize risk using confidence plus marker-level signals.
- Analyze subgroup performance by age, gender, accent/language, and recording condition where metadata allows.
- Keep all outputs framed as screening support, not diagnosis.

## Phase 5: Edge Deployment

- Export the best candidate model to ONNX, TensorFlow Lite, or a PyTorch edge runtime.
- Quantize and benchmark on Raspberry Pi 5.
- Build an offline inference path: audio capture, VAD, segmentation, inference, and local display.
- Document privacy, latency, and hardware constraints.
