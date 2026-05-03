# Project Roadmap

## Phase 1: Project Foundation

- Keep `dementia_voice_project/` as the canonical project root.
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
