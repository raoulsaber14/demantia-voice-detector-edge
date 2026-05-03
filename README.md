# Edge AI Dementia Voice Detector

This repository is the final project submission for a speech-based dementia screening-support study. It combines:

- a root-level edge inference prototype for local audio capture, preprocessing, and model wrapping
- a nested research subproject in `dementia-voice-detector/` containing the training pipeline, final reports, and evaluation documentation

This project is **screening-support only, not diagnostic**. No part of this repository should be presented as a clinically validated diagnostic system.

## Submission Status

- Official final academic baseline: `final_cleaned_logistic_regression_platt`
- Best exploratory held candidate: `max_probability_ensemble default_0.5`
- Edge prototype status: engineering wrapper and demo layer only; it is not the final validated deployment package

The short rubric-facing summary is in `SUBMISSION_OVERVIEW.md`.

## Repository Structure

```text
.
├── README.md
├── SUBMISSION_OVERVIEW.md
├── audio_pipeline.py
├── audio_pipeline_documentation.docx
├── record_and_screen.py
├── test_pipeline.py
├── test_edge_inference.py
├── verify_two_stage.py
├── web_server.py
├── generate_test_audio.py
├── Project_tree_structure.txt
├── edge_inference/
├── templates/
├── models/
├── Audio_test_samples_dementia/
├── non_dementia_Audio_test_samples/
└── dementia-voice-detector/
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── Makefile
    ├── app/
    ├── configs/
    ├── data/
    ├── docs/
    ├── reports/final/
    ├── scripts/
    ├── src/
    └── tests/
```

## Canonical Project Story

The repository should be read as one submission with two scopes:

1. Research pipeline and final evaluation archive  
   Located in `dementia-voice-detector/`. This is where the documented baseline, exploratory ensemble work, reports, and reproducibility notes live.

2. Edge prototype  
   Located at the repository root. This is the engineering layer for local preprocessing, model wrapping, CLI usage, and a prototype web interface.

The final written project conclusion is:

- the official final academic baseline is the documented classical logistic-regression system
- the best exploratory held candidate is the Phase D ensemble
- the edge prototype remains a prototype and does not by itself establish deployment readiness or clinical validity

## Official Final Outputs

Official final academic baseline documentation:

- `dementia-voice-detector/reports/final/FINAL_BASELINE_SUMMARY.md`
- `dementia-voice-detector/docs/OFFICIAL_BASELINE_CONFIGURATION.md`
- `dementia-voice-detector/docs/BASELINE_MODEL_CARD.md`

Best exploratory held candidate documentation:

- `dementia-voice-detector/reports/final/final_results.md`
- `dementia-voice-detector/reports/final/phaseD_hubert_recall_recovery_benchmark.md`

Project-level final framing and limitations:

- `dementia-voice-detector/reports/final/final_model_decision.md`
- `dementia-voice-detector/reports/final/final_limitations_and_future_work.md`

## What Runs From This Repository

### Root edge prototype

Main root-level components:

- `audio_pipeline.py`: audio cleaning and WAV I/O
- `edge_inference/`: runtime inference wrappers and backends
- `record_and_screen.py`: CLI screening path
- `web_server.py`: prototype Flask UI
- `templates/index.html`: prototype browser UI

Recommended base environment for the root prototype:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy flask sounddevice onnxruntime onnx requests pyyaml soundfile
```

Smoke test that works without external deployment artifacts:

```bash
python3 test_edge_inference.py
```

Pipeline-only validation:

```bash
python3 test_pipeline.py --input path/to/file.wav
```

### Nested research subproject

The nested research pipeline has its own packaging files and tests:

```bash
cd dementia-voice-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

The nested README explains that subproject in more detail.

## Edge Prototype Model Setup

The default root config in `edge_inference/model_config.yaml` points to ONNX deployment artifacts that are **not included in this submission**.

To run the root prototype:

1. Supply the expected external ONNX artifacts at the configured paths, or
2. Switch the config to a dummy backend for smoke testing

Because those deployment artifacts are missing, the root prototype should be treated as:

- runnable in smoke-test mode
- partially runnable with external model files supplied later
- not a complete standalone deployment package as committed here

## Included Vs Excluded Artifacts

Included in this submission:

- root edge prototype code
- sample audio used for manual root-level checks
- nested research code
- curated final reports under `dementia-voice-detector/reports/final/`
- documentation describing methodology, limitations, and responsible-use framing

Not fully included in this submission:

- raw/private training data for the research pipeline
- generated feature tables and most working `data/` outputs
- trained research model folders under `dementia-voice-detector/models/`
- deployment ONNX artifacts referenced by the root config
- many intermediate working-tree reports referenced by historical planning documents

## Known Limitations

- The official final academic baseline is documented, but the full raw-data retraining path is not self-contained in this public repo snapshot.
- The best exploratory held candidate is documented from saved artifacts; it is not promoted here as a validated final deployment model.
- The root web interface is an engineering prototype and should not be treated as the final user-facing clinical presentation layer.
- The root prototype can only run with dummy mode unless external ONNX artifacts are supplied.
- The web app can only handle one active session at a time.
- The upload flow only accepts WAV files.

## Sample Audio

The repository includes example audio under:

- `Audio_test_samples_dementia/`
- `non_dementia_Audio_test_samples/`

These clips are included for manual root-level pipeline and inference checks only. They are not a substitute for the full research dataset and do not make the full training pipeline reproducible.

## Additional Documentation

- `SUBMISSION_OVERVIEW.md`: single-page submission summary
- `audio_pipeline_documentation.docx`: root audio preprocessing write-up
- `Project_tree_structure.txt`: archived tree snapshot
