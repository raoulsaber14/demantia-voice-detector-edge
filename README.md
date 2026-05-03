# Edge AI Dementia Voice Early Detector

This repository is the final submission for a speech-based dementia screening-support project. It combines:

- research and training code for the documented academic baseline and exploratory ensemble work
- edge inference wrappers and demo applications for local audio capture and prototype screening
- final reports and supporting documentation

This project is **screening-support only, not diagnosis**. Nothing in this repository should be presented as a clinically validated diagnostic tool.

## Final Status

- Official final academic baseline: `final_cleaned_logistic_regression_platt`
- Best exploratory held candidate: `max_probability_ensemble default_0.5`
- Edge prototype status: engineering demo only; not a validated deployment package

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── Makefile
├── pyproject.toml
├── app/
│   ├── flask/
│   └── streamlit/
├── artifacts/
├── configs/
├── data/
│   ├── README.md
│   └── sample_audio/
├── docs/
├── edge_inference/
├── models/
├── reports/
│   └── final/
├── scripts/
├── src/
└── tests/
```

## Folder Guide

- `app/flask/`: Flask prototype web demo
- `app/streamlit/`: Streamlit research/demo app
- `artifacts/`: placeholder directory for large local-generated artifacts that are not committed
- `configs/`: runtime and training configuration files
- `data/`: sample audio plus notes about the omitted research data tree
- `docs/`: methodology, reproducibility notes, model cards, and project documentation
- `edge_inference/`: edge model wrapper, preprocessing bridge, and backends
- `models/`: expected location for external model binaries
- `reports/final/`: curated final submission reports and tables
- `scripts/`: runnable CLI utilities and research scripts
- `src/`: shared source code for training, inference, preprocessing, and the audio pipeline
- `tests/`: root-level automated tests

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:
- microphone capture depends on PortAudio being available for `sounddevice`
- `scripts/generate_test_audio.py` also needs `ffmpeg` on `PATH`

## Running The Project

Flask demo:

```bash
python3 -m app.flask.web_server
```

Dockerized Flask demo:

```bash
docker build -t edge-ai-dementia-voice .
docker run --rm -p 5000:5000 edge-ai-dementia-voice
```

Notes:
- the committed Docker path serves the existing Flask demo only
- the default container behavior uses the committed dummy backend, so it is a smoke-test/demo path rather than real model deployment
- microphone capture inside Docker depends on host audio pass-through; the upload flow is the more portable demo path
- if you have real external model artifacts, place them under `models/` and mount that folder into the container as needed, for example `-v "$(pwd)/models:/app/models"`

Streamlit demo:

```bash
streamlit run app/streamlit/demo_app.py
```

CLI screening flow:

```bash
python3 -m scripts.record_and_screen
```

Manual pipeline diagnostic:

```bash
python3 -m scripts.pipeline_diagnostic --input path/to/file.wav
```

## Tests

Run the root-level test suite with:

```bash
python3 -m unittest discover -s tests -v
```

The edge smoke/integration coverage lives in `tests/test_edge_inference.py`. The metadata unit test lives in `tests/test_metadata_utils.py`.

## Model Files

The committed default edge config is `configs/edge_inference.yaml`. It uses the dummy backend so the edge demo can run without a real model binary.

If you want to run external ONNX artifacts:

1. Place the binaries under `models/`
2. Start from `configs/edge_inference.onnx.example.yaml`
3. Replace the placeholder filenames with the real exported assets

See `models/README.md` for the expected layout.

The committed Flask and Streamlit demos currently expose raw screening outputs such as risk score, confidence, class probabilities, and threshold controls. They should be treated as research/demo review interfaces, not as clinician-safe or patient-facing presentation layers.

## Final Documentation

Primary final documents:

- `reports/final/FINAL_BASELINE_SUMMARY.md`
- `reports/final/final_results.md`
- `reports/final/final_model_decision.md`
- `reports/final/final_limitations_and_future_work.md`
- `docs/OFFICIAL_BASELINE_CONFIGURATION.md`
- `docs/BASELINE_MODEL_CARD.md`
- `docs/SUBMISSION_OVERVIEW.md`
- `docs/fairness_bias_statement.md`
- `docs/privacy_data_provenance.md`

Additional background:

- `docs/RESEARCH_PIPELINE_OVERVIEW.md`
- `docs/audio_pipeline_documentation.docx`
- `docs/LEGACY_PROJECT_TREE_STRUCTURE.txt`

## Data Scope

The repository includes small demo clips under `data/sample_audio/` for manual checks only.

The full research data tree is not included. Missing from the public snapshot are:

- raw/private research audio
- processed audio and feature tables
- most generated training artifacts
- trained research model folders
- deployment ONNX artifacts for real edge inference

## Known Limits

- The repo is organized as one canonical root, but it is still a submission snapshot rather than a full raw-data rerun environment.
- Real edge inference still requires external model artifacts under `models/`.
- The web prototype is a demo layer and should not be treated as the final user-facing clinical presentation.
- The Flask app handles one active session at a time.
- Subgroup fairness evaluation is limited by the available metadata coverage and should not be treated as complete fairness validation.
- Privacy, provenance, and leakage-control boundaries are documented in `docs/privacy_data_provenance.md`.
