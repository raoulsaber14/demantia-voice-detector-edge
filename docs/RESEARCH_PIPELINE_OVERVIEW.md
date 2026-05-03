# Dementia Voice Screening Research Pipeline Overview

This repository contains the research pipeline, evaluation code, and curated final reports used in the project submission.

> Disclaimer: screening-support only, not diagnostic.

## Final Status

- Official final academic baseline: `final_cleaned_logistic_regression_platt`
- Best exploratory held candidate: `max_probability_ensemble default_0.5`
- Edge deployment status: the root-level edge app is a prototype engineering layer, not the validated final model package

Key final documents:

- `reports/final/FINAL_BASELINE_SUMMARY.md`
- `reports/final/final_results.md`
- `reports/final/final_model_decision.md`
- `reports/final/final_limitations_and_future_work.md`
- `docs/OFFICIAL_BASELINE_CONFIGURATION.md`
- `docs/BASELINE_MODEL_CARD.md`

## What This Folder Contains

Included in this submission:

- `src/`: research pipeline source
- `scripts/`: experiment and audit runners
- `configs/`: configuration files
- `tests/`: unit tests for committed code
- `docs/`: methodology, governance, and final framing docs
- `reports/final/`: curated final reports and tables included in the submission
- `app/streamlit/demo_app.py`: historical local research demo

Partially or not included in this submission:

- `data/raw_audio/`, `data/processed_audio/`, `data/features/`, `data/metadata/`
- `models/`
- many intermediate reports, notebooks, and generated tables from the larger working tree

As committed here, this folder is best understood as a documented research snapshot rather than a fully self-contained rerun environment from raw data.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional editable install:

```bash
python -m pip install -e ".[dev]"
```

## Tests

Run from this folder:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## How To Read The Final Outcome

Use these documents for the final project story:

1. `reports/final/FINAL_BASELINE_SUMMARY.md`
   Official final academic baseline.

2. `reports/final/final_results.md`
   Best exploratory held candidate.

3. `reports/final/final_model_decision.md` and `reports/final/final_limitations_and_future_work.md`
   Final project framing, limitations, and non-diagnostic positioning.

4. `docs/REPRODUCIBILITY_CHECKLIST.md`
   What is and is not reproducible from the committed snapshot.

## Data And Leakage Notes

- `name`, `birth`, `death`, and `first symptoms` are not used as predictive model features.
- `speaker_id` is derived from normalized `name` for grouping and leakage checks.
- Speaker leakage is treated as a critical governance issue.
- Folder class labels are cross-check signals rather than unquestioned truth.

## Important Scope Note

Some historical planning and proposal documents in `docs/` were written before the final project packaging. They remain in the repo for traceability, but the final submission position is defined by the final reports and model-card-style documents listed above.
