# Dementia Voice Screening Project

Voice-based dementia-risk screening support from speech audio.  
Primary task: internal binary screening classification (`dementia` vs `non_dementia`) for research analysis.

> Disclaimer: This tool is for screening support only and is not a medical diagnosis.

Project proposal: `docs/PROJECT_PROPOSAL.md`  
Roadmap: `docs/ROADMAP.md`  
Data organization: `docs/DATA_ORGANIZATION.md`  
Phase 1 dataset summary (audit + splits): `reports/phase1_dataset_report.md`
Final held-model decision: `reports/final_model_decision.md`  
Final limitations and future work: `reports/final_limitations_and_future_work.md`

## Project structure

- `reports/final/` curated final narrative docs, cleanup logs, and copied final submission tables
- `reports/archive/phase4_seed_reports/` repetitive Phase 4 per-seed reports moved out of the active root
- `reports/tables/` working/generated tables kept in place for reproducibility
- `data/raw_audio/`, `data/processed_audio/`, `data/features/`, `data/metadata/` active reproducibility data
- `data/external/` original handoff snapshots kept for traceability
- `data/archive/` reserved for archived provenance or backup data
- `artifacts/phaseC_frozen_embeddings/` active reproducibility artifacts
- `artifacts/archive/` reserved for archived handoff material
- `models/` saved baseline and wav2vec models, with active checkpoints kept in their original run folders
- `models/archive/` reserved for older checkpoints kept only for provenance
- `src/`, `scripts/`, `configs/`, `tests/`, `docs/`, and `final_results/` remain active project files
- `app/demo_app.py` simple Streamlit local demo

## Data assumptions and leakage controls

- `name`, `birth`, `death`, and `first symptoms` are not used as predictive model features.
- `speaker_id` is derived from normalized `name` for grouping and leakage checks.
- Speaker leakage is flagged when one `speaker_id` appears in multiple splits.
- Folder class labels are treated as a cross-check only and conflicts are logged.
- Matching priority:
  1. `audio_filename` exact/normalized basename match
  2. normalized `name` to audio stem/folder
  3. partial normalized stem fallback

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows activation, if needed:

```bash
.venv\Scripts\activate
```

Optional editable install for development:

```bash
python -m pip install -e ".[dev]"
```

Common commands are also available through `make`, for example:

```bash
make metadata
make preprocess
make features
make baseline
make phase3
```

## Week 1: metadata pipeline and audit

Run from project root:

```bash
python -m src.metadata_utils --metadata-path data/metadata/raw_metadata.xlsx --raw-audio-dir data/raw_audio
```

Generated files:
- `data/metadata/metadata_clean.csv`
- `data/metadata/train.csv`
- `data/metadata/valid.csv`
- `data/metadata/unmatched_rows.csv`
- `data/metadata/duplicate_matches.csv`
- `data/metadata/label_conflicts.csv`
- `data/metadata/data_quality_issues.csv`

Open:
- `notebooks/01_dataset_audit.ipynb`

## Audio preprocessing

```bash
python -m src.preprocess --metadata-csv data/metadata/metadata_clean.csv --raw-audio-root data/raw_audio --processed-root data/processed_audio --trim-silence
```

Inspect:
- `data/metadata/preprocess_log.csv`
- `notebooks/02_preprocessing.ipynb`

## Feature baseline workflow

```bash
python -m src.features --metadata-csv data/metadata/metadata_clean.csv --processed-root data/processed_audio --output-csv data/features/file_features.csv
python -m src.train_baseline --feature-csv data/features/file_features.csv --model-out-dir models/baseline
```

Outputs:
- `models/baseline/*.joblib`
- `reports/tables/baseline_metrics.csv`
- `reports/figures/*_confusion_matrix.png`
- `reports/tables/feature_importance_baseline.csv`

## Phase 3 classical ML preprocessing and features

Run the Phase 3 classical baseline path from the project root:

```bash
make phase3
```

This uses `data/metadata/metadata_fixed_splits.csv`, writes standardized audio to `data/processed_audio/`, writes the cleaned feature table to `data/features/features_enhanced.csv`, and keeps preprocessing/feature QA artifacts under `reports/` and `reports/tables/`.

Documentation:
- `docs/PHASE3_PREPROCESSING_POLICY.md`
- `docs/FEATURE_DICTIONARY.md`
- `reports/phase3_preprocessing_report.md`
- `reports/phase3_feature_engineering_report.md`

## Wav2Vec workflow

```bash
python -m src.train_wav2vec --metadata-csv data/metadata/metadata_clean.csv --model-name facebook/wav2vec2-base --output-dir models/wav2vec
```

Outputs:
- `models/wav2vec/best/`
- `reports/tables/wav2vec_eval_metrics.txt`

## Inference

Baseline:
```bash
python -m src.inference --audio-path path/to/audio.wav --model-type baseline --model-path models/baseline/logistic_regression.joblib
```

Wav2Vec:
```bash
python -m src.inference --audio-path path/to/audio.wav --model-type wav2vec --model-path models/wav2vec/best
```

## Demo app

```bash
streamlit run app/demo_app.py
```

## Limitations

- This is a research-oriented screening and referral-support pipeline, not a diagnostic system.
- Label quality depends on metadata and matching quality.
- Small sample subgroups can create unstable fairness estimates.
- Audio source conditions and recording quality may strongly affect performance.
