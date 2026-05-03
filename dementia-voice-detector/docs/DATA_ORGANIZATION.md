# Data Organization

This document describes the intended research data layout relative to this folder, `dementia-voice-detector/`, inside the final submission repo.

Important scope note:

- the committed submission does **not** contain the full research data tree
- many of the paths below describe the larger working layout used during the project
- in this public snapshot, most data-bearing directories are intentionally absent because they were large, private, or generated

## Canonical Inputs

- `data/raw_audio/dementia/`: dementia-labeled raw audio files.
- `data/raw_audio/nodementia/`: non-dementia/control raw audio files.
- `data/metadata/raw_metadata.xlsx`: source metadata table.

The folder name `nodementia` is treated as the project synonym for `non_dementia`.

Phase 1 metadata generation combines the source Excel rows with folder-labeled audio-only rows for clips that are present in `data/raw_audio/` but absent from the Excel metadata. This is necessary because the provided Excel file contains dementia subjects only, while the control audio exists in the `nodementia` folder.

## Generated Outputs

- `data/metadata/*.csv`: cleaned metadata, split tables, and data quality audits.
- `data/processed_audio/`: normalized 16 kHz mono WAV files.
- `data/segments/`: speech segments and segment metadata.
- `data/features/`: hand-crafted acoustic feature tables.
- `models/`: trained baseline and wav2vec artifacts.
- `reports/tables/` and `reports/figures/`: generated evaluation outputs.

Generated outputs were typically ignored by git in the working project so the repository stayed lightweight and did not accidentally include private data. In this submission snapshot, those generated outputs are mostly not committed.

## External Snapshot

The original loose top-level audio folders were moved to:

- `data/external/top_level_audio_snapshot/dementia/`
- `data/external/top_level_audio_snapshot/nodementia/`

Those folders were retained only as a traceable source snapshot in the larger working tree. The intended pipeline input is `data/raw_audio/`.

## Reference Materials

- `docs/references/`: papers, notes, and links used during project research in the larger working tree.
- `artifacts/source_archives/`: large archive files from the original project handoff in the larger working tree.

## Phase 2 dataset audit (quality check)

Reproducible inventory and QA for `data/raw_audio/` (no raw files are modified). This is the **Phase 2** data audit and is documented in `reports/dataset_audit/phase2_dataset_audit_report.md` in the fuller working-tree history.

- **Run:** `make dataset-audit` or `python3 scripts/dataset_audit.py` from the project root.
- **Outputs:** `reports/dataset_audit/` — `master_inventory.csv`, class/duration/technical summaries, `flagged_files.csv`, `manual_review_priority.csv`, `speaker_analysis.csv`, `phase2_dataset_audit_report.md`, `split_strategy_recommendation.md`, and related tables.
- **Label mapping in audit:** physical folder `nodementia` is documented as effective ML label `non_dementia` (see report tables `raw_class_folder` vs `effective_label_name`).
- **Note:** Some controls may appear under both `data/raw_audio/nodementia/<speaker>/` and `data/raw_audio/nodementia/nodementia/<speaker>/`; the audit flags the nested layout and any byte-level duplicate candidates for manual review.
