# Reproducibility Checklist

Freeze date: 2026-04-22.

This checklist documents how to reproduce the official final baseline structure. It does not ask the current freeze run to rerun experiments.

Scope note:

- Official final academic baseline: `final_cleaned_logistic_regression_platt`
- Best exploratory held candidate: `max_probability_ensemble default_0.5`
- Root-level edge prototype: separate engineering layer, outside the scope of this baseline reproduction checklist

Public-repo limitation:

- this committed snapshot does not include the full raw-data, feature-table, and trained-model tree needed for a complete retraining rerun from scratch
- this checklist therefore describes the documented baseline structure and expected frozen outputs rather than a guaranteed full raw-data rerun from the public snapshot alone

- [ ] Environment setup: use Python `>=3.10` from `pyproject.toml` and install repository dependencies from `requirements.txt` or `pyproject.toml`.
- [ ] Dependency limitation: no lock file was found; exact package versions are therefore not frozen in this repository state. Source: `requirements.txt`; `pyproject.toml`.
- [ ] Data acquisition: use the governed project data already represented by `data/features/features_phase3_governed_pruned.csv` and the cleaned manifests in `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv` and `reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv`.
- [ ] Primary evaluation population: cleaned trusted subset, 348 rows, 94 dementia clips, 254 non-dementia clips, 191 unique speakers. Source: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv`; `reports/phase6_5_week2_phase4_rerun_comparison.md`.
- [ ] Governed sensitivity context: full governed cleaned manifest has 375 rows, with 348 `phase4_eval_allowed=True` rows and 27 caution/manual-review rows. Source: `reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv`.
- [ ] Feature source: `data/features/features_phase3_governed_pruned.csv`.
- [ ] Feature extraction policy: Phase 3 governed-pruned feature source only; no new feature engineering in the final freeze. Source: `scripts/phase4_baseline_experiments.py`; `docs/FINAL_FEATURE_SET.md`.
- [ ] Feature count: 256 official numeric model features selected after metadata/governance exclusions. Source: `scripts/phase4_baseline_experiments.py`; `reports/phaseA_classical_baseline_diagnostics.md`.
- [ ] Documentary cleanup note: future cleanup would remove 4 B2c-flagged fields and produce a 252-feature subset, but this freeze does not implement that removal. Source: `docs/FINAL_FEATURE_SET.md`; `reports/tables/phase_b2c_feature_subsets.csv`.
- [ ] Model training script: `scripts/phase4_baseline_experiments.py`.
- [ ] Model configuration: median imputation, standard scaling, balanced L2 logistic regression, `C` grid `[0.01, 0.1, 1.0, 10.0]`, and Platt calibration. Source: `scripts/phase4_baseline_experiments.py`.
- [ ] Grouping: `speaker_id` grouped cross-validation. Source: `scripts/phase4_baseline_experiments.py`.
- [ ] Main seed: 42. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/tables/phase6_5_week2b_seed_summary.csv`.
- [ ] Stability seeds: 42, 43, 44, 45, and 46. Source: `reports/phase6_5_week2b_stability_addendum.md`.
- [ ] Threshold method: sweep thresholds from `0.10` to `0.90` in `0.05` increments and select the recall-prioritized operating point subject to precision and specificity gates, with fallback marked explicitly. Source: `scripts/phase4_baseline_experiments.py`.
- [ ] Frozen threshold: `0.300`. Source: `reports/phase4_official_winner.md`; `reports/phase6_5_week2_phase4_rerun_comparison.md`; `docs/OFFICIAL_BASELINE_CONFIGURATION.md`.
- [ ] Expected cleaned seed 42 anchor metrics: recall 0.117, specificity 0.909, precision 0.324, F1 0.172, balanced accuracy 0.513, and confusion matrix `TN=231, FP=23, FN=83, TP=11`. Source: `reports/tables/phase6_5_week2b_seed_summary.csv`.
- [ ] Expected multi-seed summary: threshold mean 0.250, threshold standard deviation 0.077, recall mean 0.464, specificity mean 0.560, precision mean 0.291, F1 mean 0.292, balanced accuracy mean 0.512. Source: `reports/tables/phase6_5_week2b_comparison_update.csv`.
- [ ] Expected speaker-level seed 42 max-risk metrics: recall 0.119, specificity 0.832, precision 0.357, F1 0.179, balanced accuracy 0.475. Source: `reports/tables/phase6_5_week2b_anchor_speaker_level_metrics.csv`.
- [ ] Reproducible command shape for the historical runner: `python scripts/phase4_baseline_experiments.py --seed 42 --max-splits 5`.
- [ ] Overwrite caution: the Phase 4 script writes fixed Phase 4 output paths. Week 2 avoided overwriting official artifacts by running an unchanged script copy in an isolated temporary directory. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`.
- [ ] Demo output contract: banded outputs only; no raw score, percentage, confidence, or diagnostic label. Source: `docs/DEMO_ALIGNMENT_SPEC_FINAL.md`.
- [ ] Manual steps documented: governed/caution rows must be kept separate from trusted primary evidence, and governed results must not be used as final performance proof. Source: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`; `docs/BASELINE_MODEL_CARD.md`.
