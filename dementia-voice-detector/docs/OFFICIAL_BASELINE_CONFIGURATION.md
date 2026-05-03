# Official Baseline Configuration

Freeze date: 2026-04-22.

This document freezes one final academic baseline configuration. It does not define a production threshold policy and does not claim clinical diagnostic validity.

## Source Artifacts

- Phase 4 official historical winner: `reports/phase4_official_winner.md`.
- Cleaned Phase 4 rerun comparison: `reports/phase6_5_week2_phase4_rerun_comparison.md`.
- Week 2b multi-seed stability: `reports/phase6_5_week2b_stability_addendum.md`, `reports/tables/phase6_5_week2b_seed_summary.csv`, `reports/tables/phase6_5_week2b_comparison_update.csv`, and `reports/tables/phase6_5_week2b_anchor_speaker_level_metrics.csv`.
- B2b false-positive audit: `reports/phase_b2b_false_positive_audit.md` and `reports/tables/phase_b2b_fp_score_distribution.csv`.
- B2c feature/model comparison: `reports/phase_b2c_feature_curation_report.md` and `reports/tables/phase_b2c_experiment_results.csv`.
- Dataset manifests: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv` and `reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv`.
- Canonical feature table: `data/features/features_phase3_governed_pruned.csv`.
- Baseline runner: `scripts/phase4_baseline_experiments.py`.

## Frozen Baseline

- Official baseline name: `final_cleaned_logistic_regression_platt`.
- Primary dataset evidence: cleaned trusted evaluation subset.
- Model: `logistic_regression_platt`.
- Frozen threshold: `0.300`.
- Feature source: `data/features/features_phase3_governed_pruned.csv`.
- Feature count used by the frozen anchor: 256 numeric features selected by `scripts/phase4_baseline_experiments.py` after excluding metadata and governance columns. Source: `scripts/phase4_baseline_experiments.py`; `reports/phaseA_classical_baseline_diagnostics.md`.
- Evaluation grouping: `speaker_id`.

## Dataset Version

Primary trusted dataset:

- `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv` has 348 rows, 94 dementia clips, 254 non-dementia clips, and 191 unique speakers. Speaker counts are 84 dementia speakers and 107 non-dementia speakers. Source: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv`; also summarized in `reports/phase6_5_week2_phase4_rerun_comparison.md`.

Governed sensitivity reference:

- `reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv` has 375 rows, 121 dementia clips, 254 non-dementia clips, and 191 unique speakers. Of those rows, 348 have `phase4_eval_allowed=True`; the remaining 27 are governed/caution rows and are not primary performance evidence. Source: `reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv`.
- Week 2 and Week 2b performance reports evaluate the 348-row cleaned manifest and state that the trusted and governed evaluation manifests were identical by `metadata_index` for that cleaned evaluation population. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/phase6_5_week2b_stability_addendum.md`.

## Feature Families

The frozen anchor uses handcrafted acoustic/prosodic features from the canonical Phase 3 governed-pruned feature table. Phase A reports 256 total features with these families: 156 MFCC, 35 spectral, 23 duration/timing, 17 voice-quality/pitch, 11 energy/RMS, 9 pause/speech-timing, 4 zero-crossing-rate, and 1 other feature. Source: `reports/phaseA_classical_baseline_diagnostics.md`.

## Model Configuration

The frozen model is the already evaluated Phase 4 logistic regression with Platt calibration:

- Preprocessing: median imputation and standard scaling.
- Base estimator: `LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed)`.
- Hyperparameter grid: `C` in `[0.01, 0.1, 1.0, 10.0]`.
- Calibration: external Platt scaling with `LogisticRegression(max_iter=2000, random_state=seed)` fit from inner grouped out-of-fold scores.
- Inner calibration split: `StratifiedGroupKFold` when feasible.

Source: `scripts/phase4_baseline_experiments.py`.

## Threshold Freeze

Frozen threshold: `0.300`.

Source of threshold:

- The historical Phase 4 official winner file records `logistic_regression_platt` with official threshold `0.300`. Source: `reports/phase4_official_winner.md`.
- The cleaned Week 2 rerun evaluates the same anchor at threshold `0.300`. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`.
- Week 2b seed 42 also records `logistic_regression_platt` at threshold `0.300`. Source: `reports/tables/phase6_5_week2b_seed_summary.csv`.

Original selection method:

- Thresholds were swept from `0.10` to `0.90` in `0.05` increments.
- The selection rule maximized recall among rows with precision at least `max(prevalence, 0.25)` and specificity at least `0.20`; if no threshold met the rule, a fallback row was marked explicitly.
- Source: `scripts/phase4_baseline_experiments.py`; `reports/phase6_5_week2b_stability_addendum.md`.

This threshold is frozen for the final baseline documentation. It is not a production threshold policy.

## Evaluation Protocol

- Outer evaluation: speaker-aware grouped cross-validation using `speaker_id`.
- Default maximum splits: 5.
- Main cleaned anchor seed: 42.
- Multi-seed stability seeds: 42, 43, 44, 45, and 46.
- Positive class: dementia.

Source: `scripts/phase4_baseline_experiments.py`; `reports/phase6_5_week2b_stability_addendum.md`.

## Evidence Summary

At the frozen threshold on the cleaned seed 42 anchor, clip-level metrics were recall 0.117, specificity 0.909, precision 0.324, F1 0.172, balanced accuracy 0.513, and confusion matrix `TN=231, FP=23, FN=83, TP=11`. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/tables/phase6_5_week2b_seed_summary.csv`.

Across seeds 42-46 for the cleaned anchor, threshold mean was 0.250 with standard deviation 0.077; recall mean 0.464 with standard deviation 0.380; specificity mean 0.560 with standard deviation 0.376; precision mean 0.291 with standard deviation 0.019; F1 mean 0.292 with standard deviation 0.111; balanced accuracy mean 0.512 with standard deviation 0.012. Source: `reports/tables/phase6_5_week2b_comparison_update.csv`.

Speaker-level max-risk aggregation across seeds had recall mean 0.476, specificity mean 0.475, precision mean 0.388, F1 mean 0.375, and balanced accuracy mean 0.475. Source: `reports/tables/phase6_5_week2b_comparison_update.csv`.

## Selection Rationale

This configuration was chosen because it is the simplest already evaluated calibrated baseline, it is traceable to the original Phase 4 official winner, and it has the most complete cleaned-data stability and error-analysis package. B2c did not find any feature subset or simple-model alternative that met the screening criterion of recall >= 0.70 with specificity >= 0.40. The strongest B2c alternative, `cleaned_all_features` with `linear_svm_platt`, reached mean recall 0.311 at specificity >= 0.40 and did not pass the criterion. Source: `reports/phase_b2c_feature_curation_report.md`; `reports/tables/phase_b2c_experiment_results.csv`.

The frozen baseline is therefore a defensible academic reference point, not a deployable screening system. Its main documented weaknesses are threshold fragility, score overlap, low cleaned-data recall at the frozen seed 42 operating point, and borderline false-positive behavior. Source: `reports/phase6_5_week2b_stability_addendum.md`; `reports/phase_b2b_false_positive_audit.md`.
