# Baseline Model Card

## 1. Model Details

- Model name: `final_cleaned_logistic_regression_platt`.
- Model type: L2 logistic regression with Platt calibration.
- Freeze date: 2026-04-22.
- Version: `final_baseline_freeze_2026_04_22`.
- Frozen threshold: `0.300`.
- Hyperparameters and preprocessing: median imputation, standard scaling, `LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed)`, and `C` grid `[0.01, 0.1, 1.0, 10.0]`. Platt calibration uses `LogisticRegression(max_iter=2000, random_state=seed)`. Source: `scripts/phase4_baseline_experiments.py`.

## 2. Intended Use

This baseline is for research screening-support analysis only. It is intended to provide an academic reference point for speech-based dementia versus non-dementia classification in this repository. Source: `README.md`; `reports/phase4_official_winner.md`.

## 3. Non-Intended Use

This model must not be used for diagnosis, treatment decisions, clinical triage, unsupervised patient-facing decisions, or any use outside the dataset context without external validation. Source: `reports/phase4_official_winner.md`; `docs/PHASE6_5_DEMO_APP_ALIGNMENT_SPEC.md`.

## 4. Dataset

Primary evidence uses the cleaned trusted subset: 348 clips, 94 dementia clips, 254 non-dementia clips, and 191 unique speakers, with 84 dementia speakers and 107 non-dementia speakers. Source: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv`; `reports/phase6_5_week2_phase4_rerun_comparison.md`.

Governed sensitivity context uses `reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv`, which contains 375 rows, 121 dementia clips, 254 non-dementia clips, and 191 unique speakers; 348 rows have `phase4_eval_allowed=True` and 27 governed/caution rows are not primary performance evidence. Source: `reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv`.

The Week 2 and Week 2b cleaned evaluation reports treat the 348-row cleaned evaluation population as the fixed anchor population. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/phase6_5_week2b_stability_addendum.md`.

## 5. Features

Features are handcrafted acoustic/prosodic descriptors from `data/features/features_phase3_governed_pruned.csv`. The frozen anchor uses 256 official numeric features. Source: `scripts/phase4_baseline_experiments.py`; `reports/phaseA_classical_baseline_diagnostics.md`.

Feature families include MFCC, spectral, duration/timing, voice-quality/pitch, energy/RMS, pause/speech-timing, zero-crossing-rate, and one other feature. Source: `reports/phaseA_classical_baseline_diagnostics.md`.

The feature freeze is documentary only. B2c identified a 252-feature cleaned subset for analysis, but this final freeze does not modify the working feature table or regenerate model outputs. Source: `docs/FINAL_FEATURE_SET.md`; `reports/tables/phase_b2c_feature_subsets.csv`.

## 6. Evaluation Protocol

Evaluation is speaker-aware grouped cross-validation by `speaker_id`, with dementia as the positive class. The main cleaned anchor seed is 42, and multi-seed stability uses seeds 42, 43, 44, 45, and 46. Source: `scripts/phase4_baseline_experiments.py`; `reports/phase6_5_week2b_stability_addendum.md`.

Primary evaluation framing is recall-prioritized screening support, but the final freeze does not claim that the screening criterion is achieved. Secondary metrics include specificity, precision, F1, balanced accuracy, Brier score, and PR-AUC. Source: `reports/phase4_official_winner.md`; `reports/phase6_5_week2_phase4_rerun_comparison.md`.

## 7. Performance Summary

Cleaned seed 42 clip-level metrics at the locked threshold were recall 0.117, specificity 0.909, precision 0.324, F1 0.172, balanced accuracy 0.513, Brier score 0.197, PR-AUC 0.310, and confusion matrix `TN=231, FP=23, FN=83, TP=11`. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/tables/phase6_5_week2b_seed_summary.csv`.

Historical original Phase 4 metrics at the same official threshold were recall 0.862, specificity 0.217, precision 0.289, F1 0.433, balanced accuracy 0.539, Brier score 0.198, PR-AUC 0.332, and false negatives 13. These are historical pre-cleaning metrics, not final cleaned performance. Source: `reports/phase4_official_winner.md`; `reports/phase6_5_week2_phase4_rerun_comparison.md`.

Across seeds 42-46 on the cleaned anchor, clip-level means were recall 0.464, specificity 0.560, precision 0.291, F1 0.292, and balanced accuracy 0.512. Standard deviations were 0.380 recall, 0.376 specificity, 0.019 precision, 0.111 F1, and 0.012 balanced accuracy. Source: `reports/tables/phase6_5_week2b_comparison_update.csv`.

Speaker-level seed 42 max-risk aggregation at threshold 0.300 had recall 0.119, specificity 0.832, precision 0.357, F1 0.179, balanced accuracy 0.475, and confusion matrix `TN=89, FP=18, FN=74, TP=10`. Source: `reports/tables/phase6_5_week2b_anchor_speaker_level_metrics.csv`.

Speaker-level multi-seed max-risk aggregation had recall mean 0.476, specificity mean 0.475, precision mean 0.388, F1 mean 0.375, and balanced accuracy mean 0.475. Source: `reports/tables/phase6_5_week2b_comparison_update.csv`.

## 8. Known Failure Modes

False positives are mixed Bucket C/D: broadly distributed and mostly borderline near the decision threshold. In B2b, the seed 42 cleaned anchor had 23 false positives across 18 speakers; 87.0 percent were within 0.05 of the threshold and none were above 0.500. Source: `reports/phase_b2b_false_positive_audit.md`; `reports/tables/phase_b2b_fp_score_distribution.csv`.

False negatives dominate the cleaned seed 42 operating point: 83 of 94 dementia-labeled evaluation clips were false negatives at threshold 0.300. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/tables/phase6_5_week2b_seed_summary.csv`.

Threshold stability is a major limitation: seed-level selected thresholds ranged from 0.100 to 0.300, with mean 0.250 and standard deviation 0.077. Source: `reports/tables/phase6_5_week2b_threshold_variability.csv`.

## 9. Limitations

The model is not clinically validated. The dataset is small, speaker grouped, and not established as representative of clinical populations. The feature space has limited separability, and B2c did not find a simple classical feature/model combination that achieved recall >= 0.70 with specificity >= 0.40. Source: `reports/phase_b2c_feature_curation_report.md`; `reports/tables/phase_b2c_experiment_results.csv`.

The locked threshold is frozen for reproducibility and documentation, not because it is clinically optimal. Source: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`.

## 10. Ethical / Clinical Caution

Outputs must be banded, cautious, and non-diagnostic. The demo must not expose raw probabilities, percentages, confidence scores, diagnostic labels, or language implying that dementia is confirmed or ruled out. Source: `docs/PHASE6_5_DEMO_APP_ALIGNMENT_SPEC.md`; `docs/DEMO_ALIGNMENT_SPEC_FINAL.md`.

## 11. Governance

Trusted rows are the primary evidence set for the final baseline. Governed rows are retained for traceability and sensitivity context only. Governed/caution rows must not be used as proof of final performance. Source: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv`; `reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv`; `reports/phase6_5_week2_phase4_rerun_comparison.md`.
