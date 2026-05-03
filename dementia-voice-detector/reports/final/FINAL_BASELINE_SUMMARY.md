# Final Baseline Summary

Freeze date: 2026-04-22.

## 1. Executive Summary

The official final baseline is `final_cleaned_logistic_regression_platt`: a speaker-aware, calibrated classical logistic-regression baseline using 256 handcrafted acoustic/prosodic features from `data/features/features_phase3_governed_pruned.csv` and frozen threshold `0.300`. Source: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`; `scripts/phase4_baseline_experiments.py`.

This baseline is reproducible and well documented, but weak. On the cleaned seed 42 anchor, recall collapsed to 0.117 while specificity rose to 0.909. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/tables/phase6_5_week2b_seed_summary.csv`.

The final framing is academic baseline only: not diagnostic, not clinically validated, and not deployment-ready. Source: `docs/BASELINE_MODEL_CARD.md`; `docs/DEMO_ALIGNMENT_SPEC_FINAL.md`.

This document is distinct from the repository's best exploratory held candidate, `max_probability_ensemble default_0.5`, which is summarized separately in `final_results.md`. The root-level edge app is also separate and should be treated as a prototype engineering layer rather than the validated final model package.

## 2. Configuration Freeze

- Dataset: cleaned trusted primary subset with 348 rows, 94 dementia clips, 254 non-dementia clips, and 191 unique speakers. Source: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv`.
- Governed sensitivity reference: 375 governed rows, with 348 `phase4_eval_allowed=True` rows and 27 governed/caution rows. Source: `reports/tables/phase6_5_full_governed_subset_manifest_cleaned.csv`.
- Feature source: `data/features/features_phase3_governed_pruned.csv`.
- Feature count: 256 official numeric features. Source: `scripts/phase4_baseline_experiments.py`; `reports/phaseA_classical_baseline_diagnostics.md`.
- Model: `logistic_regression_platt`.
- Threshold: frozen at `0.300`. Source: `reports/phase4_official_winner.md`; `reports/phase6_5_week2_phase4_rerun_comparison.md`.
- Protocol: speaker-grouped CV by `speaker_id`; seed 42 for the main cleaned anchor and seeds 42-46 for stability. Source: `scripts/phase4_baseline_experiments.py`; `reports/phase6_5_week2b_stability_addendum.md`.

## 3. Performance Summary

Cleaned seed 42 clip-level metrics at threshold `0.300`: recall 0.117, specificity 0.909, precision 0.324, F1 0.172, balanced accuracy 0.513, Brier score 0.197, PR-AUC 0.310, and confusion matrix `TN=231, FP=23, FN=83, TP=11`. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/tables/phase6_5_week2b_seed_summary.csv`.

Across seeds 42-46, the cleaned anchor had threshold mean 0.250 with standard deviation 0.077, recall mean 0.464 with standard deviation 0.380, specificity mean 0.560 with standard deviation 0.376, precision mean 0.291 with standard deviation 0.019, F1 mean 0.292 with standard deviation 0.111, and balanced accuracy mean 0.512 with standard deviation 0.012. Source: `reports/tables/phase6_5_week2b_comparison_update.csv`.

Speaker-level seed 42 max-risk aggregation had recall 0.119, specificity 0.832, precision 0.357, F1 0.179, balanced accuracy 0.475, and confusion matrix `TN=89, FP=18, FN=74, TP=10`. Source: `reports/tables/phase6_5_week2b_anchor_speaker_level_metrics.csv`.

The historical original Phase 4 anchor looked much stronger before cleaning: recall 0.862, specificity 0.217, precision 0.289, F1 0.433, balanced accuracy 0.539, and false negatives 13. The cleaned rerun shows that this historical behavior should not be used as final cleaned performance. Source: `reports/phase4_official_winner.md`; `reports/phase6_5_week2_phase4_rerun_comparison.md`.

## 4. Error Summary

False positives are not concentrated in one or two speakers. B2b found 23 false positives across 18 speakers; top 1, 3, 5, and 10 speakers contributed 8.7 percent, 26.1 percent, 43.5 percent, and 65.2 percent of false positives, respectively. Source: `reports/phase_b2b_false_positive_audit.md`; `reports/tables/phase_b2b_fp_speaker_concentration.csv`.

False positives are mostly borderline: 21 of 23 were in `[0.300, 0.400)`, 2 of 23 were in `[0.400, 0.500)`, and none exceeded 0.500. Source: `reports/tables/phase_b2b_fp_score_distribution.csv`.

False negatives dominate the cleaned seed 42 anchor: 83 of 94 dementia-labeled evaluation clips were false negatives. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/tables/phase6_5_week2b_seed_summary.csv`.

The final error pattern is heavy score overlap and threshold fragility, not a single bad-speaker or obvious low-quality subset. Source: `reports/phase_b2b_false_positive_audit.md`; `docs/FINAL_ERROR_ANALYSIS.md`.

## 5. Limitations Statement

This baseline is not clinically validated, not diagnostic, and not generalizable beyond this dataset without external testing. It has weak cleaned-data recall at the frozen seed 42 threshold, unstable seed behavior, and a heavily compressed decision boundary. Source: `reports/phase6_5_week2b_stability_addendum.md`; `docs/BASELINE_MODEL_CARD.md`.

The threshold is frozen for reproducibility and documentation only. It is not a production threshold policy. Source: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`.

## 6. Comparison To Alternatives

B2c tested four feature subsets and three simple model families. No combination achieved recall >= 0.70 with specificity >= 0.40. Source: `reports/phase_b2c_feature_curation_report.md`; `reports/tables/phase_b2c_experiment_results.csv`.

The best B2c candidate, `cleaned_all_features` with `linear_svm_platt`, reached mean recall 0.311 at specificity >= 0.40 and mean recall 0.232 at specificity >= 0.50, and it did not pass the screening criterion. Source: `reports/phase_b2c_feature_curation_report.md`; `reports/tables/phase_b2c_experiment_results.csv`.

The final freeze therefore keeps the interpretable `logistic_regression_platt` anchor rather than promoting a non-passing alternative.

## 7. Next Steps

Improvement would require materially better data, stronger feature curation, external validation, or a separate deep-learning research track. This summary does not initiate those steps and does not propose further experiments in this freeze.

## 8. Conclusion

`final_cleaned_logistic_regression_platt` at threshold `0.300` is the official final baseline for the project. It is defensible as a documented academic baseline and not defensible as a clinical or deployment model. Source: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`; `docs/BASELINE_MODEL_CARD.md`.
