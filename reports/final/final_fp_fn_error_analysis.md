# Final FP/FN Error Analysis

Analysis-only phase. No model, threshold, feature, label, governance, app, deployment, or hardware behavior was changed.

## 1. Official baseline analyzed

- Official baseline: `final_cleaned_logistic_regression_platt` / `logistic_regression_platt`.
- Official threshold: `0.300`.
- Primary evidence dataset: cleaned trusted subset.
- Evaluation artifact analyzed: `reports/tables/phase4_oof_predictions_cleaned.csv` filtered to `logistic_regression_platt`.
- Governed data were not used for primary conclusions.
- Source: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`; `docs/BASELINE_MODEL_CARD.md`; `reports/final/FINAL_BASELINE_SUMMARY.md`.

## 2. Data sources actually used

- `docs/OFFICIAL_BASELINE_CONFIGURATION.md` - used.
- `docs/BASELINE_MODEL_CARD.md` - used.
- `docs/FINAL_ERROR_ANALYSIS.md` - used.
- `reports/final/FINAL_BASELINE_SUMMARY.md` - used.
- `reports/phaseB2a_specificity_floor_analysis.md` - used.
- `reports/tables/phaseB2a_specificity_floor_seed_results.csv` - used.
- `reports/tables/phaseB2a_false_positive_speaker_audit.csv` - used.
- `reports/tables/phase_b2b_fp_speaker_concentration.csv` - used.
- `reports/tables/phase_b2b_fp_vs_tn_clip_features.csv` - used.
- `reports/tables/phase_b2b_fp_score_distribution.csv` - used.
- `reports/tables/phase_b2b_top_discriminating_features.csv` - used.
- `reports/tables/phase_b2b_high_fp_speaker_detail.csv` - used.
- `reports/phase_b2b_false_positive_audit.md` - used.
- `reports/tables/phase_b2c_experiment_results.csv` - used.
- `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv` - used.
- `data/features/features_phase3_governed_pruned.csv` - used.
- `reports/tables/phase4_oof_predictions_cleaned.csv` - chosen prediction-level artifact for the official frozen baseline.
- `reports/tables/phase4_oof_predictions_week2b_seed42.csv` - checked as an equivalent Week 2b seed-42 prediction artifact for `logistic_regression_platt` scores/folds.
- `reports/tables/phase4_oof_predictions_cleaned_trusted_seed42.csv` - checked but not chosen because its threshold-0.300 confusion matrix did not match the frozen baseline evidence.
- `reports/tables/phase4_oof_predictions_cleaned_governed_seed42.csv` - checked but not chosen because it is governed sensitivity context and did not match the frozen primary evidence.

## 3. Prediction artifact chosen and why

Chosen artifact: `reports/tables/phase4_oof_predictions_cleaned.csv`.

Filtered to `logistic_regression_platt`, this file has 348 rows, 348 unique clip IDs, and 191 speakers. At threshold `0.300`, it exactly reproduces the frozen evidence confusion matrix: `TN=231, FP=23, FN=83, TP=11`. Source: `reports/tables/phase6_5_week2b_seed_summary.csv`; `reports/tables/phase4_oof_predictions_cleaned.csv`.

`reports/tables/phase4_oof_predictions_week2b_seed42.csv` has the same `logistic_regression_platt` metadata IDs, folds, and scores. I used `reports/tables/phase4_oof_predictions_cleaned.csv` because it is the same cleaned anchor artifact already cited by the final freeze error analysis.

`reports/tables/phase4_oof_predictions_cleaned_trusted_seed42.csv` was not chosen because threshold `0.300` gives `TN=227, FP=27, FN=78, TP=16`, which conflicts with the frozen official baseline metrics. `reports/tables/phase4_oof_predictions_cleaned_governed_seed42.csv` was not chosen because it is governed/sensitivity context and gives a different threshold-0.300 confusion matrix.

## 4. Overall error composition

- Total evaluated clips: 348.
- Correct predictions: 242; errors: 106.
- TP/TN/FP/FN: `TP=11`, `TN=231`, `FP=23`, `FN=83`.
- Error composition: FP=23/106 (21.7%), FN=83/106 (78.3%).
- FP rate among true negatives: 23/254 (9.1%).
- FN rate among true positives: 83/94 (88.3%).
- Precision: 0.324, recall: 0.117, specificity: 0.909, F1: 0.172, balanced accuracy: 0.513, accuracy: 0.695.

The dominant error type is false negative: nearly four out of five errors are missed dementia-labeled clips.

## 5. False-positive analysis

False positives are a minority of total errors: 23 of 106 errors (21.7%). They are spread across 18 speakers. The top FP speakers are: `bobdylan`=2, `davidallancoe`=2, `georgearomero`=2, `paulmccartney`=2, `robertvaughn`=2, `bobbarker`=1, `derekjacobi`=1, `francisfordcoppola`=1, `jeannemoreau`=1, `joebiden`=1. Source table: `reports/tables/final_fp_speaker_concentration.csv`.

FP score structure is mostly borderline: mean score 0.325, median 0.316, range 0.300 to 0.412, with 20/23 within 0.05 of threshold and 0 above 0.50. Source table: `reports/tables/final_error_score_distribution.csv`.

Top FP-vs-TN descriptive numeric differences:

| field_name | group_a_mean | group_b_mean | mean_diff_a_minus_b | effect_size_proxy |
| --- | --- | --- | --- | --- |
| score | 0.32510449296991345 | 0.2628581558810886 | 0.06224633708882488 | 3.0005722403088844 |
| score_margin_from_threshold | 0.025104492969913465 | -0.0371418441189114 | 0.06224633708882486 | 3.0005722403088835 |
| f0_analysis_duration_sec | 9.826086956521738 | 9.991341991341992 | -0.1652550348202535 | -0.6309497241699501 |
| abs_score_margin_from_threshold | 0.025104492969913465 | 0.0371418441189114 | -0.012037351148997937 | -0.580258106641554 |
| voiced_ratio | 0.6256003345618334 | 0.7019455555584981 | -0.07634522099666474 | -0.44441537874943987 |
| low_energy_ratio | 0.4996846995449139 | 0.49979723928943837 | -0.00011253974452446469 | -0.38293105157808316 |

Available quality/provenance fields do not isolate a dominant cleanup bucket. Duration, pause, RMS/energy, and provenance differences are descriptive and should not be read causally. Quality fields are existing feature-table columns, not new audio-quality metrics. Source table: `reports/tables/final_fp_vs_tn_summary.csv`.

## 6. False-negative analysis

False negatives dominate the error set: 83 of 106 errors (78.3%). They are spread across 77 speakers. The top FN speakers are: `aileenhernandez`=2, `billbuckner`=2, `davidprowse`=2, `georgeklein`=2, `jessehelms`=2, `mauricehinchey`=2, `alanramsey`=1, `allanburns`=1, `andrewsachs`=1, `annettemichelson`=1. Source table: `reports/tables/final_fn_speaker_concentration.csv`.

FN score structure is also mostly borderline: mean score 0.265, median 0.269, range 0.212 to 0.298, with 68/83 within 0.05 of threshold and 0 below 0.10. Source table: `reports/tables/final_error_score_distribution.csv`.

Top FN-vs-TP descriptive numeric differences:

| field_name | group_a_mean | group_b_mean | mean_diff_a_minus_b | effect_size_proxy |
| --- | --- | --- | --- | --- |
| score | 0.2653995757174641 | 0.32122348795594863 | -0.05582391223848454 | -2.847433431054841 |
| score_margin_from_threshold | -0.034600424282535874 | 0.02122348795594863 | -0.0558239122384845 | -2.8474334310548386 |
| abs_score_margin_from_threshold | 0.034600424282535874 | 0.02122348795594863 | 0.013376936326587246 | 0.6823229360689195 |
| rms_mean | 0.08647900687224043 | 0.07216081241793665 | 0.014318194454303781 | 0.3912225077756417 |
| rms_median | 0.07915731817932728 | 0.0647005892612717 | 0.014456728918055578 | 0.3458894938407014 |
| rms_std | 0.05999038432709897 | 0.05355174426887799 | 0.006438640058220979 | 0.3215696008229786 |

The missed-positive pattern is broad rather than a small subgroup. The FN-vs-TP feature/profile table should be treated as descriptive evidence of weak separability, not a causal explanation. Source table: `reports/tables/final_fn_vs_tp_summary.csv`.

## 7. Borderline vs confident error analysis

- FPs mostly borderline: 20/23 (87.0%) are within 0.05 of the threshold; 0 are above 0.50.
- FNs mostly borderline: 68/83 (81.9%) are within 0.05 of the threshold; 0 are below 0.10.
- More threshold-sensitive side: false negatives, because they are 78.3% of all errors and most sit just below the threshold.
- Confidently wrong cases are not the main pattern under the requested confidence definitions.

## 8. Speaker concentration analysis

FP errors are broad: 18 FP speakers, with top-1 contribution 8.7%. Repeated FP errors across multiple clips exist for 5 speakers. Source: `reports/tables/final_fp_speaker_concentration.csv`.

FN errors are even broader: 77 FN speakers, with top-1 contribution 2.4%. Repeated FN errors across multiple clips exist for 6 speakers. Source: `reports/tables/final_fn_speaker_concentration.csv`.

Speaker sensitivity is present but secondary. Neither error type is concentrated enough to justify a speaker-removal or speaker-specific governance action.

## 9. Feature-pattern analysis

Top FP-vs-TN differentiating official features by absolute standardized effect size:

| rank | feature_name | feature_family | effect_size_proxy | group_a_mean | group_b_mean |
| --- | --- | --- | --- | --- | --- |
| 1.000 | delta2_mfcc_2_std | mfcc | -0.859 | 4.727 | 5.645 |
| 2.000 | delta_mfcc_2_std | mfcc | -0.836 | 7.302 | 8.705 |
| 3.000 | spectral_centroid_std | spectral | -0.784 | 672.164 | 815.421 |
| 4.000 | delta_mfcc_2_max | mfcc | -0.765 | 25.181 | 30.003 |
| 5.000 | spectral_rolloff_std | spectral | -0.764 | 1319.854 | 1535.818 |
| 6.000 | mfcc_2_std | mfcc | -0.729 | 38.908 | 44.794 |
| 7.000 | delta2_mfcc_2_mean | mfcc | -0.721 | -0.029 | -0.000 |
| 8.000 | zcr_std | zcr | -0.676 | 0.071 | 0.090 |

Top FN-vs-TP differentiating official features by absolute standardized effect size:

| rank | feature_name | feature_family | effect_size_proxy | group_a_mean | group_b_mean |
| --- | --- | --- | --- | --- | --- |
| 1.000 | delta_mfcc_9_min | mfcc | 1.570 | -6.882 | -9.368 |
| 2.000 | delta2_mfcc_11_max | mfcc | -1.438 | 4.074 | 5.333 |
| 3.000 | delta_mfcc_11_min | mfcc | 1.177 | -6.033 | -7.780 |
| 4.000 | delta2_mfcc_12_std | mfcc | -1.153 | 1.065 | 1.289 |
| 5.000 | mfcc_12_std | mfcc | -1.145 | 8.834 | 10.593 |
| 6.000 | mfcc_9_min | mfcc | 1.136 | -46.593 | -57.342 |
| 7.000 | delta_mfcc_13_max | mfcc | -1.131 | 5.261 | 6.573 |
| 8.000 | delta_mfcc_12_std | mfcc | -1.118 | 1.588 | 1.911 |

The feature-pattern result is consistent with broad acoustic feature-space overlap. Feature-family labels are heuristic labels from existing column-name prefixes, used only for readability. The table does not identify a single broken feature, source field, or quality field that can explain the errors. Source table: `reports/tables/final_error_top_discriminating_features.csv`.

## 10. Primary diagnosis

**Final diagnosis: `mixed but primarily feature-space blind-spot`.**

The immediate error scores are borderline, but the deeper reason is weak feature-space separation: both FP and FN errors are broadly distributed, available quality/source fields do not explain the pattern, and prior B2a/B2c evidence shows threshold motion or simple classical variants do not rescue the baseline. The threshold-borderline pattern is therefore a symptom of compressed scores, not a sufficient fix path by itself.

## 11. Most justified next action

The single most justified next action is to document the classical baseline ceiling and use this FP/FN structure to guide richer-representation or new-data planning, while keeping the frozen baseline unchanged. If any frozen-baseline-adjacent work is done, it should be feature-space blind-spot analysis and not threshold retuning. Source table: `reports/tables/final_error_next_step_recommendation.csv`.

## 12. Least justified next action

The single least justified next action is retuning or replacing the frozen threshold as the next fix. The threshold is frozen by definition, and prior specificity-floor work showed threshold changes trade recall and specificity sharply rather than solving separability.

## 13. Bottom-line conclusion about remaining baseline upside

The frozen classical baseline has little realistic room for meaningful improvement without new data or a richer representation. The current errors are mostly near-threshold, broadly distributed, and not dominated by a clean quality/source/speaker bucket. This baseline is defensible as a documented academic reference, not as an improvement platform for deployment.
