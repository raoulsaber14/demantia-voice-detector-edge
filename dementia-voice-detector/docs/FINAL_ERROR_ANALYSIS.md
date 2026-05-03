# Final Error Analysis

Freeze date: 2026-04-22.

This summary uses existing completed artifacts only. No model was rerun and no threshold was changed.

## 1. False Positive Summary

At the locked threshold `0.300`, the cleaned seed 42 `logistic_regression_platt` anchor produced 23 false positives and 231 true negatives. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/tables/phase6_5_week2b_seed_summary.csv`.

B2b found those 23 false positives across 18 speakers. The top 1 speaker contributed 2 of 23 false positives (8.7 percent), the top 3 contributed 6 of 23 (26.1 percent), the top 5 contributed 10 of 23 (43.5 percent), and the top 10 contributed 15 of 23 (65.2 percent). Source: `reports/phase_b2b_false_positive_audit.md`; `reports/tables/phase_b2b_fp_speaker_concentration.csv`.

The highest false-positive-count speakers in B2b were `bobdylan`, `davidallancoe`, `georgearomero`, `paulmccartney`, and `robertvaughn`, each with 2 false-positive clips. Source: `reports/tables/phase_b2b_fp_speaker_concentration.csv`.

Score position was borderline: 21 of 23 false positives were in `[0.300, 0.400)`, 2 of 23 were in `[0.400, 0.500)`, and 0 were above 0.500. Source: `reports/tables/phase_b2b_fp_score_distribution.csv`.

## 2. False Negative Summary

At the locked threshold `0.300`, the cleaned seed 42 anchor produced 83 false negatives and 11 true positives, meaning it missed 83 of 94 dementia-labeled evaluation clips. Source: `reports/phase6_5_week2_phase4_rerun_comparison.md`; `reports/tables/phase6_5_week2b_seed_summary.csv`.

Using the existing cleaned OOF prediction file for `logistic_regression_platt` at the frozen threshold, the 83 false negatives were spread across 77 speakers. The largest per-speaker false-negative count was 2 clips. Speakers with 2 false negatives included `aileenhernandez`, `billbuckner`, `davidprowse`, `georgeklein`, `jessehelms`, and `mauricehinchey`. Source: derived from `reports/tables/phase4_oof_predictions_cleaned.csv` using threshold `0.300`; threshold source `reports/phase6_5_week2_phase4_rerun_comparison.md`.

False-negative scores were also close to the threshold from below: 68 of 83 false negatives were in `[0.250, 0.300)`, and 15 of 83 were in `[0.200, 0.250)`. Source: derived from `reports/tables/phase4_oof_predictions_cleaned.csv` using threshold `0.300`.

## 3. Borderline Case Summary

B2b reported that 87.0 percent of false positives were within 0.05 of the threshold. Source: `reports/phase_b2b_false_positive_audit.md`.

For false negatives, 68 of 83, or 81.9 percent, were within 0.05 below the threshold. Source: derived from `reports/tables/phase4_oof_predictions_cleaned.csv` using threshold `0.300`.

Across all cleaned seed 42 `logistic_regression_platt` OOF predictions, 283 of 348 predictions, or 81.3 percent, were within 0.05 of the threshold. Source: derived from `reports/tables/phase4_oof_predictions_cleaned.csv` using threshold `0.300`.

This means single-clip decisions near the threshold are not reliable. The model has heavy score overlap, not a clean boundary.

## 4. Repeated Speaker Effects

Week 2b multi-seed artifacts show repeated error speakers, but not a single dominant speaker failure. Examples derived from existing Week 2b OOF prediction files and seed-specific thresholds include `bobdylan`, `francisfordcoppola`, `jeannemoreau`, and `derekjacobi` as repeated false-positive speakers across 5 of 5 seeds, and `georgeklein` and `jessehelms` as repeated false-negative speakers across 4 of 5 seeds. Source: derived from `reports/tables/phase4_oof_predictions_week2b_seed42.csv` through `reports/tables/phase4_oof_predictions_week2b_seed46.csv` with thresholds from `reports/tables/phase6_5_week2b_seed_summary.csv`.

B2b also found that high-FP speakers often had mixed false-positive and true-negative clips rather than uniform misclassification. Source: `reports/phase_b2b_false_positive_audit.md`; `reports/tables/phase_b2b_high_fp_speaker_detail.csv`.

## 5. Quality/Length Effects

B2b did not support a simple low-quality explanation for false positives. Duration was nearly identical between false positives and true negatives: FP mean 55.18 seconds versus TN mean 54.71 seconds, with effect size 0.017. Source: `reports/tables/phase_b2b_fp_vs_tn_clip_features.csv`.

Available quality proxies were limited. B2b noted that explicit `silence_ratio` was absent, so pause/silence-duration proxies were used, and explicit `rms_energy` was absent, so `rms_mean` was used. Source: `reports/phase_b2b_false_positive_audit.md`; `reports/tables/phase_b2b_fp_vs_tn_clip_features.csv`.

The strongest FP-vs-TN feature differences were descriptive spectral differences, not causal explanations. Source: `reports/tables/phase_b2b_top_discriminating_features.csv`.

## 6. What Users Should Not Trust

Users should not trust single-clip predictions, raw probabilities, predictions close to the threshold, or predictions on speakers whose recording conditions or population context differ materially from this repository. The baseline is a research artifact only and cannot confirm or rule out dementia. Source: `reports/phase4_official_winner.md`; `docs/PHASE6_5_DEMO_APP_ALIGNMENT_SPEC.md`; `docs/BASELINE_MODEL_CARD.md`.
