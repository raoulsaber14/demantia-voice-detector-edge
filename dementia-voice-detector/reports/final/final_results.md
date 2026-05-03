# Final Results

This summary reflects the final held project decision using existing completed artifacts only. No new experiment was run for this document.

## Final held candidate

- Candidate: `max_probability_ensemble default_0.5`
- Evidence files:
  - `reports/phaseD_hubert_recall_recovery_benchmark.md`
  - `reports/tables/phaseD_best_candidate_analysis.csv`
  - `reports/tables/phaseD_model_comparison.csv`
  - `reports/tables/phaseD_ensemble_seed_summary.csv`
  - `reports/tables/phaseD_error_complementarity_summary.csv`
- Multi-seed mean speaker metrics:
  - Recall: `0.6595`
  - Specificity: `0.6804`
  - F1: `0.6381`
- Final status: `hold`

## Why this candidate was selected

- It was the strongest Phase D recall-recovery result.
- It ranked first in `reports/tables/phaseD_model_comparison.csv`.
- `reports/tables/phaseD_best_candidate_analysis.csv` marks it as better than both the Phase C frozen reference and the official classical baseline while preserving the specificity floor.
- It did not reach the multi-seed target zone, so it is treated as the best held candidate rather than a fully validated final model.

## What improved recall

- The ensemble combines:
  - HuBERT-only
  - HuBERT + handcrafted acoustic features
  - wav2vec2
- HuBERT-only and HuBERT + handcrafted were relatively redundant.
- wav2vec2 contributed the main complementary recovery signal.
- The max rule improved recall by allowing any one component to rescue a positive speaker.

## Benefit-cost summary

- Remaining false negatives across seeds 42-46: `25, 27, 31, 29, 31`
- False positives across seeds 42-46: `36, 31, 37, 33, 34`
- Derived from the saved Phase D summaries, max-style behavior recovered `70` single-component true-positive seed-speaker cases and introduced `53` single-component false-positive seed-speaker cases.
- Mean specificity remained above the minimum acceptable floor: `0.6804` vs floor `0.50`.

## Interpretation

This candidate is best understood as a screening-support or referral-support model. It improves the chance of catching dementia-positive speakers, which matters more than minimizing false positives in this project framing, but it still increases false positives and is not clinically validated.

## Risk statement

This project does not diagnose dementia. The final held model should be described as a voice-based dementia-risk flagging or referral-support model, not as a diagnostic system or medical probability estimator.

## Related final documents

- `reports/final_model_decision.md`
- `reports/final_limitations_and_future_work.md`
- `reports/final_presentation_bullets.md`
