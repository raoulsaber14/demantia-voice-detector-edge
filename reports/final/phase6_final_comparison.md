# Phase 6 Baseline vs wav2vec Comparison

This comparison uses the pre-defined Phase 6 protocol and identical speaker-level Phase 6 splits.

## Thresholds

- wav2vec threshold selected on calibration clips: `0.454939`
- wav2vec threshold note: met_target_recall_highest_precision
- same-split baseline threshold selected on calibration clips: `0.273934`
- same-split baseline threshold note: met_target_recall_highest_precision

## Metric Comparison

| metric | baseline_95ci | wav2vec_95ci | difference | verdict |
| --- | --- | --- | --- | --- |
| recall | 0.800 [0.600, 1.000] | 1.000 [1.000, 1.000] | 0.19999999999999996 | up |
| balanced_accuracy | 0.438 [0.328, 0.534] | 0.500 [0.500, 0.500] | 0.0625 | up |
| precision | 0.245 [0.125, 0.415] | 0.273 [0.143, 0.447] | 0.027829313543599243 | equivalent |
| f1 | 0.375 [0.215, 0.559] | 0.429 [0.250, 0.618] | 0.05357142857142855 | up |
| roc_auc | 0.375 [0.181, 0.578] | 0.575 [0.360, 0.795] | 0.2 | up |
| specificity | 0.075 [0.000, 0.152] | 0.000 [0.000, 0.000] | -0.075 | down |

## Final Conclusion

**Conclusion:** Based on the pre-defined evaluation protocol (primary metric = recall for atypical class, improvement threshold = +0.03), wav2vec is inconclusive compared to the baseline. Recall difference = 0.200.

The wav2vec test result should not be interpreted as a clear improvement. Although test recall increased from `0.800` to `1.000`, wav2vec classified every held-out test clip as atypical, producing specificity `0.000`. This violates the pre-defined improvement rule because a secondary metric decreased by more than `0.05` and indicates a high false-positive burden.

The same-split baseline threshold above is a Phase 6 comparison artifact only. It does not replace or modify the frozen Phase 4 winner (`logistic_regression_platt`) or the official Phase 4 threshold (`0.300`).

A null result is acceptable if the experiment is executed correctly.
