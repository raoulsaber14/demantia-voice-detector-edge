# Phase 6.5 Tier 1 Stability Gate

## Why Tier 1 Has Not Passed

Week 2 showed that data-trust cleaning materially changed Phase 4 behavior. The historical anchor `logistic_regression_platt` at threshold `0.300` dropped from recall `0.862` on the original Phase 4 setup to recall `0.117` on the cleaned setup.

Week 2b added multi-seed evidence for the cleaned anchor. Across seeds `42, 43, 44, 45, 46`, the cleaned `logistic_regression_platt` anchor had wide variation in selected threshold, recall, and specificity. The internally selected winner also changed across seeds. This means a single-seed winner is not enough for an official final model.

## Screening-First Requirement

The project remains screening-support only. The Phase 4 threshold policy prioritizes recall while requiring precision at least the observed positive-class prevalence and specificity at least `0.20`, with explicit fallback if no threshold satisfies the rule. Any future official model must be interpreted under this screening-first, non-diagnostic framing.

## Stability Gate

A future official winner must satisfy all of the following before Tier 1 can pass:

- Use grouped cross-validation with speaker leakage controls.
- Be evaluated across the approved multi-seed protocol rather than a single seed.
- Show stable threshold behavior across seeds and folds.
- Preserve screening usefulness under the existing Phase 4 threshold-selection policy.
- Report recall, specificity, precision, F1, balanced accuracy, confusion matrix counts, threshold selection, and calibration notes.
- Document any fallback threshold selections instead of treating them as strong wins.

No single-seed result should be accepted as the official winner if it fails this stability gate.

## Current Status

Tier 1 continues. The immediate prerequisite is data-trust adjudication of the 37 metadata/audio-match rows, followed by a grouped-CV, multi-seed Phase 4 rerun after any adjudication changes are finalized.
