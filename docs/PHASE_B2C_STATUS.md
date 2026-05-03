# Phase B2c Status

## Outcome

- B2c tested `4` curated feature subsets x `3` simple model families across seeds `42-46`.
- Better model found by screening criterion: `no`.
- Best candidate: `cleaned_all_features` + `linear_svm_platt`.
- Best candidate recall at specificity >=0.40: `0.311`.
- Best candidate recall at specificity >=0.50: `0.232`.
- Best candidate threshold std across seeds: `0.045`.

## Proposed Anchor

No new official anchor is proposed. Keep the current cleaned anchor only as a diagnostic baseline, not as a defensible screening model.

## Next Phase Meaning

B2c supports accepting the limits of the current classical feature space or moving to a scoped Tier 2 error-analysis plan. Further Tier 1 threshold tuning is not justified by these results.

## Deployment Consideration

Proceed to deployment consideration: `no`.