# Phase B2b Status

## What B2b Found

- Cleaned anchor audited: `logistic_regression_platt`, Week 2 seed `42`, threshold `0.300`.
- Control-clip audit set: `23` false positives and `231` true negatives.
- Speaker concentration: top 1=2/23 (8.7%), top 3=6/23 (26.1%), top 5=10/23 (43.5%), top 10=15/23 (65.2%); `18` speakers had at least one FP.
- Score position: `87.0%` of FPs were within 0.05 of the threshold; `0.0%` exceeded 0.500; `0.0%` exceeded 0.700.
- Available quality/provenance fields do not identify a single dominant low-quality or governance bucket.

## Bucket Diagnosis

- Dominant diagnosis: `Mixed C/D`.
- Bucket A, few bad speakers: `weak`.
- Bucket B, low-quality/weird-condition clips: `weak`.
- Bucket C, broad feature-space overlap: `strong`.
- Bucket D, borderline score proximity: `strong`.

## Meaning For Next Step

The current cleaned anchor is not defensible as a final screening anchor. The FP pattern is too broad for a simple speaker-removal fix and too borderline/overlapping for threshold selection to rescue recall and specificity together.

Next step should be either formal Tier 1 closeout with documented limits of the classical anchor, or a narrowly scoped Tier 2 error-analysis plan. Further Tier 1 work is only justified if it is explicitly feature-curation/error-analysis work, not broader threshold or model-policy tuning.

## Current Anchor Defensibility

The anchor remains useful as a diagnostic baseline, but it is not defensible for final performance claims or deployment. It should not be treated as a production threshold policy.