# Final Presentation Bullets

## Problem

- Early dementia-risk signs can be missed when follow-up is delayed.
- Speech offers a low-cost, accessible signal for screening-oriented research.

## Goal

- Build a voice-based dementia-risk screening-support model.
- Prioritize catching dementia-positive speakers over minimizing all false positives.

## Dataset

- Trusted cleaned Phase D subset: `191` speakers, `348` clips
- Positive speakers: `84`
- Negative speakers: `107`
- Positive-speaker evidence is sparse: `74/84` positives have only one clip

## Model Approach

- Frozen speech-embedding models plus speaker-level aggregation
- Controlled Phase D comparison of HuBERT aggregation, HuBERT + handcrafted fusion, and wav2vec2

## Why Ensemble Was Used

- Different embedding models miss different positive speakers
- Max aggregation lets any one component rescue a positive case

## Final Selected Model

- `max_probability_ensemble default_0.5`
- Components:
  - HuBERT-only
  - HuBERT + handcrafted acoustic features
  - wav2vec2

## Main Results

- Mean speaker recall: `0.6595`
- Mean speaker specificity: `0.6804`
- Mean speaker F1: `0.6381`
- Best held Phase D candidate, but not target-zone pass

## Recall Vs False-Positive Tradeoff

- Remaining false negatives across seeds: `25, 27, 31, 29, 31`
- False positives across seeds: `36, 31, 37, 33, 34`
- Max-style behavior recovered `70` single-component TP seed-speaker cases
- Max-style behavior introduced `53` single-component FP seed-speaker cases

## Why The Ensemble Helped

- HuBERT-only and HuBERT + handcrafted were relatively redundant
- wav2vec2 provided the main complementary recovery signal
- Unique TP recoveries:
  - HuBERT-only: `12`
  - HuBERT + handcrafted: `7`
  - wav2vec2: `51`

## Clinical Safety Framing

- Screening-support model, not a diagnostic tool
- Use as risk flagging or referral support only
- False negatives matter more than false positives in this project context

## Limitations

- Small, imbalanced dataset
- Sparse positive-speaker evidence
- False-positive burden increased under max aggregation
- No clinical validation yet
- No saved per-speaker ensemble artifact for full case-level review

## Future Work

- More dementia-positive speakers
- More clips per speaker
- Better calibration
- Preserved per-speaker and per-clip ensemble artifacts
- External validation
- Clinician review

## Final Conclusion

- `max_probability_ensemble default_0.5` is the best held recall-recovery candidate
- It improves the chance of catching dementia-positive speakers while staying above the specificity floor
- It should be reported as a held screening-support result, not a validated clinical model
