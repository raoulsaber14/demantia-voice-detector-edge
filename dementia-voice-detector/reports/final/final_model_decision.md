# Final Model Decision

This document finalizes the held model decision for the project using the saved Phase D artifacts only. No code was changed, no threshold was tuned, and no model was rerun for this decision note.

## Executive Conclusion

The selected held Phase D candidate is `max_probability_ensemble default_0.5`. It is the strongest recall-recovery result in the repository because it raises speaker-level dementia recall materially while keeping specificity above the minimum acceptable floor. It should remain a held screening-support candidate, not a clinically validated final model.

## Evidence Location

The main decision evidence is stored in:

- `reports/phaseD_hubert_recall_recovery_benchmark.md`
- `reports/tables/phaseD_best_candidate_analysis.csv`
- `reports/tables/phaseD_model_comparison.csv`
- `reports/tables/phaseD_ensemble_seed_summary.csv`
- `reports/tables/phaseD_error_complementarity_summary.csv`
- `reports/tables/phaseC_embedding_speaker_seed_summary.csv`
- `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv`

The most direct selection file is `reports/tables/phaseD_best_candidate_analysis.csv`, which records:

- `candidate_name = max_probability_ensemble default_0.5`
- `speaker_recall_mean = 0.6595`
- `speaker_specificity_mean = 0.6804`
- `speaker_f1_mean = 0.6381`
- `better_than_phaseC = yes`
- `better_than_classical_baseline = yes`
- `recommendation = hold`

## Why This Candidate Was Selected

`max_probability_ensemble default_0.5` was selected because it ranked first in the Phase D comparison table and produced the strongest multi-seed speaker-level recall recovery:

- Held candidate: recall `0.6595`, specificity `0.6804`, F1 `0.6381`
- Best HuBERT aggregation-only comparator: recall `0.4952`, specificity `0.7570`
- Best HuBERT + handcrafted fusion comparator: recall `0.4762`, specificity `0.7682`
- Phase C HuBERT frozen reference: recall `0.4762`, specificity `0.8673`

The candidate did not reach the target zone of recall `>= 0.70` with specificity `>= 0.50` on multi-seed means, so it remains a held best candidate rather than a validated endpoint.

## Why Recall Is Prioritized

This project is framed as dementia-risk screening support, not diagnosis. In that context, false negatives are more harmful than false positives:

- A false negative can miss a speaker who may need follow-up.
- A false positive can still be corrected later by clinical review.

Because of that project framing, the model choice gives more weight to dementia recall or sensitivity than to maximizing specificity.

## Why False Negatives Matter More Than False Positives

The project goal is early risk flagging or referral support. A conservative system that misses dementia-positive speakers undermines the main purpose of the tool. A model that produces some extra false positives can still be defensible if it catches more positive speakers and stays within a controlled specificity range.

That is why the max ensemble was held even though it is more aggressive than the safer mean-probability and majority-vote ensemble variants.

## What The False-Positive Tradeoff Means

The max ensemble increases false positives because it classifies a speaker as positive when any one component produces a strong enough positive speaker-level score.

Across seeds 42-46, the false-positive counts were:

- `36`
- `31`
- `37`
- `33`
- `34`

Average false-positive speakers per seed: `34.2`

The saved Phase D evidence supports a more nuanced interpretation than simple instability:

- Isolated one-model false positives across the five seed-speaker runs: `53`
- Multi-model-supported false positives across the five seed-speaker runs: `118`

This means the max ensemble is aggressive, but the added false positives are not mostly random single-model noise. A substantial portion had support from more than one component and therefore look more like difficult or borderline negative speakers than pure instability.

## What The Remaining False Negatives Mean

Across seeds 42-46, the remaining false-negative speaker counts were:

- `25`
- `27`
- `31`
- `29`
- `31`

Average false-negative speakers per seed: `28.6`

Under a max-probability speaker-level ensemble, a remaining false negative means that none of the component models produced enough positive speaker-level signal to rescue the speaker. These are therefore all-component misses, not cases where the ensemble suppressed a positive signal from one model.

That matters because it separates two failure modes:

- Ensemble-rule limitation: the rule ignored a useful positive signal
- Representation or data limitation: none of the components found enough positive evidence

The saved evidence supports the second interpretation. The remaining misses look more like hard low-evidence cases than aggregation failures.

## Why wav2vec2 Complementarity Matters

The ensemble combines:

1. HuBERT-only model
2. HuBERT + handcrafted acoustic feature model
3. wav2vec2 model

The complementarity evidence shows:

- HuBERT-only and HuBERT + handcrafted are relatively redundant:
  - FN overlap ratio `0.8151`
  - FP overlap ratio `0.7279`
  - Speaker disagreement count `84`
- wav2vec2 is the main complementary source:
  - HuBERT vs wav2vec2 disagreement count `271`
  - HuBERT + handcrafted vs wav2vec2 disagreement count `261`
  - Both pairings were judged `moderate` complementarity in the saved Phase D summary

Derived from the saved error-overlap and seed-summary tables, the max-style behavior recovered `70` single-component true-positive seed-speaker cases:

- HuBERT-only unique recoveries: `12`
- HuBERT + handcrafted unique recoveries: `7`
- wav2vec2 unique recoveries: `51`

This supports the claim that wav2vec2 provides the main complementary recovery signal and that the best Phase D result is not just random inflation.

## Screening-Oriented Interpretation

The final system should not be presented as if it outputs a diagnosis such as `dementia` or `non_dementia`.

Safer project wording is:

- `High dementia-risk flag: refer for clinical evaluation`
- `Low risk flag: no immediate signal detected`
- `Borderline or review-zone flag: repeat recording or clinical review recommended`

A simple risk-band design consistent with the existing Phase 5 and Phase 6.5 documentation is:

- High risk: refer to doctor or supervised clinical review
- Medium or review zone: repeat recording or clinician review
- Low risk: no immediate flag, but this does not rule out concern

The repository already contains screening-band and disclaimer guidance in:

- `reports/phase5_risk_output_definition.md`
- `reports/phase5_output_layer_policy.md`
- `docs/PHASE6_5_DEMO_APP_ALIGNMENT_SPEC.md`

This decision document does not change the code path for outputs. It records the final recommended interpretation framing.

## Why This Is Not Clinically Validated

This project should not claim clinical readiness. The current held model is:

- not externally validated
- not clinician-validated
- not calibrated as a medical probability
- not supported by preserved per-speaker ensemble artifacts for full case-level review

It is appropriate to call the result a screening-support or referral-support model, not a diagnostic or triage system.

## Final Recommendation

The project should stop broad model expansion at this point.

The final project conclusion should be:

> `max_probability_ensemble default_0.5` is the best held recall-recovery candidate in the repository. It improves the ability to catch dementia-positive speakers while maintaining specificity above the minimum acceptable floor. However, it remains a screening-support model and is not clinically validated. The correct next step for the project is final reporting and careful limitation framing, not a broader new model search.
