# Final Recall Threshold Sweep Report

## 1. Scope

This run stayed inside the locked Phase D frame: trusted cleaned subset, speaker-grouped outer folds, seeds 42-46, and the held three-component max-probability ensemble only. No new architectures, features, folds, cached embeddings, or labels were introduced.

Important implementation note: the HuBERT-only and HuBERT + handcrafted components use `majority_vote` speaker aggregation, so their speaker-level component scores are threshold-conditioned by design. This sweep preserves the original Phase D implementation rather than inventing a new decoupled rule.

## 2. Reconstruction Inputs

- Trusted manifest: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv`.
- HuBERT cache: `artifacts/phaseC_frozen_embeddings/trusted_cleaned/facebook__hubert_base_ls960_mean_std_chunk8p0.csv.gz`.
- wav2vec2 cache: `artifacts/phaseC_frozen_embeddings/trusted_cleaned/facebook__wav2vec2_base_mean_std_chunk8p0.csv.gz`.
- Handcrafted feature count: **256**.
- Execution stack: Python `3.9.6`, scikit-learn `1.6.1`, pandas `2.2.3`, numpy `2.0.2`.

## 3. Validation Checks

| check | value |
| --- | --- |
| manifest_speaker_label_inconsistencies | 0 |
| speaker_train_validation_separation_pass | True |
| speaker_leakage_overlap_total | 0 |
| component_probability_non_numeric_count | 0 |
| component_probability_out_of_bounds_count | 0 |
| component_clip_count_mismatch_rows | 0 |
| speaker_alignment_pass | True |
| duplicate_aggregated_speaker_rows | 0 |
| missing_component_probability_cells | 0 |
| historical_reproduction_pass | True |
| historical_reproduction_max_abs_delta | 0.0 |

## 4. Fixed Global Sweep

| threshold | speaker_recall_mean | speaker_specificity_mean | speaker_precision_mean | speaker_f1_mean | false_negatives_mean | false_positives_mean | target_zone_pass | selection_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.6595 | 0.6804 | 0.6184 | 0.6381 | 28.6 | 34.2 | no |  |
| 0.475 | 0.6667 | 0.6542 | 0.6021 | 0.6324 | 28.0 | 37.0 | no |  |
| 0.45 | 0.6786 | 0.6486 | 0.6025 | 0.638 | 27.0 | 37.6 | no |  |
| 0.425 | 0.6881 | 0.6336 | 0.5962 | 0.6386 | 26.2 | 39.2 | no |  |
| 0.4 | 0.7 | 0.6112 | 0.586 | 0.6376 | 25.2 | 41.6 | yes |  |
| 0.375 | 0.7024 | 0.5925 | 0.5756 | 0.6323 | 25.0 | 43.6 | yes |  |
| 0.35 | 0.7119 | 0.5701 | 0.5655 | 0.63 | 24.2 | 46.0 | yes | max_recall_subject_to_specificity_ge_0.50 |
| 0.325 | 0.7167 | 0.3925 | 0.4809 | 0.5755 | 23.8 | 65.0 | no |  |
| 0.3 | 0.7214 | 0.3813 | 0.4781 | 0.575 | 23.4 | 66.2 | no |  |
| 0.275 | 0.7238 | 0.3664 | 0.473 | 0.572 | 23.2 | 67.8 | no |  |
| 0.25 | 0.7333 | 0.3477 | 0.4689 | 0.5719 | 22.4 | 69.8 | no |  |
| 0.225 | 0.7405 | 0.3402 | 0.4684 | 0.5737 | 21.8 | 70.6 | no |  |
| 0.2 | 0.7476 | 0.3346 | 0.4687 | 0.576 | 21.2 | 71.2 | no |  |

## 5. Training-Selected Threshold Evaluation

| analysis_type | speaker_recall_mean | speaker_specificity_mean | speaker_precision_mean | speaker_f1_mean | selected_threshold_mean | selected_threshold_min | selected_threshold_max | target_zone_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_selected_threshold | 0.6595 | 0.6804 | 0.6184 | 0.6381 | 0.5 | 0.5 | 0.5 | no |

## 6. Required Answers

1. Did any threshold reach recall >= 0.70 and specificity >= 0.50? Yes. Thresholds reaching the target zone in the fixed global sweep: 0.400, 0.375, 0.350.
2. Best threshold under the predefined selection rule: `0.350`.
3. Recall improvement vs `default_0.5`: +0.0524.
4. Specificity change vs `default_0.5`: -0.1103.
5. Were the remaining false negatives borderline or deeply missed? Not mostly borderline. At the selected threshold there were 2 borderline, 3 moderate, and 116 deep false negatives.
6. Are false positives mostly single-component or multi-component supported? Mostly multi-component (85 single-component vs 145 multi-component false positives at the selected threshold).
7. Should the final project claim target-zone pass? Exploratory fixed-sweep yes, final training-selected-threshold no. The target zone was reached only in the fixed global sweep, not in the separate training-selected evaluation.
8. Should `max_probability_ensemble default_0.5` remain the final held candidate, or should the new selected threshold replace it? The fixed-sweep operating point `0.350` is the strongest exploratory threshold under the predefined selection rule. The separate training-selected-threshold evaluation reached recall 0.6595, specificity 0.6804, and F1 0.6381. Because the fixed sweep scans held-out labels directly, and because this training-selected result did not move beyond `default_0.5`, `max_probability_ensemble default_0.5` should remain the final held candidate while `0.350` is reported as an exploratory post-hoc threshold finding.

## 7. Recommended Final Framing

The fixed-sweep best threshold is `0.350` with mean speaker recall 0.7119, specificity 0.5701, precision 0.5655, and F1 0.6300.
Because at least one fixed threshold reached the target zone, the report can state that the target zone was achieved in the exploratory fixed sweep. However, the training-selected-threshold evaluation stayed at `0.500`, so the held candidate should remain `default_0.5` unless a future clean validation run also supports the lower threshold.