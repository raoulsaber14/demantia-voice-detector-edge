# Phase D HuBERT Recall Recovery Benchmark

## 1. Purpose

Phase D tests whether speaker-level recall can be recovered for the best frozen embedding path without collapsing specificity. The target zone remains speaker recall >= 0.70 and specificity >= 0.50. This is a controlled benchmark, not an open-ended experiment.

## 2. Frozen reference setup

Used frozen references: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`, `docs/BASELINE_MODEL_CARD.md`, `reports/final/FINAL_BASELINE_SUMMARY.md`, `reports/final/final_fp_fn_error_analysis.md`, `reports/phaseC_frozen_embedding_benchmark.md`, Phase C summary tables, the trusted cleaned manifest, and the canonical governed-pruned feature table. Phase C established `facebook/hubert-base-ls960` + `logistic_regression_platt` with `mean_score` as the best stable frozen embedding reference.

## 3. Data sources used

- Primary manifest: `reports/tables/phase6_5_trusted_subset_manifest_cleaned.csv`.
- Canonical handcrafted features: `data/features/features_phase3_governed_pruned.csv`.
- Fusion join key: `metadata_index`.
- Feature join: 348/348 trusted rows matched; 0 rows dropped; 256 official numeric handcrafted features.
- Cached embeddings used:

| backbone | cache_status | rows | skipped | embedding_dim | cache_csv |
| --- | --- | --- | --- | --- | --- |
| facebook/hubert-base-ls960 | loaded_existing_cache | 348 | 0 | 1536 | artifacts/phaseC_frozen_embeddings/trusted_cleaned/facebook__hubert_base_ls960_mean_std_chunk8p0.csv.gz |
| facebook/wav2vec2-base | loaded_existing_cache | 348 | 0 | 1536 | artifacts/phaseC_frozen_embeddings/trusted_cleaned/facebook__wav2vec2_base_mean_std_chunk8p0.csv.gz |

## 4. HuBERT aggregation benchmark

Fixed model: `facebook/hubert-base-ls960` + `logistic_regression_platt`. Tested `mean_score`, `max_score`, `majority_vote`, `top_2_mean`, `top_3_mean`, and `top_half_mean` under `default_0.5` and `constrained_spec_ge_0.50`.

| aggregation | threshold_policy | speaker_recall_mean | speaker_recall_std | speaker_specificity_mean | speaker_specificity_std | speaker_f1_mean | threshold_mean | target_zone_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_vote | default_0.5 | 0.4952 | 0.0247 | 0.757 | 0.0356 | 0.5491 | 0.5 | no |
| top_half_mean | default_0.5 | 0.4952 | 0.0247 | 0.6879 | 0.0253 | 0.5233 | 0.5 | no |
| max_score | default_0.5 | 0.4952 | 0.0247 | 0.5907 | 0.0259 | 0.4912 | 0.5 | no |
| mean_score | default_0.5 | 0.4762 | 0.0253 | 0.8673 | 0.0469 | 0.5793 | 0.5 | no |
| top_3_mean | default_0.5 | 0.4762 | 0.0253 | 0.8617 | 0.0425 | 0.5767 | 0.5 | no |
| top_2_mean | default_0.5 | 0.4762 | 0.0253 | 0.7794 | 0.0416 | 0.5423 | 0.5 | no |
| majority_vote | constrained_spec_ge_0.50 | 0.2714 | 0.0258 | 0.9439 | 0.0114 | 0.4038 | 0.9808 | no |
| top_half_mean | constrained_spec_ge_0.50 | 0.2714 | 0.0258 | 0.9383 | 0.0084 | 0.4017 | 0.9808 | no |
| max_score | constrained_spec_ge_0.50 | 0.2714 | 0.0258 | 0.8579 | 0.0291 | 0.3738 | 0.9808 | no |
| mean_score | constrained_spec_ge_0.50 | 0.2262 | 0.0292 | 0.985 | 0.0084 | 0.3628 | 0.9808 | no |
| top_3_mean | constrained_spec_ge_0.50 | 0.2262 | 0.0292 | 0.985 | 0.0084 | 0.3628 | 0.9808 | no |
| top_2_mean | constrained_spec_ge_0.50 | 0.2262 | 0.0292 | 0.9682 | 0.0107 | 0.3567 | 0.9808 | no |

Best aggregation selected for downstream fusion scope: `majority_vote`. Aggregation alone helped recall only modestly: `majority_vote` with `default_0.5` reached mean recall 0.495 and specificity 0.757, versus Phase C `mean_score` recall 0.476.

## 5. Fusion benchmark

Fusion concatenates the pooled HuBERT embedding with the official 256 handcrafted acoustic/prosodic features. Only `logistic_regression_platt` and `linear_svm_platt` were tested. Fusion aggregations were `mean_score` and `majority_vote`.

| feature_mode | classifier | aggregation | threshold_policy | speaker_recall_mean | speaker_specificity_mean | speaker_f1_mean | threshold_mean | target_zone_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hubert_only | logistic_regression_platt | majority_vote | default_0.5 | 0.4952 | 0.757 | 0.5491 | 0.5 | no |
| hubert_only | logistic_regression_platt | mean_score | default_0.5 | 0.4762 | 0.8673 | 0.5793 | 0.5 | no |
| hubert_plus_handcrafted | logistic_regression_platt | majority_vote | default_0.5 | 0.4762 | 0.7682 | 0.5374 | 0.5 | no |
| hubert_plus_handcrafted | logistic_regression_platt | mean_score | default_0.5 | 0.4595 | 0.8598 | 0.561 | 0.5 | no |
| hubert_plus_handcrafted | linear_svm_platt | majority_vote | default_0.5 | 0.4476 | 0.7963 | 0.524 | 0.5 | no |
| hubert_only | linear_svm_platt | majority_vote | default_0.5 | 0.4381 | 0.8019 | 0.5185 | 0.5 | no |
| hubert_plus_handcrafted | linear_svm_platt | mean_score | default_0.5 | 0.4167 | 0.871 | 0.5275 | 0.5 | no |
| hubert_only | linear_svm_platt | mean_score | default_0.5 | 0.4 | 0.8804 | 0.5158 | 0.5 | no |
| hubert_plus_handcrafted | logistic_regression_platt | majority_vote | constrained_spec_ge_0.50 | 0.2762 | 0.9308 | 0.405 | 0.982 | no |
| hubert_only | logistic_regression_platt | majority_vote | constrained_spec_ge_0.50 | 0.2714 | 0.9439 | 0.4038 | 0.9808 | no |
| hubert_plus_handcrafted | logistic_regression_platt | mean_score | constrained_spec_ge_0.50 | 0.2262 | 0.9869 | 0.3639 | 0.982 | no |
| hubert_only | logistic_regression_platt | mean_score | constrained_spec_ge_0.50 | 0.2262 | 0.985 | 0.3628 | 0.9808 | no |
| hubert_only | linear_svm_platt | majority_vote | constrained_spec_ge_0.50 | 0.1833 | 0.9757 | 0.3015 | 0.9596 | no |
| hubert_plus_handcrafted | linear_svm_platt | majority_vote | constrained_spec_ge_0.50 | 0.1595 | 0.972 | 0.267 | 0.956 | no |
| hubert_only | linear_svm_platt | mean_score | constrained_spec_ge_0.50 | 0.1405 | 0.9944 | 0.2442 | 0.9596 | no |
| hubert_plus_handcrafted | linear_svm_platt | mean_score | constrained_spec_ge_0.50 | 0.1238 | 1.0 | 0.2202 | 0.956 | no |

Fusion did not improve over the HuBERT-only control. The best fused setup was `logistic_regression_platt` + `majority_vote` + `default_0.5`, with recall 0.476 and specificity 0.768.

## 6. Ensemble benchmark

The ensemble used at most three controlled components: the best HuBERT aggregation-only model from Phase D, the best HuBERT+handcrafted fusion model from Phase D, and the best second embedding-only Phase C candidate (`facebook/wav2vec2-base` + logistic mean-score).

| ensemble_method | threshold_policy | speaker_recall_mean | speaker_specificity_mean | speaker_f1_mean | threshold_mean | target_zone_pass |
| --- | --- | --- | --- | --- | --- | --- |
| max_probability_ensemble | default_0.5 | 0.6595 | 0.6804 | 0.6381 | 0.5 | no |
| majority_vote_ensemble | default_0.5 | 0.4929 | 0.7794 | 0.5554 | 0.5 | no |
| mean_probability_ensemble | default_0.5 | 0.4619 | 0.8654 | 0.5652 | 0.5 | no |
| max_probability_ensemble | constrained_spec_ge_0.50 | 0.3548 | 0.9794 | 0.5135 | 0.9844 | no |
| majority_vote_ensemble | constrained_spec_ge_0.50 | 0.2 | 0.9944 | 0.3313 | 0.9816 | no |
| mean_probability_ensemble | constrained_spec_ge_0.50 | 0.0833 | 0.9963 | 0.1525 | 0.9804 | no |

The ensemble helped recall most: `max_probability_ensemble` with `default_0.5` reached mean recall 0.660 and specificity 0.680.

## 7. Threshold policy and leakage controls

- Outer evaluation used 5-fold `StratifiedGroupKFold` grouped by `speaker_id` for seeds 42, 43, 44, 45, and 46.
- Training and held-out speaker sets were disjoint in every fold.
- Platt calibration was fit from training speakers only, matching the Phase C in-sample training-speaker calibration style.
- `default_0.5` used a fixed threshold of 0.5.
- `constrained_spec_ge_0.50` selected a threshold on training speakers only, maximizing recall subject to specificity >= 0.50.
- Held-out fold labels were not used for threshold selection.

## 8. Comparison against Phase C best model

Phase C HuBERT baseline speaker recall mean was 0.476 with specificity 0.867. Phase D comparison:

| decision_ranking | candidate_type | candidate_name | speaker_recall_mean | speaker_recall_std | speaker_specificity_mean | speaker_specificity_std | speaker_f1_mean | threshold_mean | target_zone_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | phaseD_ensemble | max_probability_ensemble default_0.5 | 0.6595 | 0.031 | 0.6804 | 0.0223 | 0.6381 | 0.5 | no |
| 2 | phaseD_aggregation_only | HuBERT logistic majority_vote default_0.5 | 0.4952 | 0.0247 | 0.757 | 0.0356 | 0.5491 | 0.5 | no |
| 3 | phaseD_fusion | HuBERT+handcrafted logistic_regression_platt majority_vote default_0.5 | 0.4762 | 0.0367 | 0.7682 | 0.0276 | 0.5374 | 0.5 | no |
| 4 | phaseC_reference | Phase C HuBERT logistic mean_score default_0.5 | 0.4762 | 0.0253 | 0.8673 | 0.0469 | 0.5793 | 0.5 | no |

## 9. Comparison against official classical baseline

The official classical reference remains `final_cleaned_logistic_regression_platt` at threshold 0.300. Its available multi-seed speaker reference uses `max_risk_score_seed_selected_threshold` with recall 0.476 and specificity 0.475. Phase D did not alter that baseline.

## 10. Error complementarity findings

| model_a | model_b | shared_fn_count | shared_fp_count | fn_overlap_ratio | fp_overlap_ratio | speaker_disagreement_count | complementarity_judgment | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HuBERT logistic majority_vote default_0.5 | HuBERT+handcrafted logistic_regression_platt majority_vote default_0.5 | 194 | 107 | 0.8151 | 0.7279 | 84 | low | Overlap ratios are Jaccard overlap across seed-speaker error sets; aligned rows=955, disagreement_rate=0.088. |
| HuBERT logistic majority_vote default_0.5 | facebook/wav2vec2-base logistic mean_score default_0.5 | 150 | 42 | 0.495 | 0.2625 | 271 | moderate | Overlap ratios are Jaccard overlap across seed-speaker error sets; aligned rows=955, disagreement_rate=0.284. |
| HuBERT+handcrafted logistic_regression_platt majority_vote default_0.5 | facebook/wav2vec2-base logistic mean_score default_0.5 | 155 | 43 | 0.5065 | 0.281 | 261 | moderate | Overlap ratios are Jaccard overlap across seed-speaker error sets; aligned rows=955, disagreement_rate=0.273. |

## 11. Best Phase D candidate

| candidate_name | feature_mode | classifier | aggregation | threshold_policy | speaker_recall_mean | speaker_recall_std | speaker_specificity_mean | speaker_specificity_std | speaker_f1_mean | speaker_f1_std | target_zone_pass | better_than_phaseC | better_than_classical_baseline | recommendation | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_probability_ensemble default_0.5 | ensemble | ensemble | speaker_level_component_scores | default_0.5 | 0.6595 | 0.031 | 0.6804 | 0.0223 | 0.6381 | 0.0236 | no | yes | yes | hold | Improved recall over Phase C while preserving the specificity floor, but did not reach the target zone. |

## 12. Did anything reach the target zone?

Number of comparison rows reaching speaker recall >= 0.70 and specificity >= 0.50: **0**. At seed level, target-zone passes across all Phase D rows: **1**; the headline decision uses multi-seed means, so no Phase D setup is treated as having reached the target zone.

## 13. What helped recall most?

The strongest ranked setup was `max_probability_ensemble default_0.5`. Whether that is enough is captured by the target-zone and recommendation fields above.

## 14. What failed and why?

Aggregation alone recovered only a small amount of recall and traded away specificity. Fusion failed because the handcrafted features did not add enough complementary signal to recover missed positives. The ensemble was the closest setup, but its multi-seed recall still stayed below 0.70. Constrained-threshold variants preserved high specificity but reduced recall sharply.

## 15. Single most justified next step

Inspect the remaining false negatives for the held Phase D candidate before adding model complexity; the next experiment should be error-driven, not a broader sweep.
