# Phase 3 Feature Acceptance Policy

Scope: candidate hand-crafted acoustic features for the classical ML baseline.

## Acceptance Rules

- Drop a feature when missingness is greater than `0.25` of rows.
- Drop a feature when all values are missing, infinite, non-numeric, or effectively one unique non-null value.
- Drop a feature when it is numerically unstable enough to produce non-finite values after feature cleaning.
- Retain a feature with caution when it has low but non-zero missingness and passes the drop threshold.
- Retain a feature with caution when its IQR outlier rate is greater than `0.05` unless the values are non-finite or clearly extraction failures.
- Retain a feature with caution when the missingness gap between classes is at least `0.10`; if the gap is caused by extraction failure concentrated in one class, drop or investigate before final reporting.
- Proxy features are allowed only when they are deterministic, documented, and named as proxies.
- No global winsorization or clipping is applied in Phase 3. Outliers are documented and retained unless they violate missingness/finite-value/variance rules.
- Remaining missing values are not imputed globally in the feature table. They are imputed inside sklearn model pipelines fitted on training data only.

## Final Status Labels

- `kept`: passes acceptance checks and is a direct acoustic descriptor or QC feature.
- `kept_with_caution`: passes hard checks but needs interpretation caution because it is a proxy, has low missingness, or has a notable outlier rate.
- `exploratory_only`: not used in the official baseline feature table, but retained for future analysis notes if a feature is not defensible for the frozen baseline.
- `dropped`: excluded from the final baseline feature set by the hard policy.

## Applied Manifest

The machine-readable application of this policy is saved to `reports/phase3_feature_manifest.csv`. That file records every candidate feature, its family, status, missingness, variance, outlier rate, proxy/direct label, and decision reason.

## Redundancy Pruning

- After quality acceptance, train-split-only Spearman correlations are used to identify redundant feature pairs with absolute correlation at or above `0.98`.
- One feature from each redundant pair is removed from the official model-input table, preferring direct acoustic descriptors over proxies, lower missingness, and deterministic tie-breaks.
- Held-out validation/test feature distributions and labels are not used to choose redundant features.
- Redundancy decisions are saved to `reports/phase3_redundancy_pruning_manifest.csv`; the official pruned model input is `data/features/features_phase3_governed_pruned.csv`.

## Jitter/Shimmer Decision

The current jitter and shimmer columns are kept with caution as frame-level proxies only. They are not clinical cycle-level jitter or shimmer because this repo does not use glottal-cycle pitch-marking tooling such as Praat/parselmouth. They may be useful as classical voice-quality descriptors, but they must be reported as proxy features.
