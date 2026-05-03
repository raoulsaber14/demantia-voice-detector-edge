# Final Feature Set

Freeze date: 2026-04-22.

This is a documentary feature freeze. No feature table, feature extraction code, or model configuration was changed in this run.

## Source Artifacts

- Canonical feature table: `data/features/features_phase3_governed_pruned.csv`.
- Baseline feature selector: `scripts/phase4_baseline_experiments.py`.
- Feature-quality audit: `reports/phaseA_classical_baseline_diagnostics.md`.
- B2c feature-subset definitions: `reports/tables/phase_b2c_feature_subsets.csv`.
- B2c feature curation report: `reports/phase_b2c_feature_curation_report.md`.

## Official Feature Source

The frozen baseline uses `data/features/features_phase3_governed_pruned.csv`. The Phase 4 runner defines this as `FEATURE_CSV` and selects official numeric model features after excluding metadata and governance columns. Source: `scripts/phase4_baseline_experiments.py`.

The official frozen anchor uses 256 numeric features. Source: `scripts/phase4_baseline_experiments.py`; `reports/phaseA_classical_baseline_diagnostics.md`.

## Feature Families

Phase A reports this family breakdown for the 256 official features:

| Feature family | Feature count | Source |
|---|---:|---|
| MFCC | 156 | `reports/phaseA_classical_baseline_diagnostics.md` |
| Spectral | 35 | `reports/phaseA_classical_baseline_diagnostics.md` |
| Duration/timing | 23 | `reports/phaseA_classical_baseline_diagnostics.md` |
| Voice-quality/pitch | 17 | `reports/phaseA_classical_baseline_diagnostics.md` |
| Energy/RMS | 11 | `reports/phaseA_classical_baseline_diagnostics.md` |
| Pause/speech-timing | 9 | `reports/phaseA_classical_baseline_diagnostics.md` |
| Zero-crossing-rate | 4 | `reports/phaseA_classical_baseline_diagnostics.md` |
| Other | 1 | `reports/phaseA_classical_baseline_diagnostics.md` |

## Broken Features Identified For Future Cleanup

B2c defined a `cleaned_all_features` subset with 252 features after mechanically removing 4 features from the 256-feature anchor. Source: `reports/tables/phase_b2c_feature_subsets.csv`; `reports/phase_b2c_feature_curation_report.md`.

Features identified for future documentary cleanup:

| Feature | Reason | Source |
|---|---|---|
| `spectral_flatness_min` | Variance below `1e-6` in B2c cleanup rule | `reports/phase_b2c_feature_curation_report.md` |
| `low_energy_ratio` | Variance below `1e-6` in B2c cleanup rule | `reports/phase_b2c_feature_curation_report.md` |
| `f0_analysis_duration_sec` | More than 90 percent identical values in B2c cleanup rule | `reports/phase_b2c_feature_curation_report.md` |
| `pitch_period_std_proxy` | Variance below `1e-6` in B2c cleanup rule | `reports/phase_b2c_feature_curation_report.md` |

Important discrepancy note: Phase A reported 0 near-zero features and 0 duplicate-like features in its audit, while B2c later applied a stricter mechanical cleaned-subset rule and removed the 4 fields above. Source: `reports/phaseA_classical_baseline_diagnostics.md`; `reports/phase_b2c_feature_curation_report.md`.

## Redundant Features

No exact duplicate-like baseline features were reported in Phase A. Source: `reports/phaseA_classical_baseline_diagnostics.md`.

Phase A did report 13 high-correlation features, mainly percentile summaries in timing, RMS, spectral, and F0 families. These are redundancy risks but not exact duplicates. Source: `reports/phaseA_classical_baseline_diagnostics.md`.

## Proposed Future Cleanup Count

If the 4 B2c mechanical cleanup features are removed in a future implemented cleanup, the proposed feature count would be 252. Source: `reports/tables/phase_b2c_feature_subsets.csv`.

This final freeze does not implement that removal. The official frozen anchor remains the already evaluated 256-feature `logistic_regression_platt` configuration at threshold `0.300`. Source: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`.

## Interpretation Boundary

Feature-family and cleanup notes are descriptive. They do not claim causality, biomarker status, or clinical interpretability. The features are handcrafted acoustic/prosodic descriptors used for a research baseline only.
