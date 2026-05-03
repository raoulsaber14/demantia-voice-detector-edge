# Fairness And Bias Statement

This repository does not claim a complete subgroup fairness evaluation. The
available public submission snapshot is enough to document the main modeling and
leakage-control choices, but it is not enough to support strong fairness claims
across all relevant demographic or recording-condition subgroups.

## Current Boundary

- The project is a screening-support research system, not a diagnosis system.
- The repository includes final reports and leakage-control documentation, but
  it does not present a complete public subgroup fairness benchmark.
- Some metadata fields exist in the documented manifests, but metadata coverage
  and subgroup sample sizes are not sufficient here to claim robust fairness
  performance across all populations.

## Potential Bias Factors

Future subgroup review should explicitly consider at least:

- age
- gender
- accent
- primary language and code-switching behavior
- recording device and microphone quality
- background noise and room acoustics
- dataset imbalance across dementia/non-dementia classes
- imbalance in the number of clips per speaker

These factors can shift acoustic feature distributions, degrade audio quality,
or change the way a model generalizes across populations and collection
conditions.

## What Is Documented Now

- The official baseline documentation is explicit that the final baseline is an
  academic reference configuration rather than a clinically validated model.
- The evaluation protocol is speaker-grouped to reduce same-speaker leakage.
- The official feature set excludes metadata and governance columns from the
  numeric model feature table.

Those controls improve methodological rigor, but they are not a substitute for a
proper subgroup fairness audit.

## Future Work Needed

The following work would be required for a stronger fairness assessment:

- establish complete and permissioned subgroup metadata where ethically and
  legally appropriate
- report subgroup sample counts before reporting subgroup metrics
- evaluate recall, specificity, precision, calibration, and error rates by
  subgroup
- test robustness across recording devices, recording environments, and
  background-noise conditions
- validate performance on external data rather than relying on one project
  dataset snapshot

Until that work exists, this repository should be interpreted as a research and
engineering submission with documented fairness limitations, not as evidence of
fair deployment readiness.
