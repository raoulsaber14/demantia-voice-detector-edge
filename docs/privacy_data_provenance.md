# Privacy And Data Provenance

This repository is an academic project submission for dementia-risk
screening-support research. It is not a diagnosis system and should not be used
to make clinical decisions without qualified human review.

## Data Included In The Public Repo

- small sample audio clips are committed under `data/sample_audio/` for manual
  demo and smoke-test use
- final reports, tables, and documentation are included for submission review
- the full raw/private research audio tree is not included
- most generated intermediate artifacts are not included

Because the full research audio is not committed, this repository is a
documentation and demo snapshot rather than a complete raw-data redistribution
package.

## Provenance Boundary

The repository documents the project structure, the expected research data tree,
and the final evaluation artifacts, but it does not include a complete public
license and redistribution package for all original audio sources. This means:

- do not assume that every original research audio source can be redistributed
  from this repository
- do not assume that demographic metadata are complete enough for public
  external reuse decisions
- do not upload personally identifying labels or notes into the demo unless
  that is explicitly permitted by your own governance process

The Flask demo already warns users not to enter full names or identifying
information in session labels.

## Privacy And Leakage Controls Documented In The Repo

The repository documents several methodological controls intended to reduce
leakage and evaluation contamination:

- speaker-grouped evaluation by `speaker_id`
- documented separation of training and held-out speakers
- calibration carved from training speakers only where described in the
  training code and reports
- exclusion of metadata and governance columns from the official numeric model
  feature set

These controls matter because same-speaker overlap between train and evaluation
splits, or direct metadata leakage into model features, can overstate
performance.

## Responsible Use Boundary

- screening-support only
- not a diagnosis
- not clinically validated for independent medical use

The committed demos and reports should be interpreted within those boundaries.

## What Is Still Missing

This repository does not claim:

- a full public license audit for all audio sources
- a complete de-identification or regulatory compliance package
- a full raw-data reproducibility bundle for outside redistribution

Those items would need separate documentation before any broader public or
production use.
