# Submission Overview

## Project Summary

This repository is a speech-based dementia screening-support project. It explores whether acoustic speech patterns can support early screening or referral-style flagging while remaining explicitly non-diagnostic.

The repository contains:

- a root-level edge prototype for local audio preprocessing and inference wrapping
- a nested research subproject in `dementia-voice-detector/` with the training pipeline, reports, and final documentation

## Final Project Position

- Official final academic baseline: `final_cleaned_logistic_regression_platt`
- Best exploratory held candidate: `max_probability_ensemble default_0.5`
- Edge prototype status: engineering prototype only, not a validated deployment artifact

## Research Outcome

The official final academic baseline is the documented classical logistic-regression system frozen in the final baseline documents. It is defensible as a documented academic baseline, but it is not clinically validated and is not deployment-ready.

The strongest exploratory held candidate is the Phase D ensemble. It improved recall relative to the baseline, but it remained a held exploratory candidate rather than a promoted final deployment model.

## Repository Boundaries

Included in this submission:

- edge prototype code
- research code
- curated final reports
- methodology, limitations, and responsible-use documentation

Not fully included in this submission:

- full raw/private research data
- generated feature tables and most intermediate data products
- trained research model folders
- external ONNX deployment artifacts referenced by the root prototype

## Responsible Use

This project is screening-support only. It must not be described as diagnosing dementia, ruling out dementia, or producing a clinically validated medical probability.
