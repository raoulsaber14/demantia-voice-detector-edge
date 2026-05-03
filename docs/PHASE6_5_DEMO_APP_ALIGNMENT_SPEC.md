# Phase 6.5 Demo App Alignment Spec

Historical planning document reconciled to the committed repository state.

This file originally described a stricter future-state UI that would hide raw
screening outputs. That future-state was not implemented in the committed demo
code for this submission snapshot.

## Source Policy

This document now serves two purposes:

- preserve the fact that a safer qualitative-only demo was discussed during
  planning
- make clear that the committed repository still uses a research/demo review UI
  with raw output fields visible

## Current Committed Demo Behavior

The current demo code displays raw screening outputs.

Flask demo:

- risk band
- risk score
- screening indicator label
- confidence percentage
- per-class screening probabilities
- optional marker summaries
- timing and disclaimer

Streamlit demo:

- predicted label
- confidence
- dementia probability
- threshold control and threshold used
- optional ensemble component details
- explanation and disclaimer

## Interpretation

This means the committed UI should be described as:

- a research/demo review interface
- non-diagnostic
- not a clinician-safe production presentation
- useful for transparency, debugging, and demonstration of the existing model
  wrapper

It should not be described as a final responsible end-user UX.

## Historical Planning Note

The stricter qualitative-only output policy remained planning guidance rather
than implemented repository behavior in this submission snapshot.

If a future version of the project needs a safer end-user presentation layer,
that would require a separate UI revision rather than documentation changes
alone.
