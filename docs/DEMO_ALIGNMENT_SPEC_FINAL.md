# Demo Alignment Spec Final

Freeze date: 2026-04-22.

This document records the current committed demo behavior in the submission
repository. It does not request UI changes.

## Source Artifacts

- `app/flask/templates/index.html`
- `app/flask/web_server.py`
- `app/streamlit/demo_app.py`
- `docs/OFFICIAL_BASELINE_CONFIGURATION.md`
- `docs/BASELINE_MODEL_CARD.md`

## Current Committed Demo Contract

The repository includes two demo surfaces:

- a Flask edge-demo UI for microphone/upload sessions
- a Streamlit research/demo UI for manual local inference

These demos are honest about the model being non-diagnostic, but they are not
minimal clinician-facing interfaces. They expose raw screening outputs for
research/demo review.

## Flask Demo Output

The committed Flask result screen shows:

- qualitative risk band
- risk score out of 100
- screening indicator label
- confidence percentage
- per-class screening probabilities
- optional marker rows when available
- preprocessing and inference timing
- non-diagnostic disclaimer

The downloadable Flask session report follows the same pattern and includes the
same raw screening output fields.

## Streamlit Demo Output

The committed Streamlit demo shows:

- model type selector
- model path field
- decision threshold control
- predicted label
- confidence
- dementia probability
- threshold used
- optional component details for the ensemble runtime
- explanation text
- non-diagnostic disclaimer

## Interpretation Boundary

The current demos should be interpreted as research/demo review interfaces.
They are not the same thing as a clinician-safe or patient-facing product UX.

Important boundaries:

- screening-support only
- not a diagnosis
- not clinically validated
- raw scores and confidence-style outputs must not be treated as medical
  probabilities
- the presence of these fields in the demo does not change the frozen model or
  make the interface deployment-ready

## Why This Document Matters

Earlier planning docs described a stricter future-state demo that would hide
raw scores, confidence values, and diagnostic-style labels. That stricter UI
was not the behavior committed in this repository snapshot.

For submission review, the source of truth is the current code above, not the
unimplemented future-state wording from earlier planning notes.
