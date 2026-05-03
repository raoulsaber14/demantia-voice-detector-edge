# Demo Alignment Spec Final

Freeze date: 2026-04-22.

This is documentation only. It does not edit `app/streamlit/demo_app.py`.

## Source Artifacts

- Frozen configuration: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`.
- Prior demo alignment spec: `docs/PHASE6_5_DEMO_APP_ALIGNMENT_SPEC.md`.
- Error analysis: `docs/FINAL_ERROR_ANALYSIS.md`.
- Threshold source: `reports/phase4_official_winner.md`; `reports/phase6_5_week2_phase4_rerun_comparison.md`.

## UI Constraints

The demo must not show raw scores, percentages, confidence displays, diagnostic labels, or exact threshold details to end users. Source: `docs/PHASE6_5_DEMO_APP_ALIGNMENT_SPEC.md`.

Allowed user-facing result elements:

- Qualitative screening band.
- Short cautious explanation.
- Non-diagnostic disclaimer.
- Conservative next-step language.

## Internal Band Definition

Frozen internal threshold: `0.300`. Source: `docs/OFFICIAL_BASELINE_CONFIGURATION.md`.

Internal display margin: `0.050`. The margin is a demo-safety display rule based on the final error analysis showing heavy score concentration within 0.05 of the threshold. Source: `docs/FINAL_ERROR_ANALYSIS.md`.

Internal routing:

| Internal score condition | User-facing band |
|---|---|
| score < 0.250 | Low likelihood |
| 0.250 <= score < 0.350 | Borderline |
| score >= 0.350 | Elevated likelihood |

Any score at or above 0.300 is internally threshold-positive. If it is also within the borderline margin, the display should still use the `Borderline` band to avoid overstating a near-threshold result.

## Exact Wording

### Low likelihood

Band label:

> Low likelihood

Text:

> This screening analysis did not show an elevated speech-pattern signal in this recording. This does not rule out cognitive or memory concerns.

Next step:

> If concerns are present, consider professional evaluation regardless of this screening result.

### Borderline

Band label:

> Borderline

Text:

> This result is close to the model's decision boundary and should be treated as uncertain. It is not a diagnosis.

Next step:

> Consider repeat recording review or professional screening if memory, language, behavior, or daily-function concerns are present.

### Elevated likelihood

Band label:

> Elevated likelihood

Text:

> This screening analysis suggests speech-pattern features that may warrant further evaluation. Not a diagnosis.

Next step:

> Consider discussing concerns with a qualified health professional or using formal cognitive screening. Do not use this result alone for decisions.

## Disclaimer Text

Short disclaimer:

> This result is for screening support only. It is not diagnostic, is not clinically validated, and does not confirm or rule out dementia.

Expanded disclaimer:

> This speech-based output is a non-diagnostic screening signal from a research prototype. It is not a medical probability, confidence score, diagnosis, or all-clear. False positives and false negatives are possible. The result should be interpreted only with appropriate professional judgment and additional assessment.

The short disclaimer must be shown on every result screen. The expanded disclaimer should be shown in any details or review view.

## Prohibited Elements

- Raw probability.
- Percentage risk.
- Confidence score.
- Gauge implying severity.
- Diagnostic labels such as `dementia` or `non_dementia`.
- Buttons or text implying diagnosis, such as "Dementia detected" or "No dementia detected".
- Any statement that the model confirms or rules out dementia.
- Any claim that the threshold is clinically validated.
- Feature-level causal explanations or biomarker language.

## Implementation Boundary

This spec only aligns the demo language and display contract with the frozen baseline. It does not request deployment, hardware integration, model changes, threshold changes, or demo code edits in this freeze run.
