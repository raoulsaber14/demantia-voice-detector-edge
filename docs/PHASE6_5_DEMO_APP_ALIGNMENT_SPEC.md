# Phase 6.5 Demo App Alignment Spec

Planning and documentation only. This spec does not modify `app/streamlit/demo_app.py` or `src/inference.py`.

## Source Policy

This spec maps the demo to existing Phase 5 wording:

- `reports/phase5_final_summary.md`
- `reports/phase5_score_display_policy.md`
- `reports/phase5_output_layer_policy.md`
- `reports/phase5_score_band_definition.md`
- `reports/phase5_mock_outputs.md`

## Current Demo Risk

The current demo code displays:

- `Predicted label: ...`
- `Confidence: ...`

This conflicts with Phase 5 because end users should not see raw scores, percentage-style confidence, or diagnostic-style labels.

## UI Constraints

The user-facing demo must show:

- Screening-signal band.
- Short qualitative explanation.
- Non-diagnostic disclaimer.
- Suggested next step.

The user-facing demo must not show:

- Raw model score.
- Percentage.
- Confidence score.
- Predicted label such as `dementia` or `non_dementia`.
- Text such as "dementia detected".
- Text such as "healthy" or "no dementia".
- Exact threshold logic.
- Model coefficients or feature contributions.
- Clinical probability.
- Diagnostic or rule-out language.

## Band Mapping

| Internal band ID | Score range for internal thresholding | User-facing label |
|---|---:|---|
| low | 0.000 to less than 0.300 | Lower screening signal |
| moderate | 0.300 to less than 0.400 | Elevated screening signal |
| high | 0.400 to 1.000 | Stronger screening signal |

The score range is implementation/internal review context. It should not be shown to end users.

## Exact User-Facing Text

### Lower Screening Signal

Band label:

> Lower screening signal

Short explanation:

> This recording did not show a strong speech-pattern screening signal at the model's selected threshold.

Uncertainty statement:

> A lower screening signal does not rule out cognitive or memory concerns. False negatives are possible.

Next step:

> If there are concerns about memory, language, behavior, or daily functioning, consider discussing them with a qualified health professional despite the lower screening signal.

### Elevated Screening Signal

Band label:

> Elevated screening signal

Short explanation:

> This recording crossed the model's screening threshold and shows a possible elevated screening pattern. This does not mean dementia is present.

Uncertainty statement:

> This result is suggestive, not confirmatory. False positives and false negatives are possible, and the score is not an externally validated medical probability.

Next step:

> Consider professional review or formal cognitive screening if memory, language, or daily-function concerns are present.

### Stronger Screening Signal

Band label:

> Stronger screening signal

Short explanation:

> This recording showed a stronger speech-pattern screening signal in the frozen Phase 4 model. The result remains non-diagnostic.

Uncertainty statement:

> A stronger screening signal may warrant follow-up, but it cannot confirm dementia. The current model has limited validation, weak specificity, and possible false positives.

Next step:

> Professional clinical review is recommended if concerns are present. The result should be interpreted with recording quality, medical context, cognitive assessment, and clinician judgment.

## Exact Disclaimer Text

Short disclaimer:

> This result is for screening support only. It is not a diagnosis and does not confirm or rule out dementia. It should not be used without professional clinical judgment and appropriate assessment.

Expanded disclaimer:

> This speech-based result is a non-diagnostic screening signal from a research prototype. It is not a medical probability, does not confirm or rule out dementia, and should not replace clinical history, cognitive testing, neurological assessment, or clinician judgment. False positives and false negatives are possible.

Minimum demo requirement:

- Show at least the short disclaimer on every result screen.
- Use the expanded disclaimer in any detail or review screen.

## Optional Marker Summary Text

Only show marker summaries if the app can do so without raw feature values or causal claims.

Approved marker wording:

- "The recording showed elevated pause or low-speech burden."
- "The recording showed a lower acoustic speech-activity rate."
- "The recording showed speech-flow patterns that may warrant review."
- "The recording showed pitch or prosody variability captured by the acoustic model."

Required marker caveat:

> Marker summaries are model-aligned acoustic indicators. They are not clinical biomarkers and should not be interpreted as direct evidence of dementia.

## Prohibited UI Elements / Content

- Raw probability.
- Confidence meter.
- Percent risk.
- Gauge implying severity.
- Binary diagnosis label.
- "Dementia detected."
- "No dementia detected."
- "Healthy."
- "Patient has dementia."
- "Patient does not have dementia."
- Any text implying the model confirms or rules out dementia.
- Any text implying the score is externally validated as a clinical probability.

## Logging / Telemetry Constraints

Not specified in the current assessment. This spec does not add telemetry requirements.

If technical logging is later added, user-facing displays still must not expose raw scores, confidence-style outputs, diagnostic labels, or feature-level causal explanations.

## Phase 5 To UI Mapping

| Phase 5 requirement | UI element |
|---|---|
| Display band | Result header: Lower / Elevated / Stronger screening signal |
| Brief qualitative explanation | One short paragraph below the band |
| Uncertainty statement | Caution text below the explanation |
| Non-diagnostic disclaimer | Persistent disclaimer block on result screen |
| Suggested next step | Next-step paragraph below disclaimer |
| Hide raw score | No visible score field |
| Hide percentage confidence | No confidence field or percent display |
| Avoid diagnostic claims | No predicted `dementia` / `non_dementia` label shown to end user |
