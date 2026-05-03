# Final Limitations And Future Work

This project should be presented honestly as an academic dementia-risk screening study, not as a clinically validated diagnostic system.

## Final Limitations

- The dataset is small and imbalanced.
- The trusted cleaned Phase D subset contains `191` speakers and `348` clips, with only `84` dementia-positive speakers.
- Positive-speaker evidence is especially sparse:
  - `84` positive speakers have only `94` clips total
  - `74/84` positive speakers have one clip
  - `10/84` positive speakers have two clips
- The remaining false negatives under the max ensemble likely reflect representation or data limitations, because a missed positive under max means none of the component models produced enough positive speaker-level signal.
- False positives increased because the selected ensemble uses aggressive max aggregation.
- Mean specificity remained above the minimum acceptable floor (`0.6804` vs `0.50`), but the false-positive burden is still substantial and prevents overclaiming.
- The model has not been clinically validated and should not be treated as a diagnostic system.
- The saved Phase D artifacts do not preserve:
  - per-speaker ensemble prediction table
  - per-speaker component score table
  - clip-level component probabilities for the final ensemble decision
  - exact named remaining false negatives
  - exact named new false positives
- Because of that artifact gap, detailed per-case ensemble narratives cannot be reconstructed without rerunning Phase D.

## What These Limitations Mean

- The final held candidate is appropriate for research reporting as a recall-prioritized screening-support model.
- It is not appropriate to claim that the model diagnoses dementia, rules out dementia, or gives a clinically validated medical probability.
- The project conclusion should stay at the system level: best held recall-recovery candidate, but not a final validated clinical model.

## Future Work

Future work should focus on strengthening evidence quality rather than opening another broad model sweep:

- Collect more dementia-positive speakers.
- Collect more clips per speaker.
- Preserve per-speaker and per-clip ensemble prediction artifacts for future case-level analysis.
- Improve calibration and probability reliability.
- Consider review-zone or risk-band outputs instead of exposing binary diagnostic-style labels.
- Validate wording and usefulness with clinicians.
- Evaluate on an external dataset.

## Final Framing

The most defensible final project message is:

> The selected max-probability ensemble improves speaker-level dementia recall and remains above the minimum specificity floor, but its evidence base is still limited by dataset size, sparse positive-speaker coverage, false-positive cost, missing ensemble-level case artifacts, and lack of external clinical validation.
