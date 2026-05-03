# Phase 6.5 Pre-Deployment Strengthening Plan

Planning and documentation only. This file does not change training code, model selection logic, thresholds, or model artifacts.

## Repository Context

- Workspace root: `/Users/celinesadaka/Desktop/ml project`
- Project folder: `/Users/celinesadaka/Desktop/ml project/dementia_voice_project`
- Project-relative paths below assume the project folder as root.

## Source Of Record

This plan relies on existing repository artifacts only:

- `reports/phase4_threshold_analysis.md`
- `reports/tables/phase4_threshold_analysis.csv`
- `reports/phase4_official_winner.md`
- `reports/phase4_model_selection.md`
- `reports/phase4_executive_summary.md`
- `reports/phase5_final_summary.md`
- `reports/phase5_score_display_policy.md`
- `reports/phase5_output_layer_policy.md`
- `dementia_voice_project/reports/phase5_score_band_definition.md`
- `reports/phase5_mock_outputs.md`
- `reports/phase5_clinician_review_pack.md`
- `reports/phase6_final_comparison.md`
- `reports/phase6_focused/phase6_focused_summary.md`
- `final_results/reliability_frozen_fusion/decision_summary.json`
- `final_results/reliability_frozen_fusion/usability_criteria.csv`
- `app/demo_app.py`
- `src/inference.py`
- `scripts/phase4_baseline_experiments.py`

## Current Defensible Anchor

- Final model anchor: Phase 4 `logistic_regression_platt`
- Official threshold: `0.300`
- Framing: recall-prioritized screening baseline only
- Current Phase 4 selected-threshold metrics: recall `0.862`, precision `0.289`, specificity `0.217`, balanced accuracy `0.539`, PR-AUC `0.332`, Brier `0.198`, false negatives `13`
- wav2vec status: explored but not accepted because the initial comparison was inconclusive and specificity collapsed
- reliability fusion status: explored but not accepted; latest final decision says `Not usable`

## Non-Goals For This Run

- Do not train new models.
- Do not change thresholds.
- Do not change Phase 4 model selection logic.
- Do not modify app code.
- Do not claim deployment readiness.
- Do not present raw probabilities, confidence percentages, or diagnostic labels as user-facing output.

## Duration Choice

Use a 6-week Phase 6.5 pass. The work has strict dependencies: data trust must be addressed before trusted-subset evaluation; stability gates must be formalized before final model justification; error/confound review must happen before specificity-recall tradeoff decisions; demo alignment and clinical review should happen only after the model framing is locked.

## Roadmap Overview

| Week | Focus | Must depend on | Main deliverables |
|---:|---|---|---|
| 1 | Resolve dataset trust blockers | Existing Phase 1/2/3 governance artifacts | Trusted subset definition, full governed subset confirmation, manual review logs |
| 2 | Rerun baseline evaluation plan on trusted vs full governed | Week 1 dataset version definitions | Trusted vs full comparison tables, threshold variability summaries |
| 3 | Lock model anchor and formalize acceptance gates | Week 2 evaluation outputs | Final model justification draft, acceptance gate report, rejected-approaches summary |
| 4 | Detailed FP/FN and confound analysis | Week 2 predictions and Week 3 model anchor | Case review tables, error-pattern summary, confound experiment report |
| 5 | Specificity-recall tradeoff review and demo alignment | Weeks 3-4 analysis | Threshold tradeoff report, demo alignment checklist/spec |
| 6 | Clinical framing review package and final Phase 6.5 closeout | Weeks 1-5 outputs | Dr. Sidani one-pager, Tier 1/2/3 report pack, final go/no-go note |

## Week 1: Dataset Trust And Review Set Definition

### Tasks

- Review the `37` uncertain metadata/audio rows listed by the existing match/governance artifacts.
- Review flagged cases from the dataset audit: duplicates, low-volume outlier, path anomalies, suspicious speaker identity, format/channel/sample-rate outliers.
- Mark each reviewed case as trusted, excluded, or retained with caution.
- Define two dataset versions:
  - Trusted subset: doubtful cases excluded.
  - Full governed subset: existing Phase 3/4 governed records retained with documented caution flags.
- Audit speaker identity quality for key analysis subsets, especially folder-inferred speaker IDs.

### Expected Outputs

- `reports/phase6_5_dataset_trust_review.md` - narrative summary of manual review decisions and unresolved risks.
- `reports/tables/phase6_5_uncertain_metadata_audio_review.csv` - row-level decisions for the 37 uncertain metadata/audio cases.
- `reports/tables/phase6_5_flagged_audio_case_review.csv` - row-level decisions for duplicates, low-volume outlier, path anomalies, and format outliers.
- `reports/tables/phase6_5_dataset_versions.csv` - trusted subset vs full governed subset definition and counts.
- `reports/tables/phase6_5_speaker_identity_audit.csv` - speaker identity quality notes for key analysis subsets.

### Outputs Produced By Week 1

- [ ] Row-level review table for all 37 uncertain metadata/audio rows.
- [ ] Row-level review table for flagged audio cases.
- [ ] Trusted subset definition table.
- [ ] Full governed subset confirmation table.
- [ ] Speaker identity audit table.
- [ ] Week 1 narrative summary.

### Definition Of Done

- Every uncertain or flagged case has an explicit decision.
- Doubtful cases are either excluded from the trusted subset or documented with a caution reason.
- Trusted and full governed subsets are mechanically reproducible from a table.
- No speaker identity issue remains undocumented for key analysis subsets.

## Week 2: Trusted vs Full Evaluation And Threshold Stability

### Tasks

- Rerun the Phase 4 evaluation protocol on both dataset versions when execution is approved:
  - trusted subset
  - full governed subset
- Keep grouped CV only.
- Require multi-seed evaluation for stability reporting.
- Analyze threshold variability fold-by-fold and seed-by-seed.
- Compare the current Phase 4 anchor with the trusted-only result and the full governed result.

### Expected Outputs

- `reports/phase6_5_trusted_vs_full_evaluation.md` - comparison report for trusted vs full governed evaluation.
- `reports/tables/phase6_5_trusted_vs_full_metrics.csv` - metrics table for both dataset versions.
- `reports/tables/phase6_5_threshold_variability_by_fold_seed.csv` - fold/seed threshold variability.
- `reports/tables/phase6_5_model_stability_summary.csv` - multi-seed stability summary.
- `reports/figures/phase6_5_threshold_variability.png` - threshold variability plot, if figures are generated.

### Outputs Produced By Week 2

- [ ] Trusted subset evaluation table.
- [ ] Full governed evaluation table.
- [ ] Fold-by-fold threshold variability table.
- [ ] Seed-by-seed threshold variability table.
- [ ] Stability summary table.
- [ ] Trusted vs full narrative report.

### Definition Of Done

- Both dataset versions use the same grouped CV framing.
- Multi-seed results are reported.
- Threshold variability is visible by fold and seed.
- Any result that changes materially between trusted and full governed subsets is documented as a dataset-trust risk.

## Week 3: Final Model Anchor And Acceptance Gates

### Tasks

- Keep the Phase 4 `logistic_regression_platt` model at threshold `0.300` as the final model anchor unless Week 2 evidence clearly invalidates it.
- Document the final anchor as a recall-prioritized screening baseline only.
- Document wav2vec and reliability fusion as explored but not accepted.
- Formalize model acceptance gates for Tier 1 to Tier 2, Tier 2 to Tier 3, and Tier 3 to deployment consideration.
- Require grouped CV, multi-seed reporting, threshold stability reporting, and demo safety.

### Expected Outputs

- `reports/phase6_5_final_model_anchor.md` - final anchor justification and caveats.
- `reports/phase6_5_rejected_approaches_summary.md` - wav2vec and reliability fusion non-acceptance rationale.
- `reports/phase6_5_acceptance_gates.md` - objective gate criteria.
- `reports/tables/phase6_5_acceptance_gate_status.csv` - pass/fail status for gates.

### Outputs Produced By Week 3

- [ ] Final model anchor report.
- [ ] Rejected approaches summary.
- [ ] Acceptance gates report.
- [ ] Gate status table.
- [ ] Threshold stability requirements recorded in final model documentation.

### Definition Of Done

- The final model anchor is stated without diagnostic claims.
- Rejected deep-learning and reliability approaches are documented without overclaiming.
- Gate criteria are concrete enough that a reviewer can mark pass/fail from tables.
- Demo safety requirements are included in the gate criteria.

## Week 4: False Positive/Negative Review And Confound Analysis

### Tasks

- Perform detailed false-positive and false-negative case review using the reusable template.
- Listen to false-positive and false-negative clips where allowed by local review workflow.
- Record audio quality notes, source type, duration, pauses, noise, accents, speaker style, and suspected confounds.
- Compare classical vs wav2vec failure overlap where existing artifacts support it.
- Run confound-focused experiments when execution is approved:
  - duration-matched evaluation
  - loudness-matched evaluation
  - source-quality-stratified evaluation
  - trusted-only vs full governed comparison
  - speaker/source style checks
  - optional label permutation sanity check
  - age/gender confounds if possible from existing metadata

### Expected Outputs

- `reports/phase6_5_error_analysis.md` - FP/FN narrative and summary patterns.
- `reports/tables/phase6_5_false_positive_case_review.csv` - reviewed false-positive cases.
- `reports/tables/phase6_5_false_negative_case_review.csv` - reviewed false-negative cases.
- `reports/tables/phase6_5_error_pattern_summary.csv` - rolled-up error patterns.
- `reports/phase6_5_confound_experiments.md` - confound experiment report.
- `reports/tables/phase6_5_confound_experiments.csv` - confound experiment table.

### Outputs Produced By Week 4

- [ ] False-positive review table.
- [ ] False-negative review table.
- [ ] Error-pattern summary table.
- [ ] Confound experiment table.
- [ ] Confound experiment narrative report.
- [ ] Classical vs wav2vec failure overlap note, where existing outputs support it.

### Definition Of Done

- FP/FN cases have structured review notes.
- Error patterns are summarized without claiming causality.
- Confound analyses report protocol, dataset slice, threshold policy, and pass/fail criteria.
- Limitations are documented when metadata is missing or incomplete.

## Week 5: Specificity-Recall Tradeoff And Demo Alignment

### Tasks

- Review specificity vs recall tradeoff using a finer threshold sweep when execution is approved.
- Compare top Phase 4 candidates already present in repository artifacts.
- Check whether trusted-only evaluation changes specificity/recall behavior.
- Consider small ensemble/voting only if it is interpretable and stable.
- Reject candidates that do not improve specificity without collapsing recall and stability across seeds.
- Align demo behavior with Phase 5:
  - no raw scores
  - no percentages
  - no confidence-style display
  - no predicted diagnostic label
  - show only band, short explanation, disclaimer, and next step

### Expected Outputs

- `reports/phase6_5_specificity_recall_tradeoff.md` - threshold/candidate tradeoff report.
- `reports/tables/phase6_5_specificity_recall_tradeoff.csv` - candidate and threshold comparison table.
- `reports/tables/phase6_5_candidate_rejection_log.csv` - rejected candidates and reasons.
- `reports/phase6_5_demo_alignment_review.md` - demo compliance review against Phase 5.
- `reports/tables/phase6_5_demo_alignment_checklist.csv` - app wording and UI safety checklist.

### Outputs Produced By Week 5

- [ ] Specificity-recall tradeoff table.
- [ ] Candidate rejection log.
- [ ] Demo alignment review.
- [ ] Demo alignment checklist.
- [ ] Explicit record that raw scores, percentages, confidence display, and predicted diagnostic labels are not user-facing.

### Definition Of Done

- No candidate is accepted unless it preserves stability and does not collapse recall.
- The Phase 4 anchor remains the default if tradeoff evidence is not clearly stronger.
- Demo wording is mapped directly to Phase 5 policy.
- Demo risks are documented if implementation still differs from policy.

## Week 6: Clinical Review Package And Closeout

### Tasks

- Prepare a one-page clinical framing review for Dr. Sidani.
- Include intended use, non-intended uses, output bands, disclaimer, next steps, misuse risks, mitigations, and review questions.
- Consolidate Tier 1, Tier 2, and Tier 3 report outlines and status.
- Write final Phase 6.5 go/no-go statement.
- Keep external validation, linguistic features, explainability, and hardware deployment as post-blocker items.

### Expected Outputs

- `reports/phase6_5_clinical_review_pack.md` - clinician review packet.
- `reports/phase6_5_tier1_closeout.md` - Tier 1 closeout report.
- `reports/phase6_5_tier2_readiness.md` - Tier 2 readiness report.
- `reports/phase6_5_tier3_readiness.md` - Tier 3 readiness report.
- `reports/phase6_5_final_go_no_go.md` - final Phase 6.5 go/no-go statement.

### Outputs Produced By Week 6

- [ ] Dr. Sidani review one-pager.
- [ ] Tier 1 closeout report.
- [ ] Tier 2 readiness report.
- [ ] Tier 3 readiness report.
- [ ] Final Phase 6.5 go/no-go statement.
- [ ] Explicit statement of what is evaluated, what is verified, and what is not clinically validated.

### Definition Of Done

- Clinical wording is ready for review.
- Deployment is not considered unless all gates pass.
- The final closeout clearly states whether the project remains a research prototype or has evidence for further review.

## Tier Report Outlines

### Tier 1 Report Outline: Must Fix Before Clinical Demo Or Serious Presentation

1. Executive Summary
   - State whether Tier 1 passed.
   - State final model anchor and screening-only framing.
2. Dataset Trust Review
   - 37 uncertain metadata/audio rows.
   - Duplicate, low-volume, path anomaly, and speaker identity findings.
3. Dataset Versions
   - Trusted subset definition.
   - Full governed subset definition.
4. Trusted vs Full Evaluation
   - Metrics, threshold behavior, and dataset sensitivity.
5. Threshold Stability
   - Grouped CV only.
   - Multi-seed fold/seed variability.
6. Final Anchor Decision
   - Phase 4 `logistic_regression_platt`, threshold `0.300`.
   - Caveats and non-diagnostic framing.
7. Demo Safety Status
   - Band-only display.
   - No raw score or confidence-style display.
8. Tier 1 Gate Status
   - Pass/fail criteria and blockers.

### Tier 2 Report Outline: Strengthen Before Clinical Review Or Thesis Defense

1. Executive Summary
   - Whether Tier 2 is ready for clinical review or thesis defense.
2. Specificity-Recall Tradeoff
   - Finer threshold sweep.
   - Top Phase 4 candidate comparisons.
   - Trusted-only effect.
3. Error Analysis
   - FP/FN listening notes.
   - Error patterns by quality, pauses, duration, source type, accent, and speaker style.
4. Confound Experiments
   - Duration-matched.
   - Loudness-matched.
   - Source-quality-stratified.
   - Trusted-only vs full.
   - Speaker/source style checks.
   - Optional label permutation sanity check.
   - Age/gender confounds if possible.
5. Recording Quality Metadata
   - SNR/noise level, microphone/source type, environment flags, source platform/category.
6. Speaker Identity Quality
   - Folder-inferred speaker ID limitations and tightened subsets.
7. Clinical Framing Review
   - Dr. Sidani questions and response summary.
8. Tier 2 Gate Status
   - Pass/fail criteria and blockers.

### Tier 3 Report Outline: Academically Stronger After Blockers

1. Executive Summary
   - What remains before deployment consideration.
2. External Validation Plan And Results
   - External dataset status.
   - Report no clinical validation unless actually completed.
3. Linguistic/Cognitive-Linguistic Features
   - Lexical diversity, pronoun-to-noun ratio, syntactic complexity, filled vs silent pauses, articulation ratio, speech fluency markers.
4. Explainability
   - SHAP or careful feature importance.
   - Avoid causal or diagnostic claims.
5. Evaluation Framing
   - What is verified.
   - What is evaluated.
   - What is not clinically validated.
6. Tier 3 Gate Status
   - Pass/fail criteria and remaining blockers.

## Acceptance Criteria Gates

### Tier 1 To Tier 2

Required:

- All 37 uncertain metadata/audio rows have explicit review decisions.
- Duplicate, low-volume, path anomaly, and suspicious speaker identity cases have explicit decisions.
- Trusted subset and full governed subset are defined in machine-readable tables.
- Phase 4 evaluation is rerun on both trusted and full governed subsets when execution is approved.
- Grouped CV is mandatory; random clip splitting is not permitted.
- Multi-seed evaluation is reported.
- Fold/seed threshold variability is reported.
- Final model anchor remains Phase 4 `logistic_regression_platt` at threshold `0.300` unless repo evidence clearly contradicts it.
- wav2vec and reliability fusion are documented as explored but not accepted.
- Demo safety review confirms no raw scores, percentages, confidence-style display, or predicted diagnostic label are user-facing.

Fail if:

- Any uncertain or flagged case remains unreviewed.
- Speaker leakage is present.
- Threshold variability is not reported.
- Demo output implies diagnosis or confidence.

### Tier 2 To Tier 3

Required:

- FP/FN review is complete with structured case notes.
- Error patterns are summarized without causal claims.
- Confound experiments are documented with dataset slice, matching method, grouped CV/seeds, threshold policy, and pass/fail criteria.
- Specificity-recall tradeoff review is complete.
- No candidate is accepted unless it improves specificity without collapsing recall and remains stable across seeds.
- Recording-quality metadata gaps are documented.
- Speaker identity quality is audited for key analysis subsets.
- Dr. Sidani clinical framing review is prepared and review questions are answerable.

Fail if:

- Specificity improvement depends on unstable thresholds.
- Recall collapses without being explicitly accepted as a conservative tradeoff.
- Error/confound findings are hidden or not summarized.
- Clinical wording still presents the score as probability, confidence, diagnosis, or rule-out.

### Tier 3 To Deployment Consideration

Required:

- External validation has been completed and documented.
- Linguistic/cognitive-linguistic features are added only after stability blockers are resolved.
- Explainability is reported carefully and without causal claims.
- Evaluation framing clearly separates verified, evaluated, and not clinically validated claims.
- Demo remains Phase 5 aligned.
- Model stability gates continue to pass under grouped CV and multi-seed evaluation.

Current status:

- Not ready for deployment consideration.
- Hardware/edge work should not be prioritized until data trust, stability, specificity-recall balance, error analysis, final model justification, and safe clinical framing are complete.

## Risks And Mitigation Table

| Risk | Root cause | User harm scenario | Likelihood | Impact | Mitigation | Residual risk |
|---|---|---|---|---|---|---|
| Demo looks diagnostic | Current demo displays predicted label and confidence | Viewer interprets output as diagnosis or probability of dementia | High | High | Align demo with Phase 5: band, qualitative explanation, disclaimer, next step only | Residual misunderstanding remains possible even with cautious wording |
| Excess false positives | Phase 4 specificity is weak at `0.217` | User or reviewer treats elevated signal as proof of dementia | High | High | Use screening-only framing, no raw probability, document false positives and weak specificity | False positives remain expected at the current operating point |
| Missed positives | Lower screening signal can be false negative | User is reassured incorrectly | Medium | High | State that lower signal does not rule out cognitive concerns | False negatives remain possible |
| Threshold instability | Fold/seed variability and rejected reliability fusion | Model appears stronger than it is | High | High | Require grouped CV, multi-seed evaluation, and threshold variability summaries | Stability may still fail after review |
| Dataset trust weakness | 37 uncertain metadata/audio rows and flagged audio cases | Results are driven by label/match errors | High | High | Manual review, trusted subset, full governed subset comparison | Some uncertainty may remain if source evidence is insufficient |
| Speaker identity uncertainty | Speaker IDs are metadata/path-derived | Leakage or identity mismatch biases results | Medium | High | Audit speaker identity quality and use grouped CV only | Folder-derived identity remains weaker than biometric verification |
| Confounding by duration, loudness, source quality, or speaker style | Heterogeneous public/source audio | Model learns recording/source artifacts instead of dementia-related patterns | High | High | Duration/loudness/source-quality/confound experiments and error analysis | Confounding may not be fully removable with current data |
| Deep learning overclaiming | wav2vec recall looked high but specificity collapsed | Project presents wav2vec as success despite unusable behavior | Medium | High | Document wav2vec as explored but not accepted | Audience may still focus on recall unless caveats are prominent |
| Reliability fusion overclaiming | Reliability final says `Not usable` | Unstable model is treated as final improvement | Medium | High | Document reliability fusion as rejected by usability criteria | Future reruns still need formal gates |
| No external validation | Current evidence is internal only | Clinical-style demo implies generalization | High | High | State no external clinical validation; keep deployment blocked | Generalization remains unproven |
| Missing language/accent/recording metadata | Current metadata does not support fairness or quality stratification well | Subgroup performance is overstated | Medium | Medium | Add recording-quality metadata and document language/accent limitations | Fairness remains uncertain until metadata/data improve |
