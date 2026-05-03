# Phase 3 Feature Dictionary

Scope: final hand-crafted features for the Phase 3 classical ML baseline. These features support a screening baseline only; they are not clinical diagnostic measurements.

## Metadata Columns

| Column | Meaning | Modeling use |
| --- | --- | --- |
| `metadata_index` | Row identifier from the cleaned metadata table. | Traceability only. |
| `speaker_id` | Metadata/path-derived speaker grouping key. | Grouped splitting, leakage checks, and speaker-level evaluation only. |
| `audio_file` | Source raw audio path. | Traceability and Phase 2 governance matching only. |
| `processed_file` | Standardized WAV used for feature extraction. | Traceability only. |
| `label`, `label_name` | Binary target and readable class label. | Target/audit only, never input features. |
| `datasplit` | Speaker-grouped split assignment. | Train/validation/test filtering only. |
| `dementia_type`, `gender`, `ethnicity` | Source metadata fields when available. | Not model features in the Phase 3 baseline. |
| `feature_extraction_status`, `error` | Extraction QC fields. | QC filtering only. |

## Preprocessing Assumptions

All features are extracted from standardized audio under `data/processed_audio/`: 16 kHz, mono, WAV, PCM_16. The preprocessing order is load/resample/mono, finite-value cleanup, DC removal, conservative high-pass filtering, leading/trailing silence trim, capped RMS normalization, WAV write, and QC logging.

Aggressive spectral denoising is excluded because it can distort pitch, shimmer, jitter proxies, and spectral shape in a way that is hard to defend on a small heterogeneous speech dataset.

## Feature Families

| Feature family | Columns | What it measures | Why it may matter | Unit/scale | Direct or proxy | High-level computation | Cautions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Duration | `duration_sec` | Length of the processed clip. | Useful QC and possible confound; clip length can affect feature stability. | Seconds | Direct QC | Samples divided by sample rate. | Interpret as QC/confound context, not dementia evidence. |
| MFCC summaries | `mfcc_1_*` through `mfcc_13_*` | Vocal-tract/spectral-envelope shape. | Standard classical speech descriptors for articulation and spectral consistency. | MFCC coefficient scale | Direct acoustic descriptor | `librosa.feature.mfcc`, summarized by mean/std/min/max. | Sensitive to recording conditions. |
| Delta MFCC summaries | `delta_mfcc_1_*` through `delta_mfcc_13_*` | First-order MFCC change over time. | Captures acoustic dynamics and articulation change. | MFCC change/frame | Direct acoustic descriptor | Safe-width `librosa.feature.delta`, summarized. | Less stable on very short clips. |
| Delta-delta MFCC summaries | `delta2_mfcc_1_*` through `delta2_mfcc_13_*` | Second-order MFCC change. | Captures acceleration/instability in acoustic trajectories. | MFCC acceleration/frame | Direct acoustic descriptor | Second-order delta, summarized. | Some outlier rates are documented in the feature manifest. |
| Zero crossing rate | `zcr_mean`, `zcr_std`, `zcr_min`, `zcr_max` | Rate of sign changes in the waveform. | Reflects noisiness, frication, and voicing balance. | Fraction/frame | Direct acoustic descriptor | `librosa.feature.zero_crossing_rate`, summarized. | Noise-sensitive. |
| Spectral centroid | `spectral_centroid_*` | Frequency center of mass. | Captures spectral brightness/noisiness. | Hz | Direct acoustic descriptor | `librosa.feature.spectral_centroid`, summarized by distribution stats. | Recording and microphone sensitive. |
| Spectral bandwidth | `spectral_bandwidth_*` | Spread of spectral energy. | May reflect articulation, breathiness, or noise. | Hz | Direct acoustic descriptor | `librosa.feature.spectral_bandwidth`, summarized. | Not disease-specific. |
| Spectral rolloff | `spectral_rolloff_*` | Frequency below which 85 percent of energy lies. | Captures high-frequency energy distribution. | Hz | Direct acoustic descriptor | `librosa.feature.spectral_rolloff(roll_percent=0.85)`, summarized. | Noise-sensitive. |
| Spectral flatness | `spectral_flatness_*` | Tone-like versus noise-like spectrum. | Helps characterize noisy/breathy regions and degraded recordings. | Unitless ratio | Direct acoustic descriptor | `librosa.feature.spectral_flatness`, summarized. | Low-level values can have outliers. |
| F0 statistics | `f0_mean`, `f0_median`, `f0_p10`, `f0_p25`, `f0_p75`, `f0_p90`, `f0_std`, `f0_min`, `f0_max`, `f0_range`, `f0_iqr`, `f0_semitone_std` | Fundamental-frequency estimates on voiced frames. | Prosody and pitch variability may relate to speech planning, affect, and voice quality. Percentiles reduce dependence on extreme F0 estimates. | Hz or semitones | Acoustic estimate | `librosa.pyin` on a bounded analysis window; NaNs excluded and summarized. | Estimate only; not laryngoscopic measurement. |
| F0 coverage | `f0_voiced_frame_count`, `voiced_ratio`, `f0_analysis_duration_sec` | Amount of usable voiced pitch evidence. | Low voiced coverage flags less stable pitch features. | Count, ratio, seconds | QC/proxy | pYIN voiced-frame count and ratio. | Coverage depends on audio quality and voicing. |
| Jitter proxies | `jitter_local_proxy`, `jitter_rap_proxy`, `pitch_period_std_proxy` | Frame-to-frame pitch-period variability on voiced frames. | May capture voice instability as a classical descriptor. | Unitless ratio or seconds | Proxy | Convert voiced F0 estimates to periods and summarize local variation. | Final decision: kept with caution as proxy only, not clinical jitter. |
| Shimmer proxies | `shimmer_local_proxy`, `shimmer_db_proxy` | Frame-to-frame RMS amplitude variability on voiced frames. | May capture amplitude instability as a classical descriptor. | Unitless ratio or dB change | Proxy | Align RMS to voiced F0 frames and summarize amplitude changes. | Final decision: kept with caution as proxy only, not clinical shimmer. |
| RMS energy | `rms_mean`, `rms_std`, `rms_min`, `rms_p10`, `rms_p25`, `rms_median`, `rms_p75`, `rms_p90`, `rms_max` | Short-time waveform energy distribution. | Captures loudness dynamics and supports pause/low-energy analysis. | Linear RMS amplitude | Direct acoustic descriptor | `librosa.feature.rms`, summarized by distribution stats. | Microphone distance and normalization matter. |
| Energy modulation | `energy_modulation_mean_abs`, `energy_modulation_std`, `log_energy_modulation_std` | Frame-to-frame RMS change. | Captures local loudness dynamics and speech-flow variation without transcripts. | Linear or log RMS change/frame | Proxy | First differences of RMS and log-RMS frame sequences. | Sensitive to background noise and gain changes. |
| Low energy ratio | `low_energy_ratio` | Fraction of frames below the file-level median RMS. | Simple low-energy distribution marker. | Ratio | Proxy | Count RMS frames below median RMS. | File-relative threshold, not absolute loudness. |
| Pause/silence ratios | `silence_ratio`, `pause_ratio`, `unvoiced_ratio`, `energy_voiced_ratio` | Fraction of clip classified as low-energy or speech-like. | Pause behavior is a common fluency marker in cognitive speech analysis. | Ratio | Energy-based proxy | Frame-RMS energy intervals define speech regions; silence is the complement. | Sensitive to background noise and threshold choice. |
| Pause durations/counts | `pause_count`, `avg_pause_duration_sec`, `std_pause_duration_sec`, `median_pause_duration_sec`, pause duration percentiles, `max_pause_duration_sec`, `total_silence_duration_sec`, `pause_rate_per_min` | Number and duration distribution of pauses at least 0.20 seconds. | Approximates pausing and hesitations without transcripts; percentiles are more robust than max alone. | Count, seconds, rate/min | Energy-based proxy | Low-energy gaps between speech intervals are counted as pauses when above threshold and summarized. | Not transcript-based hesitation annotation. |
| Silence event summaries | `silence_event_count`, `avg_silence_duration_sec`, `std_silence_duration_sec`, `median_silence_duration_sec`, silence duration percentiles, `max_silence_duration_sec` | All detected low-energy gaps, including brief gaps. | Separates brief gaps from longer pauses. | Count, seconds | Energy-based proxy | Durations between energy-based speech intervals. | May merge/split regions under noise. |
| Speech segment summaries | `speech_segment_count`, `avg_speech_segment_duration_sec`, `std_speech_segment_duration_sec`, `median_speech_segment_duration_sec`, speech-segment duration percentiles, `max_speech_segment_duration_sec`, `speech_duration_sec`, `speech_segment_rate_per_min` | Number and duration distribution of detected speech-like regions. | Fragmented or highly variable speech regions can be a fluency proxy. | Count, seconds, rate/min | Energy-based proxy | Durations of frame-RMS speech intervals. | Energy-based, not phonetic segmentation. |
| Transition proxies | `pause_to_speech_transition_count`, `speech_to_pause_transition_count`, `pause_speech_transition_rate_per_min` | Energy-state transitions around pauses. | Captures coarse flow/continuity without ASR or transcripts. | Count, rate/min | Energy-based proxy | Counts boundaries around pauses lasting at least 0.20 seconds. | Threshold-sensitive and not a phonetic turn-taking measure. |
| Speaking-rate proxies | `speaking_rate_proxy`, `onset_count`, `onset_rate_per_speech_sec` | Acoustic onset density. | Approximates syllable/utterance activity when transcripts are unavailable. | Count or rate/sec | Proxy | `librosa.onset` events per total duration and per speech duration. | Not words per minute. |

Distribution suffixes such as `_p10`, `_p25`, `_median`, `_p75`, and `_p90` summarize frame-level or event-duration feature distributions. Suffixes `_mean`, `_std`, `_min`, and `_max` summarize central tendency, variability, and range.

## Final Proxy-Feature Policy For Phase 4

Option 1 is selected for Phase 4 readiness: proxy/acoustic-estimate features already present in the official Phase 3 table are retained, but they must be labeled and interpreted cautiously. The official modeling table is not changed by this policy.

- `proxy_but_retained`: F0/acoustic estimates, jitter/shimmer proxies, pause/VAD/fluency proxies, low-energy ratio, and energy-modulation speech-dynamics proxies that remain in `data/features/features_phase3_governed_pruned.csv`.
- `exploratory_only`: quality-accepted features that were removed from the official table by redundancy pruning; these may be discussed only as exploratory/audit variables unless the manifest is regenerated.
- `should_be_excluded`: features excluded by the hard Phase 3 feature policy, currently including `voiced_unvoiced_ratio`.
- `direct`: retained acoustic or QC descriptors that are not proxy-labeled by the Phase 4 policy.

Machine-readable status for every candidate feature is saved in `reports/phase3_proxy_feature_status.csv`. Proxy features are acoustic approximations, not clinical biomarkers, and should not be presented as direct measures of dementia.

## Excluded Feature

`voiced_unvoiced_ratio` is excluded from the final baseline feature set because it exceeded the Phase 3 missingness threshold when silence duration approached zero. The more stable component ratios and duration features are retained instead.

## Scaling and Missing Values

Infinite values are converted to missing during feature cleaning. Remaining missing values are not globally imputed in the CSV. Median imputation and scaling are fitted inside sklearn pipelines on training data only. Logistic Regression and SVM use imputation plus `StandardScaler`; Random Forest uses imputation without scaling. The official model input is the governed table after train-split redundancy pruning.
