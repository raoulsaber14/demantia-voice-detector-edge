# Phase 3 Preprocessing Policy

This policy applies to the classical ML baseline only. Raw audio under `data/raw_audio/` is never modified.

## Targets

- Sample rate: `16000 Hz`
- Channels: `mono`
- Output format: `WAV`, subtype `PCM_16`
- Output location: mirrored paths under `data/processed_audio/`

## Step Order

1. load/resample/mono -> finite-value cleanup -> DC removal -> conservative high-pass filtering -> optional leading/trailing silence trim -> capped RMS normalization -> PCM_16 WAV write -> QC logging.

## Rationale

- Resampling and mono conversion make downstream hand-crafted features comparable across files.
- DC removal and conservative high-pass filtering reduce electrical offset and low-frequency rumble without broad speech enhancement.
- Leading/trailing silence trimming removes obvious padding while preserving internal pauses needed for fluency features.
- RMS normalization uses capped gain and peak limiting, so very quiet clips are not amplified without bound.
- Energy-based VAD is used for quality-control summaries and pause features, not as a hard deletion step for internal non-speech regions.
- Aggressive spectral denoising is excluded because it can alter F0, shimmer, jitter proxies, and spectral descriptors in ways that are hard to defend on this dataset.

## Effective Parameters

| parameter | value |
| --- | --- |
| `target_sample_rate_hz` | 16000 |
| `target_channels` | mono |
| `target_file_format` | wav |
| `target_subtype` | PCM_16 |
| `dc_offset_removal` | True |
| `highpass_filter_hz` | 50.0 |
| `silence_trim_top_db` | 30 |
| `loudness_normalization` | RMS target with capped gain and peak limiting |
| `target_rms_dbfs` | -20.0 |
| `max_gain_db` | 12.0 |
| `peak_limit` | 0.99 |
| `aggressive_spectral_denoising` | False |
| `vad_for_qc` | frame RMS energy intervals, top_db=30 |
| `silence_trim_enabled` | True |
| `step_order` | load/resample/mono -> finite-value cleanup -> DC removal -> conservative high-pass filtering -> optional leading/trailing silence trim -> capped RMS normalization -> PCM_16 WAV write -> QC logging |
| `reuse_existing_valid_processed_audio` | True |
