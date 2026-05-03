"""Generate a Phase 1 pipeline audit report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(".")
RAW_AUDIO_ROOT = PROJECT_ROOT / "data/raw_audio"
METADATA_CSV = PROJECT_ROOT / "data/metadata/metadata_clean.csv"
PREPROCESS_LOG_CSV = PROJECT_ROOT / "data/metadata/preprocess_log.csv"
FEATURES_CSV = PROJECT_ROOT / "data/features/file_features.csv"
BASELINE_METRICS_CSV = PROJECT_ROOT / "reports/tables/baseline_metrics.csv"
PHASE1_REPORT = PROJECT_ROOT / "reports/phase1_audit_summary.md"
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}


def real_audio_files(label_dir: str) -> list[Path]:
    root = RAW_AUDIO_ROOT / label_dir
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS and not path.name.startswith("._")
    )


def simple_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    return df.to_string(index=False)


def audio_info_sample(paths: list[Path], limit: int = 8) -> pd.DataFrame:
    rows = []
    for path in paths[:limit]:
        try:
            info = sf.info(str(path))
            rows.append(
                {
                    "file": str(path),
                    "sample_rate": info.samplerate,
                    "duration_sec": round(float(info.duration), 3),
                    "frames": info.frames,
                }
            )
        except Exception as exc:
            rows.append({"file": str(path), "error": str(exc)})
    return pd.DataFrame(rows)


def audio_info_all(paths: list[Path]) -> tuple[pd.DataFrame, list[tuple[Path, str]]]:
    rows = []
    failures = []
    for path in paths:
        try:
            info = sf.info(str(path))
            rows.append(
                {
                    "file": str(path),
                    "sample_rate": info.samplerate,
                    "duration_sec": float(info.duration),
                    "frames": info.frames,
                }
            )
        except Exception as exc:
            failures.append((path, str(exc)))
    return pd.DataFrame(rows), failures


def main() -> None:
    meta = pd.read_csv(METADATA_CSV)
    preprocess_log = pd.read_csv(PREPROCESS_LOG_CSV)
    features = pd.read_csv(FEATURES_CSV)
    metrics = pd.read_csv(BASELINE_METRICS_CSV)

    dementia_files = real_audio_files("dementia")
    nodementia_files = real_audio_files("nodementia")
    all_real_audio = dementia_files + nodementia_files
    apple_double_count = sum(
        1
        for path in RAW_AUDIO_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS and path.name.startswith("._")
    )
    zero_byte_files = [path for path in all_real_audio if path.stat().st_size == 0]

    meta_path_exists = meta["audio_file"].map(lambda value: Path(str(value)).exists())
    meta_duplicates = int(meta["audio_file"].duplicated().sum())
    unexpected_labels = sorted(set(meta["label_name"].dropna()) - {"dementia", "non_dementia"})
    nodementia_meta = meta[meta["audio_file"].astype(str).str.contains("/nodementia/", regex=False)]
    dementia_meta = meta[meta["audio_file"].astype(str).str.contains("/dementia/", regex=False)]

    processed_paths = preprocess_log.loc[preprocess_log["status"] == "ok", "processed_file"].map(Path).tolist()
    processed_exists = [path for path in processed_paths if path.exists()]
    processed_missing = [path for path in processed_paths if not path.exists()]
    processed_info, processed_read_failures = audio_info_all(processed_exists)
    processed_sample = audio_info_sample(processed_exists)

    numeric_features = features.select_dtypes(include=["number"])
    nan_counts = numeric_features.isna().sum().sort_values(ascending=False)
    nan_counts = nan_counts[nan_counts > 0].head(12)
    inf_count = int(np.isinf(numeric_features.to_numpy(dtype=float)).sum())

    model_files = sorted(str(path) for path in (PROJECT_ROOT / "models/baseline").glob("*.joblib"))
    figure_files = sorted(str(path) for path in (PROJECT_ROOT / "reports/figures").glob("*_confusion_matrix.png"))

    lines = [
        "# Phase 1 Audit Summary",
        "",
        "## Raw Audio Discovery",
        "",
        f"- Real audio files discovered: {len(all_real_audio)}",
        f"- Dementia files: {len(dementia_files)}",
        f"- Non-dementia files from `nodementia`: {len(nodementia_files)}",
        f"- AppleDouble sidecar audio-extension files skipped: {apple_double_count}",
        f"- Zero-byte real audio files: {len(zero_byte_files)}",
        "",
        "## Metadata Output",
        "",
        f"- Rows: {len(meta)}",
        f"- Columns: {', '.join(meta.columns)}",
        f"- Audio paths that exist: {int(meta_path_exists.sum())}/{len(meta)}",
        f"- Duplicate `audio_file` rows: {meta_duplicates}",
        f"- Unexpected labels: {unexpected_labels if unexpected_labels else 'none'}",
        "",
        "Class counts:",
        "",
        simple_table(meta["label_name"].value_counts(dropna=False).rename_axis("label_name").reset_index(name="rows")),
        "",
        "Split by class:",
        "",
        simple_table(pd.crosstab(meta["datasplit"], meta["label_name"], dropna=False).reset_index()),
        "",
        f"- Dementia-folder rows labeled dementia: {int((dementia_meta['label_name'] == 'dementia').sum())}/{len(dementia_meta)}",
        f"- Nodementia-folder rows labeled non_dementia: {int((nodementia_meta['label_name'] == 'non_dementia').sum())}/{len(nodementia_meta)}",
        "",
        "Sample metadata rows:",
        "",
        simple_table(
            meta[
                [
                    "metadata_index",
                    "name",
                    "datasplit",
                    "audio_class_from_folder",
                    "label",
                    "label_name",
                    "audio_file",
                ]
            ].head(8)
        ),
        "",
        "## Preprocessing Output",
        "",
        f"- Log rows: {len(preprocess_log)}",
        "",
        simple_table(preprocess_log["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="rows")),
        "",
        f"- Processed files referenced by ok rows: {len(processed_paths)}",
        f"- Processed files found on disk: {len(processed_exists)}",
        f"- Missing processed files referenced by log: {len(processed_missing)}",
        f"- Processed files readable by soundfile: {len(processed_info)}/{len(processed_exists)}",
        f"- Processed read failures: {len(processed_read_failures)}",
        f"- Processed sample-rate counts: {processed_info['sample_rate'].value_counts().to_dict() if not processed_info.empty else {}}",
        (
            "- Processed duration seconds min/median/max: "
            f"{processed_info['duration_sec'].min():.3f}/"
            f"{processed_info['duration_sec'].median():.3f}/"
            f"{processed_info['duration_sec'].max():.3f}"
            if not processed_info.empty
            else "- Processed duration seconds min/median/max: unavailable"
        ),
        "",
        "Readable processed-audio sample:",
        "",
        simple_table(processed_sample),
        "",
        "## Feature Output",
        "",
        f"- Feature shape: {features.shape[0]} rows x {features.shape[1]} columns",
        "",
        "Feature class counts:",
        "",
        simple_table(features["label_name"].value_counts(dropna=False).rename_axis("label_name").reset_index(name="rows")),
        "",
        "Feature split by class:",
        "",
        simple_table(pd.crosstab(features["datasplit"], features["label_name"], dropna=False).reset_index()),
        "",
        f"- Numeric feature columns: {len(numeric_features.columns)}",
        f"- Infinite numeric values: {inf_count}",
        "",
        "Top numeric NaN counts:",
        "",
        simple_table(nan_counts.rename_axis("feature").reset_index(name="nan_rows")),
        "",
        "## Baseline Output",
        "",
        f"- Metrics file: {BASELINE_METRICS_CSV}",
        f"- Model files: {len(model_files)}",
        f"- Confusion matrix figures: {len(figure_files)}",
        "",
        "Baseline metrics:",
        "",
        simple_table(metrics),
        "",
        "Model artifacts:",
        "",
        "\n".join(f"- {path}" for path in model_files),
        "",
        "Confusion matrix artifacts:",
        "",
        "\n".join(f"- {path}" for path in figure_files),
        "",
        "## Phase 1 Notes",
        "",
        "- The raw Excel metadata contains dementia subjects only; non-dementia controls are now added from folder-labeled audio-only rows.",
        "- `nodementia` is normalized to `non_dementia` in metadata labels.",
        "- AppleDouble `._*.wav` sidecar files are skipped during audio scanning.",
        "- F0/pitch extraction is bounded to the first 10 seconds per clip for Phase 1 runtime practicality.",
        "- Baseline figures use Matplotlib's non-GUI Agg backend to avoid macOS backend aborts in batch runs.",
    ]

    PHASE1_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PHASE1_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {PHASE1_REPORT}")


if __name__ == "__main__":
    main()
