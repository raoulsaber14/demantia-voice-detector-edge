"""Project-wide path and runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved filesystem paths and constants used across the project."""

    root_dir: Path = Path(".")
    metadata_xlsx: Path = Path("data/metadata/raw_metadata.xlsx")
    metadata_clean_csv: Path = Path("data/metadata/metadata_clean.csv")
    train_csv: Path = Path("data/metadata/train.csv")
    valid_csv: Path = Path("data/metadata/valid.csv")
    unmatched_rows_csv: Path = Path("data/metadata/unmatched_rows.csv")
    duplicate_matches_csv: Path = Path("data/metadata/duplicate_matches.csv")
    label_conflicts_csv: Path = Path("data/metadata/label_conflicts.csv")
    data_quality_issues_csv: Path = Path("data/metadata/data_quality_issues.csv")
    raw_audio_dir: Path = Path("data/raw_audio")
    processed_audio_dir: Path = Path("data/processed_audio")
    segments_dir: Path = Path("data/segments")
    features_dir: Path = Path("data/features")
    reports_figures_dir: Path = Path("reports/figures")
    reports_tables_dir: Path = Path("reports/tables")
    baseline_model_dir: Path = Path("models/baseline")
    wav2vec_model_dir: Path = Path("models/wav2vec")
    allowed_audio_exts: Tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac")
    target_sample_rate: int = 16_000
    target_splits: Tuple[str, ...] = ("train", "valid")
    enhanced_splits: Tuple[str, ...] = ("train", "valid", "test")
    random_seed: int = 42

    def resolve(self, path_like: Path) -> Path:
        """Resolve a path relative to the project root."""
        return (self.root_dir / path_like).resolve()


CFG = ProjectConfig()
