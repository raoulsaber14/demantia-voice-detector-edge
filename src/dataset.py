"""Dataset helpers for wav2vec2 training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from src.config import CFG


def _processed_file_from_raw(raw_audio_file: str) -> str:
    """Map raw audio path to expected processed WAV path."""
    raw = Path(str(raw_audio_file))
    rel = raw.resolve().relative_to(CFG.raw_audio_dir.resolve())
    return str((CFG.processed_audio_dir.resolve() / rel.with_suffix(".wav")).resolve())


def load_hf_dataset(metadata_csv: Path = CFG.metadata_clean_csv) -> DatasetDict:
    """Build HuggingFace train/valid datasets from metadata splits."""
    meta = pd.read_csv(metadata_csv)
    meta = meta[meta["label"].isin([0, 1])].copy()
    meta = meta[meta["datasplit"].isin(["train", "valid"])].copy()
    meta["processed_file"] = meta["audio_file"].map(_processed_file_from_raw)
    meta = meta[meta["processed_file"].map(lambda p: Path(p).exists())].copy()

    keep_cols = [
        "metadata_index",
        "speaker_id",
        "processed_file",
        "label",
        "label_name",
        "datasplit",
        "dementia_type",
        "gender",
        "ethnicity",
    ]
    meta = meta[keep_cols]
    train_ds = Dataset.from_pandas(meta[meta["datasplit"] == "train"].reset_index(drop=True), preserve_index=False)
    valid_ds = Dataset.from_pandas(meta[meta["datasplit"] == "valid"].reset_index(drop=True), preserve_index=False)
    return DatasetDict({"train": train_ds, "valid": valid_ds})
