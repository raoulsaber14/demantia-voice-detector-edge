"""Silence-based segmentation for processed audio."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.config import CFG


def segment_audio_file(
    audio_path: Path,
    out_dir: Path,
    top_db: int = 30,
    min_duration_sec: float = 0.5,
) -> List[Dict[str, object]]:
    """Create speech segments from one file and save each segment as WAV."""
    y, sr = librosa.load(audio_path, sr=CFG.target_sample_rate, mono=True)
    intervals = librosa.effects.split(y, top_db=top_db)
    rows: List[Dict[str, object]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    seg_idx = 0
    for start, end in intervals:
        dur = (end - start) / sr
        if dur < min_duration_sec:
            continue
        segment = y[start:end]
        seg_name = f"{stem}_seg_{seg_idx:04d}.wav"
        seg_path = out_dir / seg_name
        sf.write(seg_path, segment, sr)
        rows.append(
            {
                "segment_file": str(seg_path.resolve()),
                "source_audio_file": str(audio_path.resolve()),
                "segment_index": seg_idx,
                "start_sec": start / sr,
                "end_sec": end / sr,
                "duration_sec": dur,
                "sample_rate": sr,
            }
        )
        seg_idx += 1
    return rows


def build_segments(
    metadata_csv: Path = CFG.metadata_clean_csv,
    processed_root: Path = CFG.processed_audio_dir,
    segments_root: Path = CFG.segments_dir,
) -> pd.DataFrame:
    """Generate segments for all available processed files and save mapping."""
    meta = pd.read_csv(metadata_csv)
    rows: List[Dict[str, object]] = []

    for _, mrow in tqdm(meta.iterrows(), total=len(meta), desc="Segmenting"):
        src_raw = Path(str(mrow.get("audio_file", "")))
        if not src_raw.exists():
            continue
        rel = src_raw.resolve().relative_to(CFG.raw_audio_dir.resolve())
        proc_file = processed_root.resolve() / rel.with_suffix(".wav")
        if not proc_file.exists():
            continue
        seg_dir = segments_root.resolve() / rel.parent / rel.stem
        try:
            seg_rows = segment_audio_file(proc_file, seg_dir)
            for r in seg_rows:
                r["metadata_index"] = mrow.get("metadata_index")
                r["speaker_id"] = mrow.get("speaker_id")
                r["label"] = mrow.get("label")
                r["label_name"] = mrow.get("label_name")
                r["datasplit"] = mrow.get("datasplit")
            rows.extend(seg_rows)
        except Exception as exc:
            rows.append(
                {
                    "metadata_index": mrow.get("metadata_index"),
                    "speaker_id": mrow.get("speaker_id"),
                    "source_audio_file": str(proc_file),
                    "segment_file": "",
                    "error": str(exc),
                }
            )

    seg_meta = pd.DataFrame(rows)
    out_csv = Path("data/segments/segment_metadata.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    seg_meta.to_csv(out_csv, index=False)
    return seg_meta


def parse_args() -> argparse.Namespace:
    """CLI arguments for segmentation."""
    parser = argparse.ArgumentParser(description="Split processed audio into speech segments.")
    parser.add_argument("--metadata-csv", type=str, default=str(CFG.metadata_clean_csv))
    parser.add_argument("--processed-root", type=str, default=str(CFG.processed_audio_dir))
    parser.add_argument("--segments-root", type=str, default=str(CFG.segments_dir))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seg_df = build_segments(
        metadata_csv=Path(args.metadata_csv),
        processed_root=Path(args.processed_root),
        segments_root=Path(args.segments_root),
    )
    print(f"Segment rows: {len(seg_df)}")
