"""Metadata ingestion, audio matching, and data quality checks."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import soundfile as sf

from src.config import CFG


def snake_case(text: str) -> str:
    """Convert a column name into snake_case."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
    return re.sub(r"_+", "_", normalized)


def clean_text(value: object) -> str:
    """Normalize string values for consistent matching and analysis."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(text: str) -> str:
    """Aggressive normalization for file/name matching keys."""
    text = clean_text(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def derive_speaker_id(name: object) -> str:
    """Derive a de-identified speaker key from metadata name."""
    base = normalize_key(clean_text(name))
    return base or "unknown_speaker"


def normalize_datasplit(value: object) -> str:
    """Normalize split labels to train/valid if possible."""
    split = clean_text(value).lower()
    if split in {"val", "validation"}:
        return "valid"
    return split


def derive_binary_label(dementia_type: object) -> Tuple[Optional[int], str]:
    """
    Map dementia subtype metadata to binary screening labels.

    Any non-empty dementia type is treated as dementia for the primary task.
    If no dementia type is present, label is unresolved.
    """
    dtype = clean_text(dementia_type).lower()
    if not dtype:
        return None, "unknown"
    if dtype in {"non_dementia", "control", "healthy", "none", "no dementia"}:
        return 0, "non_dementia"
    return 1, "dementia"


def binary_label_from_folder(folder_label: object) -> Tuple[Optional[int], str]:
    """Map normalized audio folder labels to binary labels."""
    label = clean_text(folder_label).lower()
    if label == "dementia":
        return 1, "dementia"
    if label == "non_dementia":
        return 0, "non_dementia"
    return None, "unknown"


def deterministic_split_for_speaker(speaker_id: str, train_ratio: float = 0.8) -> str:
    """Assign a stable train/valid split for folder-only speakers."""
    digest = hashlib.sha256(speaker_id.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if score < train_ratio else "valid"


def preferred_time_buckets(row: pd.Series) -> List[int]:
    """Infer preferred clip timing buckets from source metadata URL columns."""
    buckets: List[int] = []
    column_to_bucket = [
        ("urls_after_symptoms", 0),
        ("5_years", 5),
        ("5_10_years", 10),
        ("10_15_years", 15),
    ]
    for column, bucket in column_to_bucket:
        if clean_text(row.get(column, "")):
            buckets.append(bucket)
    return buckets


def filename_time_bucket(path: Path) -> Optional[int]:
    """Extract timing suffixes such as _0, _5, _10, or _15 from a filename."""
    matches = re.findall(r"(?:^|_)(0|5|10|15)(?:_|$)", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def candidate_score(row: pd.Series, audio_row: pd.Series) -> Tuple[int, float, str]:
    """Rank candidate audio files by transparent deterministic rules."""
    name_key = normalize_key(row.get("name", ""))
    stem_key = clean_text(audio_row.get("norm_stem", ""))
    parent_key = clean_text(audio_row.get("norm_parent", ""))
    audio_path = Path(audio_row["audio_file"])
    preferred_buckets = preferred_time_buckets(row)
    bucket = filename_time_bucket(audio_path)

    if bucket is not None and bucket in preferred_buckets:
        return 0, 1.0, f"preferred_time_bucket_{bucket}"
    if parent_key == name_key:
        similarity = SequenceMatcher(None, name_key, stem_key).ratio() if stem_key else 0.0
        return 1, similarity, "exact_speaker_folder"
    if stem_key == name_key:
        return 2, 1.0, "exact_normalized_stem"
    similarity = SequenceMatcher(None, name_key, stem_key).ratio() if stem_key else 0.0
    return 3, similarity, "filename_similarity"


@dataclass
class AudioRecord:
    """Single discovered audio file with matching metadata keys."""

    audio_file: Path
    audio_class_from_folder: str
    base_name: str
    norm_stem: str
    norm_parent: str
    duration_sec: Optional[float]
    sample_rate: Optional[int]

    @property
    def norm_filename(self) -> str:
        return normalize_key(self.base_name)


@dataclass
class MatchDecision:
    """Deterministic audio-match decision for one metadata row."""

    best_idx: Optional[int]
    status: str
    candidates: List[int]
    rule_used: str
    confidence: str
    needs_review: bool
    score_summary: str


def safe_audio_info(path: Path) -> Tuple[Optional[float], Optional[int]]:
    """Read duration and sample-rate without loading waveform into memory."""
    try:
        info = sf.info(str(path))
        return float(info.duration), int(info.samplerate)
    except Exception:
        return None, None


def infer_folder_label(path: Path, raw_audio_dir: Path) -> str:
    """Infer binary label from folder convention for cross-checking only."""
    rel = path.relative_to(raw_audio_dir).parts
    if not rel:
        return ""
    top = rel[0].lower()
    if top == "dementia":
        return "dementia"
    if top in {"nodementia", "nondementia", "non_dementia", "control", "healthy"}:
        return "non_dementia"
    return top


def scan_audio_files(raw_audio_dir: Path) -> pd.DataFrame:
    """Recursively scan audio files and build a searchable index."""
    records: List[AudioRecord] = []
    if not raw_audio_dir.exists():
        return pd.DataFrame()

    for path in raw_audio_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() not in CFG.allowed_audio_exts:
            continue

        duration, sr = safe_audio_info(path)
        records.append(
            AudioRecord(
                audio_file=path.resolve(),
                audio_class_from_folder=infer_folder_label(path, raw_audio_dir.resolve()),
                base_name=path.name,
                norm_stem=normalize_key(path.stem),
                norm_parent=normalize_key(path.parent.name),
                duration_sec=duration,
                sample_rate=sr,
            )
        )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame([r.__dict__ | {"norm_filename": r.norm_filename} for r in records])


def append_audio_only_rows(df: pd.DataFrame, audio_df: pd.DataFrame) -> pd.DataFrame:
    """Add one row per raw audio file not already represented by metadata."""
    if audio_df.empty:
        return df

    matched_audio = {int(idx) for idx in df["audio_row_idx"].dropna().tolist()}
    speaker_split_lookup = (
        df[df["datasplit"].fillna("").astype(str) != ""]
        .drop_duplicates("speaker_id")
        .set_index("speaker_id")["datasplit"]
        .to_dict()
    )

    audio_only_rows: List[Dict[str, object]] = []
    for audio_idx, audio_row in audio_df.iterrows():
        if int(audio_idx) in matched_audio:
            continue

        audio_file = Path(audio_row["audio_file"])
        speaker_name = clean_text(audio_file.parent.name)
        speaker_id = derive_speaker_id(speaker_name)
        folder_label = clean_text(audio_row.get("audio_class_from_folder", ""))
        label, label_name = binary_label_from_folder(folder_label)
        datasplit = speaker_split_lookup.get(speaker_id, deterministic_split_for_speaker(speaker_id))

        row = {col: "" for col in df.columns}
        row.update(
            {
                "metadata_index": f"audio_only_{audio_idx}",
                "name": speaker_name,
                "dementia_type": folder_label if folder_label == "dementia" else "",
                "source_datasplit": "audio_only",
                "datasplit": datasplit,
                "speaker_id": speaker_id,
                "label": label,
                "label_name": label_name,
                "datasplit_valid": datasplit in CFG.target_splits,
                "audio_row_idx": int(audio_idx),
                "match_status": "audio_only_from_folder",
                "match_candidate_count": 1,
                "audio_exists": True,
                "audio_file": audio_file.resolve(),
                "audio_class_from_folder": folder_label,
                "duration_sec": audio_row.get("duration_sec"),
                "sample_rate": audio_row.get("sample_rate"),
                "label_from_audio_folder": label,
                "label_conflict": False,
                "match_rule_used": "audio_only_from_folder",
                "match_confidence": "high",
                "match_needs_review": False,
                "match_score_summary": "",
            }
        )
        audio_only_rows.append(row)

    if not audio_only_rows:
        return df
    return pd.concat([df, pd.DataFrame(audio_only_rows)], ignore_index=True)


def build_audio_lookup(audio_df: pd.DataFrame) -> Dict[str, List[int]]:
    """Create reverse index from normalized keys to candidate audio rows."""
    lookup: Dict[str, List[int]] = {}
    if audio_df.empty:
        return lookup

    for idx, row in audio_df.iterrows():
        keys = {
            clean_text(row.get("base_name", "")),
            normalize_key(row.get("base_name", "")),
            normalize_key(Path(clean_text(row.get("base_name", ""))).stem),
            clean_text(row.get("norm_stem", "")),
            clean_text(row.get("norm_parent", "")),
        }
        keys = {k for k in keys if k}
        for key in keys:
            lookup.setdefault(key, []).append(idx)
    return lookup


def choose_match(
    row: pd.Series,
    audio_df: pd.DataFrame,
    lookup: Dict[str, List[int]],
) -> MatchDecision:
    """Return best audio row index, match status, and all candidates."""
    if audio_df.empty:
        return MatchDecision(None, "no_audio_index", [], "no_audio_index", "none", True, "")

    candidates: List[int] = []
    audio_filename = clean_text(row.get("audio_filename", ""))
    name = clean_text(row.get("name", ""))
    norm_name = normalize_key(name)

    if audio_filename:
        basename = Path(audio_filename).name
        direct = lookup.get(clean_text(basename), [])
        normalized = lookup.get(normalize_key(basename), [])
        candidates = list(dict.fromkeys(direct + normalized))
        if candidates:
            return MatchDecision(
                candidates[0],
                "matched_by_audio_filename",
                candidates,
                "exact_audio_filename",
                "high",
                False,
                f"candidate_count={len(candidates)}",
            )

    by_parent = lookup.get(norm_name, [])
    by_stem = [i for i in lookup.get(norm_name, []) if clean_text(audio_df.loc[i, "norm_stem"]) == norm_name]
    if by_stem:
        candidates = by_stem
        status = "matched_by_normalized_stem"
    elif by_parent:
        candidates = by_parent
        status = "matched_by_folder_name"
    else:
        starts = audio_df.index[
            audio_df["norm_stem"].fillna("").astype(str).str.startswith(norm_name)
        ].tolist() if norm_name else []
        if starts:
            candidates = starts
            status = "matched_by_partial_stem"
        else:
            return MatchDecision(None, "unmatched", [], "no_candidate", "none", True, "")

    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return MatchDecision(
            unique[0],
            status,
            unique,
            status,
            "high",
            False,
            "candidate_count=1",
        )

    scored_rows = []
    for candidate_idx in unique:
        priority, similarity, rule = candidate_score(row, audio_df.loc[candidate_idx])
        audio_path = Path(audio_df.loc[candidate_idx, "audio_file"])
        scored_rows.append(
            {
                "candidate_idx": candidate_idx,
                "priority": priority,
                "similarity": similarity,
                "rule": rule,
                "path": str(audio_path),
            }
        )
    scored_rows = sorted(scored_rows, key=lambda item: (item["priority"], -item["similarity"], item["path"]))
    best = scored_rows[0]
    tied_best = [
        item
        for item in scored_rows
        if item["priority"] == best["priority"] and abs(item["similarity"] - best["similarity"]) < 1e-9
    ]
    confidence = "medium" if best["priority"] <= 1 else "low"
    needs_review = len(tied_best) > 1 or confidence == "low"
    rule_used = best["rule"] if len(tied_best) == 1 else f"{best['rule']}_lexicographic_tiebreak"
    summary = "; ".join(
        f"{item['candidate_idx']}:{item['rule']}:{item['similarity']:.3f}:{Path(item['path']).name}"
        for item in scored_rows
    )
    return MatchDecision(
        int(best["candidate_idx"]),
        f"{status}_ambiguous_resolved",
        unique,
        rule_used,
        confidence,
        needs_review,
        summary,
    )


def collect_quality_issues(issue_rows: List[Dict[str, object]], issue_type: str, detail: str, row: pd.Series) -> None:
    """Append one structured issue to issue collection."""
    issue_rows.append(
        {
            "issue_type": issue_type,
            "detail": detail,
            "metadata_index": row.get("metadata_index"),
            "speaker_id": row.get("speaker_id"),
            "name": row.get("name"),
            "datasplit": row.get("datasplit"),
            "audio_file": row.get("audio_file"),
        }
    )


def enforce_speaker_split_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all clips from the same speaker stay in a single split."""
    speaker_split_nunique = (
        df[df["datasplit"].isin(CFG.enhanced_splits)]
        .groupby("speaker_id")["datasplit"]
        .nunique()
    )
    leaking_speakers = speaker_split_nunique[speaker_split_nunique > 1].index.tolist()
    if not leaking_speakers:
        return pd.DataFrame(columns=df.columns)
    return df[df["speaker_id"].isin(leaking_speakers)].copy()


def assign_speaker_stratified_splits(
    df: pd.DataFrame,
    seed: int = CFG.random_seed,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Assign reproducible speaker-level splits with class balance where feasible."""
    split_df = df.copy()
    speaker_labels = (
        split_df[split_df["label"].isin([0, 1])]
        .groupby("speaker_id")
        .agg(label=("label", "first"), label_nunique=("label", "nunique"), label_name=("label_name", "first"))
        .reset_index()
    )
    mixed_label_speakers = speaker_labels[speaker_labels["label_nunique"] > 1]["speaker_id"].tolist()
    if mixed_label_speakers:
        raise ValueError(f"Speakers with multiple labels cannot be split safely: {mixed_label_speakers[:10]}")

    by_label: Dict[int, List[str]] = {}
    for label in [0, 1]:
        speakers = sorted(speaker_labels[speaker_labels["label"] == label]["speaker_id"].tolist())
        rng = random.Random(seed + label)
        rng.shuffle(speakers)
        by_label[label] = speakers

    min_class_speakers = min((len(v) for v in by_label.values()), default=0)
    if min_class_speakers >= 3:
        kept_splits = ["train", "valid", "test"]
        fallback_reason = ""
    elif min_class_speakers >= 2:
        kept_splits = ["train", "valid"]
        fallback_reason = "Test split dropped because at least one class has fewer than 3 unique speakers."
    else:
        raise ValueError("Cannot create leakage-aware evaluation split with both classes represented.")

    assignments: Dict[str, str] = {}
    for label, speakers in by_label.items():
        n = len(speakers)
        if kept_splits == ["train", "valid", "test"]:
            n_valid = max(1, round(n * valid_ratio))
            n_test = max(1, round(n * (1.0 - train_ratio - valid_ratio)))
            if n_valid + n_test >= n:
                n_valid = 1
                n_test = 1
            valid_speakers = speakers[:n_valid]
            test_speakers = speakers[n_valid : n_valid + n_test]
            train_speakers = speakers[n_valid + n_test :]
            for speaker_id in train_speakers:
                assignments[speaker_id] = "train"
            for speaker_id in valid_speakers:
                assignments[speaker_id] = "valid"
            for speaker_id in test_speakers:
                assignments[speaker_id] = "test"
        else:
            n_valid = max(1, round(n * 0.20))
            if n_valid >= n:
                n_valid = 1
            valid_speakers = speakers[:n_valid]
            train_speakers = speakers[n_valid:]
            for speaker_id in train_speakers:
                assignments[speaker_id] = "train"
            for speaker_id in valid_speakers:
                assignments[speaker_id] = "valid"

    split_df["datasplit"] = split_df["speaker_id"].map(assignments)
    split_df["datasplit_valid"] = split_df["datasplit"].isin(kept_splits)
    if split_df["datasplit"].isna().any():
        missing = split_df[split_df["datasplit"].isna()]["speaker_id"].drop_duplicates().tolist()
        raise ValueError(f"Could not assign split for speakers: {missing[:10]}")

    split_summary = (
        split_df.groupby(["datasplit", "label_name"], dropna=False)
        .agg(clips=("audio_file", "count"), speakers=("speaker_id", "nunique"))
        .reset_index()
        .sort_values(["datasplit", "label_name"])
    )
    missing_representations: List[str] = []
    for split in kept_splits:
        labels_in_split = set(split_df[split_df["datasplit"] == split]["label"].astype(int).tolist())
        if labels_in_split != {0, 1}:
            missing_representations.append(f"{split}: {sorted(labels_in_split)}")
    if missing_representations:
        raise ValueError(
            "Leakage-aware split failed class-representation check: "
            + "; ".join(missing_representations)
        )

    split_decision = {
        "seed": seed,
        "kept_splits": ",".join(kept_splits),
        "fallback_reason": fallback_reason,
        "speaker_leakage_count": int(split_df.groupby("speaker_id")["datasplit"].nunique().gt(1).sum()),
    }
    return split_df, split_summary, split_decision


def write_split_report(split_summary: pd.DataFrame, split_decision: Dict[str, object], out_dir: Path) -> None:
    """Save split summary table and Markdown decision report."""
    reports_dir = Path("reports")
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_summary.to_csv(tables_dir / "split_summary.csv", index=False)

    kept_splits = str(split_decision["kept_splits"]).split(",")
    lines = [
        "# Phase 1 Split Fix Report",
        "",
        f"- Fixed random seed: {split_decision['seed']}",
        "- Split unit: `speaker_id`",
        f"- Kept splits: {', '.join(kept_splits)}",
        f"- Speaker leakage count: {split_decision['speaker_leakage_count']}",
    ]
    if split_decision.get("fallback_reason"):
        lines.append(f"- Fallback decision: {split_decision['fallback_reason']}")
    else:
        lines.append("- Fallback decision: none; three-way speaker-level split was feasible.")
    lines.extend(
        [
            "",
            "Split summary:",
            "",
            split_summary.to_string(index=False),
            "",
            "Requirements checked:",
            "",
            "- No speaker appears in more than one split.",
            "- Every kept split contains both `dementia` and `non_dementia`.",
            "- Original source split values are preserved in `source_datasplit`.",
            "",
            f"Corrected metadata was saved to `{out_dir / 'metadata_fixed_splits.csv'}`.",
        ]
    )
    (reports_dir / "phase1_split_fix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_match_resolution_report(resolution_log: pd.DataFrame, out_dir: Path) -> None:
    """Save transparent metadata-to-audio match resolution artifacts."""
    reports_dir = Path("reports")
    tables_dir = reports_dir / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    log_path = tables_dir / "ambiguous_match_resolution_log.csv"
    resolution_log.to_csv(log_path, index=False)

    unresolved = resolution_log[resolution_log["needs_review"].astype(bool)].copy()
    unresolved.to_csv(out_dir / "unresolved_match_review.csv", index=False)

    ambiguous = resolution_log[resolution_log["candidate_count"] > 1].copy()
    if ambiguous.empty:
        rule_counts = pd.DataFrame(columns=["rule_used", "count"])
    else:
        rule_counts = ambiguous["rule_used"].fillna("unknown").value_counts().reset_index()
        rule_counts.columns = ["rule_used", "count"]

    lines = [
        "# Phase 1 Match Resolution Report",
        "",
        "Metadata-to-audio matching uses deterministic ranked rules.",
        "",
        "Rule order:",
        "",
        "1. Exact audio filename when metadata provides one.",
        "2. Preferred timing bucket from metadata URL columns when candidate filenames contain `_0`, `_5`, `_10`, or `_15`.",
        "3. Exact speaker folder match.",
        "4. Exact normalized filename stem.",
        "5. Normalized filename similarity with deterministic lexicographic tie-breaks.",
        "",
        f"- Total source metadata rows audited: {len(resolution_log)}",
        f"- Ambiguous rows with multiple candidates: {len(ambiguous)}",
        f"- Rows still requiring manual review: {len(unresolved)}",
        f"- Resolution log: `{log_path}`",
        f"- Manual review file: `{out_dir / 'unresolved_match_review.csv'}`",
        "",
        "Ambiguous rule counts:",
        "",
        rule_counts.to_string(index=False) if not rule_counts.empty else "No ambiguous rows were found.",
        "",
        "Rows marked `needs_review=True` are retained only with an explicit audit trail.",
    ]
    (reports_dir / "phase1_match_resolution_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_metadata_pipeline(
    metadata_path: Path = CFG.metadata_xlsx,
    raw_audio_dir: Path = CFG.raw_audio_dir,
) -> pd.DataFrame:
    """Clean metadata, deterministically match audio, assign splits, and export outputs."""
    metadata_path = metadata_path.resolve()
    raw_audio_dir = raw_audio_dir.resolve()
    out_dir = metadata_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_excel(metadata_path, engine="openpyxl")
    df.columns = [snake_case(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(clean_text)

    if "name" not in df.columns:
        raise ValueError("Metadata must include a 'name' column.")

    df = df.reset_index(names="metadata_index")
    df["speaker_id"] = df["name"].map(derive_speaker_id)

    if "dementia_type" not in df.columns:
        df["dementia_type"] = ""
    labels = df["dementia_type"].map(derive_binary_label)
    df["label"] = labels.map(lambda x: x[0])
    df["label_name"] = labels.map(lambda x: x[1])

    if "datasplit" not in df.columns:
        df["datasplit"] = ""
    df["datasplit"] = df["datasplit"].map(normalize_datasplit)
    df["source_datasplit"] = df["datasplit"]
    df["datasplit_valid"] = df["datasplit"].isin(CFG.target_splits)

    audio_df = scan_audio_files(raw_audio_dir)
    lookup = build_audio_lookup(audio_df)

    previous_audio_by_metadata: Dict[str, str] = {}
    previous_metadata_path = out_dir / "metadata_clean.csv"
    if previous_metadata_path.exists():
        try:
            previous_df = pd.read_csv(previous_metadata_path)
            if {"metadata_index", "audio_file"}.issubset(previous_df.columns):
                previous_audio_by_metadata = dict(
                    zip(previous_df["metadata_index"].astype(str), previous_df["audio_file"].astype(str))
                )
        except Exception:
            previous_audio_by_metadata = {}

    matched_audio_idx: List[Optional[int]] = []
    match_statuses: List[str] = []
    all_candidates: List[List[int]] = []
    match_rules: List[str] = []
    match_confidences: List[str] = []
    match_review_flags: List[bool] = []
    match_score_summaries: List[str] = []
    resolution_rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        decision = choose_match(row, audio_df, lookup)
        matched_audio_idx.append(decision.best_idx)
        match_statuses.append(decision.status)
        all_candidates.append(decision.candidates)
        match_rules.append(decision.rule_used)
        match_confidences.append(decision.confidence)
        match_review_flags.append(decision.needs_review)
        match_score_summaries.append(decision.score_summary)

        selected_audio_file = ""
        if decision.best_idx is not None and not audio_df.empty:
            selected_audio_file = str(Path(audio_df.loc[int(decision.best_idx), "audio_file"]).resolve())
        candidate_audio_files = [
            str(Path(audio_df.loc[int(candidate_idx), "audio_file"]).resolve())
            for candidate_idx in decision.candidates
            if not audio_df.empty
        ]
        metadata_key = str(row.get("metadata_index"))
        previous_audio_file = previous_audio_by_metadata.get(metadata_key, "")
        resolution_rows.append(
            {
                "metadata_index": row.get("metadata_index"),
                "speaker_id": row.get("speaker_id"),
                "name": row.get("name"),
                "candidate_count": len(decision.candidates),
                "selected_audio_file": selected_audio_file,
                "previous_audio_file": previous_audio_file,
                "changed_from_previous": bool(previous_audio_file and selected_audio_file != previous_audio_file),
                "rule_used": decision.rule_used,
                "confidence": decision.confidence,
                "needs_review": decision.needs_review,
                "match_status": decision.status,
                "candidate_audio_files": " | ".join(candidate_audio_files),
                "score_summary": decision.score_summary,
            }
        )

    df["audio_row_idx"] = matched_audio_idx
    df["match_status"] = match_statuses
    df["match_candidate_count"] = [len(c) for c in all_candidates]
    df["match_rule_used"] = match_rules
    df["match_confidence"] = match_confidences
    df["match_needs_review"] = match_review_flags
    df["match_score_summary"] = match_score_summaries
    df["audio_exists"] = df["audio_row_idx"].notna()

    def pick_audio_field(row: pd.Series, field: str) -> object:
        idx = row.get("audio_row_idx")
        if pd.isna(idx) or audio_df.empty:
            return None
        return audio_df.loc[int(idx), field]

    df["audio_file"] = df.apply(lambda r: pick_audio_field(r, "audio_file"), axis=1)
    df["audio_class_from_folder"] = df.apply(lambda r: pick_audio_field(r, "audio_class_from_folder"), axis=1)
    df["duration_sec"] = df.apply(lambda r: pick_audio_field(r, "duration_sec"), axis=1)
    df["sample_rate"] = df.apply(lambda r: pick_audio_field(r, "sample_rate"), axis=1)

    df["label_from_audio_folder"] = df["audio_class_from_folder"].map(lambda x: binary_label_from_folder(x)[0])
    df["label_conflict"] = (
        df["label"].notna()
        & df["label_from_audio_folder"].notna()
        & (df["label"] != df["label_from_audio_folder"])
    )

    df = append_audio_only_rows(df, audio_df)
    df, split_summary, split_decision = assign_speaker_stratified_splits(df)
    write_split_report(split_summary, split_decision, out_dir)
    kept_splits = str(split_decision["kept_splits"]).split(",")

    resolution_log_df = pd.DataFrame(resolution_rows)
    write_match_resolution_report(resolution_log_df, out_dir)

    issue_rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        if not row["audio_exists"]:
            collect_quality_issues(issue_rows, "unmatched_audio", "No audio file matched to metadata row", row)
        if "ambiguous" in clean_text(row["match_status"]):
            issue_type = (
                "ambiguous_match_needs_review"
                if bool(row.get("match_needs_review"))
                else "ambiguous_match_resolved"
            )
            collect_quality_issues(
                issue_rows,
                issue_type,
                f"Rule: {row.get('match_rule_used')}; confidence: {row.get('match_confidence')}",
                row,
            )
        if row["match_candidate_count"] > 1:
            collect_quality_issues(issue_rows, "duplicate_match_candidates", "Metadata row matched multiple files", row)
        if bool(row["label_conflict"]):
            collect_quality_issues(issue_rows, "label_conflict", "Metadata label conflicts with folder label", row)
        if not bool(row["datasplit_valid"]):
            collect_quality_issues(issue_rows, "invalid_datasplit", f"datasplit is not one of {kept_splits}", row)
        if pd.isna(row.get("label")):
            collect_quality_issues(issue_rows, "missing_label", "Could not derive binary label from dementia_type", row)

    speaker_leak_df = enforce_speaker_split_consistency(df)
    for _, row in speaker_leak_df.iterrows():
        collect_quality_issues(
            issue_rows,
            "speaker_split_leakage",
            "Speaker appears in more than one split",
            row,
        )

    unmatched_df = df[~df["audio_exists"]].copy()
    duplicates_df = df[df["match_candidate_count"] > 1].copy()
    label_conflicts_df = df[df["label_conflict"]].copy()
    issue_columns = ["issue_type", "detail", "metadata_index", "speaker_id", "name", "datasplit", "audio_file"]
    issues_df = pd.DataFrame(issue_rows, columns=issue_columns).drop_duplicates()

    export_df = df.copy()
    export_df["audio_file"] = export_df["audio_file"].map(lambda x: str(x) if pd.notna(x) else "")
    export_df.to_csv(out_dir / "metadata_clean.csv", index=False)
    export_df.to_csv(out_dir / "metadata_fixed_splits.csv", index=False)
    export_df[export_df["datasplit"] == "train"].to_csv(out_dir / "train.csv", index=False)
    export_df[export_df["datasplit"] == "valid"].to_csv(out_dir / "valid.csv", index=False)
    export_df[export_df["datasplit"] == "test"].to_csv(out_dir / "test.csv", index=False)
    unmatched_df.to_csv(out_dir / "unmatched_rows.csv", index=False)
    duplicates_df.to_csv(out_dir / "duplicate_matches.csv", index=False)
    label_conflicts_df.to_csv(out_dir / "label_conflicts.csv", index=False)
    issues_df.to_csv(out_dir / "data_quality_issues.csv", index=False)
    review_needed_issue_types = {
        "ambiguous_match_needs_review",
        "unmatched_audio",
        "label_conflict",
        "missing_label",
        "speaker_split_leakage",
        "invalid_datasplit",
    }
    resolved_issues_df = issues_df.copy()
    resolved_issues_df["resolution_status"] = resolved_issues_df["issue_type"].map(
        lambda issue_type: "manual_review_needed"
        if issue_type in review_needed_issue_types
        else "deterministically_resolved_or_informational"
    )
    resolved_issues_df.to_csv(out_dir / "data_quality_issues_resolved.csv", index=False)

    print("Metadata pipeline completed.")
    print(f"Rows: {len(df)}")
    print(f"Matched audio: {int(df['audio_exists'].sum())}/{len(df)}")
    print(f"Unmatched: {len(unmatched_df)}")
    print(f"Ambiguous or duplicate matches: {len(duplicates_df)}")
    print(f"Label conflicts: {len(label_conflicts_df)}")
    print(f"Invalid splits: {int((~df['datasplit_valid']).sum())}")
    print(f"Kept splits: {', '.join(kept_splits)}")
    print(f"Speaker leakage rows: {len(speaker_leak_df)}")
    return df


def parse_args() -> argparse.Namespace:
    """CLI arguments for week-1 metadata pipeline."""
    parser = argparse.ArgumentParser(description="Run metadata cleaning and audio matching pipeline.")
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(CFG.metadata_xlsx),
        help="Path to raw metadata xlsx file.",
    )
    parser.add_argument(
        "--raw-audio-dir",
        type=str,
        default=str(CFG.raw_audio_dir),
        help="Path to raw audio root directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_metadata_pipeline(Path(args.metadata_path), Path(args.raw_audio_dir))
