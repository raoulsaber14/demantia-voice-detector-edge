"""Expand unresolved metadata/audio candidate rows into one-audio-per-row tables.

This is a governance proposal generator only. It does not update official
manifests, adjudication decisions, Phase 4 logic, thresholds, or models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADJUDICATION_CSV = ROOT / "reports/tables/phase6_5_uncertain_row_adjudication.csv"
DEFAULT_RESOLUTION_LOG_CSV = ROOT / "reports/tables/ambiguous_match_resolution_log.csv"
DEFAULT_TRUSTED_MANIFEST_CSV = ROOT / "reports/tables/trusted_subset_manifest.csv"
DEFAULT_FULL_GOVERNED_MANIFEST_CSV = ROOT / "reports/tables/full_governed_subset_manifest.csv"
DEFAULT_FEATURE_LOG_CSV = ROOT / "reports/tables/phase3_feature_extraction_log.csv"
DEFAULT_GOVERNANCE_MANIFEST_CSV = ROOT / "reports/phase3_governance_manifest.csv"
DEFAULT_MANUAL_OBSERVATIONS_CSV = ROOT / "reports/tables/phase6_5_audio_triage_manual_observations.csv"
DEFAULT_EXPANDED_CSV = ROOT / "reports/tables/phase6_5_expanded_unresolved_audio_rows.csv"
DEFAULT_PROPOSAL_CSV = ROOT / "reports/tables/phase6_5_governed_manifest_expansion_proposal.csv"
DEFAULT_SUMMARY_MD = ROOT / "reports/phase6_5_expanded_unresolved_audio_rows_summary.md"

AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".mp3", ".m4a", ".ogg", ".aif", ".aiff"}
GOVERNANCE_NOTE = (
    "EXPANDED_UNRESOLVED_METADATA_AUDIO_MATCH: governed-only/caution proposal; "
    "not trusted; human verification required before use."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand unresolved candidate audio rows into one row per real audio file.")
    parser.add_argument("--adjudication-csv", type=Path, default=DEFAULT_ADJUDICATION_CSV)
    parser.add_argument("--resolution-log-csv", type=Path, default=DEFAULT_RESOLUTION_LOG_CSV)
    parser.add_argument("--trusted-manifest-csv", type=Path, default=DEFAULT_TRUSTED_MANIFEST_CSV)
    parser.add_argument("--full-governed-manifest-csv", type=Path, default=DEFAULT_FULL_GOVERNED_MANIFEST_CSV)
    parser.add_argument("--feature-log-csv", type=Path, default=DEFAULT_FEATURE_LOG_CSV)
    parser.add_argument("--governance-manifest-csv", type=Path, default=DEFAULT_GOVERNANCE_MANIFEST_CSV)
    parser.add_argument("--manual-observations-csv", type=Path, default=DEFAULT_MANUAL_OBSERVATIONS_CSV)
    parser.add_argument("--expanded-csv", type=Path, default=DEFAULT_EXPANDED_CSV)
    parser.add_argument("--proposal-csv", type=Path, default=DEFAULT_PROPOSAL_CSV)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    return parser.parse_args()


def is_sidecar_path(path_text: str) -> bool:
    return Path(str(path_text)).name.startswith("._")


def resolve_audio_path(path_text: Any) -> Path:
    text = "" if path_text is None or pd.isna(path_text) else str(path_text).strip()
    if not text:
        return ROOT / "__missing_audio_path__"
    path = Path(text)
    if path.is_absolute():
        return path
    direct = ROOT / path
    if direct.exists():
        return direct
    data_pos = text.find("data/")
    if data_pos >= 0:
        candidate = ROOT / text[data_pos:]
        if candidate.exists():
            return candidate
    return direct


def repo_relative_path(path_text: Any) -> str:
    path = resolve_audio_path(path_text)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalized_path_key(path_text: Any) -> str:
    return str(resolve_audio_path(path_text))


def split_candidate_list(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    parts = [p.strip() for p in str(value).split("|")]
    return [p for p in parts if p]


def extract_note_value(notes: Any, key: str, next_keys: list[str]) -> str:
    if notes is None or pd.isna(notes):
        return ""
    text = str(notes)
    start = text.find(f"{key}=")
    if start < 0:
        return ""
    start += len(key) + 1
    end = len(text)
    for next_key in next_keys:
        marker = f"; {next_key}="
        pos = text.find(marker, start)
        if pos >= 0:
            end = min(end, pos)
    return text[start:end].strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    show = df.where(pd.notna(df), "")
    lines = [
        "| " + " | ".join(str(c) for c in show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in show.columns) + " |")
    return "\n".join(lines)


def load_resolution_by_id(path: Path) -> dict[str, pd.Series]:
    if not path.exists():
        return {}
    log = pd.read_csv(path)
    if "metadata_index" not in log.columns:
        return {}
    return {str(row["metadata_index"]).strip(): row for _, row in log.iterrows()}


def load_manual_observations(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    obs = pd.read_csv(path)
    if "row_id" not in obs.columns or "observation" not in obs.columns:
        return {}
    out: dict[str, list[str]] = {}
    for _, row in obs.iterrows():
        row_id = str(row.get("row_id", "")).strip()
        observation = str(row.get("observation", "")).strip()
        if row_id and observation:
            out.setdefault(row_id, []).append(observation)
    return out


def row_candidate_record(row: pd.Series, resolution_by_id: dict[str, pd.Series]) -> dict[str, Any]:
    row_id = str(row.get("row_id", "")).strip()
    log_row = resolution_by_id.get(row_id)
    notes = row.get("notes", "")

    selected = ""
    candidates: list[str] = []
    speaker_name = ""
    rule_used = ""
    confidence = ""
    match_status = ""
    score_summary = ""
    candidate_count_reported: int | None = None

    if log_row is not None:
        selected = str(log_row.get("selected_audio_file", "")).strip()
        candidates = split_candidate_list(log_row.get("candidate_audio_files", ""))
        speaker_name = str(log_row.get("name", "")).strip()
        rule_used = str(log_row.get("rule_used", "")).strip()
        confidence = str(log_row.get("confidence", "")).strip()
        match_status = str(log_row.get("match_status", "")).strip()
        score_summary = str(log_row.get("score_summary", "")).strip()
        candidate_count_reported = safe_int(log_row.get("candidate_count", None), default=0)

    if not selected:
        selected = extract_note_value(notes, "selected_audio_file", ["candidate_audio_files", "score_summary"])
    if not selected:
        selected = str(row.get("audio_path", "")).strip()
    if not candidates:
        candidates = split_candidate_list(
            extract_note_value(notes, "candidate_audio_files", ["score_summary", "required_manual_action"])
        )
    if not speaker_name:
        folder = Path(selected).parent.name if selected else ""
        speaker_name = folder
    if not rule_used:
        rule_used = extract_note_value(notes, "rule_used", ["confidence"])
    if not confidence:
        confidence = extract_note_value(notes, "confidence", ["needs_review_in_match_log"])
    if not match_status:
        match_status = extract_note_value(notes, "match_status", ["match_rank_by_score_summary"])
    if not score_summary:
        score_summary = extract_note_value(notes, "score_summary", ["required_manual_action"])
    if not candidate_count_reported:
        candidate_count_reported = safe_int(extract_note_value(notes, "candidate_count", ["rule_used"]), default=len(candidates))

    return {
        "row_id": row_id,
        "selected_audio_file": selected,
        "candidate_audio_files": candidates,
        "candidate_count_reported": candidate_count_reported,
        "speaker_name": speaker_name,
        "rule_used": rule_used,
        "confidence": confidence,
        "match_status": match_status,
        "score_summary": score_summary,
    }


def build_reference_sets(paths: argparse.Namespace) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {
        "trusted": set(),
        "full_governed": set(),
        "feature_log": set(),
        "governance_manifest": set(),
    }
    if paths.trusted_manifest_csv.exists():
        trusted = pd.read_csv(paths.trusted_manifest_csv, usecols=lambda c: c in {"audio_file"})
        refs["trusted"] = {normalized_path_key(v) for v in trusted.get("audio_file", [])}
    if paths.full_governed_manifest_csv.exists():
        governed = pd.read_csv(paths.full_governed_manifest_csv, usecols=lambda c: c in {"audio_file"})
        refs["full_governed"] = {normalized_path_key(v) for v in governed.get("audio_file", [])}
    if paths.feature_log_csv.exists():
        flog = pd.read_csv(paths.feature_log_csv, usecols=lambda c: c in {"audio_file"})
        refs["feature_log"] = {normalized_path_key(v) for v in flog.get("audio_file", [])}
    if paths.governance_manifest_csv.exists():
        govm = pd.read_csv(paths.governance_manifest_csv, usecols=lambda c: c in {"audio_file"})
        refs["governance_manifest"] = {normalized_path_key(v) for v in govm.get("audio_file", [])}
    return refs


def label_to_int(label_text: Any) -> int | None:
    text = "" if label_text is None or pd.isna(label_text) else str(label_text).strip().lower()
    if text in {"dementia", "1", "positive"}:
        return 1
    if text in {"control", "non_dementia", "nodementia", "no_dementia", "0", "negative"}:
        return 0
    return None


def build_expanded_rows(
    unresolved: pd.DataFrame,
    resolution_by_id: dict[str, pd.Series],
    refs: dict[str, set[str]],
    manual_obs: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    summary = {
        "source_rows": int(len(unresolved)),
        "sidecar_ignored": 0,
        "within_row_repeated_paths_removed": 0,
        "missing_or_non_audio_skipped": 0,
    }

    for _, row in unresolved.sort_values("row_id").iterrows():
        record = row_candidate_record(row, resolution_by_id)
        selected_key = normalized_path_key(record["selected_audio_file"])
        candidates = record["candidate_audio_files"] or []

        ordered: list[tuple[str, str, bool]] = []
        # Always consider selected first, then candidate list.
        ordered.append((record["selected_audio_file"], "selected", True))
        for candidate in candidates:
            ordered.append((candidate, "candidate", normalized_path_key(candidate) == selected_key))

        seen_within_row: set[str] = set()
        unique_items: list[tuple[str, str, bool, bool]] = []
        for path_text, source_type, is_selected in ordered:
            if is_sidecar_path(path_text):
                summary["sidecar_ignored"] += 1
                continue
            resolved = resolve_audio_path(path_text)
            key = str(resolved)
            if key in seen_within_row:
                summary["within_row_repeated_paths_removed"] += 1
                # If the selected file appears again in the candidate list, the
                # selected row notes capture that below.
                continue
            seen_within_row.add(key)
            ext = resolved.suffix.lower()
            exists = resolved.exists() and resolved.is_file()
            if not exists or ext not in AUDIO_EXTENSIONS:
                summary["missing_or_non_audio_skipped"] += 1
                continue
            unique_items.append((path_text, source_type, is_selected, any(normalized_path_key(c) == key for c in candidates)))

        candidate_group_size = len(unique_items)
        for ordinal, (path_text, source_type, is_selected, also_in_candidate_list) in enumerate(unique_items, start=1):
            resolved = resolve_audio_path(path_text)
            key = str(resolved)
            label_name = str(row.get("label_current", "")).strip()
            label = label_to_int(label_name)
            expansion_status = "expanded_from_selected_file" if is_selected else "expanded_from_candidate_file"
            obs = manual_obs.get(record["row_id"], [])
            notes = [
                "same-speaker multiple rows are allowed",
                "not a duplicate solely because speaker_id repeats",
                f"also_in_candidate_list={also_in_candidate_list}",
            ]
            if obs:
                notes.append("manual_observations=" + "|".join(obs))
            rows.append(
                {
                    "row_id": safe_int(record["row_id"]),
                    "expanded_row_id": f"phase6_5_unresolved_{int(record['row_id']):03d}_{ordinal:02d}",
                    "speaker_name": record["speaker_name"],
                    "speaker_id": str(row.get("speaker_id_current", "")).strip(),
                    "label": label,
                    "label_name": label_name,
                    "audio_path": repo_relative_path(path_text),
                    "audio_file_abs": str(resolved),
                    "file_exists": True,
                    "file_size_bytes": resolved.stat().st_size,
                    "source_type": "selected" if is_selected else "candidate",
                    "was_original_selected": bool(is_selected),
                    "original_selected_audio_file": repo_relative_path(record["selected_audio_file"]),
                    "candidate_group_size": candidate_group_size,
                    "reported_candidate_count": record["candidate_count_reported"],
                    "source_of_expansion": "phase6_5_uncertain_row_adjudication + ambiguous_match_resolution_log",
                    "expansion_status": expansion_status,
                    "governance_note": GOVERNANCE_NOTE,
                    "manual_verification_status": "needs_manual_review",
                    "decision_carryover": str(row.get("decision", "")),
                    "review_status_carryover": str(row.get("review_status", "")),
                    "match_rule_used": record["rule_used"],
                    "match_confidence": record["confidence"],
                    "match_status": record["match_status"],
                    "audio_in_trusted_manifest": key in refs["trusted"],
                    "audio_in_full_governed_manifest": key in refs["full_governed"],
                    "audio_in_phase3_feature_log": key in refs["feature_log"],
                    "audio_in_phase3_governance_manifest": key in refs["governance_manifest"],
                    "notes": "; ".join(notes),
                }
            )
    return pd.DataFrame(rows), summary


def build_proposal_table(full_governed_path: Path, expanded_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    existing_audio_keys: set[str] = set()
    if full_governed_path.exists():
        full = pd.read_csv(full_governed_path)
        for _, row in full.iterrows():
            key = normalized_path_key(row.get("audio_file", ""))
            existing_audio_keys.add(key)
            rows.append(
                {
                    "proposal_record_type": "current_full_governed_row",
                    "proposal_record_id": str(row.get("metadata_index", "")),
                    "metadata_index": row.get("metadata_index", ""),
                    "original_unresolved_row_id": "",
                    "expanded_row_id": "",
                    "speaker_id": row.get("speaker_id", ""),
                    "label": row.get("label", ""),
                    "label_name": row.get("label_name", ""),
                    "audio_path": repo_relative_path(row.get("audio_file", "")),
                    "source_type": "existing_manifest",
                    "feature_row_available": True,
                    "phase4_ready_without_upstream_rebuild": True,
                    "manual_verification_status": "existing_governed_status",
                    "governance_note": row.get("governance_note", "Existing current full governed row."),
                    "notes": "Existing row copied into compact proposal index; official manifest was not modified.",
                }
            )

    for _, row in expanded_df.iterrows():
        key = normalized_path_key(row["audio_file_abs"])
        if key in existing_audio_keys:
            continue
        rows.append(
            {
                "proposal_record_type": "proposed_expanded_unresolved_audio_row",
                "proposal_record_id": row["expanded_row_id"],
                "metadata_index": "",
                "original_unresolved_row_id": row["row_id"],
                "expanded_row_id": row["expanded_row_id"],
                "speaker_id": row["speaker_id"],
                "label": row["label"],
                "label_name": row["label_name"],
                "audio_path": row["audio_path"],
                "source_type": row["source_type"],
                "feature_row_available": bool(row["audio_in_phase3_feature_log"]),
                "phase4_ready_without_upstream_rebuild": False,
                "manual_verification_status": row["manual_verification_status"],
                "governance_note": row["governance_note"],
                "notes": (
                    "Proposal-only expanded row; requires upstream governance/feature-table rebuild "
                    "before any model evaluation use."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    path: Path,
    expanded_df: pd.DataFrame,
    proposal_df: pd.DataFrame,
    summary_counts: dict[str, int],
    args: argparse.Namespace,
    manual_observation_row_count: int,
) -> None:
    per_speaker = (
        expanded_df.groupby(["speaker_id", "speaker_name"], as_index=False)
        .agg(expanded_rows=("expanded_row_id", "count"), source_rows=("row_id", "nunique"))
        .sort_values(["expanded_rows", "speaker_id"], ascending=[False, True])
    )
    status_counts = expanded_df["expansion_status"].value_counts().rename_axis("expansion_status").reset_index(name="rows")
    source_counts = expanded_df["source_type"].value_counts().rename_axis("source_type").reset_index(name="rows")
    represented_counts = pd.DataFrame(
        [
            {"reference": "trusted_manifest", "expanded_rows_already_present": int(expanded_df["audio_in_trusted_manifest"].sum())},
            {"reference": "full_governed_manifest", "expanded_rows_already_present": int(expanded_df["audio_in_full_governed_manifest"].sum())},
            {"reference": "phase3_feature_log", "expanded_rows_already_present": int(expanded_df["audio_in_phase3_feature_log"].sum())},
            {"reference": "phase3_governance_manifest", "expanded_rows_already_present": int(expanded_df["audio_in_phase3_governance_manifest"].sum())},
        ]
    )
    proposal_counts = proposal_df["proposal_record_type"].value_counts().rename_axis("proposal_record_type").reset_index(name="rows")

    lines = [
        "# Phase 6.5 Expanded Unresolved Audio Rows Summary",
        "",
        "## Inputs",
        "",
        f"- Unresolved adjudication file: `{args.adjudication_csv.relative_to(ROOT)}`.",
        f"- Ambiguous match log: `{args.resolution_log_csv.relative_to(ROOT)}`.",
        f"- Trusted manifest reference: `{args.trusted_manifest_csv.relative_to(ROOT)}`.",
        f"- Full governed manifest reference: `{args.full_governed_manifest_csv.relative_to(ROOT)}`.",
        "",
        "## Expansion Result",
        "",
        f"- Unresolved source rows read: **{summary_counts['source_rows']}**.",
        f"- Expanded real audio rows created: **{len(expanded_df)}**.",
        f"- Fake Mac sidecar files ignored: **{summary_counts['sidecar_ignored']}**.",
        f"- Exact repeated file paths removed within source row: **{summary_counts['within_row_repeated_paths_removed']}**.",
        f"- Missing or non-audio paths skipped: **{summary_counts['missing_or_non_audio_skipped']}**.",
        "",
        "## Expansion Status Counts",
        "",
        markdown_table(status_counts),
        "",
        "## Source Type Counts",
        "",
        markdown_table(source_counts),
        "",
        "## Existing Representation Checks",
        "",
        markdown_table(represented_counts),
        "",
        "## Expanded Rows Per Speaker",
        "",
        markdown_table(per_speaker),
        "",
        "## Why This Does Not Mean Duplicate Records",
        "",
        "- The expansion follows a one-audio-per-row rule.",
        "- Same speaker appearing in multiple rows is allowed and expected.",
        "- Same-speaker rows are not collapsed because speaker identity is the grouping field for later speaker-safe evaluation, not a uniqueness constraint.",
        "- Similarity between same-speaker clips is not treated as duplication evidence.",
        "- Only exact repeated file paths within the same unresolved source row are removed from the expanded table.",
        "",
        "## Governance Status",
        "",
        "- Expanded rows are **not trusted**.",
        "- Expanded rows are **not inserted into official manifests**.",
        f"- Manual real-audio/non-duplicate observations are recorded for **{manual_observation_row_count}** unresolved source rows.",
        "- Expanded rows remain governed-only/caution proposal rows pending final source-reference/governance adjudication.",
        "- Any future model use would require upstream governance and feature-table regeneration.",
        "",
        "## Governed Manifest Expansion Proposal",
        "",
        f"- Proposal table: `{args.proposal_csv.relative_to(ROOT)}`.",
        "- This is a compact proposal index, not a Phase 4 feature-ready manifest.",
        "- Existing full-governed rows are listed once; proposed expanded rows whose exact audio path is not already represented in the full governed manifest are appended as proposal-only rows.",
        "",
        markdown_table(proposal_counts),
        "",
        "## Output Tables",
        "",
        f"- Expanded one-audio-per-row table: `{args.expanded_csv.relative_to(ROOT)}`.",
        f"- Governed manifest expansion proposal: `{args.proposal_csv.relative_to(ROOT)}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.adjudication_csv.exists():
        raise FileNotFoundError(args.adjudication_csv)
    adjudication = pd.read_csv(args.adjudication_csv)
    required = {"row_id", "audio_path", "speaker_id_current", "label_current", "decision", "review_status", "notes"}
    missing = sorted(required - set(adjudication.columns))
    if missing:
        raise ValueError(f"Adjudication CSV missing required columns: {missing}")

    unresolved = adjudication[
        adjudication["decision"].astype(str).eq("needs_more_review")
        | adjudication["review_status"].astype(str).isin(["needs_more_info", "cannot_resolve"])
    ].copy()
    if unresolved.empty:
        raise ValueError("No unresolved adjudication rows found.")

    refs = build_reference_sets(args)
    resolution_by_id = load_resolution_by_id(args.resolution_log_csv)
    manual_obs = load_manual_observations(args.manual_observations_csv)
    expanded_df, summary_counts = build_expanded_rows(unresolved, resolution_by_id, refs, manual_obs)
    proposal_df = build_proposal_table(args.full_governed_manifest_csv, expanded_df)

    args.expanded_csv.parent.mkdir(parents=True, exist_ok=True)
    args.expanded_csv.write_text(expanded_df.to_csv(index=False), encoding="utf-8")
    args.proposal_csv.write_text(proposal_df.to_csv(index=False), encoding="utf-8")
    write_summary(
        args.summary_md,
        expanded_df,
        proposal_df,
        summary_counts,
        args,
        manual_observation_row_count=len(manual_obs),
    )

    print(f"Unresolved source rows: {len(unresolved)}")
    print(f"Expanded audio rows: {len(expanded_df)}")
    print(f"Sidecar files ignored: {summary_counts['sidecar_ignored']}")
    print(f"Within-row repeated paths removed: {summary_counts['within_row_repeated_paths_removed']}")
    print(f"Proposal rows: {len(proposal_df)}")
    print(f"Wrote: {args.expanded_csv}")
    print(f"Wrote: {args.proposal_csv}")
    print(f"Wrote: {args.summary_md}")


if __name__ == "__main__":
    main()
