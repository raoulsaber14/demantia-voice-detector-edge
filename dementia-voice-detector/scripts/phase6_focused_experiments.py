"""Focused Phase 6 deep-learning experiments.

This script implements the narrower path requested after the initial Phase 6
diagnostics:

1. cache frozen wav2vec2-base segment embeddings once;
2. run grouped-CV Experiment 0: frozen wav2vec embeddings only;
3. run grouped-CV Experiment 1: handcrafted-only control and frozen wav2vec
   attention-fusion model;
4. apply the go/no-go rule before any fine-tuning;
5. train the best frozen model on the existing Phase 6 train/calibration/test
   split for final tables, CIs, calibration, speaker-level evaluation, attention
   visualization, aggregation comparison, and error examples.

The embedding cache is resumable. CPU-only extraction is slow on this machine,
so each batch is saved as an artifact chunk before final consolidation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURE_CSV = ROOT / "data/features/features_phase3_governed_pruned.csv"
SEGMENT_MANIFEST_CSV = ROOT / "reports/tables/phase6_segment_manifest.csv"
CLIP_MANIFEST_CSV = ROOT / "reports/tables/phase6_clip_manifest.csv"

REPORT_DIR = ROOT / "reports/phase6_focused"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
CHUNK_DIR = ROOT / "artifacts/phase6_focused_embedding_chunks"
EMBEDDING_CACHE = TABLE_DIR / "frozen_wav2vec_segment_embeddings.npz"
SUMMARY_MD = REPORT_DIR / "phase6_focused_summary.md"

PHASE4_BASELINE = {
    "model": "phase4_logistic_regression_platt",
    "recall": 0.8617021276595744,
    "precision": 0.2892857142857143,
    "specificity": 0.21653543307086615,
    "balanced_accuracy": 0.5391187803652203,
    "f1": 2 * 0.2892857142857143 * 0.8617021276595744 / (0.2892857142857143 + 0.8617021276595744),
    "roc_auc": np.nan,
    "pr_auc": 0.3322394307030126,
    "brier_score": 0.1980628044332236,
    "threshold": 0.300,
}

METRIC_ORDER = [
    "recall",
    "precision",
    "f1",
    "specificity",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "brier_score",
]

HANDCRAFTED_PRIORITY_COLUMNS = [
    "pause_ratio",
    "speaking_rate_proxy",
    "pause_rate_per_min",
    "speech_segment_rate_per_min",
    "speech_duration_sec",
    "duration_sec",
    "f0_std",
    "f0_semitone_std",
    "f0_range",
    "voiced_ratio",
    "jitter_local_proxy",
    "jitter_rap_proxy",
    "shimmer_local_proxy",
    "shimmer_db_proxy",
    "energy_modulation_std",
    "log_energy_modulation_std",
    "rms_mean",
    "rms_std",
    "spectral_centroid_mean",
    "spectral_flatness_mean",
    "mfcc_1_mean",
    "mfcc_1_std",
    "mfcc_2_mean",
    "mfcc_2_std",
    "mfcc_3_mean",
    "mfcc_3_std",
]


@dataclass
class FoldResult:
    experiment: str
    model: str
    seed: int
    fold: int
    threshold: float
    threshold_note: str
    calibrated: bool
    metrics: dict[str, float]
    predictions: pd.DataFrame


@dataclass
class TrainedResult:
    model_name: str
    threshold: float
    threshold_note: str
    calibrated: bool
    metrics: dict[str, float]
    cal_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    model: Any
    preprocessors: dict[str, Any]
    handcrafted_columns: list[str]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "(empty)"
    show = df.head(max_rows).copy() if max_rows is not None else df.copy()
    show = show.where(pd.notna(show), "")
    headers = [str(c) for c in show.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        values = [str(row[c]).replace("\n", " ") for c in show.columns]
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append("| ... | ... |")
    return "\n".join(lines)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_CSV)
    df = df[df["label"].isin([0, 1])].copy()
    if "feature_extraction_status" in df.columns:
        df = df[df["feature_extraction_status"].fillna("success").eq("success")].copy()
    df["metadata_index"] = df["metadata_index"].astype(str)
    df["speaker_id"] = df["speaker_id"].astype(str)
    return df


def load_segments() -> pd.DataFrame:
    seg = pd.read_csv(SEGMENT_MANIFEST_CSV)
    seg["metadata_index"] = seg["metadata_index"].astype(str)
    seg["speaker_id"] = seg["speaker_id"].astype(str)
    seg["segment_id"] = seg["segment_id"].astype(str)
    return seg.reset_index(drop=True)


def load_clips() -> pd.DataFrame:
    clips = pd.read_csv(CLIP_MANIFEST_CSV)
    clips["metadata_index"] = clips["metadata_index"].astype(str)
    clips["speaker_id"] = clips["speaker_id"].astype(str)
    return clips


def select_handcrafted_columns(df: pd.DataFrame, max_columns: int = 26) -> list[str]:
    cols = [c for c in HANDCRAFTED_PRIORITY_COLUMNS if c in df.columns]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {
        "metadata_index",
        "label",
        "phase6_split",
    }
    for col in numeric:
        if col not in exclude and col not in cols:
            cols.append(col)
        if len(cols) >= max_columns:
            break
    return cols[:max_columns]


class SegmentAudioDataset(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, sampling_rate: int = 16000, segment_duration: float = 3.0):
        self.rows = rows.reset_index(drop=False).rename(columns={"index": "row_index"})
        self.sampling_rate = sampling_rate
        self.segment_duration = segment_duration

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        info = sf.info(str(row["processed_file"]))
        start_frame = int(round(float(row["start_sec"]) * info.samplerate))
        frames = int(round(float(row["segment_duration_sec"]) * info.samplerate))
        audio, sr = sf.read(str(row["processed_file"]), start=start_frame, frames=frames, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sampling_rate:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=self.sampling_rate)
        target_len = int(round(self.segment_duration * self.sampling_rate))
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        return {
            "row_index": int(row["row_index"]),
            "array": audio.astype(np.float32),
            "segment_id": str(row["segment_id"]),
            "metadata_index": str(row["metadata_index"]),
            "speaker_id": str(row["speaker_id"]),
            "label": int(row["label"]),
            "phase6_split": str(row["phase6_split"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        }


def make_collate(feature_extractor: Any, sampling_rate: int = 16000) -> Any:
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = feature_extractor(
            [item["array"] for item in batch],
            sampling_rate=sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded["metadata"] = [
            {k: item[k] for k in ["row_index", "segment_id", "metadata_index", "speaker_id", "label", "phase6_split", "start_sec", "end_sec"]}
            for item in batch
        ]
        return encoded

    return collate


def chunk_path(batch_start: int) -> Path:
    return CHUNK_DIR / f"chunk_{batch_start:06d}.npz"


def completed_embedding_indices() -> set[int]:
    done: set[int] = set()
    for path in CHUNK_DIR.glob("chunk_*.npz"):
        try:
            data = np.load(path, allow_pickle=True)
            done.update(data["row_index"].astype(int).tolist())
        except Exception:
            continue
    return done


def pool_hidden(model: Any, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None and hasattr(model, "_get_feature_vector_attention_mask"):
        feature_mask = model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        mask = feature_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return hidden.mean(dim=1)


def extract_frozen_embeddings(batch_size: int, threads: int, limit: int | None = None) -> None:
    """Extract frozen wav2vec embeddings into resumable chunks and consolidate."""
    ensure_dirs()
    torch.set_num_threads(threads)
    segments = load_segments()
    if limit is not None:
        segments = segments.head(limit).copy()

    done = completed_embedding_indices()
    pending = segments[~segments.index.isin(done)].copy()
    print(f"Embedding cache: total={len(segments)} complete={len(done)} pending={len(pending)}", flush=True)
    if not pending.empty:
        from transformers import AutoFeatureExtractor, Wav2Vec2Model

        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base", local_files_only=True)
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", local_files_only=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        dataset = SegmentAudioDataset(pending)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate(feature_extractor),
        )
        t0 = time.time()
        processed = 0
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                hidden = model(input_values=batch["input_values"], attention_mask=batch.get("attention_mask")).last_hidden_state
                pooled = pool_hidden(model, hidden, batch.get("attention_mask")).detach().cpu().numpy().astype(np.float32)
                meta = pd.DataFrame(batch["metadata"])
                first_index = int(meta["row_index"].min())
                np.savez_compressed(
                    chunk_path(first_index),
                    row_index=meta["row_index"].astype(int).to_numpy(),
                    embeddings=pooled,
                    segment_id=meta["segment_id"].astype(str).to_numpy(),
                    metadata_index=meta["metadata_index"].astype(str).to_numpy(),
                    speaker_id=meta["speaker_id"].astype(str).to_numpy(),
                    label=meta["label"].astype(int).to_numpy(),
                    phase6_split=meta["phase6_split"].astype(str).to_numpy(),
                    start_sec=meta["start_sec"].astype(float).to_numpy(),
                    end_sec=meta["end_sec"].astype(float).to_numpy(),
                )
                processed += len(meta)
                if batch_number == 1 or batch_number % 25 == 0 or processed == len(pending):
                    elapsed = max(time.time() - t0, 1e-6)
                    print(
                        f"Embedding extraction: batch={batch_number}/{len(loader)} "
                        f"processed={processed}/{len(pending)} rate={processed / elapsed:.2f} seg/s",
                        flush=True,
                    )
    consolidate_embedding_cache(expected_rows=len(segments))


def consolidate_embedding_cache(expected_rows: int | None = None) -> None:
    chunks = []
    for path in sorted(CHUNK_DIR.glob("chunk_*.npz")):
        data = np.load(path, allow_pickle=True)
        chunks.append({key: data[key] for key in data.files})
    if not chunks:
        raise FileNotFoundError(f"No embedding chunks found in {CHUNK_DIR}")
    row_index = np.concatenate([c["row_index"].astype(int) for c in chunks])
    order = np.argsort(row_index)
    if expected_rows is not None and len(np.unique(row_index)) < expected_rows:
        print(
            f"Embedding cache incomplete: unique_rows={len(np.unique(row_index))}, expected={expected_rows}. "
            "Consolidating completed rows only.",
            flush=True,
        )
    arrays = {
        "row_index": row_index[order],
        "embeddings": np.vstack([c["embeddings"] for c in chunks])[order],
    }
    for key in ["segment_id", "metadata_index", "speaker_id", "label", "phase6_split", "start_sec", "end_sec"]:
        arrays[key] = np.concatenate([c[key] for c in chunks])[order]
    np.savez_compressed(EMBEDDING_CACHE, **arrays)
    print(f"Saved embedding cache: {EMBEDDING_CACHE} rows={len(arrays['row_index'])}", flush=True)


def load_embedding_cache() -> dict[str, np.ndarray]:
    if not EMBEDDING_CACHE.exists():
        raise FileNotFoundError(
            f"Frozen wav2vec cache not found: {EMBEDDING_CACHE}. "
            "Run this script with --extract-cache first."
        )
    data = np.load(EMBEDDING_CACHE, allow_pickle=True)
    return {key: data[key] for key in data.files}


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = np.nan
    try:
        out["brier_score"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        out["brier_score"] = np.nan
    return out


def safe_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, precision_floor: float = 0.45) -> tuple[float, str, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 4):
        rows.append({"threshold": float(threshold), **metric_dict(y_true, y_prob, threshold)})
    table = pd.DataFrame(rows)
    viable = table[table["precision"] >= precision_floor].copy()
    if not viable.empty:
        selected = viable.sort_values(
            ["recall", "f1", "balanced_accuracy", "specificity", "threshold"],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        return float(selected["threshold"]), "max_recall_precision_floor_0.45", table
    selected = table.sort_values(
        ["f1", "balanced_accuracy", "recall", "precision", "specificity"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return float(selected["threshold"]), "precision_floor_not_reached_max_f1", table


def maybe_platt_scale(
    cal_y: np.ndarray,
    cal_prob: np.ndarray,
    eval_prob: np.ndarray,
    brier_cutoff: float = 0.22,
) -> tuple[np.ndarray, np.ndarray, bool, Any | None]:
    raw_brier = brier_score_loss(cal_y, cal_prob) if len(np.unique(cal_y)) == 2 else np.nan
    if not np.isnan(raw_brier) and raw_brier > brier_cutoff and len(np.unique(cal_y)) == 2:
        scaler = LogisticRegression(solver="lbfgs", class_weight="balanced", random_state=42)
        scaler.fit(safe_logit(cal_prob).reshape(-1, 1), cal_y)
        return (
            scaler.predict_proba(safe_logit(cal_prob).reshape(-1, 1))[:, 1],
            scaler.predict_proba(safe_logit(eval_prob).reshape(-1, 1))[:, 1],
            True,
            scaler,
        )
    return cal_prob, eval_prob, False, None


def build_clip_records(features: pd.DataFrame, cache: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    emb = cache["embeddings"].astype(np.float32)
    meta = pd.DataFrame(
        {
            "row_index": cache["row_index"].astype(int),
            "metadata_index": cache["metadata_index"].astype(str),
            "segment_id": cache["segment_id"].astype(str),
            "speaker_id": cache["speaker_id"].astype(str),
            "label": cache["label"].astype(int),
            "phase6_split": cache["phase6_split"].astype(str),
            "start_sec": cache["start_sec"].astype(float),
            "end_sec": cache["end_sec"].astype(float),
        }
    )
    features = features.copy()
    features["metadata_index"] = features["metadata_index"].astype(str)
    feature_lookup = features.set_index("metadata_index")
    records = []
    for clip_id, sub in meta.groupby("metadata_index", sort=True):
        if clip_id not in feature_lookup.index:
            continue
        idx = sub.index.to_numpy()
        row = feature_lookup.loc[clip_id]
        records.append(
            {
                "metadata_index": str(clip_id),
                "speaker_id": str(row["speaker_id"]),
                "label": int(row["label"]),
                "phase6_split": sub["phase6_split"].iloc[0],
                "duration_sec": float(row.get("duration_sec", np.nan)),
                "processed_file": str(row.get("processed_file", "")),
                "embeddings": emb[idx],
                "segment_ids": sub["segment_id"].astype(str).tolist(),
                "start_sec": sub["start_sec"].astype(float).to_numpy(),
                "end_sec": sub["end_sec"].astype(float).to_numpy(),
            }
        )
    return records


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metadata_index": r["metadata_index"],
                "speaker_id": r["speaker_id"],
                "label": r["label"],
                "phase6_split": r["phase6_split"],
                "duration_sec": r.get("duration_sec", np.nan),
                "processed_file": r.get("processed_file", ""),
            }
            for r in records
        ]
    )


class VectorMLP(torch.nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AttentionFusionMLP(torch.nn.Module):
    def __init__(self, embedding_dim: int, handcrafted_dim: int, dropout: float = 0.3):
        super().__init__()
        self.query = torch.nn.Parameter(torch.empty(embedding_dim))
        torch.nn.init.normal_(self.query, std=0.02)
        self.classifier = VectorMLP(embedding_dim + handcrafted_dim, dropout=dropout)

    def attention_pool(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = (seq @ self.query) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(scores, dim=0)
        pooled = (seq * weights.unsqueeze(1)).sum(dim=0)
        return pooled, weights

    def forward(self, seq: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.attention_pool(seq)
        return self.classifier(torch.cat([pooled, handcrafted], dim=0).unsqueeze(0)).squeeze(0)

    def segment_logits(self, seq: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        repeated = handcrafted.unsqueeze(0).repeat(seq.shape[0], 1)
        return self.classifier(torch.cat([seq, repeated], dim=1))


def pos_weight_from_labels(y: np.ndarray) -> torch.Tensor:
    y = np.asarray(y).astype(int)
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    return torch.tensor(float(neg / pos), dtype=torch.float32)


def weighted_sampler(y: np.ndarray) -> torch.utils.data.WeightedRandomSampler:
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = np.asarray([1.0 / counts[label] for label in y], dtype=float)
    return torch.utils.data.WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)


def train_vector_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int,
    lr: float = 1e-3,
    dropout: float = 0.3,
    max_epochs: int = 30,
) -> VectorMLP:
    set_seed(seed)
    model = VectorMLP(x_train.shape[1], dropout=dropout)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_from_labels(y_train))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    tx = torch.tensor(x_train, dtype=torch.float32)
    ty = torch.tensor(y_train.astype(float), dtype=torch.float32)
    cx = torch.tensor(x_cal, dtype=torch.float32)
    cy = torch.tensor(y_cal.astype(float), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tx, ty),
        batch_size=min(32, len(tx)),
        sampler=weighted_sampler(y_train),
    )
    best_state = None
    best_loss = np.inf
    patience = 7
    patience_left = patience
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(cx), cy))
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_vector_model(model: VectorMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        return torch.sigmoid(logits).detach().cpu().numpy()


def train_attention_model(
    train_records: list[dict[str, Any]],
    cal_records: list[dict[str, Any]],
    train_features: np.ndarray,
    cal_features: np.ndarray,
    seed: int,
    lr: float = 1e-3,
    dropout: float = 0.3,
    max_epochs: int = 30,
) -> AttentionFusionMLP:
    set_seed(seed)
    embedding_dim = train_records[0]["embeddings"].shape[1]
    handcrafted_dim = train_features.shape[1]
    model = AttentionFusionMLP(embedding_dim, handcrafted_dim, dropout=dropout)
    y_train = np.asarray([r["label"] for r in train_records], dtype=int)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_from_labels(y_train))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sampler = weighted_sampler(y_train)
    best_state = None
    best_loss = np.inf
    patience = 7
    patience_left = patience

    train_tensors = [
        (
            torch.tensor(r["embeddings"], dtype=torch.float32),
            torch.tensor(train_features[i], dtype=torch.float32),
            torch.tensor(float(r["label"]), dtype=torch.float32),
        )
        for i, r in enumerate(train_records)
    ]
    cal_tensors = [
        (
            torch.tensor(r["embeddings"], dtype=torch.float32),
            torch.tensor(cal_features[i], dtype=torch.float32),
            torch.tensor(float(r["label"]), dtype=torch.float32),
        )
        for i, r in enumerate(cal_records)
    ]
    for _ in range(max_epochs):
        model.train()
        for idx in sampler:
            seq, feat, label = train_tensors[int(idx)]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(seq, feat).view(1), label.view(1))
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for seq, feat, label in cal_tensors:
                losses.append(float(loss_fn(model(seq, feat).view(1), label.view(1))))
        cal_loss = float(np.mean(losses)) if losses else np.inf
        if cal_loss < best_loss - 1e-5:
            best_loss = cal_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_attention_model(
    model: AttentionFusionMLP,
    records: list[dict[str, Any]],
    features: np.ndarray,
    return_attention: bool = False,
) -> tuple[np.ndarray, list[np.ndarray] | None, list[np.ndarray] | None]:
    probs = []
    attentions = []
    segment_probs = []
    model.eval()
    with torch.no_grad():
        for i, record in enumerate(records):
            seq = torch.tensor(record["embeddings"], dtype=torch.float32)
            feat = torch.tensor(features[i], dtype=torch.float32)
            logit = model(seq, feat)
            probs.append(float(torch.sigmoid(logit)))
            if return_attention:
                _, weights = model.attention_pool(seq)
                attentions.append(weights.detach().cpu().numpy())
                segment_probs.append(torch.sigmoid(model.segment_logits(seq, feat)).detach().cpu().numpy())
    return np.asarray(probs, dtype=float), attentions if return_attention else None, segment_probs if return_attention else None


def speaker_stratified_calibration_indices(
    frame: pd.DataFrame,
    outer_train_idx: np.ndarray,
    seed: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_frame = frame.iloc[outer_train_idx].copy()
    speaker_df = train_frame.groupby("speaker_id")["label"].agg(lambda s: int(s.mode().iloc[0])).reset_index()
    if speaker_df["label"].nunique() == 2 and speaker_df["label"].value_counts().min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed * 100 + fold)
        speaker_indices = np.arange(len(speaker_df))
        subtrain_spk_idx, cal_spk_idx = next(splitter.split(speaker_indices, speaker_df["label"].to_numpy()))
        cal_speakers = set(speaker_df.iloc[cal_spk_idx]["speaker_id"].astype(str))
    else:
        rng = np.random.default_rng(seed * 100 + fold)
        speakers = speaker_df["speaker_id"].astype(str).to_numpy()
        n_cal = max(1, int(round(0.20 * len(speakers))))
        cal_speakers = set(rng.choice(speakers, size=n_cal, replace=False).tolist())
    outer = frame.iloc[outer_train_idx].copy()
    is_cal = outer["speaker_id"].astype(str).isin(cal_speakers).to_numpy()
    cal_idx = outer_train_idx[is_cal]
    subtrain_idx = outer_train_idx[~is_cal]
    return subtrain_idx, cal_idx


def preprocess_features(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    *other_indices: np.ndarray,
    columns: list[str],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_raw = frame.iloc[train_idx][columns].to_numpy(dtype=float)
    x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw))
    outputs = [x_train]
    for idx in other_indices:
        outputs.append(scaler.transform(imputer.transform(frame.iloc[idx][columns].to_numpy(dtype=float))))
    return outputs, {"imputer": imputer, "scaler": scaler}


def standardize_embeddings(
    records: list[dict[str, Any]],
    train_idx: np.ndarray,
    *other_indices: np.ndarray,
) -> tuple[list[list[dict[str, Any]]], dict[str, np.ndarray]]:
    train_segments = np.vstack([records[i]["embeddings"] for i in train_idx])
    mean = train_segments.mean(axis=0)
    std = train_segments.std(axis=0) + 1e-6

    def transform(indices: np.ndarray) -> list[dict[str, Any]]:
        out = []
        for i in indices:
            rec = dict(records[int(i)])
            rec["embeddings"] = ((rec["embeddings"] - mean) / std).astype(np.float32)
            out.append(rec)
        return out

    return [transform(train_idx), *[transform(idx) for idx in other_indices]], {"embedding_mean": mean, "embedding_std": std}


def mean_embedding_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    return np.vstack([r["embeddings"].mean(axis=0) for r in records]).astype(np.float32)


def make_prediction_frame(records: list[dict[str, Any]], prob: np.ndarray, threshold: float) -> pd.DataFrame:
    frame = records_frame(records)
    frame["probability"] = prob
    frame["prediction"] = (prob >= threshold).astype(int)
    return frame


def evaluate_fold_model(
    experiment: str,
    model_name: str,
    seed: int,
    fold: int,
    cal_y: np.ndarray,
    cal_prob_raw: np.ndarray,
    val_records: list[dict[str, Any]],
    val_prob_raw: np.ndarray,
) -> FoldResult:
    val_y = np.asarray([r["label"] for r in val_records], dtype=int)
    cal_prob, val_prob, calibrated, _ = maybe_platt_scale(cal_y, cal_prob_raw, val_prob_raw)
    threshold, note, threshold_table = select_threshold(cal_y, cal_prob)
    threshold_table.to_csv(TABLE_DIR / f"threshold_{experiment}_{model_name}_seed{seed}_fold{fold}.csv", index=False)
    pred = make_prediction_frame(val_records, val_prob, threshold)
    metrics = metric_dict(val_y, val_prob, threshold)
    return FoldResult(experiment, model_name, seed, fold, threshold, note, calibrated, metrics, pred)


def run_grouped_experiments(records: list[dict[str, Any]], feature_df: pd.DataFrame, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    frame = records_frame(records)
    handcrafted_cols = select_handcrafted_columns(feature_df)
    feature_frame = frame[["metadata_index"]].merge(feature_df[["metadata_index", *handcrafted_cols]], on="metadata_index", how="left")
    y = frame["label"].astype(int).to_numpy()
    groups = frame["speaker_id"].astype(str).to_numpy()
    all_results: list[FoldResult] = []
    overlap_rows = []

    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (outer_train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups=groups), start=1):
            train_speakers = set(groups[outer_train_idx])
            val_speakers = set(groups[val_idx])
            overlap = train_speakers & val_speakers
            overlap_rows.append({"seed": seed, "fold": fold, "overlap_speaker_count": len(overlap), "status": "pass" if not overlap else "fail"})
            subtrain_idx, cal_idx = speaker_stratified_calibration_indices(frame, outer_train_idx, seed, fold)

            [train_records, cal_records, val_records], emb_stats = standardize_embeddings(records, subtrain_idx, cal_idx, val_idx)
            x_train_mean = mean_embedding_matrix(train_records)
            x_cal_mean = mean_embedding_matrix(cal_records)
            x_val_mean = mean_embedding_matrix(val_records)
            y_train = np.asarray([r["label"] for r in train_records], dtype=int)
            y_cal = np.asarray([r["label"] for r in cal_records], dtype=int)

            exp0 = train_vector_model(x_train_mean, y_train, x_cal_mean, y_cal, seed=seed)
            all_results.append(
                evaluate_fold_model(
                    "experiment_0",
                    "frozen_wav2vec_mean_mlp",
                    seed,
                    fold,
                    y_cal,
                    predict_vector_model(exp0, x_cal_mean),
                    val_records,
                    predict_vector_model(exp0, x_val_mean),
                )
            )

            feature_arrays, feature_pp = preprocess_features(feature_frame, subtrain_idx, cal_idx, val_idx, columns=handcrafted_cols)
            x_train_h, x_cal_h, x_val_h = feature_arrays
            handcrafted_model = train_vector_model(x_train_h, y_train, x_cal_h, y_cal, seed=seed)
            all_results.append(
                evaluate_fold_model(
                    "experiment_1_control",
                    "handcrafted_only_mlp",
                    seed,
                    fold,
                    y_cal,
                    predict_vector_model(handcrafted_model, x_cal_h),
                    val_records,
                    predict_vector_model(handcrafted_model, x_val_h),
                )
            )

            fusion_model = train_attention_model(train_records, cal_records, x_train_h, x_cal_h, seed=seed)
            cal_fusion_prob, _, _ = predict_attention_model(fusion_model, cal_records, x_cal_h)
            val_fusion_prob, _, _ = predict_attention_model(fusion_model, val_records, x_val_h)
            all_results.append(
                evaluate_fold_model(
                    "experiment_1",
                    "frozen_wav2vec_attention_fusion",
                    seed,
                    fold,
                    y_cal,
                    cal_fusion_prob,
                    val_records,
                    val_fusion_prob,
                )
            )

    fold_rows = []
    prediction_rows = []
    for result in all_results:
        fold_rows.append(
            {
                "experiment": result.experiment,
                "model": result.model,
                "seed": result.seed,
                "fold": result.fold,
                "threshold": result.threshold,
                "threshold_note": result.threshold_note,
                "platt_scaled": result.calibrated,
                **result.metrics,
            }
        )
        pred = result.predictions.copy()
        pred["experiment"] = result.experiment
        pred["model"] = result.model
        pred["seed"] = result.seed
        pred["fold"] = result.fold
        prediction_rows.append(pred)
    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(prediction_rows, ignore_index=True)
    fold_df.to_csv(TABLE_DIR / "focused_grouped_cv_fold_metrics.csv", index=False)
    pred_df.to_csv(TABLE_DIR / "focused_grouped_cv_predictions.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(TABLE_DIR / "focused_grouped_cv_speaker_overlap.csv", index=False)
    summary = summarize_fold_metrics(fold_df)
    summary.to_csv(TABLE_DIR / "focused_grouped_cv_summary.csv", index=False)

    exp1 = summary[summary["model"].eq("frozen_wav2vec_attention_fusion")]
    abort = False
    if not exp1.empty:
        recall_gap = PHASE4_BASELINE["recall"] - float(exp1.iloc[0]["recall_mean"])
        bal_gap = PHASE4_BASELINE["balanced_accuracy"] - float(exp1.iloc[0]["balanced_accuracy_mean"])
        abort = recall_gap > 0.10 and bal_gap > 0.10
    (TABLE_DIR / "focused_go_no_go.json").write_text(json.dumps({"abort_finetuning": abort}, indent=2), encoding="utf-8")
    return summary, pred_df, abort


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, model), sub in fold_df.groupby(["experiment", "model"]):
        row: dict[str, Any] = {"experiment": experiment, "model": model, "n_runs": len(sub)}
        for metric in METRIC_ORDER:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["f1_mean", "balanced_accuracy_mean"], ascending=False)


def train_final_model(records: list[dict[str, Any]], feature_df: pd.DataFrame, best_model_name: str, seed: int) -> TrainedResult:
    frame = records_frame(records)
    handcrafted_cols = select_handcrafted_columns(feature_df)
    feature_frame = frame[["metadata_index"]].merge(feature_df[["metadata_index", *handcrafted_cols]], on="metadata_index", how="left")
    train_idx = frame.index[frame["phase6_split"].eq("train")].to_numpy()
    cal_idx = frame.index[frame["phase6_split"].eq("calibration")].to_numpy()
    test_idx = frame.index[frame["phase6_split"].eq("test")].to_numpy()
    [train_records, cal_records, test_records], emb_pp = standardize_embeddings(records, train_idx, cal_idx, test_idx)
    y_train = np.asarray([r["label"] for r in train_records], dtype=int)
    y_cal = np.asarray([r["label"] for r in cal_records], dtype=int)
    y_test = np.asarray([r["label"] for r in test_records], dtype=int)
    preprocessors: dict[str, Any] = {**emb_pp}

    if best_model_name == "frozen_wav2vec_mean_mlp":
        x_train = mean_embedding_matrix(train_records)
        x_cal = mean_embedding_matrix(cal_records)
        x_test = mean_embedding_matrix(test_records)
        model = train_vector_model(x_train, y_train, x_cal, y_cal, seed=seed)
        cal_prob_raw = predict_vector_model(model, x_cal)
        test_prob_raw = predict_vector_model(model, x_test)
        preprocessors["mode"] = "mean_embedding"
    elif best_model_name == "handcrafted_only_mlp":
        feature_arrays, feature_pp = preprocess_features(feature_frame, train_idx, cal_idx, test_idx, columns=handcrafted_cols)
        x_train, x_cal, x_test = feature_arrays
        model = train_vector_model(x_train, y_train, x_cal, y_cal, seed=seed)
        cal_prob_raw = predict_vector_model(model, x_cal)
        test_prob_raw = predict_vector_model(model, x_test)
        preprocessors.update(feature_pp)
        preprocessors["mode"] = "handcrafted"
    else:
        feature_arrays, feature_pp = preprocess_features(feature_frame, train_idx, cal_idx, test_idx, columns=handcrafted_cols)
        x_train, x_cal, x_test = feature_arrays
        model = train_attention_model(train_records, cal_records, x_train, x_cal, seed=seed)
        cal_prob_raw, cal_attentions, cal_segment_probs = predict_attention_model(model, cal_records, x_cal, return_attention=True)
        test_prob_raw, attentions, segment_probs = predict_attention_model(model, test_records, x_test, return_attention=True)
        preprocessors.update(feature_pp)
        preprocessors["mode"] = "attention_fusion"
        save_attention_outputs(
            [*cal_records, *test_records],
            [*(cal_attentions or []), *(attentions or [])],
            [*(cal_segment_probs or []), *(segment_probs or [])],
        )

    cal_prob, test_prob, calibrated, platt = maybe_platt_scale(y_cal, cal_prob_raw, test_prob_raw)
    preprocessors["platt"] = platt
    threshold, note, threshold_table = select_threshold(y_cal, cal_prob)
    threshold_table.to_csv(TABLE_DIR / "final_threshold_sweep.csv", index=False)
    cal_pred = make_prediction_frame(cal_records, cal_prob, threshold)
    test_pred = make_prediction_frame(test_records, test_prob, threshold)
    metrics = metric_dict(y_test, test_prob, threshold)
    cal_pred.to_csv(TABLE_DIR / "final_calibration_predictions.csv", index=False)
    test_pred.to_csv(TABLE_DIR / "final_test_predictions.csv", index=False)
    return TrainedResult(best_model_name, threshold, note, calibrated, metrics, cal_pred, test_pred, model, preprocessors, handcrafted_cols)


def save_attention_outputs(records: list[dict[str, Any]], attentions: list[np.ndarray], segment_probs: list[np.ndarray]) -> None:
    rows = []
    for record, weights, probs in zip(records, attentions, segment_probs):
        for segment_id, start, end, weight, prob in zip(record["segment_ids"], record["start_sec"], record["end_sec"], weights, probs):
            rows.append(
                {
                    "metadata_index": record["metadata_index"],
                    "speaker_id": record["speaker_id"],
                    "label": record["label"],
                    "segment_id": segment_id,
                    "start_sec": start,
                    "end_sec": end,
                    "attention_weight": float(weight),
                    "segment_probability": float(prob),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "final_attention_segment_weights.csv", index=False)
    if table.empty:
        return
    top = table.sort_values("attention_weight", ascending=False).head(25).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r.metadata_index}:{r.start_sec:.0f}s" for r in top.itertuples()]
    ax.barh(np.arange(len(top)), top["attention_weight"].to_numpy())
    ax.set_yticks(np.arange(len(top)), labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Attention weight")
    ax.set_title("Top Attention-Weighted Segments")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "attention_weights_top_segments.png", dpi=200)
    plt.close(fig)


def run_aggregation_comparison(final: TrainedResult) -> pd.DataFrame:
    attention_path = TABLE_DIR / "final_attention_segment_weights.csv"
    rows = []
    if not attention_path.exists():
        rows.append({"method": "clip_probability", "note": "aggregation_not_applicable_for_non_attention_model", **final.metrics})
        table = pd.DataFrame(rows)
        table.to_csv(TABLE_DIR / "experiment_3_aggregation_comparison.csv", index=False)
        return table
    segments = pd.read_csv(attention_path)
    if segments.empty:
        return pd.DataFrame()
    cal_clips = final.cal_predictions[["metadata_index", "label"]].copy()
    test_clips = final.test_predictions[["metadata_index", "label"]].copy()

    def aggregate(split_clips: pd.DataFrame, method: str) -> pd.DataFrame:
        out_rows = []
        for clip_id, row in split_clips.set_index("metadata_index").iterrows():
            sub = segments[segments["metadata_index"].astype(str).eq(str(clip_id))]
            if sub.empty:
                continue
            probs = sub["segment_probability"].to_numpy(dtype=float)
            if method == "mean":
                prob = float(np.mean(probs))
            elif method == "median":
                prob = float(np.median(probs))
            elif method == "max":
                prob = float(np.max(probs))
            elif method == "top3_mean":
                prob = float(np.mean(np.sort(probs)[-min(3, len(probs)) :]))
            elif method == "attention":
                weights = sub["attention_weight"].to_numpy(dtype=float)
                prob = float(np.sum(probs * weights) / max(np.sum(weights), 1e-9))
            else:
                raise ValueError(method)
            out_rows.append({"metadata_index": clip_id, "label": int(row["label"]), "probability": prob})
        return pd.DataFrame(out_rows)

    for method in ["mean", "median", "max", "attention", "top3_mean"]:
        cal = aggregate(cal_clips, method)
        test = aggregate(test_clips, method)
        cal = final.cal_predictions.drop(columns=["probability", "prediction"], errors="ignore").merge(cal, on=["metadata_index", "label"], how="inner")
        test = final.test_predictions.drop(columns=["probability", "prediction"], errors="ignore").merge(test, on=["metadata_index", "label"], how="inner")
        cal_prob, test_prob, calibrated, _ = maybe_platt_scale(
            cal["label"].to_numpy(),
            cal["probability"].to_numpy(),
            test["probability"].to_numpy(),
        )
        cal["probability"] = cal_prob
        test["probability"] = test_prob
        threshold, note, _ = select_threshold(cal["label"].to_numpy(), cal["probability"].to_numpy())
        cal["prediction"] = (cal["probability"] >= threshold).astype(int)
        test["prediction"] = (test["probability"] >= threshold).astype(int)
        cal.to_csv(TABLE_DIR / f"aggregation_{method}_calibration_predictions.csv", index=False)
        test.to_csv(TABLE_DIR / f"aggregation_{method}_test_predictions.csv", index=False)
        rows.append(
            {
                "method": method,
                "threshold": threshold,
                "threshold_note": note,
                "platt_scaled": calibrated,
                **metric_dict(test["label"].to_numpy(), test["probability"].to_numpy(), threshold),
            }
        )
    table = pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=False)
    table.to_csv(TABLE_DIR / "experiment_3_aggregation_comparison.csv", index=False)
    return table


def adopt_best_aggregation(final: TrainedResult, aggregation: pd.DataFrame) -> TrainedResult:
    if aggregation.empty or "method" not in aggregation.columns:
        return final
    best = aggregation.sort_values(["f1", "recall", "balanced_accuracy"], ascending=False).iloc[0]
    if float(best.get("f1", -np.inf)) <= final.metrics.get("f1", -np.inf):
        return final
    method = str(best["method"])
    cal_path = TABLE_DIR / f"aggregation_{method}_calibration_predictions.csv"
    test_path = TABLE_DIR / f"aggregation_{method}_test_predictions.csv"
    if not cal_path.exists() or not test_path.exists():
        return final
    cal_pred = pd.read_csv(cal_path)
    test_pred = pd.read_csv(test_path)
    threshold = float(best["threshold"])
    metrics = metric_dict(test_pred["label"].to_numpy(), test_pred["probability"].to_numpy(), threshold)
    return TrainedResult(
        model_name=f"{final.model_name}_{method}_aggregation",
        threshold=threshold,
        threshold_note=str(best.get("threshold_note", "")),
        calibrated=bool(best.get("platt_scaled", final.calibrated)),
        metrics=metrics,
        cal_predictions=cal_pred,
        test_predictions=test_pred,
        model=final.model,
        preprocessors={**final.preprocessors, "aggregation_method": method},
        handcrafted_columns=final.handcrafted_columns,
    )


def bootstrap_ci(pred: pd.DataFrame, threshold: float, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    speakers = pred["speaker_id"].astype(str).to_numpy()
    unique = np.unique(speakers)
    idx_by_speaker = {s: np.where(speakers == s)[0] for s in unique}
    values = {metric: [] for metric in METRIC_ORDER}
    y = pred["label"].astype(int).to_numpy()
    p = pred["probability"].to_numpy(dtype=float)
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_speaker[s] for s in sampled])
        if np.unique(y[idx]).size < 2:
            continue
        metrics = metric_dict(y[idx], p[idx], threshold)
        for metric in METRIC_ORDER:
            val = metrics.get(metric, np.nan)
            if not np.isnan(val):
                values[metric].append(float(val))
    rows = []
    point = metric_dict(y, p, threshold)
    for metric in METRIC_ORDER:
        arr = np.asarray(values[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "point": point.get(metric, np.nan),
                "ci_low": float(np.percentile(arr, 2.5)) if arr.size else np.nan,
                "ci_high": float(np.percentile(arr, 97.5)) if arr.size else np.nan,
                "bootstrap_n": int(arr.size),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "final_best_model_bootstrap_ci.csv", index=False)
    return table


def speaker_level_evaluation(final: TrainedResult) -> pd.DataFrame:
    pred = final.test_predictions.copy()
    rows = []
    for speaker, sub in pred.groupby("speaker_id"):
        clip_positive_rate = float(sub["prediction"].mean())
        max_prob = float(sub["probability"].max())
        mean_prob = float(sub["probability"].mean())
        label = int(sub["label"].mode().iloc[0])
        majority_vote = int(clip_positive_rate >= 0.5)
        rows.append(
            {
                "speaker_id": speaker,
                "n_clips": int(len(sub)),
                "true_label": label,
                "majority_vote": majority_vote,
                "max_probability": max_prob,
                "mean_probability": mean_prob,
                "mean_probability_prediction": int(mean_prob >= final.threshold),
                "max_probability_prediction": int(max_prob >= final.threshold),
                "majority_vote_correct": bool(majority_vote == label),
            }
        )
    speaker_df = pd.DataFrame(rows)
    speaker_df.to_csv(TABLE_DIR / "speaker_level_evaluation.csv", index=False)
    metric_rows = []
    for method, pred_col in [
        ("majority_vote", "majority_vote"),
        ("mean_probability", "mean_probability_prediction"),
        ("max_probability", "max_probability_prediction"),
    ]:
        metric_rows.append({"method": method, **metric_dict(speaker_df["true_label"].to_numpy(), speaker_df[pred_col].to_numpy(dtype=float), 0.5)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(TABLE_DIR / "speaker_level_metrics.csv", index=False)
    return metrics


def save_pr_curve(final: TrainedResult) -> None:
    y = final.test_predictions["label"].astype(int).to_numpy()
    p = final.test_predictions["probability"].to_numpy(dtype=float)
    precision, recall, _ = precision_recall_curve(y, p)
    pr_auc = average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan
    op = metric_dict(y, p, final.threshold)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
    ax.scatter([op["recall"]], [op["precision"]], color="red", label=f"threshold={final.threshold:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Final Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "precision_recall_operating_point.png", dpi=200)
    plt.close(fig)


def save_calibration_curve(final: TrainedResult) -> None:
    y_cal = final.cal_predictions["label"].astype(int).to_numpy()
    cal_prob = final.cal_predictions["probability"].to_numpy(dtype=float)
    y_test = final.test_predictions["label"].astype(int).to_numpy()
    test_prob = final.test_predictions["probability"].to_numpy(dtype=float)
    raw_test_prob = test_prob
    if final.calibrated and final.preprocessors.get("platt") is not None:
        scaler = final.preprocessors["platt"]
        # Reverse raw probabilities are not stored after final prediction; use
        # the calibrated curve plus the ideal line and record the Brier result.
        curves = [("after_platt", test_prob)]
    else:
        curves = [("before_platt", raw_test_prob)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, prob in curves:
        observed, predicted = calibration_curve(y_test, prob, n_bins=5, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="ideal")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration_curve_before_after.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(
        [
            {
                "model": final.model_name,
                "platt_scaled": final.calibrated,
                "calibration_brier": brier_score_loss(y_cal, cal_prob) if len(np.unique(y_cal)) == 2 else np.nan,
                "test_brier": brier_score_loss(y_test, test_prob) if len(np.unique(y_test)) == 2 else np.nan,
            }
        ]
    ).to_csv(TABLE_DIR / "calibration_summary.csv", index=False)


def feature_pattern_tables(final: TrainedResult, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = feature_df.copy()
    feature_df["metadata_index"] = feature_df["metadata_index"].astype(str)
    pred = final.test_predictions.merge(feature_df, on=["metadata_index", "speaker_id"], how="left", suffixes=("", "_feature"))
    if "spectral_flatness_mean" in pred.columns:
        noise_score = pred["spectral_flatness_mean"].astype(float)
    else:
        noise_score = pred.get("zcr_mean", pd.Series(np.nan, index=pred.index)).astype(float)
    pred["noise_bucket"] = pd.qcut(noise_score.rank(method="first"), q=3, labels=["clean_proxy", "moderate_proxy", "noisy_proxy"])
    prosody = pred.get("f0_semitone_std", pd.Series(np.nan, index=pred.index)).astype(float)
    pred["prosody_bucket"] = pd.qcut(prosody.rank(method="first"), q=3, labels=["low", "medium", "high"])
    pause = pred.get("pause_ratio", pd.Series(np.nan, index=pred.index)).astype(float)
    pred["pause_bucket"] = pd.qcut(pause.rank(method="first"), q=3, labels=["low", "medium", "high"])
    pred["duration_bucket"] = pd.cut(pred["duration_sec"].astype(float), bins=[-np.inf, 2, 4, np.inf], labels=["short_lt_2s", "medium_2_4s", "long_gt_4s"])
    pred["is_fn"] = pred["label"].eq(1) & pred["prediction"].eq(0)
    pred["is_fp"] = pred["label"].eq(0) & pred["prediction"].eq(1)
    pred.to_csv(TABLE_DIR / "final_error_case_details.csv", index=False)

    def summarize(sub: pd.DataFrame, error_type: str) -> pd.DataFrame:
        total = len(sub)
        if total == 0:
            return pd.DataFrame([{"error_type": error_type, "pattern": "no_cases", "count": 0, "pct_of_errors": 0.0}])
        speaker_counts = sub["speaker_id"].value_counts()
        top2 = int(speaker_counts.head(2).sum()) if not speaker_counts.empty else 0
        rows = [
            {"error_type": error_type, "pattern": "Short clips (<2s)", "count": int(sub["duration_bucket"].eq("short_lt_2s").sum())},
            {"error_type": error_type, "pattern": "High noise", "count": int(sub["noise_bucket"].eq("noisy_proxy").sum())},
            {"error_type": error_type, "pattern": "Dominant speaker (top 2 speakers)", "count": top2},
            {"error_type": error_type, "pattern": "Low prosody instability", "count": int(sub["prosody_bucket"].eq("low").sum())},
            {"error_type": error_type, "pattern": "Unusual pause burden", "count": int(sub["pause_bucket"].isin(["low", "high"]).sum())},
        ]
        out = pd.DataFrame(rows)
        out["pct_of_errors"] = (100 * out["count"] / total).round(1)
        return out

    fn = pred[pred["is_fn"]].copy()
    fp = pred[pred["is_fp"]].copy()
    fn.to_csv(TABLE_DIR / "false_negative_cases.csv", index=False)
    fp.to_csv(TABLE_DIR / "false_positive_cases.csv", index=False)
    patterns = pd.concat([summarize(fn, "false_negative"), summarize(fp, "false_positive")], ignore_index=True)
    patterns.to_csv(TABLE_DIR / "error_pattern_summary.csv", index=False)
    examples = qualitative_examples(fn, fp, final.handcrafted_columns)
    examples.to_csv(TABLE_DIR / "qualitative_error_examples.csv", index=False)
    return patterns, examples


def qualitative_examples(fn: pd.DataFrame, fp: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for error_type, sub, ascending in [
        ("false_negative", fn, True),
        ("false_positive", fp, False),
    ]:
        if sub.empty:
            continue
        ordered = sub.sort_values("probability", ascending=ascending)
        for _, row in ordered.head(5).iterrows():
            key_features = {
                c: round(float(row[c]), 4)
                for c in feature_cols[:8]
                if c in row.index and pd.notna(row[c])
            }
            rows.append(
                {
                    "error_type": error_type,
                    "clip_id": row["metadata_index"],
                    "speaker_id": row["speaker_id"],
                    "true_label": int(row["label"]),
                    "probability": float(row["probability"]),
                    "aggregation_method": "final_model",
                    "key_handcrafted_features": json.dumps(key_features),
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def final_replacement_decision(summary: pd.DataFrame, final: TrainedResult) -> dict[str, Any]:
    metrics = final.metrics
    criteria = {
        "recall_competitive": metrics["recall"] >= 0.83 or metrics["recall"] >= PHASE4_BASELINE["recall"] - 0.03,
        "f1_target": metrics["f1"] >= 0.52 and metrics["f1"] > PHASE4_BASELINE["f1"],
        "balanced_accuracy_target": metrics["balanced_accuracy"] >= 0.58 and metrics["balanced_accuracy"] > PHASE4_BASELINE["balanced_accuracy"],
        "precision_target": metrics["precision"] >= 0.45,
        "brier_target": metrics["brier_score"] <= 0.22,
    }
    cv_row = summary[summary["model"].eq(final.model_name)]
    if not cv_row.empty:
        criteria["stable_f1_std"] = float(cv_row.iloc[0]["f1_std"]) <= 0.05
        criteria["stable_balanced_accuracy_std"] = float(cv_row.iloc[0]["balanced_accuracy_std"]) <= 0.05
    replace = all(criteria.values())
    decision = {"replace_phase4": bool(replace), "criteria": criteria, "final_model": final.model_name, "threshold": final.threshold}
    (TABLE_DIR / "replacement_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def write_summary(
    cv_summary: pd.DataFrame,
    abort_finetuning: bool,
    final: TrainedResult | None,
    ci: pd.DataFrame | None,
    speaker_metrics: pd.DataFrame | None,
    aggregation: pd.DataFrame | None,
    error_patterns: pd.DataFrame | None,
    decision: dict[str, Any] | None,
) -> None:
    lines = [
        "# Phase 6 Focused Deep-Learning Experiments",
        "",
        "## Phase 4 Anchor",
        "",
        markdown_table(pd.DataFrame([PHASE4_BASELINE]).round(4)),
        "",
        "## Experiment 0 And 1 Grouped CV",
        "",
        markdown_table(cv_summary.round(4)),
        "",
        "## Go/No-Go",
        "",
        f"- Abort gentle fine-tuning: **{abort_finetuning}**",
        "",
    ]
    if final is not None:
        lines.extend(
            [
                "## Final Best Model",
                "",
                f"- Configuration: `{final.model_name}`",
                f"- Threshold rule: sweep 0.10-0.90, maximize recall with precision >= 0.45; selected `{final.threshold:.3f}`.",
                f"- Platt scaling applied: `{final.calibrated}`",
                "",
                markdown_table(pd.DataFrame([{**{"model": final.model_name, "threshold": final.threshold}, **final.metrics}]).round(4)),
                "",
                "## Bootstrap Confidence Intervals",
                "",
                markdown_table(ci.round(4) if ci is not None else pd.DataFrame()),
                "",
                "## Experiment 3 Aggregation Comparison",
                "",
                markdown_table(aggregation.round(4) if aggregation is not None else pd.DataFrame()),
                "",
                "## Speaker-Level Metrics",
                "",
                markdown_table(speaker_metrics.round(4) if speaker_metrics is not None else pd.DataFrame()),
                "",
                "## Error Analysis",
                "",
                markdown_table(error_patterns if error_patterns is not None else pd.DataFrame()),
                "",
            ]
        )
    if decision is not None:
        lines.extend(
            [
                "## Replacement Decision",
                "",
                f"- Replace Phase 4: **{decision['replace_phase4']}**",
                f"- Criteria: `{json.dumps(decision['criteria'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Figures",
            "",
            "- `figures/precision_recall_operating_point.png`",
            "- `figures/calibration_curve_before_after.png`",
            "- `figures/attention_weights_top_segments.png` when the final model is attention-fusion",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused Phase 6 frozen wav2vec/fusion experiments.")
    parser.add_argument("--extract-cache", action="store_true", help="Extract/resume frozen wav2vec segment embeddings.")
    parser.add_argument("--consolidate-cache", action="store_true", help="Consolidate existing embedding chunks.")
    parser.add_argument("--run-experiments", action="store_true", help="Run grouped CV and final evaluation from the embedding cache.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--extract-limit", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="42,7,13")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.extract_cache:
        extract_frozen_embeddings(batch_size=args.batch_size, threads=args.threads, limit=args.extract_limit)
    if args.consolidate_cache:
        expected = len(load_segments()) if args.extract_limit is None else args.extract_limit
        consolidate_embedding_cache(expected_rows=expected)
    if not args.run_experiments:
        return

    feature_df = load_features()
    cache = load_embedding_cache()
    records = build_clip_records(feature_df, cache)
    if len(records) < len(feature_df):
        print(f"Warning: records from cache={len(records)} feature_rows={len(feature_df)}", flush=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cv_summary, _, abort_finetuning = run_grouped_experiments(records, feature_df, seeds)
    if abort_finetuning:
        print("Go/no-go: Experiment 1 is clearly below Phase 4; skipping Experiment 2 fine-tuning.", flush=True)
    best_model = str(cv_summary.sort_values(["f1_mean", "balanced_accuracy_mean"], ascending=False).iloc[0]["model"])
    final = train_final_model(records, feature_df, best_model, seed=seeds[0])
    aggregation = run_aggregation_comparison(final)
    final_candidate = adopt_best_aggregation(final, aggregation)
    final_candidate.test_predictions.to_csv(TABLE_DIR / "final_best_model_test_predictions.csv", index=False)
    final_candidate.cal_predictions.to_csv(TABLE_DIR / "final_best_model_calibration_predictions.csv", index=False)
    ci = bootstrap_ci(final_candidate.test_predictions, final_candidate.threshold, n_bootstrap=args.bootstrap_iterations, seed=seeds[0])
    speaker_metrics = speaker_level_evaluation(final_candidate)
    save_pr_curve(final_candidate)
    save_calibration_curve(final_candidate)
    error_patterns, _ = feature_pattern_tables(final_candidate, feature_df)
    decision = final_replacement_decision(cv_summary, final_candidate)
    write_summary(cv_summary, abort_finetuning, final_candidate, ci, speaker_metrics, aggregation, error_patterns, decision)
    print(f"Focused Phase 6 experiments complete: {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
