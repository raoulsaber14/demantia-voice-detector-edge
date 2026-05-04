"""Evaluation helpers for baseline and wav2vec models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
    roc_auc_score,
)


def positive_class_probability(y_prob: np.ndarray) -> np.ndarray:
    """Return a 1-D dementia-positive probability vector."""
    arr = np.asarray(y_prob)
    if arr.ndim == 2:
        if arr.shape[1] < 2:
            return arr[:, 0]
        return arr[:, 1]
    return arr.ravel()


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity for the non-dementia class."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = cm.ravel()
    denom = tn + fp
    return float(tn / denom) if denom else np.nan


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute binary screening metrics with dementia as the positive class."""
    y_prob_pos = positive_class_probability(y_prob)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
    except Exception:
        metrics["roc_auc"] = np.nan
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob_pos))
    except Exception:
        metrics["pr_auc"] = np.nan
    try:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob_pos))
    except Exception:
        metrics["brier_score"] = np.nan
    return metrics


def bootstrap_metric_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    speaker_ids: Optional[np.ndarray] = None,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence intervals for key metrics, sampling speakers when provided."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = positive_class_probability(y_prob)
    rng = np.random.default_rng(seed)
    metric_names = ["roc_auc", "pr_auc", "recall", "specificity"]
    values_by_metric = {name: [] for name in metric_names}

    if speaker_ids is not None:
        speaker_ids = np.asarray(speaker_ids).astype(str)
        unique_speakers = np.unique(speaker_ids)
        speaker_to_idx = {speaker: np.where(speaker_ids == speaker)[0] for speaker in unique_speakers}
    else:
        unique_speakers = None
        speaker_to_idx = None

    for _ in range(n_bootstrap):
        if unique_speakers is not None and len(unique_speakers) > 0:
            sampled_speakers = rng.choice(unique_speakers, size=len(unique_speakers), replace=True)
            indices = np.concatenate([speaker_to_idx[speaker] for speaker in sampled_speakers])
        else:
            indices = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)

        sample_true = y_true[indices]
        if np.unique(sample_true).size < 2:
            continue
        sample_pred = y_pred[indices]
        sample_prob = y_prob[indices]
        metrics = classification_metrics(sample_true, sample_pred, sample_prob)
        for metric_name in metric_names:
            value = metrics.get(metric_name, np.nan)
            if not np.isnan(value):
                values_by_metric[metric_name].append(value)

    cis: Dict[str, float] = {}
    for metric_name, values in values_by_metric.items():
        if len(values) < max(20, n_bootstrap * 0.20):
            cis[f"{metric_name}_ci_low"] = np.nan
            cis[f"{metric_name}_ci_high"] = np.nan
            cis[f"{metric_name}_ci_n"] = len(values)
            continue
        arr = np.asarray(values, dtype=float)
        cis[f"{metric_name}_ci_low"] = float(np.percentile(arr, 2.5))
        cis[f"{metric_name}_ci_high"] = float(np.percentile(arr, 97.5))
        cis[f"{metric_name}_ci_n"] = len(values)
    return cis


def save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Render and save confusion matrix image."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=["non_dementia", "dementia"])
    ax.set_yticks([0, 1], labels=["non_dementia", "dementia"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_roc_curve_figure(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    """Render and save ROC curve."""
    y_prob_pos = positive_class_probability(y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    if np.unique(y_true).size >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
        roc_auc = roc_auc_score(y_true, y_prob_pos)
        ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC unavailable: one class only", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_pr_curve_figure(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    """Render and save precision-recall curve."""
    y_prob_pos = positive_class_probability(y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    if np.unique(y_true).size >= 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
        pr_auc = average_precision_score(y_true, y_prob_pos)
        ax.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
        ax.legend(loc="lower left")
    else:
        ax.text(0.5, 0.5, "PR unavailable: one class only", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_calibration_curve_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
    title: str,
    n_bins: int = 5,
) -> None:
    """Render and save reliability diagram."""
    y_prob_pos = positive_class_probability(y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    if np.unique(y_true).size >= 2:
        prob_true, prob_pred = calibration_curve(y_true, y_prob_pos, n_bins=n_bins, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label="Observed")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Ideal")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "Calibration unavailable: one class only", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed dementia rate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_metrics_table(metrics_rows: Dict[str, Dict[str, float]], out_csv: Path) -> pd.DataFrame:
    """Save model metrics as a table."""
    df = pd.DataFrame.from_dict(metrics_rows, orient="index").reset_index(names="model")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
