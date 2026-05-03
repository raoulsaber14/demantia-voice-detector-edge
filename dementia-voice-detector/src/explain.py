"""Explainability and subgroup error analysis helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def top_feature_importance(importance_csv: Path, top_k: int = 20) -> pd.DataFrame:
    """Load feature importance table and return top rows."""
    df = pd.read_csv(importance_csv)
    return df.sort_values("importance", ascending=False).head(top_k)


def subgroup_metrics(
    predictions_df: pd.DataFrame,
    group_col: str,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Compute subgroup-level accuracy and positive rate where sample size allows."""
    rows: List[Dict[str, object]] = []
    for value, g in predictions_df.groupby(group_col):
        if len(g) < min_samples:
            continue
        acc = (g["y_true"] == g["y_pred"]).mean()
        rows.append(
            {
                "group_col": group_col,
                "group_value": value,
                "n": len(g),
                "accuracy": float(acc),
                "positive_prediction_rate": float((g["y_pred"] == 1).mean()),
            }
        )
    return pd.DataFrame(rows)


LIMITATION_WARNING = (
    "Bias and fairness warning: subgroup metrics in small samples are unstable. "
    "This tool is for screening support research only and must not be used as a diagnosis."
)
