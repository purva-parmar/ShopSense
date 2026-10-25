"""Evaluation helpers for ShopSense repeat-purchase classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: Any, y_pred: Any, y_probability: Any
) -> dict[str, float | None]:
    """Calculate metrics for positive class 1; return None for undefined AUCs."""
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    probability_series = pd.Series(y_probability)

    roc_auc: float | None = None
    average_precision: float | None = None
    if y_true_series.nunique() == 2:
        roc_auc = float(roc_auc_score(y_true_series, probability_series))
        average_precision = float(
            average_precision_score(y_true_series, probability_series)
        )

    return {
        "accuracy": float(accuracy_score(y_true_series, y_pred_series)),
        "precision": float(
            precision_score(y_true_series, y_pred_series, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_series, y_pred_series, pos_label=1, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true_series, y_pred_series, pos_label=1, zero_division=0)
        ),
        "roc_auc": roc_auc,
        "pr_auc": average_precision,
    }


def save_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a labelled confusion-matrix figure and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No repeat purchase (0)", "Repeat purchase (1)"],
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    display.plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def save_model_comparison_plot(
    comparison: pd.DataFrame,
    metric: str,
    split: str,
    output_path: str | Path,
) -> Path:
    """Save a bar chart comparing model metric values for validation or test."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    metric_column = f"{split}_{metric}"
    if metric_column not in comparison.columns:
        raise ValueError(f"Missing metric column: {metric_column}")

    plot_data = comparison[["model", metric_column]].dropna().sort_values(
        metric_column, ascending=False
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(plot_data["model"], plot_data[metric_column])
    axis.set_title(f"{split.title()} model comparison — {metric.upper()}")
    axis.set_xlabel("Model")
    axis.set_ylabel(metric.upper())
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path
