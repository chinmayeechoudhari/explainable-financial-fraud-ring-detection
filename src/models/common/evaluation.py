"""Standard classification metrics shared by all model experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_recall_at_k(
    y_true: Iterable[int],
    scores: Iterable[float],
    k: int,
) -> tuple[float, float]:
    """Compute Precision@K and Recall@K from prediction scores."""
    y_true = np.asarray(list(y_true), dtype=np.int8)
    scores = np.asarray(list(scores), dtype=float)

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        return 0.0, 0.0

    k = max(1, min(int(k), len(y_true)))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_positive = int(y_true[top_indices].sum())
    total_positive = int(y_true.sum())

    precision = top_positive / k
    recall = top_positive / total_positive if total_positive else 0.0
    return float(precision), float(recall)


def classification_metrics(
    y_true: Iterable[int],
    scores: Iterable[float],
    threshold: float = 0.5,
    k_values: tuple[int, ...] = (100, 500, 1000),
) -> dict[str, float]:
    """Return the project's common binary-classification metrics."""
    y_true = np.asarray(list(y_true), dtype=np.int8)
    scores = np.asarray(list(scores), dtype=float)

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores contain NaN or infinite values")

    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    metrics = {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    # ROC-AUC is undefined if a split contains only one class.
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, scores))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )

    for k in k_values:
        precision_k, recall_k = precision_recall_at_k(y_true, scores, k)
        metrics[f"precision_at_{k}"] = precision_k
        metrics[f"recall_at_{k}"] = recall_k

    return metrics
