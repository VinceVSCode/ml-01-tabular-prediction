"""The metrics, each with what it means here and where it is weak."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

# name -> (label, True if higher is better)
METRICS: dict[str, tuple[str, bool]] = {
    "f1": ("F1 score", True),
    "roc_auc": ("ROC AUC", True),
    "rmse": ("RMSE", False),
    "mae": ("MAE", False),
    "accuracy": ("Accuracy", True),
    "precision": ("Precision", True),
    "recall": ("Recall", True),
}


def compute_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return every metric for one set of predictions.

    ``probability`` is the predicted probability of income above 50K. F1,
    accuracy, precision and recall use the yes/no answer at ``threshold``.
    ROC AUC, RMSE and MAE use the probability itself: RMSE and MAE measure how
    far it is from the true answer (0 or 1).
    """
    y_true = np.asarray(y_true)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, probability))),
        "mae": float(mean_absolute_error(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def is_better(metric: str, candidate: float, current: float) -> bool:
    """Return whether ``candidate`` beats ``current`` for this metric."""
    higher_is_better = METRICS[metric][1]
    return candidate > current if higher_is_better else candidate < current
