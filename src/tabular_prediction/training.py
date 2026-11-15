"""Training: compare the models with cross-validation, keep the best, test it once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from tabular_prediction.config import FEATURES, TARGET, Settings
from tabular_prediction.metrics import METRICS, compute_metrics, is_better
from tabular_prediction.models import BASELINE, build_models


@dataclass
class TrainingResult:
    """Everything one run produced."""

    cv: dict[str, dict[str, dict[str, float]]]  # model -> metric -> mean / std
    best_model: str
    test: dict[str, dict[str, float]]  # "best" and "baseline" -> metric -> value
    model: Pipeline
    test_frame: pd.DataFrame  # the test rows, with "probability" and "predicted" added
    train_rows: int
    test_rows: int


def split(frame: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test). The test rows are set aside and used once, at the end."""
    if settings.max_rows is not None and len(frame) > settings.max_rows:
        frame, _ = train_test_split(
            frame,
            train_size=settings.max_rows,
            stratify=frame[TARGET],
            random_state=settings.seed,
        )
    train, test = train_test_split(
        frame,
        test_size=settings.test_size,
        stratify=frame[TARGET],
        random_state=settings.seed,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability of income above 50K for each row."""
    return model.predict_proba(features)[:, 1]


def cross_validate(
    models: dict[str, Pipeline], train: pd.DataFrame, settings: Settings
) -> dict[str, dict[str, dict[str, float]]]:
    """Score every model on the same folds. Return the mean and spread per metric."""
    folds = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.seed)
    features, target = train[list(FEATURES)], train[TARGET].to_numpy()
    fold_indices = list(folds.split(features, target))
    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, pipeline in models.items():
        scores: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for fit_rows, check_rows in fold_indices:
            fitted = clone(pipeline).fit(features.iloc[fit_rows], target[fit_rows])
            probability = positive_probability(fitted, features.iloc[check_rows])
            fold_scores = compute_metrics(target[check_rows], probability, settings.threshold)
            for metric, value in fold_scores.items():
                scores[metric].append(value)
        results[name] = {
            metric: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for metric, values in scores.items()
        }
    return results


def choose_best(cv: dict[str, dict[str, dict[str, float]]], metric: str) -> str:
    """Return the name of the model with the best mean cross-validated ``metric``."""
    best: str | None = None
    for name, scores in cv.items():
        if name == BASELINE:
            continue
        if best is None or is_better(metric, scores[metric]["mean"], cv[best][metric]["mean"]):
            best = name
    if best is None:
        raise ValueError("There is no model to choose from.")
    return best


def train_and_evaluate(frame: pd.DataFrame, settings: Settings) -> TrainingResult:
    """Run the whole comparison on a cleaned table."""
    train, test = split(frame, settings)
    models = build_models(settings)
    cv = cross_validate(models, train, settings)
    best_name = choose_best(cv, settings.selection_metric)

    train_features, train_target = train[list(FEATURES)], train[TARGET].to_numpy()
    test_features, test_target = test[list(FEATURES)], test[TARGET].to_numpy()

    best = clone(models[best_name]).fit(train_features, train_target)
    baseline = clone(models[BASELINE]).fit(train_features, train_target)
    probability = positive_probability(best, test_features)

    test_frame = test.copy()
    test_frame["probability"] = probability
    test_frame["predicted"] = (probability >= settings.threshold).astype(int)

    return TrainingResult(
        cv=cv,
        best_model=best_name,
        test={
            "best": compute_metrics(test_target, probability, settings.threshold),
            "baseline": compute_metrics(
                test_target, positive_probability(baseline, test_features), settings.threshold
            ),
        },
        model=best,
        test_frame=test_frame,
        train_rows=len(train),
        test_rows=len(test),
    )


def rounded(value: Any, digits: int = 4) -> Any:
    """Round every number in a nested structure, so saved results compare exactly."""
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(item, digits) for item in value]
    if isinstance(value, float | np.floating):
        # "Not a number" (a rate over zero people) is saved as null.
        return None if np.isnan(value) else round(float(value), digits)
    if isinstance(value, np.integer):
        return int(value)
    return value
