"""The candidate models. Each is a full pipeline: features first, then the model."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tabular_prediction.config import Settings
from tabular_prediction.features import build_preprocessor

BASELINE = "baseline"


def build_models(settings: Settings) -> dict[str, Pipeline]:
    """Return the pipelines to compare, by name.

    The baseline always predicts the share of high incomes seen in training. A
    real model has to beat it. The other three go from simple to flexible.
    """
    estimators = {
        BASELINE: DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=settings.logistic_max_iter, random_state=settings.seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=settings.forest_trees,
            max_depth=settings.forest_max_depth,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=settings.seed,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=settings.boosting_iterations,
            learning_rate=settings.boosting_learning_rate,
            early_stopping=False,
            random_state=settings.seed,
        ),
    }
    return {
        name: Pipeline([("features", build_preprocessor(settings)), ("model", estimator)])
        for name, estimator in estimators.items()
    }
