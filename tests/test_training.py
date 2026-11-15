"""Tests for the features, the metrics, the training and the whole run."""

from __future__ import annotations

import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from tabular_prediction.cleaning import clean
from tabular_prediction.config import FEATURES, TARGET, Settings
from tabular_prediction.errors import analyse, confusion
from tabular_prediction.features import build_preprocessor
from tabular_prediction.metrics import METRICS, compute_metrics, is_better
from tabular_prediction.models import BASELINE, build_models
from tabular_prediction.pipeline import run
from tabular_prediction.training import choose_best, rounded, split, train_and_evaluate

# --- features ------------------------------------------------------------------


def test_preprocessor_gives_numbers_without_gaps(raw: pd.DataFrame, settings: Settings) -> None:
    frame, _ = clean(raw)
    assert frame[list(FEATURES)].isna().any().any()  # the input does have gaps
    matrix = build_preprocessor(settings).fit_transform(frame[list(FEATURES)])
    assert matrix.shape[0] == len(frame)
    assert np.isfinite(matrix).all()


def test_preprocessor_accepts_a_category_it_never_saw(
    raw: pd.DataFrame, settings: Settings
) -> None:
    frame, _ = clean(raw)
    preprocessor = build_preprocessor(settings).fit(frame[list(FEATURES)])
    new = frame[list(FEATURES)].head(3).copy()
    new["occupation"] = "Astronaut"
    assert np.isfinite(preprocessor.transform(new)).all()


def test_preprocessor_scales_the_number_columns(raw: pd.DataFrame, settings: Settings) -> None:
    frame, _ = clean(raw)
    preprocessor = build_preprocessor(settings)
    matrix = preprocessor.fit_transform(frame[list(FEATURES)])
    names = list(preprocessor.get_feature_names_out())
    age = matrix[:, names.index("age")]
    assert age.mean() == pytest.approx(0, abs=1e-9)
    assert age.std() == pytest.approx(1, abs=1e-9)


# --- metrics -------------------------------------------------------------------


def test_metrics_on_a_case_worked_out_by_hand() -> None:
    actual = np.array([1, 1, 0, 0])
    probability = np.array([0.9, 0.4, 0.6, 0.1])
    # At 0.5: one true positive, one false negative, one false positive, one true negative.
    scores = compute_metrics(actual, probability)
    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["precision"] == pytest.approx(0.5)
    assert scores["recall"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(0.5)
    assert scores["roc_auc"] == pytest.approx(0.75)
    assert scores["mae"] == pytest.approx((0.1 + 0.6 + 0.6 + 0.1) / 4)
    assert scores["rmse"] == pytest.approx(np.sqrt((0.01 + 0.36 + 0.36 + 0.01) / 4))
    assert set(scores) == set(METRICS)


def test_perfect_predictions() -> None:
    scores = compute_metrics(np.array([1, 0, 1]), np.array([1.0, 0.0, 1.0]))
    assert (scores["f1"], scores["roc_auc"], scores["rmse"], scores["mae"]) == (1, 1, 0, 0)


def test_the_threshold_changes_the_yes_or_no_metrics_only() -> None:
    actual, probability = np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.6, 0.1])
    low = compute_metrics(actual, probability, threshold=0.3)
    assert low["recall"] == 1.0
    assert low["roc_auc"] == compute_metrics(actual, probability)["roc_auc"]


def test_f1_is_zero_when_nothing_is_predicted_positive() -> None:
    assert compute_metrics(np.array([1, 0]), np.array([0.2, 0.1]))["f1"] == 0


def test_is_better_knows_the_direction_of_each_metric() -> None:
    assert is_better("f1", 0.7, 0.6)
    assert is_better("rmse", 0.3, 0.4)
    assert not is_better("mae", 0.4, 0.3)


# --- training ------------------------------------------------------------------


def test_split_is_stratified_and_repeatable(raw: pd.DataFrame, settings: Settings) -> None:
    frame, _ = clean(raw)
    train, test = split(frame, settings)
    again_train, again_test = split(frame, settings)
    assert len(train) + len(test) == len(frame)
    assert train.equals(again_train) and test.equals(again_test)
    assert train[TARGET].mean() == pytest.approx(test[TARGET].mean(), abs=0.03)


def test_split_can_use_fewer_rows(raw: pd.DataFrame, settings: Settings) -> None:
    frame, _ = clean(raw)
    train, test = split(frame, replace(settings, max_rows=100))
    assert len(train) + len(test) == 100


def test_there_are_three_models_and_a_baseline(settings: Settings) -> None:
    models = build_models(settings)
    assert BASELINE in models
    assert len(models) == 4


def test_choose_best_respects_the_metric_direction_and_skips_the_baseline() -> None:
    cv = {
        BASELINE: {"f1": {"mean": 0.99}, "rmse": {"mean": 0.01}},
        "a": {"f1": {"mean": 0.6}, "rmse": {"mean": 0.30}},
        "b": {"f1": {"mean": 0.7}, "rmse": {"mean": 0.35}},
    }
    assert choose_best(cv, "f1") == "b"
    assert choose_best(cv, "rmse") == "a"


def test_training_beats_the_baseline(raw: pd.DataFrame, settings: Settings) -> None:
    frame, _ = clean(raw)
    result = train_and_evaluate(frame, settings)
    assert result.best_model != BASELINE
    assert set(result.cv) == set(build_models(settings))
    assert result.test["best"]["roc_auc"] > 0.7 > result.test["baseline"]["roc_auc"]
    assert result.test["best"]["rmse"] < result.test["baseline"]["rmse"]
    assert len(result.test_frame) == result.test_rows
    assert result.test_frame["probability"].between(0, 1).all()


def test_error_tables_add_up(raw: pd.DataFrame, settings: Settings) -> None:
    frame, _ = clean(raw)
    result = train_and_evaluate(frame, settings)
    assert sum(confusion(result.test_frame).values()) == result.test_rows
    tables = analyse(result.test_frame)
    assert set(tables) == {"sex", "age_band", "education_band", "hours_band"}
    for table in tables.values():
        assert table["rows"].sum() == result.test_rows


def test_rounded_handles_nested_values() -> None:
    value = {"a": [0.123456, np.float64(1.99999)], "b": np.int64(3), "c": "x", "d": float("nan")}
    assert rounded(value) == {"a": [0.1235, 2.0], "b": 3, "c": "x", "d": None}


# --- the whole run -------------------------------------------------------------


def test_run_writes_every_output(raw: pd.DataFrame, settings: Settings) -> None:
    summary = run(settings, raw)
    assert settings.clean_file.is_file()
    assert (settings.reports_dir / "metrics.md").read_text(encoding="utf-8").startswith("# Metrics")
    saved = (settings.reports_dir / "metrics.json").read_text(encoding="utf-8")
    assert json.loads(saved) == summary
    for name in ("data_overview.png", "model_comparison.png", "test_results.png"):
        assert (settings.figures_dir / name).stat().st_size > 5000


def test_the_saved_model_predicts_from_cleaned_rows(raw: pd.DataFrame, settings: Settings) -> None:
    run(settings, raw)
    model = joblib.load(settings.model_file)
    frame = pd.read_csv(settings.clean_file)
    probability = model.predict_proba(frame[list(FEATURES)].head(5))[:, 1]
    assert ((probability >= 0) & (probability <= 1)).all()


def test_the_same_seed_gives_exactly_the_same_metrics(
    raw: pd.DataFrame, settings: Settings
) -> None:
    assert run(settings, raw) == run(settings, raw)


def test_a_different_seed_gives_different_metrics(raw: pd.DataFrame, settings: Settings) -> None:
    first = run(settings, raw)["cross_validation"]
    second = run(replace(settings, seed=7), raw)["cross_validation"]
    assert first != second
