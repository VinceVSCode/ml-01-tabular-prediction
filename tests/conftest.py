"""Shared test data: a small made-up table in the shape of the raw Adult files.

The tests never download anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabular_prediction.config import RAW_COLUMNS, Settings


def make_raw(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Return ``rows`` made-up people as text, the way the raw files are read."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    education_num = rng.integers(1, 17, rows)
    hours = rng.integers(10, 70, rows)
    capital_gain = np.where(rng.random(rows) < 0.1, rng.integers(1000, 50000, rows), 0)
    score = 0.35 * (education_num - 9) + 0.04 * (age - 40) + 0.03 * (hours - 40)
    score = score + capital_gain / 10000 + rng.normal(0, 1, rows)
    high = score > 1.0
    suffix = np.where(np.arange(rows) % 3 == 0, ".", "")  # the test file adds a full stop
    frame = pd.DataFrame(
        {
            "age": age.astype(str),
            "workclass": rng.choice(
                ["Private", "Self-emp", "Gov", "?"], rows, p=[0.6, 0.2, 0.1, 0.1]
            ),
            "fnlwgt": rng.integers(10000, 900000, rows).astype(str),
            "education": "level-" + pd.Series(education_num).astype(str),
            "education_num": education_num.astype(str),
            "marital_status": rng.choice(["Married", "Never-married", "Divorced"], rows),
            "occupation": rng.choice(["Sales", "Tech", "Craft", "?"], rows),
            "relationship": rng.choice(["Husband", "Wife", "Own-child", "Not-in-family"], rows),
            "race": rng.choice(["White", "Black", "Other"], rows),
            "sex": rng.choice(["Male", "Female"], rows),
            "capital_gain": capital_gain.astype(str),
            "capital_loss": np.where(rng.random(rows) < 0.05, "1500", "0"),
            "hours_per_week": hours.astype(str),
            "native_country": rng.choice(
                ["United-States", "Mexico", "?"], rows, p=[0.85, 0.1, 0.05]
            ),
            "income": np.where(high, ">50K", "<=50K") + suffix,
        }
    )
    return frame[list(RAW_COLUMNS)]


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Small, fast settings that write only inside a temporary folder."""
    return Settings(
        root=tmp_path,
        cv_folds=3,
        forest_trees=10,
        forest_max_depth=4,
        boosting_iterations=10,
        logistic_max_iter=200,
        min_category_share=0.02,
    )
