"""Feature engineering as one scikit-learn ``ColumnTransformer``.

Everything is learned from the training rows only (medians, scales, category
lists), because the transformer sits inside the model pipeline.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from tabular_prediction.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SKEWED_FEATURES,
    Settings,
)


def build_preprocessor(settings: Settings) -> ColumnTransformer:
    """Return the transformer that turns the cleaned columns into model inputs.

    - numbers: fill gaps with the median, then scale;
    - capital gain and loss (mostly zero, with a long tail): ``log(1 + x)``, then scale;
    - categories: fill gaps with the label "missing", group rare categories,
      then one column per category.
    """
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    skewed = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=settings.min_category_share,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("skewed", skewed, list(SKEWED_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        verbose_feature_names_out=False,
    )
