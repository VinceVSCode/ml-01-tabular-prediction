"""Configuration: every setting of the project, in one place.

The values are chosen for a short run on a laptop CPU. To train harder on a
stronger machine, change the "training" block below and run the same command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tabular_prediction.metrics import METRICS

# --- data --------------------------------------------------------------------

DATA_URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}
# SHA-256 of the files as downloaded on 2026-11-15. A different file is refused.
DATA_SHA256 = {
    "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
}

RAW_COLUMNS = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
)
TARGET = "income_over_50k"
NUMERIC_FEATURES = ("age", "education_num", "hours_per_week")
SKEWED_FEATURES = ("capital_gain", "capital_loss")
CATEGORICAL_FEATURES = (
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
)
FEATURES = NUMERIC_FEATURES + SKEWED_FEATURES + CATEGORICAL_FEATURES
# Left out on purpose: "fnlwgt" is a census sampling weight, not a property of the
# person, and "education" says the same as "education_num".
DROPPED_COLUMNS = ("fnlwgt", "education")


# --- training ----------------------------------------------------------------


class SettingsError(ValueError):
    """Raised when a value in ``Settings`` cannot be used. The message names the setting."""


@dataclass(frozen=True)
class Settings:
    """Settings of one run. The defaults are the ones the README results use."""

    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    threshold: float = 0.5
    # The model with the best cross-validated value of this metric is kept.
    selection_metric: str = "f1"
    # CPU limits: models train one at a time, on at most this many threads.
    max_threads: int = 2
    # Use only this many rows (None = all). Tests use a small number.
    max_rows: int | None = None
    # Categories rarer than this share of the rows are grouped together.
    min_category_share: float = 0.01
    logistic_max_iter: int = 500
    forest_trees: int = 100
    forest_max_depth: int = 14
    boosting_iterations: int = 150
    boosting_learning_rate: float = 0.1
    root: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        problems = settings_problems(self)
        if problems:
            raise SettingsError("invalid settings in config.py: " + "; ".join(problems))

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def clean_file(self) -> Path:
        return self.root / "data" / "processed" / "adult_clean.csv"

    @property
    def model_file(self) -> Path:
        return self.root / "models" / "income_model.joblib"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.root / "reports" / "figures"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def settings_problems(settings: Settings) -> list[str]:
    """Return one sentence for each setting that cannot be used (empty if all are fine)."""
    problems = []
    if settings.selection_metric not in METRICS:
        problems.append(
            f"selection_metric is {settings.selection_metric!r}, "
            f"but must be one of: {', '.join(METRICS)}"
        )
    minimums = {
        "seed": 0,
        "cv_folds": 2,
        "max_threads": 1,
        "logistic_max_iter": 1,
        "forest_trees": 1,
        "forest_max_depth": 1,
        "boosting_iterations": 1,
    }
    for name, minimum in minimums.items():
        value = getattr(settings, name)
        if not _is_int(value) or value < minimum:
            problems.append(f"{name} is {value!r}, but must be a whole number of {minimum} or more")
    if settings.max_rows is not None and (not _is_int(settings.max_rows) or settings.max_rows < 1):
        problems.append(
            f"max_rows is {settings.max_rows!r}, but must be None or a whole number of 1 or more"
        )
    for name in ("test_size", "threshold"):
        value = getattr(settings, name)
        if not _is_number(value) or not 0 < value < 1:
            problems.append(f"{name} is {value!r}, but must be a number between 0 and 1")
    share = settings.min_category_share
    if not _is_number(share) or not 0 <= share < 1:
        problems.append(
            f"min_category_share is {share!r}, but must be a number from 0 up to (not including) 1"
        )
    rate = settings.boosting_learning_rate
    if not _is_number(rate) or rate <= 0:
        problems.append(f"boosting_learning_rate is {rate!r}, but must be a number above 0")
    return problems
