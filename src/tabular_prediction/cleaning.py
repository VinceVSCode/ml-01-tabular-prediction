"""Cleaning: from the raw text table to a typed table ready for modelling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tabular_prediction.config import (
    CATEGORICAL_FEATURES,
    DROPPED_COLUMNS,
    FEATURES,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    SKEWED_FEATURES,
    TARGET,
)

MISSING_MARK = "?"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"


@dataclass(frozen=True)
class CleaningReport:
    """What the cleaning changed, for the README and the notebook."""

    rows_in: int
    duplicate_rows_removed: int
    rows_out: int
    missing_by_column: dict[str, int]
    positive_share: float


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return the cleaned table and a report of what was done.

    Steps: trim text, turn "?" into missing values, read the target (the test
    file writes it with a full stop), convert the numbers, drop exact duplicate
    rows, and drop the columns that are not used.
    """
    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"The raw table has no column(s): {', '.join(missing)}")

    frame = raw[list(RAW_COLUMNS)].copy()
    for column in frame.columns:
        frame[column] = frame[column].astype("string").str.strip()
    frame = frame.replace(MISSING_MARK, pd.NA)

    labels = frame["income"].str.rstrip(".")
    unknown = sorted(set(labels.dropna()) - {POSITIVE_LABEL, NEGATIVE_LABEL})
    if unknown or labels.isna().any():
        raise ValueError(f"Unexpected income labels: {unknown or 'missing values'}")
    frame["income"] = labels  # so the same person in both spellings counts as a duplicate
    frame[TARGET] = (labels == POSITIVE_LABEL).astype("int8")

    for column in NUMERIC_FEATURES + SKEWED_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("object").where(frame[column].notna(), None)

    rows_in = len(frame)
    frame = frame.drop_duplicates(ignore_index=True)
    frame = frame.drop(columns=[*DROPPED_COLUMNS, "income"])
    frame = frame[[*FEATURES, TARGET]]

    report = CleaningReport(
        rows_in=rows_in,
        duplicate_rows_removed=rows_in - len(frame),
        rows_out=len(frame),
        missing_by_column={
            column: int(count) for column, count in frame.isna().sum().items() if count
        },
        positive_share=float(frame[TARGET].mean()),
    )
    return frame, report
