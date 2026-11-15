"""Error analysis: where the chosen model is wrong on the test rows."""

from __future__ import annotations

import pandas as pd

from tabular_prediction.config import TARGET

AGE_BANDS = ([0, 25, 35, 45, 55, 65, 200], ["under 25", "25-34", "35-44", "45-54", "55-64", "65+"])
EDUCATION_BANDS = (
    [0, 8, 9, 12, 13, 20],
    ["below high school", "high school", "some college", "bachelor", "above bachelor"],
)
HOURS_BANDS = ([0, 34, 40, 50, 200], ["under 35", "35-40", "41-50", "over 50"])


def with_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Add readable group columns (age, education, weekly hours) to a table."""
    banded = frame.copy()
    banded["age_band"] = pd.cut(frame["age"], AGE_BANDS[0], labels=AGE_BANDS[1], right=False)
    banded["education_band"] = pd.cut(
        frame["education_num"], EDUCATION_BANDS[0], labels=EDUCATION_BANDS[1]
    )
    banded["hours_band"] = pd.cut(frame["hours_per_week"], HOURS_BANDS[0], labels=HOURS_BANDS[1])
    return banded


def confusion(test_frame: pd.DataFrame) -> dict[str, int]:
    """Return the four counts of right and wrong answers."""
    actual, predicted = test_frame[TARGET], test_frame["predicted"]
    return {
        "true_negative": int(((actual == 0) & (predicted == 0)).sum()),
        "false_positive": int(((actual == 0) & (predicted == 1)).sum()),
        "false_negative": int(((actual == 1) & (predicted == 0)).sum()),
        "true_positive": int(((actual == 1) & (predicted == 1)).sum()),
    }


def _group_row(group: pd.DataFrame) -> dict[str, float]:
    counts = confusion(group)
    positives = counts["true_positive"] + counts["false_negative"]
    negatives = counts["true_negative"] + counts["false_positive"]
    flagged = counts["true_positive"] + counts["false_positive"]
    return {
        "rows": len(group),
        "actual_share": positives / len(group),
        "predicted_share": flagged / len(group),
        # Of the people who earn above 50K, the share the model misses.
        "miss_rate": counts["false_negative"] / positives if positives else float("nan"),
        # Of the people who do not, the share the model wrongly flags.
        "false_alarm_rate": counts["false_positive"] / negatives if negatives else float("nan"),
        "error_rate": (counts["false_positive"] + counts["false_negative"]) / len(group),
    }


def errors_by_group(test_frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return one row of error figures per value of ``column``."""
    rows = {
        str(value): _group_row(group)
        for value, group in test_frame.groupby(column, observed=True, sort=True)
        if len(group)
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = column
    return table


def analyse(test_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the error tables used in the report, by group column."""
    banded = with_bands(test_frame)
    return {
        column: errors_by_group(banded, column)
        for column in ("sex", "age_band", "education_band", "hours_band")
    }
