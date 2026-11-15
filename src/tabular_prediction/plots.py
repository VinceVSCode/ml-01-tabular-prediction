"""The three figures: the data, the model comparison, and the test results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from tabular_prediction.config import TARGET  # noqa: E402
from tabular_prediction.errors import confusion, with_bands  # noqa: E402
from tabular_prediction.metrics import METRICS  # noqa: E402
from tabular_prediction.training import TrainingResult  # noqa: E402

SIZE = (12, 6.3)  # 1200 x 630 pixels at 100 dpi
DPI = 100
BLUE, ORANGE, GREY = "#2f6fab", "#e08a2c", "#9aa3ad"
MODEL_COLOURS = {
    "baseline": GREY,
    "logistic_regression": "#2f6fab",
    "random_forest": "#3c9d6b",
    "gradient_boosting": "#e08a2c",
}


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return path


def _share_bars(axis: plt.Axes, frame: pd.DataFrame, column: str, title: str) -> None:
    share = frame.groupby(column, observed=True)[TARGET].mean() * 100
    axis.barh([str(item) for item in share.index], share.to_numpy(), color=BLUE)
    axis.invert_yaxis()
    axis.set_xlabel("Share earning above 50K (%)")
    axis.set_title(title)
    for position, value in enumerate(share.to_numpy()):
        axis.text(value + 1, position, f"{value:.0f}%", va="center", fontsize=9)
    axis.set_xlim(0, max(share.max() * 1.18, 10))


def plot_data_overview(frame: pd.DataFrame, path: Path) -> Path:
    """The data: how unbalanced the target is, and how it changes with education and age."""
    banded = with_bands(frame)
    figure, axes = plt.subplots(1, 3, figsize=SIZE, constrained_layout=True)
    counts = frame[TARGET].value_counts().sort_index()
    bars = axes[0].bar(["50K or less", "Above 50K"], counts.to_numpy(), color=[GREY, ORANGE])
    for bar, count in zip(bars, counts.to_numpy(), strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{count:,}\n({count / len(frame):.0%})",
            ha="center",
            va="bottom",
        )
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[0].set_ylabel("People")
    axes[0].set_title("Income classes")
    _share_bars(axes[1], banded, "education_band", "By education")
    _share_bars(axes[2], banded, "age_band", "By age")
    figure.suptitle(f"Adult Income data after cleaning ({len(frame):,} people)", fontsize=14)
    return _save(figure, path)


def plot_model_comparison(result: TrainingResult, folds: int, path: Path) -> Path:
    """The metrics: every model on every metric, from cross-validation."""
    figure, axis = plt.subplots(figsize=SIZE, constrained_layout=True)
    names = list(result.cv)
    width = 0.8 / len(names)
    positions = np.arange(len(METRICS))
    for index, name in enumerate(names):
        means = [result.cv[name][metric]["mean"] for metric in METRICS]
        spreads = [result.cv[name][metric]["std"] for metric in METRICS]
        offset = (index - (len(names) - 1) / 2) * width
        bars = axis.bar(
            positions + offset,
            means,
            width,
            yerr=spreads,
            capsize=2,
            label=_label(name) + (" (chosen)" if name == result.best_model else ""),
            color=MODEL_COLOURS.get(name, BLUE),
        )
        for bar, mean in zip(bars, means, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.015,
                f"{mean:.2f}",
                ha="center",
                fontsize=7,
                rotation=90,
            )
    ticks = [
        f"{label}\n({'higher' if higher else 'lower'} is better)"
        for label, higher in METRICS.values()
    ]
    axis.set_xticks(positions, ticks)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel(f"Mean over {folds} folds")
    axis.set_title(f"Model comparison, {folds}-fold cross-validation on the training rows")
    axis.legend(ncols=len(names), loc="upper center", frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    return _save(figure, path)


def plot_test_results(result: TrainingResult, path: Path) -> Path:
    """The chosen model on the test rows: ROC curve, right and wrong answers, probabilities."""
    test = result.test_frame
    actual, probability = test[TARGET].to_numpy(), test["probability"].to_numpy()
    figure, axes = plt.subplots(1, 3, figsize=SIZE, constrained_layout=True)

    false_alarm, found, _ = roc_curve(actual, probability)
    axes[0].plot(false_alarm, found, color=ORANGE, label=_label(result.best_model))
    axes[0].plot([0, 1], [0, 1], color=GREY, linestyle="--", label="Guessing")
    axes[0].set_xlabel("False alarm rate")
    axes[0].set_ylabel("Share of high incomes found")
    axes[0].set_title(f"ROC curve (AUC {result.test['best']['roc_auc']:.3f})")
    axes[0].legend(loc="lower right", frameon=False)

    counts = confusion(test)
    grid = np.array(
        [
            [counts["true_negative"], counts["false_positive"]],
            [counts["false_negative"], counts["true_positive"]],
        ]
    )
    axes[1].imshow(grid, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                f"{grid[row, column]:,}",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if grid[row, column] > grid.max() / 2 else "black",
            )
    axes[1].set_xticks([0, 1], ["Predicted\n50K or less", "Predicted\nabove 50K"])
    axes[1].set_yticks([0, 1], ["Actual\n50K or less", "Actual\nabove 50K"])
    axes[1].set_title(f"Answers on the test rows (F1 {result.test['best']['f1']:.3f})")

    bins = np.linspace(0, 1, 26)
    axes[2].hist(probability[actual == 0], bins, color=GREY, alpha=0.8, label="Actual 50K or less")
    axes[2].hist(probability[actual == 1], bins, color=ORANGE, alpha=0.8, label="Actual above 50K")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Predicted probability of above 50K")
    axes[2].set_ylabel("People (log scale)")
    axes[2].set_title(
        f"Predicted probabilities\n(RMSE {result.test['best']['rmse']:.3f}, "
        f"MAE {result.test['best']['mae']:.3f})",
        fontsize=11,
    )
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"{_label(result.best_model)} on {result.test_rows:,} test rows it never saw",
        fontsize=14,
    )
    return _save(figure, path)
