"""The whole run, from raw files to the saved model, the reports and the figures."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from tabular_prediction import __version__, plots
from tabular_prediction.cleaning import CleaningReport, clean
from tabular_prediction.config import Settings
from tabular_prediction.data import load_raw
from tabular_prediction.errors import analyse, confusion
from tabular_prediction.metrics import METRICS
from tabular_prediction.training import TrainingResult, rounded, train_and_evaluate


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def metrics_markdown(summary: dict[str, Any]) -> str:
    """Return the results as Markdown tables (this text is saved as ``reports/metrics.md``)."""
    labels = [label for label, _ in METRICS.values()]
    cv_rows = [
        [name + (" **(chosen)**" if name == summary["best_model"] else "")]
        + [f"{scores[m]['mean']:.4f} ± {scores[m]['std']:.4f}" for m in METRICS]
        for name, scores in summary["cross_validation"].items()
    ]
    test_rows = [
        [name] + [f"{summary['test'][key][m]:.4f}" for m in METRICS]
        for name, key in ((summary["best_model"], "best"), ("baseline", "baseline"))
    ]
    parts = [
        "# Metrics",
        f"Seed {summary['settings']['seed']}, {summary['rows']['train']:,} training rows, "
        f"{summary['rows']['test']:,} test rows. Created by `python scripts/train.py`.",
        f"## Cross-validation ({summary['settings']['cv_folds']} folds, mean ± standard deviation)",
        _table(["Model", *labels], cv_rows),
        "## Test rows (used once, for the chosen model)",
        _table(["Model", *labels], test_rows),
        "## Errors by group (chosen model, test rows)",
    ]
    for column, table in summary["errors_by_group"].items():
        rows = [
            [
                group,
                f"{values['rows']:,}",
                f"{values['actual_share']:.1%}",
                f"{values['predicted_share']:.1%}",
                _share(values["miss_rate"]),
                _share(values["false_alarm_rate"]),
                f"{values['error_rate']:.1%}",
            ]
            for group, values in table.items()
        ]
        header = [
            column,
            "Rows",
            "Actually above 50K",
            "Predicted above 50K",
            "High incomes missed",
            "False alarms",
            "All errors",
        ]
        parts.append(_table(header, rows))
    return "\n\n".join(parts) + "\n"


def summarise(
    result: TrainingResult, report: CleaningReport, settings: Settings
) -> dict[str, Any]:
    """Return every number of the run in one JSON-ready structure."""
    settings_shown = {
        key: value for key, value in asdict(settings).items() if key not in ("root",)
    }
    return rounded(
        {
            "version": __version__,
            "settings": settings_shown,
            "cleaning": asdict(report),
            "rows": {"train": result.train_rows, "test": result.test_rows},
            "cross_validation": result.cv,
            "best_model": result.best_model,
            "test": result.test,
            "confusion": confusion(result.test_frame),
            "errors_by_group": {
                column: table.to_dict(orient="index")
                for column, table in analyse(result.test_frame).items()
            },
        }
    )


def run(settings: Settings, raw: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run everything and write the outputs. Return the summary that was saved.

    Outputs: the cleaned table, the fitted model, ``reports/metrics.json``,
    ``reports/metrics.md`` and three figures in ``reports/figures``.
    """
    with threadpool_limits(limits=settings.max_threads):
        if raw is None:
            raw = load_raw(settings.raw_dir)
        frame, report = clean(raw)
        settings.clean_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.clean_file, index=False)

        result = train_and_evaluate(frame, settings)
        summary = summarise(result, report, settings)

        settings.model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.model, settings.model_file)

        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        (settings.reports_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (settings.reports_dir / "metrics.md").write_text(
            metrics_markdown(summary), encoding="utf-8", newline="\n"
        )
        figures = settings.figures_dir
        plots.plot_data_overview(frame, figures / "data_overview.png")
        plots.plot_model_comparison(result, settings.cv_folds, figures / "model_comparison.png")
        plots.plot_test_results(result, figures / "test_results.png")
    return summary
