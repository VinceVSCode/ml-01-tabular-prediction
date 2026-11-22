"""Reproduce everything: download (if needed), clean, compare models, test, report.

Usage: python scripts/train.py

It works in the folder it is run from: the data, the reports and the model are
read from and written to data/, reports/ and models/ there. Run it from the
project folder. With the same settings it writes the same reports/metrics.json
every time (the seed is fixed in config.py).
"""

from __future__ import annotations

import sys
import time

from tabular_prediction.config import Settings, SettingsError
from tabular_prediction.console import describe_os_error, use_utf8_output
from tabular_prediction.data import DataError, download_raw
from tabular_prediction.metrics import METRICS
from tabular_prediction.pipeline import run


def main() -> int:
    use_utf8_output()
    started = time.perf_counter()
    try:
        settings = Settings()
        download_raw(settings.raw_dir)
        summary = run(settings)
    except (SettingsError, DataError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: {describe_os_error(error)}", file=sys.stderr)
        return 1

    cleaning = summary["cleaning"]
    print(
        f"Rows: {cleaning['rows_in']:,} raw, {cleaning['duplicate_rows_removed']:,} duplicates "
        f"removed, {summary['rows']['train']:,} for training, {summary['rows']['test']:,} for test"
    )
    print(f"\nCross-validation ({settings.cv_folds} folds, mean):")
    names = list(METRICS)
    print(f"{'model':<22}" + "".join(f"{name:>10}" for name in names))
    for model, scores in summary["cross_validation"].items():
        mark = " *" if model == summary["best_model"] else ""
        print(
            f"{model + mark:<22}" + "".join(f"{scores[name]['mean']:>10.4f}" for name in names)
        )
    print(f"\n* chosen by {settings.selection_metric}. On the test rows:")
    for key, label in (("best", summary["best_model"]), ("baseline", "baseline")):
        print(
            f"{label:<22}" + "".join(f"{summary['test'][key][name]:>10.4f}" for name in names)
        )
    print("\nWrote reports/metrics.json, reports/metrics.md, reports/figures/ and the model.")
    seconds = time.perf_counter() - started
    print(f"Done in {seconds:.0f} s on at most {settings.max_threads} threads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
