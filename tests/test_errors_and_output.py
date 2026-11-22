"""Tests for the errors found in the quality review (version 1.0.1).

Each problem must end in one clear "Error: ..." line and exit code 1, never in a
traceback. None of these tests uses the network.
"""

from __future__ import annotations

import http.client
import io
import sys
from pathlib import Path

import pytest

import download_data
import train
from tabular_prediction import data
from tabular_prediction.config import Settings, SettingsError, settings_problems
from tabular_prediction.console import describe_os_error, use_utf8_output
from tabular_prediction.data import DataError, download_raw

# --- settings ------------------------------------------------------------------


def test_the_default_settings_are_valid() -> None:
    assert settings_problems(Settings()) == []


def test_an_unknown_selection_metric_names_the_choices() -> None:
    with pytest.raises(SettingsError, match="selection_metric is 'F1'.*f1, roc_auc, rmse, mae"):
        Settings(selection_metric="F1")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("cv_folds", 1),
        ("cv_folds", 2.5),
        ("seed", -1),
        ("max_threads", 0),
        ("forest_trees", 0),
        ("boosting_iterations", True),
        ("max_rows", 0),
        ("test_size", 1.5),
        ("test_size", 0),
        ("threshold", 1),
        ("min_category_share", 1),
        ("boosting_learning_rate", 0),
        ("boosting_learning_rate", "0.1"),
    ],
)
def test_a_value_that_cannot_be_used_is_refused_by_name(name: str, value: object) -> None:
    with pytest.raises(SettingsError, match=f"{name} is {value!r}"):
        Settings(**{name: value})


def test_every_problem_is_reported_at_once() -> None:
    with pytest.raises(SettingsError) as raised:
        Settings(cv_folds=1, threshold=2.0)
    assert "cv_folds" in str(raised.value)
    assert "threshold" in str(raised.value)


# --- downloading and reading the raw files -------------------------------------


def test_a_download_that_breaks_off_is_reported_and_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenResponse:
        def __enter__(self) -> BrokenResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            raise http.client.IncompleteRead(b"39, State-gov", 3_000_000)

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: BrokenResponse())
    with pytest.raises(DataError, match=r"Cannot download .*\(IncompleteRead"):
        download_raw(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_raw_file_that_cannot_be_read_is_reported(tmp_path: Path) -> None:
    (tmp_path / "adult.data").mkdir()  # a folder where the file should be
    with pytest.raises(DataError, match="Cannot read"):
        download_raw(tmp_path)


# --- the scripts ---------------------------------------------------------------


def legacy_stream() -> io.TextIOWrapper:
    """A stream like Windows redirected output: a code page without Greek letters."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


def written(stream: io.TextIOWrapper) -> str:
    stream.flush()
    return stream.buffer.getvalue().decode("utf-8")  # type: ignore[attr-defined]


def test_use_utf8_output_switches_a_legacy_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = legacy_stream()
    monkeypatch.setattr(sys, "stdout", stdout)
    use_utf8_output()
    print("φάκελος")
    assert written(stdout).strip() == "φάκελος"


def test_train_reports_invalid_settings_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(train, "Settings", lambda: Settings(selection_metric="F1"))
    assert train.main() == 1
    error = capsys.readouterr().err
    assert error.startswith("Error: invalid settings in config.py: selection_metric is 'F1'")
    assert "Traceback" not in error


def test_train_reports_a_file_it_cannot_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def locked(*args: object) -> None:
        raise PermissionError(13, "Permission denied", "reports/metrics.json")

    monkeypatch.setattr(train, "Settings", lambda: Settings(root=tmp_path))
    monkeypatch.setattr(train, "download_raw", locked)
    assert train.main() == 1
    error = capsys.readouterr().err
    assert error == "Error: cannot read or write reports/metrics.json: Permission denied\n"


def test_an_error_with_a_greek_folder_name_is_printed_on_redirected_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object) -> None:
        raise DataError("Cannot read C:/δοκιμή/data/raw/adult.data: Permission denied")

    stderr = legacy_stream()
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(download_data, "download_raw", missing)
    assert download_data.main() == 1
    assert "δοκιμή" in written(stderr)


def test_describe_os_error_without_a_file_name() -> None:
    assert describe_os_error(OSError("disk full")) == "cannot read or write: disk full"
