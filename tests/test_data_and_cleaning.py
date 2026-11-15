"""Tests for reading the raw files and for cleaning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tabular_prediction import data
from tabular_prediction.cleaning import clean
from tabular_prediction.config import DATA_URLS, FEATURES, RAW_COLUMNS, TARGET
from tabular_prediction.data import DataError, download_raw, load_raw, read_raw_file

ROW_A = "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K"  # noqa: E501
ROW_B = "52, ?, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 45, ?, >50K."  # noqa: E501


def write_raw_files(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adult.data").write_text(f"{ROW_A}\n{ROW_B}\n\n", encoding="utf-8")
    (folder / "adult.test").write_text(
        f"|1x3 Cross validator\n{ROW_B}\n{ROW_A}.\n", encoding="utf-8"
    )


# --- reading -------------------------------------------------------------------


def test_read_raw_file_skips_the_note_line_and_blank_lines(tmp_path: Path) -> None:
    write_raw_files(tmp_path)
    frame = read_raw_file(tmp_path / "adult.test")
    assert list(frame.columns) == list(RAW_COLUMNS)
    assert len(frame) == 2
    assert frame.loc[0, "workclass"] == "?"
    assert frame.loc[0, "income"] == ">50K."


def test_load_raw_joins_both_files(tmp_path: Path) -> None:
    write_raw_files(tmp_path)
    assert len(load_raw(tmp_path)) == 4


def test_load_raw_says_how_to_get_missing_files(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="download_data.py"):
        load_raw(tmp_path)


def test_download_keeps_existing_files_and_checks_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_raw_files(tmp_path)

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("the network must not be used when the files exist")

    monkeypatch.setattr(data.urllib.request, "urlopen", no_network)
    assert [path.name for path in download_raw(tmp_path, check=False)] == list(DATA_URLS)
    with pytest.raises(DataError, match="not the expected file"):
        download_raw(tmp_path)


def test_download_failure_is_reported_and_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(*args: object, **kwargs: object) -> None:
        raise OSError("no connection")

    monkeypatch.setattr(data.urllib.request, "urlopen", offline)
    with pytest.raises(DataError, match="Cannot download"):
        download_raw(tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- cleaning ------------------------------------------------------------------


def test_clean_returns_the_feature_columns_and_the_target(raw: pd.DataFrame) -> None:
    frame, report = clean(raw)
    assert list(frame.columns) == [*FEATURES, TARGET]
    assert report.rows_out == len(frame)
    assert set(frame[TARGET].unique()) == {0, 1}


def test_clean_reads_labels_with_and_without_a_full_stop(tmp_path: Path) -> None:
    write_raw_files(tmp_path)
    frame, _ = clean(load_raw(tmp_path))
    assert sorted(frame[TARGET].tolist()) == [0, 1]


def test_clean_removes_exact_duplicates(tmp_path: Path) -> None:
    write_raw_files(tmp_path)
    frame, report = clean(load_raw(tmp_path))
    assert (report.rows_in, report.duplicate_rows_removed, report.rows_out) == (4, 2, 2)
    assert len(frame) == 2


def test_clean_turns_question_marks_into_missing_values(tmp_path: Path) -> None:
    write_raw_files(tmp_path)
    frame, report = clean(load_raw(tmp_path))
    assert report.missing_by_column == {"workclass": 1, "native_country": 1}
    assert not (frame == "?").any().any()


def test_clean_converts_numbers(raw: pd.DataFrame) -> None:
    frame, _ = clean(raw)
    for column in ("age", "education_num", "hours_per_week", "capital_gain", "capital_loss"):
        assert frame[column].dtype == "float64"


def test_clean_reports_the_share_of_high_incomes(raw: pd.DataFrame) -> None:
    frame, report = clean(raw)
    assert report.positive_share == pytest.approx(frame[TARGET].mean())
    assert 0 < report.positive_share < 1


def test_clean_rejects_an_unknown_label(raw: pd.DataFrame) -> None:
    raw.loc[0, "income"] = "maybe"
    with pytest.raises(ValueError, match="Unexpected income labels"):
        clean(raw)


def test_clean_rejects_a_table_without_a_needed_column(raw: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no column"):
        clean(raw.drop(columns=["age"]))


def test_clean_rejects_text_in_a_number_column(raw: pd.DataFrame) -> None:
    raw.loc[0, "age"] = "old"
    with pytest.raises(ValueError):
        clean(raw)
