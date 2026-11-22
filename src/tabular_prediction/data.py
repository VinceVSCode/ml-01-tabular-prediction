"""Getting the raw Adult Income files and reading them into one table."""

from __future__ import annotations

import hashlib
import http.client
import urllib.request
from pathlib import Path

import pandas as pd

from tabular_prediction.config import DATA_SHA256, DATA_URLS, RAW_COLUMNS


class DataError(Exception):
    """Raised when the raw data is missing, changed, or cannot be downloaded."""


def sha256_of(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise DataError(f"Cannot read {path}: {error.strerror or error}") from error


def download_raw(raw_dir: Path, *, check: bool = True) -> list[Path]:
    """Download the raw files that are not there yet. Return the paths of all of them."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in DATA_URLS.items():
        path = raw_dir / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                    temporary.write_bytes(response.read())
            # OSError: no connection, a timeout, or a failed write to disk.
            # HTTPException: the connection broke off in the middle of the file.
            except (OSError, http.client.HTTPException) as error:
                temporary.unlink(missing_ok=True)
                reason = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
                raise DataError(f"Cannot download {url} ({reason})") from error
            temporary.replace(path)
        if check and sha256_of(path) != DATA_SHA256[name]:
            raise DataError(
                f"{path} is not the expected file (its checksum differs). "
                "Delete it and run again, or update DATA_SHA256 in config.py "
                "if the source has published a new version."
            )
        paths.append(path)
    return paths


def read_raw_file(path: Path) -> pd.DataFrame:
    """Read one raw file. Lines that are not data rows (blank, or a note) are skipped."""
    frame = pd.read_csv(
        path,
        header=None,
        names=list(RAW_COLUMNS),
        skipinitialspace=True,
        comment="|",
        skip_blank_lines=True,
        dtype=str,
        keep_default_na=False,
    )
    return frame[frame["income"] != ""].reset_index(drop=True)


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Read ``adult.data`` and ``adult.test`` and return them as one table of text."""
    frames = []
    for name in DATA_URLS:
        path = raw_dir / name
        if not path.exists():
            raise DataError(f"{path} is missing. Run: python scripts/download_data.py")
        frames.append(read_raw_file(path))
    return pd.concat(frames, ignore_index=True)
