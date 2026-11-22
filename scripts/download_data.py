"""Download the Adult Income files from the UCI repository into data/raw/.

Usage: python scripts/download_data.py
The files go into data/raw/ inside the folder it is run from.
Files that are already there are kept. Each file's checksum is verified.
"""

from __future__ import annotations

import sys

from tabular_prediction.config import Settings, SettingsError
from tabular_prediction.console import use_utf8_output
from tabular_prediction.data import DataError, download_raw


def main() -> int:
    use_utf8_output()
    try:
        paths = download_raw(Settings().raw_dir)
    except (SettingsError, DataError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"ready: {path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
