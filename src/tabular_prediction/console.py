"""Printing from the scripts, the same way on every system."""

from __future__ import annotations

import io
import sys


def use_utf8_output() -> None:
    """Print UTF-8 text even when the output is redirected to a file or a pipe.

    On Windows, redirected output uses the system's legacy code page, which
    cannot hold Greek or most other non-Latin text. A project folder with such a
    name would then raise ``UnicodeEncodeError`` as soon as a script printed a
    path, for example in an error message.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")


def describe_os_error(error: OSError) -> str:
    """Return a short sentence for a file that could not be read or written."""
    where = f" {error.filename}" if error.filename else ""
    return f"cannot read or write{where}: {error.strerror or error}"
