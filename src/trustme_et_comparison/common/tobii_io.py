"""Shared Tobii file I/O helpers.

This module provides robust readers for Tobii TSV exports that contain a
human-readable preamble before the tabular header row.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_tobii_header_offset(tsv_path: Path) -> int:
    """Return the zero-based line index of the Tobii tabular header.

    The expected first tabular token is ``TimeStamp``. A UTF-8 BOM is handled.

    Args:
        tsv_path: Path to one Tobii ``.tsv`` file.

    Returns:
        Header line index used with ``skiprows``.

    Raises:
        ValueError: If no tabular header is found.
    """

    with tsv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, raw_line in enumerate(handle):
            if raw_line.lstrip("\ufeff").strip().startswith("TimeStamp"):
                return index
    raise ValueError(f"Could not find 'TimeStamp' header in {tsv_path}")


def read_tobii_tsv_with_preamble(tsv_path: Path) -> pd.DataFrame:
    """Load a Tobii TSV while skipping the textual preamble.

    Args:
        tsv_path: Path to one Tobii ``.tsv`` file.

    Returns:
        Parsed Tobii frame. Empty files return an empty frame.
    """

    if tsv_path.stat().st_size == 0:
        return pd.DataFrame()

    header_offset = find_tobii_header_offset(tsv_path)
    return pd.read_csv(
        tsv_path,
        sep="\t",
        skiprows=header_offset,
        low_memory=False,
    )

