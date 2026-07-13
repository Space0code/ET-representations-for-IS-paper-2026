"""Create per-session screen-coordinate-normalized Tobii representation trees.

Each ``<subject>/ml/tobii`` directory is copied to a sibling
``<subject>/ml/tobii_coordinate_normalized`` directory.  The raw-sample CSV is
rewritten in chunks: all available gaze x columns are divided by the source
recording's display width and all gaze y columns by its display height.  The
display resolution is read from the original ``.tsv`` header and validated
against ``display_resolutions.yaml``; this avoids guessing when a participant
used more than one display.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_TRUSTME_ROOT = Path("/home/ppg/eyetracking/TrustME-ET/data/raw/TrustMe")
DEFAULT_SOURCE_DIRECTORY = "tobii"
DEFAULT_DESTINATION_DIRECTORY = "tobii_coordinate_normalized"
DEFAULT_RAW_SAMPLES_FILENAME = "tobii_raw_samples.csv"
DEFAULT_CHUNK_SIZE = 100_000
RESOLUTION_PATTERN = re.compile(r"^(?P<width>\d+)x(?P<height>\d+)$")
DISPLAY_HEADER_PATTERN = re.compile(
    r"^display resolution:\s*(?P<width>\d+)x(?P<height>\d+)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Resolution:
    """A display resolution in pixels."""

    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class SubjectNormalizationReport:
    """Summary of one normalized subject raw-sample CSV."""

    subject: str
    row_count: int
    source_count: int
    resolutions: tuple[Resolution, ...]
    x_min: float | None
    x_max: float | None
    y_min: float | None
    y_max: float | None
    in_unit_square_count: int
    finite_coordinate_pair_count: int


def parse_resolution(value: str) -> Resolution:
    """Parse a ``WIDTHxHEIGHT`` display-resolution string."""
    match = RESOLUTION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid display resolution {value!r}; expected WIDTHxHEIGHT.")
    return Resolution(width=int(match["width"]), height=int(match["height"]))


def load_subject_resolutions(path: Path) -> dict[str, set[Resolution]]:
    """Load the allowed per-subject resolutions from the YAML metadata file."""
    with path.open(encoding="utf-8") as stream:
        raw_data = yaml.safe_load(stream)
    if not isinstance(raw_data, Mapping):
        raise ValueError(f"Resolution metadata must be a mapping: {path}")

    subject_resolutions: dict[str, set[Resolution]] = {}
    for resolution_name, subjects in raw_data.items():
        if not isinstance(resolution_name, str) or not isinstance(subjects, list):
            raise ValueError(f"Invalid entry in resolution metadata: {resolution_name!r}")
        resolution = parse_resolution(resolution_name)
        for subject in subjects:
            if not isinstance(subject, str):
                raise ValueError(f"Invalid subject name for {resolution}: {subject!r}")
            subject_resolutions.setdefault(subject, set()).add(resolution)
    return subject_resolutions


def read_display_resolution(tsv_path: Path) -> Resolution:
    """Read the display resolution declared in a source Tobii TSV header."""
    try:
        with tsv_path.open(encoding="utf-8", errors="replace") as stream:
            for _ in range(10):
                line = stream.readline()
                if not line:
                    break
                match = DISPLAY_HEADER_PATTERN.fullmatch(line.strip())
                if match is not None:
                    return Resolution(width=int(match["width"]), height=int(match["height"]))
    except OSError as error:
        raise ValueError(f"Could not read source recording {tsv_path}: {error}") from error
    raise ValueError(f"No 'display resolution: WIDTHxHEIGHT' header in {tsv_path}")


def source_filename_to_tsv(source_filename: str) -> str:
    """Map a processed ``*.tsv.parquet`` source name back to its raw TSV name."""
    if source_filename.endswith(".tsv.parquet"):
        return source_filename[: -len(".parquet")]
    if source_filename.endswith(".tsv"):
        return source_filename
    raise ValueError(
        f"Unsupported source_file value {source_filename!r}; expected a .tsv or .tsv.parquet name."
    )


def resolve_source_resolutions(
    subject_directory: Path,
    source_filenames: Iterable[str],
    allowed_resolutions: set[Resolution],
) -> dict[str, Resolution]:
    """Resolve and validate a display resolution for every processed source file."""
    resolved: dict[str, Resolution] = {}
    raw_tobii_directory = subject_directory / "tobii"
    for source_filename in sorted(set(source_filenames)):
        tsv_path = raw_tobii_directory / source_filename_to_tsv(source_filename)
        resolution = read_display_resolution(tsv_path)
        if resolution not in allowed_resolutions:
            allowed = ", ".join(str(item) for item in sorted(allowed_resolutions, key=str))
            raise ValueError(
                f"{subject_directory.name}/{source_filename}: TSV declares {resolution}, "
                f"which is absent from display_resolutions.yaml ({allowed})."
            )
        resolved[source_filename] = resolution
    return resolved


def _coordinate_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return available gaze x and y columns from raw Tobii samples."""
    column_set = set(columns)
    x_columns = ["GazePointXLeft", "GazePointXRight", "GazePointX"]
    y_columns = ["GazePointYLeft", "GazePointYRight", "GazePointY"]
    return (
        [column for column in x_columns if column in column_set],
        [column for column in y_columns if column in column_set],
    )


def normalize_raw_samples(
    raw_samples_path: Path,
    output_path: Path,
    subject_directory: Path,
    allowed_resolutions: set[Resolution],
    chunk_size: int,
) -> SubjectNormalizationReport:
    """Normalize raw gaze coordinates by per-source display dimensions."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    header = pd.read_csv(raw_samples_path, nrows=0)
    if "source_file" not in header.columns:
        raise ValueError(f"Raw samples lack required source_file column: {raw_samples_path}")
    x_columns, y_columns = _coordinate_columns(header.columns)
    if not x_columns or not y_columns:
        raise ValueError(f"Raw samples lack gaze coordinate columns: {raw_samples_path}")

    source_files = pd.read_csv(raw_samples_path, usecols=["source_file"])["source_file"]
    if source_files.isna().any():
        raise ValueError(f"Raw samples contain missing source_file values: {raw_samples_path}")
    source_resolutions = resolve_source_resolutions(
        subject_directory,
        source_files.astype(str).unique(),
        allowed_resolutions,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    row_count = 0
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    in_unit_square_count = 0
    finite_coordinate_pair_count = 0
    try:
        for chunk_index, chunk in enumerate(pd.read_csv(raw_samples_path, chunksize=chunk_size)):
            for source_file, resolution in source_resolutions.items():
                source_mask = chunk["source_file"].astype(str).eq(source_file)
                if not source_mask.any():
                    continue
                for column in x_columns:
                    chunk.loc[source_mask, column] = (
                        pd.to_numeric(chunk.loc[source_mask, column], errors="raise")
                        / resolution.width
                    )
                for column in y_columns:
                    chunk.loc[source_mask, column] = (
                        pd.to_numeric(chunk.loc[source_mask, column], errors="raise")
                        / resolution.height
                    )

            primary_x = pd.to_numeric(chunk["GazePointX"], errors="raise")
            primary_y = pd.to_numeric(chunk["GazePointY"], errors="raise")
            finite_mask = primary_x.notna() & primary_y.notna()
            finite_coordinate_pair_count += int(finite_mask.sum())
            in_unit_square_count += int(
                (finite_mask & primary_x.between(0.0, 1.0) & primary_y.between(0.0, 1.0)).sum()
            )
            if finite_mask.any():
                chunk_x_min = float(primary_x[finite_mask].min())
                chunk_x_max = float(primary_x[finite_mask].max())
                chunk_y_min = float(primary_y[finite_mask].min())
                chunk_y_max = float(primary_y[finite_mask].max())
                x_min = chunk_x_min if x_min is None else min(x_min, chunk_x_min)
                x_max = chunk_x_max if x_max is None else max(x_max, chunk_x_max)
                y_min = chunk_y_min if y_min is None else min(y_min, chunk_y_min)
                y_max = chunk_y_max if y_max is None else max(y_max, chunk_y_max)

            chunk.to_csv(
                temporary_path,
                index=False,
                mode="w" if chunk_index == 0 else "a",
                header=chunk_index == 0,
                quoting=csv.QUOTE_MINIMAL,
            )
            row_count += len(chunk)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return SubjectNormalizationReport(
        subject=subject_directory.name,
        row_count=row_count,
        source_count=len(source_resolutions),
        resolutions=tuple(sorted(set(source_resolutions.values()), key=str)),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        in_unit_square_count=in_unit_square_count,
        finite_coordinate_pair_count=finite_coordinate_pair_count,
    )


def copy_non_raw_files(source_directory: Path, destination_directory: Path, raw_filename: str) -> None:
    """Copy all other local representation CSVs and manifest files unchanged."""
    for path in source_directory.iterdir():
        if path.name == raw_filename:
            continue
        if path.is_file():
            shutil.copy2(path, destination_directory / path.name)


def normalize_subject(
    source_directory: Path,
    destination_directory_name: str,
    subject_resolutions: Mapping[str, set[Resolution]],
    raw_filename: str,
    chunk_size: int,
    overwrite: bool,
) -> SubjectNormalizationReport:
    """Create one normalized sibling directory from a subject's ``tobii`` directory."""
    subject_directory = source_directory.parents[1]
    subject = subject_directory.name
    if subject not in subject_resolutions:
        raise ValueError(f"Subject {subject!r} is missing from display-resolutions metadata.")
    raw_samples_path = source_directory / raw_filename
    if not raw_samples_path.is_file():
        raise ValueError(f"Missing raw samples CSV: {raw_samples_path}")

    destination_directory = source_directory.parent / destination_directory_name
    if destination_directory.exists() and not overwrite:
        raise FileExistsError(
            f"Destination exists: {destination_directory}. Use --overwrite to replace its files."
        )
    destination_directory.mkdir(parents=True, exist_ok=True)
    copy_non_raw_files(source_directory, destination_directory, raw_filename)
    return normalize_raw_samples(
        raw_samples_path=raw_samples_path,
        output_path=destination_directory / raw_filename,
        subject_directory=subject_directory,
        allowed_resolutions=subject_resolutions[subject],
        chunk_size=chunk_size,
    )


def load_config(path: Path | None) -> dict[str, Any]:
    """Load an optional YAML configuration mapping."""
    if path is None:
        return {}
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line options, allowing YAML defaults and CLI overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional YAML configuration file.")
    parser.add_argument("--trustme-root", type=Path, help="Root containing subject directories.")
    parser.add_argument(
        "--display-resolutions",
        type=Path,
        help="Path to display_resolutions.yaml (defaults to <trustme-root>/display_resolutions.yaml).",
    )
    parser.add_argument("--source-directory", help="Source directory name under each subject's ml/.")
    parser.add_argument("--destination-directory", help="Sibling output directory name under each subject's ml/.")
    parser.add_argument("--raw-samples-filename", help="Raw samples CSV filename.")
    parser.add_argument("--chunk-size", type=int, help="Rows processed per CSV chunk.")
    parser.add_argument("--overwrite", action="store_true", help="Replace files in existing output directories.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without creating output files.")
    args = parser.parse_args()
    config = load_config(args.config)

    def get_value(name: str, default: Any) -> Any:
        cli_value = getattr(args, name)
        return cli_value if cli_value is not None else config.get(name, default)

    args.trustme_root = Path(get_value("trustme_root", DEFAULT_TRUSTME_ROOT))
    args.display_resolutions = Path(
        get_value("display_resolutions", args.trustme_root / "display_resolutions.yaml")
    )
    args.source_directory = str(get_value("source_directory", DEFAULT_SOURCE_DIRECTORY))
    args.destination_directory = str(
        get_value("destination_directory", DEFAULT_DESTINATION_DIRECTORY)
    )
    args.raw_samples_filename = str(get_value("raw_samples_filename", DEFAULT_RAW_SAMPLES_FILENAME))
    args.chunk_size = int(get_value("chunk_size", DEFAULT_CHUNK_SIZE))
    args.overwrite = args.overwrite or bool(config.get("overwrite", False))
    args.dry_run = args.dry_run or bool(config.get("dry_run", False))
    return args


def main() -> int:
    """Normalize every discovered local Tobii subject tree and print a concise audit."""
    args = parse_args()
    if not args.trustme_root.is_dir():
        raise ValueError(f"TrustMe root does not exist: {args.trustme_root}")
    if not args.display_resolutions.is_file():
        raise ValueError(f"Display-resolution metadata does not exist: {args.display_resolutions}")
    if args.source_directory == args.destination_directory:
        raise ValueError("source_directory and destination_directory must differ")

    subject_resolutions = load_subject_resolutions(args.display_resolutions)
    source_directories = sorted(args.trustme_root.glob(f"*/ml/{args.source_directory}"))
    if not source_directories:
        raise ValueError(f"No */ml/{args.source_directory} directories under {args.trustme_root}")

    reports: list[SubjectNormalizationReport] = []
    for source_directory in source_directories:
        subject = source_directory.parents[1].name
        if args.dry_run:
            raw_samples_path = source_directory / args.raw_samples_filename
            source_files = pd.read_csv(raw_samples_path, usecols=["source_file"])["source_file"]
            resolve_source_resolutions(
                source_directory.parents[1],
                source_files.astype(str).unique(),
                subject_resolutions[subject],
            )
            print(f"VALID {subject}: {source_files.nunique()} source recordings")
            continue
        report = normalize_subject(
            source_directory=source_directory,
            destination_directory_name=args.destination_directory,
            subject_resolutions=subject_resolutions,
            raw_filename=args.raw_samples_filename,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )
        reports.append(report)
        in_range_percentage = (
            100.0 * report.in_unit_square_count / report.finite_coordinate_pair_count
            if report.finite_coordinate_pair_count
            else 0.0
        )
        print(
            f"NORMALIZED {report.subject}: rows={report.row_count}, sources={report.source_count}, "
            f"resolutions={','.join(map(str, report.resolutions))}, "
            f"x=[{report.x_min:.6g},{report.x_max:.6g}], "
            f"y=[{report.y_min:.6g},{report.y_max:.6g}], "
            f"unit-square={report.in_unit_square_count}/{report.finite_coordinate_pair_count} "
            f"({in_range_percentage:.2f}%)"
        )

    if not args.dry_run:
        total_rows = sum(report.row_count for report in reports)
        print(f"COMPLETE subjects={len(reports)}, raw_sample_rows={total_rows}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError, OSError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
