"""Configuration types and loading for descriptive analysis."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


REPRESENTATION_ORDER = (
    "raw",
    "handcrafted features",
    "GazeMAE embeddings",
    "MOMENT embeddings",
)

SUBJECT_CSV_FILES = {
    "raw": "tobii_raw_samples.csv",
    "features": "tobii_features.csv",
    "gazemae": "tobii_gazemae_embeddings.csv",
    "moment": "tobii_moment_embeddings.csv",
}

EXPORT_METADATA_COLUMNS = (
    "window_id",
    "subject",
    "source_file",
    "start_timestamp",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "sleep_feedback",
    "prompt_id",
    "prompt_time",
)


@dataclass(frozen=True)
class TableConfig:
    """Tabular input and its join identifier."""

    path: Path | None = None
    id_column: str = "window_uid"
    paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class RepresentationConfig:
    """One window-level representation input."""

    name: str
    path: Path | None = None
    id_column: str = "window_uid"
    feature_prefixes: tuple[str, ...] = ()
    array_key: str = "embeddings"
    id_key: str = "window_uid"
    kind: str = "table"
    channels: tuple[str, ...] = ()
    sequence_length: int = 120
    paths: tuple[Path, ...] = ()
    exclude_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisConfig:
    """Complete descriptive-analysis configuration."""

    metadata: TableConfig
    raw_samples: TableConfig | None
    representations: tuple[RepresentationConfig, ...]
    output_dir: Path
    target_columns: tuple[str, ...]
    target_name: str
    subject_column: str
    pupil_columns: tuple[str, ...]
    gaze_columns: tuple[str, str]
    methods: tuple[str, ...]
    color_by: tuple[str, ...]
    max_raw_rows: int
    max_projection_points: int
    random_state: int
    trustme_root: Path | None = None
    subject_export_dir: str = "tobii"
    subjects: tuple[str, ...] = ()


def _expand_path(value: str, base_dir: Path) -> Path:
    """Expand environment variables and resolve a path relative to the YAML file."""

    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded.resolve() if expanded.is_absolute() else (base_dir / expanded).resolve()


def _table(raw: dict[str, Any], base_dir: Path) -> TableConfig:
    """Parse a table input mapping."""

    return TableConfig(
        path=_expand_path(str(raw["path"]), base_dir),
        id_column=str(raw.get("id_column", "window_uid")),
    )


def _discover_subject_csvs(
    root: Path,
    subject_export_dir: str,
) -> tuple[tuple[str, ...], dict[str, tuple[Path, ...]]]:
    """Discover and strictly validate one named export directory per subject."""

    if not root.is_dir():
        raise ValueError(f"trustme_root is not a directory: {root}")

    subject_dirs = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not subject_dirs:
        raise ValueError(f"No subject directories found in trustme_root: {root}")

    discovered: dict[str, list[Path]] = {key: [] for key in SUBJECT_CSV_FILES}
    missing: list[str] = []
    subjects: list[str] = []
    for subject_dir in subject_dirs:
        export_dir = subject_dir / "ml" / subject_export_dir
        subject_missing = [
            filename
            for filename in SUBJECT_CSV_FILES.values()
            if not (export_dir / filename).is_file()
        ]
        if subject_missing:
            missing.append(f"{subject_dir.name}: {', '.join(subject_missing)}")
            continue
        subjects.append(subject_dir.name)
        for key, filename in SUBJECT_CSV_FILES.items():
            discovered[key].append(export_dir / filename)

    if missing:
        details = "\n  - ".join(missing)
        raise ValueError(f"Missing subject CSV exports:\n  - {details}")
    return tuple(subjects), {key: tuple(paths) for key, paths in discovered.items()}


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    """Load and validate descriptive-analysis YAML configuration."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Analysis config must be a YAML mapping.")

    base_dir = config_path.parent
    trustme_root_raw = raw.get("trustme_root")
    trustme_root = _expand_path(str(trustme_root_raw), base_dir) if trustme_root_raw else None
    subject_export_dir = str(raw.get("subject_export_dir", "tobii"))
    if Path(subject_export_dir).name != subject_export_dir:
        raise ValueError("subject_export_dir must be a directory name, not a path.")
    subjects: tuple[str, ...] = ()
    if trustme_root is not None:
        subjects, files = _discover_subject_csvs(trustme_root, subject_export_dir)
        metadata = TableConfig(paths=files["features"])
        raw_samples = TableConfig(paths=files["raw"])
        representations = [
            RepresentationConfig(
                name="raw",
                kind="raw_samples",
                paths=files["raw"],
                channels=("PupilSizeLeft", "PupilSizeRight", "GazePointX", "GazePointY", "AverageDistance"),
                sequence_length=120,
            ),
            RepresentationConfig(
                name="handcrafted features",
                paths=files["features"],
                exclude_columns=("window_uid", *EXPORT_METADATA_COLUMNS),
            ),
            RepresentationConfig(
                name="GazeMAE embeddings",
                paths=files["gazemae"],
                feature_prefixes=("z_pos_", "z_vel_"),
            ),
            RepresentationConfig(
                name="MOMENT embeddings",
                paths=files["moment"],
                feature_prefixes=("moment_",),
            ),
        ]
    else:
        metadata = _table(raw["metadata"], base_dir)
        raw_samples_raw = raw.get("raw_samples")
        raw_samples = _table(raw_samples_raw, base_dir) if raw_samples_raw else None
        representations = []
        for item in raw.get("representations", []):
            representations.append(
                RepresentationConfig(
                    name=str(item["name"]),
                    path=_expand_path(str(item["path"]), base_dir),
                    id_column=str(item.get("id_column", "window_uid")),
                    feature_prefixes=tuple(str(value) for value in item.get("feature_prefixes", [])),
                    array_key=str(item.get("array_key", "embeddings")),
                    id_key=str(item.get("id_key", "window_uid")),
                    kind=str(item.get("kind", "table")),
                    channels=tuple(str(value) for value in item.get("channels", [])),
                    sequence_length=int(item.get("sequence_length", 120)),
                    exclude_columns=tuple(str(value) for value in item.get("exclude_columns", [])),
                )
            )
    names = [item.name for item in representations]
    unknown = sorted(set(names) - set(REPRESENTATION_ORDER))
    if unknown:
        raise ValueError(f"Unknown representation names: {unknown}")
    if len(names) != len(set(names)):
        raise ValueError("Representation names must be unique.")
    invalid_kinds = sorted({item.kind for item in representations} - {"table", "raw_samples"})
    if invalid_kinds:
        raise ValueError(f"Unsupported representation kinds: {invalid_kinds}")
    for item in representations:
        if item.kind == "raw_samples" and not item.channels:
            raise ValueError("A raw_samples representation requires channels.")
    representations.sort(key=lambda item: REPRESENTATION_ORDER.index(item.name))

    plots = raw.get("plots", {})
    methods = tuple(str(value).lower() for value in plots.get("projection_methods", ["pca", "tsne", "umap"]))
    invalid_methods = sorted(set(methods) - {"pca", "tsne", "umap"})
    if invalid_methods:
        raise ValueError(f"Unsupported projection methods: {invalid_methods}")

    gaze_columns = tuple(str(value) for value in plots.get("gaze_columns", ["GazePointX", "GazePointY"]))
    if len(gaze_columns) != 2:
        raise ValueError("plots.gaze_columns must contain exactly two column names.")

    return AnalysisConfig(
        metadata=metadata,
        raw_samples=raw_samples,
        representations=tuple(representations),
        output_dir=_expand_path(str(raw.get("output_dir", "../results/data_analysis")), base_dir),
        target_columns=tuple(str(value) for value in raw.get("target_columns", ["5"])),
        target_name=str(raw.get("target_name", "q5")),
        subject_column=str(raw.get("subject_column", "Subject")),
        pupil_columns=tuple(str(value) for value in plots.get("pupil_columns", ["PupilSizeLeft", "PupilSizeRight"])),
        gaze_columns=(gaze_columns[0], gaze_columns[1]),
        methods=methods,
        color_by=tuple(str(value) for value in plots.get("color_by", ["5", "Subject"])),
        max_raw_rows=int(plots.get("max_raw_rows", 250_000)),
        max_projection_points=int(plots.get("max_projection_points", 5_000)),
        random_state=int(raw.get("random_state", 42)),
        trustme_root=trustme_root,
        subject_export_dir=subject_export_dir,
        subjects=subjects,
    )
