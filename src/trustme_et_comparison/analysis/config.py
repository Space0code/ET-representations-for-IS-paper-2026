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


@dataclass(frozen=True)
class TableConfig:
    """Tabular input and its join identifier."""

    path: Path
    id_column: str = "window_uid"


@dataclass(frozen=True)
class RepresentationConfig:
    """One window-level representation input."""

    name: str
    path: Path
    id_column: str = "window_uid"
    feature_prefixes: tuple[str, ...] = ()
    array_key: str = "embeddings"
    id_key: str = "window_uid"
    kind: str = "table"
    channels: tuple[str, ...] = ()
    sequence_length: int = 120


@dataclass(frozen=True)
class AnalysisConfig:
    """Complete descriptive-analysis configuration."""

    metadata: TableConfig
    raw_samples: TableConfig | None
    representations: tuple[RepresentationConfig, ...]
    output_dir: Path
    target_columns: tuple[str, ...]
    subject_column: str
    pupil_columns: tuple[str, ...]
    gaze_columns: tuple[str, str]
    methods: tuple[str, ...]
    color_by: tuple[str, ...]
    max_raw_rows: int
    max_projection_points: int
    random_state: int


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


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    """Load and validate descriptive-analysis YAML configuration."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Analysis config must be a YAML mapping.")

    base_dir = config_path.parent
    metadata = _table(raw["metadata"], base_dir)
    raw_samples_raw = raw.get("raw_samples")
    raw_samples = _table(raw_samples_raw, base_dir) if raw_samples_raw else None

    representations: list[RepresentationConfig] = []
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
        subject_column=str(raw.get("subject_column", "Subject")),
        pupil_columns=tuple(str(value) for value in plots.get("pupil_columns", ["PupilSizeLeft", "PupilSizeRight"])),
        gaze_columns=(gaze_columns[0], gaze_columns[1]),
        methods=methods,
        color_by=tuple(str(value) for value in plots.get("color_by", ["5", "Subject"])),
        max_raw_rows=int(plots.get("max_raw_rows", 250_000)),
        max_projection_points=int(plots.get("max_projection_points", 5_000)),
        random_state=int(raw.get("random_state", 42)),
    )
