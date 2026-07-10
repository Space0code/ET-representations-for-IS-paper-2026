"""Orchestration for descriptive analysis and representation projections."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import load_analysis_config
from .data import load_representation, project_matrix, read_sampled_table, read_table, sample_rows
from .plotting import (
    plot_class_distribution,
    plot_gaze_density,
    plot_missingness,
    plot_projection,
    plot_pupil_distributions,
    plot_subject_class_distribution,
)


def run_analysis(config_path: str | Path) -> Path:
    """Run configured analyses and return the output directory."""

    config = load_analysis_config(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_table(config.metadata)
    required_metadata = {config.metadata.id_column, config.subject_column, *config.target_columns}
    missing_metadata = sorted(required_metadata - set(metadata.columns))
    if missing_metadata:
        raise ValueError(f"Metadata is missing required columns: {missing_metadata}")
    metadata = metadata.drop_duplicates(config.metadata.id_column, keep="first").copy()
    metadata[config.metadata.id_column] = metadata[config.metadata.id_column].astype(str)

    outputs: list[Path] = []
    for target in config.target_columns:
        outputs.append(plot_class_distribution(metadata, target, config.output_dir))
        outputs.append(
            plot_subject_class_distribution(metadata, target, config.subject_column, config.output_dir)
        )

    if config.raw_samples is not None:
        raw_columns = list(
            dict.fromkeys(
                [
                    config.raw_samples.id_column,
                    *config.pupil_columns,
                    *config.gaze_columns,
                ]
            )
        )
        raw = read_sampled_table(
            config.raw_samples,
            raw_columns,
            config.max_raw_rows,
            config.random_state,
        )
        if config.raw_samples.id_column in raw.columns:
            label_columns = [config.metadata.id_column, *config.target_columns]
            label_frame = metadata[label_columns].rename(
                columns={config.metadata.id_column: config.raw_samples.id_column}
            )
            raw = raw.drop(columns=list(config.target_columns), errors="ignore").merge(
                label_frame,
                on=config.raw_samples.id_column,
                how="left",
            )
        descriptive_columns = [column for column in [*config.pupil_columns, *config.gaze_columns] if column in raw.columns]
        if descriptive_columns:
            outputs.append(plot_missingness(raw[descriptive_columns], config.output_dir))
        if all(column in raw.columns for column in config.gaze_columns):
            outputs.append(plot_gaze_density(raw, config.gaze_columns, config.output_dir))
        if all(column in raw.columns for column in config.pupil_columns):
            for target in config.target_columns:
                if target in raw.columns:
                    outputs.append(plot_pupil_distributions(raw, config.pupil_columns, target, config.output_dir))

    for representation in config.representations:
        allowed_ids: set[str] | None = None
        if representation.kind == "raw_samples":
            sampled_metadata = sample_rows(metadata, config.max_projection_points, config.random_state)
            allowed_ids = set(sampled_metadata[config.metadata.id_column].astype(str))
        ids, matrix = load_representation(representation, allowed_ids=allowed_ids)
        index_frame = pd.DataFrame({config.metadata.id_column: ids, "matrix_index": range(len(ids))})
        joined = index_frame.merge(metadata, on=config.metadata.id_column, how="inner")
        joined = sample_rows(joined, config.max_projection_points, config.random_state)
        selected_matrix = matrix[joined["matrix_index"].to_numpy()]
        for method in config.methods:
            coordinates = project_matrix(selected_matrix, method, config.random_state)
            projection = joined.reset_index(drop=True).copy()
            projection[["component_1", "component_2"]] = coordinates
            projection_path = config.output_dir / (
                f"projection_{representation.name.lower().replace(' ', '_')}_{method}.csv"
            )
            projection.to_csv(projection_path, index=False)
            outputs.append(projection_path)
            for color_column in config.color_by:
                if color_column in projection.columns:
                    outputs.append(
                        plot_projection(
                            projection,
                            representation.name,
                            method,
                            color_column,
                            config.output_dir,
                        )
                    )

    manifest = {
        "config": str(Path(config_path).resolve()),
        "metadata_windows": len(metadata),
        "outputs": [str(path) for path in outputs],
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return config.output_dir
