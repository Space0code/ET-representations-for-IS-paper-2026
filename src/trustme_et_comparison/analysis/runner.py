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
    plot_projection,
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
    if len(config.target_columns) != 1:
        raise ValueError("The paper analysis currently supports exactly one q5 target column.")
    target_column = config.target_columns[0]
    target_values = pd.to_numeric(metadata[target_column], errors="raise")
    global_median = float(target_values.median())
    binary_column = f"{config.target_name}_binary_global_median"
    metadata[binary_column] = (target_values > global_median).astype(int)

    outputs: list[Path] = []
    raw_output_dir = config.output_dir / f"raw_{config.target_name}"
    binary_output_dir = config.output_dir / f"binary_{config.target_name}"
    outputs.append(
        plot_class_distribution(
            metadata,
            target_column,
            "Raw q5 engagement rating",
            raw_output_dir / "class_distribution.png",
        )
    )
    outputs.append(
        plot_class_distribution(
            metadata,
            binary_column,
            "Binary q5 engagement rating",
            binary_output_dir / "class_distribution.png",
        )
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
            label_columns = [config.metadata.id_column]
            label_frame = metadata[label_columns].rename(
                columns={config.metadata.id_column: config.raw_samples.id_column}
            )
            raw = raw.merge(
                label_frame,
                on=config.raw_samples.id_column,
                how="left",
            )
        if all(column in raw.columns for column in config.gaze_columns):
            outputs.append(
                plot_gaze_density(raw, config.gaze_columns, raw_output_dir / "descriptive")
            )

    sampled_metadata = sample_rows(metadata, config.max_projection_points, config.random_state)
    allowed_ids = set(sampled_metadata[config.metadata.id_column].astype(str))
    for representation in config.representations:
        ids, matrix = load_representation(representation, allowed_ids=allowed_ids)
        index_frame = pd.DataFrame({config.metadata.id_column: ids, "matrix_index": range(len(ids))})
        joined = index_frame.merge(metadata, on=config.metadata.id_column, how="inner")
        joined = sample_rows(joined, config.max_projection_points, config.random_state)
        selected_matrix = matrix[joined["matrix_index"].to_numpy()]
        for method in config.methods:
            coordinates = project_matrix(selected_matrix, method, config.random_state)
            projection = joined.reset_index(drop=True).copy()
            projection[["component_1", "component_2"]] = coordinates
            for output_dir, color_column, target_label, continuous in [
                (raw_output_dir, target_column, "Raw q5 engagement rating", True),
                (binary_output_dir, binary_column, "Binary q5 engagement rating", False),
            ]:
                method_dir = output_dir / method
                projection_path = method_dir / (
                    f"projection_{representation.name.lower().replace(' ', '_')}.csv"
                )
                method_dir.mkdir(parents=True, exist_ok=True)
                projection.to_csv(projection_path, index=False)
                outputs.append(projection_path)
                outputs.append(
                    plot_projection(
                        projection,
                        representation.name,
                        method,
                        color_column,
                        target_label,
                        continuous,
                        method_dir,
                    )
                )

    manifest = {
        "config": str(Path(config_path).resolve()),
        "trustme_root": str(config.trustme_root) if config.trustme_root else None,
        "subject_export_dir": config.subject_export_dir,
        "subjects": list(config.subjects),
        "subject_count": len(config.subjects) or metadata[config.subject_column].nunique(),
        "metadata_windows": len(metadata),
        "raw_target_column": target_column,
        "global_median": global_median,
        "binary_target_definition": f"{target_column} > {global_median}",
        "outputs": [str(path) for path in outputs],
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return config.output_dir
