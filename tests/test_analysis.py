"""Data-free tests for the analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trustme_et_comparison.analysis.config import REPRESENTATION_ORDER, load_analysis_config
from trustme_et_comparison.analysis.data import load_representation, project_matrix
from trustme_et_comparison.analysis.runner import run_analysis


def _write_fixture_data(root: Path) -> None:
    """Write small synthetic inputs for all representation families."""

    rng = np.random.default_rng(42)
    ids = [f"s{index // 6}|file|{index}" for index in range(12)]
    subjects = [f"s{index // 6}" for index in range(12)]
    labels = [index % 2 for index in range(12)]
    metadata = pd.DataFrame(
        {"window_uid": ids, "Subject": subjects, "5": labels, "f_0": rng.normal(size=12), "f_1": rng.normal(size=12)}
    )
    metadata.to_csv(root / "features.csv", index=False)

    raw_rows: list[dict[str, object]] = []
    for window_id, label in zip(ids, labels):
        for step in range(5):
            raw_rows.append(
                {
                    "window_uid": window_id,
                    "PupilSizeLeft": 3.0 + label + step / 10,
                    "PupilSizeRight": 3.1 + label + step / 10,
                    "GazePointX": step / 5,
                    "GazePointY": (5 - step) / 5,
                }
            )
    pd.DataFrame(raw_rows).to_csv(root / "raw.csv", index=False)
    for name, prefix in [("gazemae", "z_pos_"), ("moment", "moment_")]:
        frame = pd.DataFrame(rng.normal(size=(12, 4)), columns=[f"{prefix}{index}" for index in range(4)])
        frame.insert(0, "window_uid", ids)
        frame.to_csv(root / f"{name}.csv", index=False)


def _write_config(root: Path) -> Path:
    """Write a synthetic PCA-only analysis config."""

    config = root / "analysis.yaml"
    config.write_text(
        f"""
random_state: 42
output_dir: {root / 'output'}
subject_column: Subject
target_columns: [\"5\"]
metadata: {{path: {root / 'features.csv'}, id_column: window_uid}}
raw_samples: {{path: {root / 'raw.csv'}, id_column: window_uid}}
representations:
  - {{name: MOMENT embeddings, path: {root / 'moment.csv'}, feature_prefixes: [moment_]}}
  - name: raw
    kind: raw_samples
    path: {root / 'raw.csv'}
    channels: [PupilSizeLeft, PupilSizeRight, GazePointX, GazePointY]
    sequence_length: 4
  - {{name: GazeMAE embeddings, path: {root / 'gazemae.csv'}, feature_prefixes: [z_pos_]}}
  - {{name: handcrafted features, path: {root / 'features.csv'}, feature_prefixes: [f_]}}
plots:
  pupil_columns: [PupilSizeLeft, PupilSizeRight]
  gaze_columns: [GazePointX, GazePointY]
  projection_methods: [pca]
  color_by: [\"5\", Subject]
  max_raw_rows: 100
  max_projection_points: 100
""".strip(),
        encoding="utf-8",
    )
    return config


def test_config_enforces_representation_order(tmp_path: Path) -> None:
    """Canonical order is independent of the YAML input order."""

    _write_fixture_data(tmp_path)
    config = load_analysis_config(_write_config(tmp_path))
    assert tuple(item.name for item in config.representations) == REPRESENTATION_ORDER


def test_raw_samples_are_resampled_per_window(tmp_path: Path) -> None:
    """Raw samples become one fixed-width vector per window."""

    _write_fixture_data(tmp_path)
    config = load_analysis_config(_write_config(tmp_path))
    ids, matrix = load_representation(config.representations[0])
    assert len(ids) == 12
    assert matrix.shape == (12, 16)


def test_project_matrix_returns_two_components() -> None:
    """PCA projection has the expected shape and finite values."""

    matrix = np.arange(60, dtype=np.float32).reshape(12, 5)
    projection = project_matrix(matrix, "pca", random_state=42)
    assert projection.shape == (12, 2)
    assert np.isfinite(projection).all()


def test_analysis_smoke_run(tmp_path: Path) -> None:
    """The complete analysis creates plots, coordinates, and a manifest."""

    _write_fixture_data(tmp_path)
    output_dir = run_analysis(_write_config(tmp_path))
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "gaze_density.png").exists()
    assert len(list(output_dir.glob("projection_*.csv"))) == 4
