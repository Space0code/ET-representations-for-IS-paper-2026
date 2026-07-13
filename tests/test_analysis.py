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
                    "AverageDistance": 60.0 + step,
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


def _write_subject_tree(root: Path) -> Path:
    """Write two standard per-subject export directories."""

    source = root / "source"
    source.mkdir()
    _write_fixture_data(source)
    trustme_root = root / "TrustMe"
    features = pd.read_csv(source / "features.csv")
    raw = pd.read_csv(source / "raw.csv")
    gazemae = pd.read_csv(source / "gazemae.csv")
    moment = pd.read_csv(source / "moment.csv")
    for subject in ["s0", "s1"]:
        output = trustme_root / subject / "ml" / "tobii"
        output.mkdir(parents=True)
        ids = set(features.loc[features["Subject"] == subject, "window_uid"])
        features.loc[features["window_uid"].isin(ids)].rename(columns={"Subject": "subject"}).to_csv(
            output / "tobii_features.csv", index=False
        )
        raw.loc[raw["window_uid"].isin(ids)].to_csv(output / "tobii_raw_samples.csv", index=False)
        gazemae.loc[gazemae["window_uid"].isin(ids)].to_csv(
            output / "tobii_gazemae_embeddings.csv", index=False
        )
        moment.loc[moment["window_uid"].isin(ids)].to_csv(
            output / "tobii_moment_embeddings.csv", index=False
        )
    return trustme_root


def _write_subject_tree_config(root: Path, trustme_root: Path) -> Path:
    """Write a root-only subject-tree analysis config."""

    path = root / "tree_analysis.yaml"
    path.write_text(
        f"""
trustme_root: {trustme_root}
output_dir: {root / 'tree_output'}
subject_column: subject
target_columns: [\"5\"]
plots:
  pupil_columns: [PupilSizeLeft, PupilSizeRight]
  gaze_columns: [GazePointX, GazePointY]
  projection_methods: [pca]
  color_by: [\"5\", subject]
  max_raw_rows: 100
  max_projection_points: 100
""".strip(),
        encoding="utf-8",
    )
    return path


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
    assert (output_dir / "raw_q5" / "descriptive" / "gaze_density.png").exists()
    assert (output_dir / "raw_q5" / "class_distribution.png").exists()
    assert (output_dir / "binary_q5" / "class_distribution.png").exists()
    assert len(list(output_dir.glob("**/projection_*.csv"))) == 8


def test_subject_tree_is_discovered_and_combined(tmp_path: Path) -> None:
    """One root parameter discovers all subjects and standard representations."""

    trustme_root = _write_subject_tree(tmp_path)
    config_path = _write_subject_tree_config(tmp_path, trustme_root)
    config = load_analysis_config(config_path)
    assert config.subjects == ("s0", "s1")
    assert all(len(representation.paths) == 2 for representation in config.representations)

    output_dir = run_analysis(config_path)
    manifest = pd.read_json(output_dir / "manifest.json", typ="series")
    assert manifest["subject_count"] == 2


def test_subject_tree_reports_all_missing_exports(tmp_path: Path) -> None:
    """Incomplete subject exports fail before analysis starts."""

    trustme_root = tmp_path / "TrustMe"
    (trustme_root / "s0" / "ml" / "tobii").mkdir(parents=True)
    config_path = _write_subject_tree_config(tmp_path, trustme_root)
    try:
        load_analysis_config(config_path)
    except ValueError as exc:
        message = str(exc)
        assert "s0" in message
        assert "tobii_raw_samples.csv" in message
        assert "tobii_moment_embeddings.csv" in message
    else:
        raise AssertionError("Expected missing subject exports to raise ValueError")
