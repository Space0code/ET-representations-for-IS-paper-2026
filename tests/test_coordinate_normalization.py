"""Tests for per-recording Tobii screen-coordinate normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tobii_experiments.normalize_tobii_coordinates import (
    Resolution,
    load_subject_resolutions,
    normalize_subject,
)


def _write_source_tsv(path: Path, resolution: str) -> None:
    """Write the header fields required by the normalizer."""
    path.write_text(
        f"pygaze initiation report start\ndisplay resolution: {resolution}\n",
        encoding="utf-8",
    )


def test_normalization_uses_the_resolution_of_each_source_file(tmp_path: Path) -> None:
    """A subject with two displays is normalized session by session."""
    root = tmp_path / "TrustMe"
    subject_directory = root / "participant" 
    source_directory = subject_directory / "ml" / "tobii"
    raw_tobii_directory = subject_directory / "tobii"
    source_directory.mkdir(parents=True)
    raw_tobii_directory.mkdir()

    first_source = "recording_one.tsv.parquet"
    second_source = "recording_two.tsv.parquet"
    _write_source_tsv(raw_tobii_directory / "recording_one.tsv", "100x200")
    _write_source_tsv(raw_tobii_directory / "recording_two.tsv", "200x100")
    raw = pd.DataFrame(
        {
            "source_file": [first_source, second_source],
            "GazePointXLeft": [50.0, 100.0],
            "GazePointYLeft": [100.0, 50.0],
            "GazePointXRight": [25.0, 50.0],
            "GazePointYRight": [50.0, 25.0],
            "GazePointX": [50.0, 100.0],
            "GazePointY": [100.0, 50.0],
            "PupilSizeLeft": [3.1, 3.2],
        }
    )
    raw.to_csv(source_directory / "tobii_raw_samples.csv", index=False)
    (source_directory / "tobii_features.csv").write_text("preserved\n", encoding="utf-8")

    metadata_path = root / "display_resolutions.yaml"
    metadata_path.write_text(
        "100x200:\n  - participant\n200x100:\n  - participant\n",
        encoding="utf-8",
    )
    report = normalize_subject(
        source_directory=source_directory,
        destination_directory_name="tobii_coordinate_normalized",
        subject_resolutions=load_subject_resolutions(metadata_path),
        raw_filename="tobii_raw_samples.csv",
        chunk_size=1,
        overwrite=False,
    )

    normalized_directory = subject_directory / "ml" / "tobii_coordinate_normalized"
    normalized = pd.read_csv(normalized_directory / "tobii_raw_samples.csv")
    assert normalized["GazePointX"].tolist() == [0.5, 0.5]
    assert normalized["GazePointY"].tolist() == [0.5, 0.5]
    assert normalized["GazePointXLeft"].tolist() == [0.5, 0.5]
    assert normalized["GazePointYRight"].tolist() == [0.25, 0.25]
    assert normalized["PupilSizeLeft"].tolist() == [3.1, 3.2]
    assert (normalized_directory / "tobii_features.csv").read_text(encoding="utf-8") == "preserved\n"
    assert report.resolutions == (Resolution(100, 200), Resolution(200, 100))
    assert report.in_unit_square_count == 2
