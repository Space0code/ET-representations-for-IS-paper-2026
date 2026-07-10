"""Data assembly helpers for TrustMe Tobii ET protocol evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...common.types import RepresentationDataset
from ..data_loading import (
    _normalize_window_id_value,
    build_variant_canonical_segments,
    load_embedding_payload,
    load_embeddings_representation,
    load_features_representation,
    load_raw_representation,
)
from .types import ZojaExperimentConfig


@dataclass
class _PathAdapter:
    """Minimal adapter expected by existing representation loaders."""

    features_csv: Path | None


@dataclass
class _LoaderAdapter:
    """Duck-typed adapter expected by existing representation loaders."""

    raw: Any
    features: Any
    paths: _PathAdapter


@dataclass
class ProtocolDatasetBundle:
    """Common aligned metadata and representation matrices."""

    metadata: pd.DataFrame
    representations: dict[str, RepresentationDataset]


def _resolve_time_column(frame: pd.DataFrame) -> str | None:
    """Return preferred timestamp column if available."""

    for candidate in ["start_ts", "TimeStamp", "timestamp", "time", "prompt_time"]:
        if candidate in frame.columns:
            return candidate
    return None


def _compute_per_window_targets(
    frame: pd.DataFrame,
    target_columns: list[str],
) -> pd.DataFrame:
    """Aggregate target columns at window level."""

    for col in target_columns:
        if col not in frame.columns:
            raise ValueError(f"Missing target column {col!r} in processed data.")

    grouped = frame.groupby("window_id_norm", sort=False)
    out = grouped[target_columns].first().reset_index()
    return out


def build_protocol_canonical(
    cfg: ZojaExperimentConfig,
    *,
    target_columns: list[str],
) -> pd.DataFrame:
    """Build canonical segment table with target source columns and temporal order."""

    rows: list[dict[str, object]] = []

    subject_dirs = sorted(path for path in cfg.paths.processed_root.iterdir() if path.is_dir())
    if not subject_dirs:
        raise ValueError(f"No subject directories found under {cfg.paths.processed_root}")

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        tobii_dir = subject_dir / "tobii"
        if not tobii_dir.exists():
            raise ValueError(f"Missing required directory: {tobii_dir}")

        parquet_files = sorted(tobii_dir.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {tobii_dir}")

        for file_idx, parquet_path in enumerate(parquet_files):
            cols = ["window_id", *target_columns]
            frame = pd.read_parquet(parquet_path)
            missing = sorted(set(cols) - set(frame.columns))
            if missing:
                raise ValueError(
                    f"File {parquet_path} missing required columns for targets: {missing}"
                )

            time_col = _resolve_time_column(frame)
            working = frame.copy()
            working["window_id_norm"] = working["window_id"].map(_normalize_window_id_value)

            per_window = _compute_per_window_targets(working, target_columns=target_columns)

            if time_col is not None:
                times = (
                    working.groupby("window_id_norm", sort=False)[time_col]
                    .min()
                    .reset_index()
                    .rename(columns={time_col: "start_ts"})
                )
                per_window = per_window.merge(times, on="window_id_norm", how="left")
            else:
                per_window["start_ts"] = np.arange(len(per_window), dtype=np.int64)

            # Keep deterministic subject-local temporal ordering even when timestamps are absent/duplicate.
            per_window["_window_order"] = np.arange(len(per_window), dtype=np.int64)

            for rec in per_window.to_dict(orient="records"):
                window_id_str = str(rec["window_id_norm"])
                segment_id = f"{subject}|{parquet_path.name}|{window_id_str}"
                row = {
                    "segment_id": segment_id,
                    "Subject": subject,
                    "Label": str(rec[target_columns[0]]),
                    "file_path": str(parquet_path),
                    "window_id": window_id_str,
                    "file_index": file_idx,
                    "window_order": int(rec["_window_order"]),
                    "start_ts": rec["start_ts"],
                }
                for col in target_columns:
                    row[col] = rec[col]
                rows.append(row)

    if not rows:
        raise ValueError("No canonical segments produced.")

    canonical = pd.DataFrame(rows)
    canonical.sort_values(["Subject", "file_index", "window_order"], inplace=True)
    canonical.reset_index(drop=True, inplace=True)
    return canonical


def _subset_dataset(dataset: RepresentationDataset, indices: np.ndarray) -> RepresentationDataset:
    """Return representation dataset subset by index array."""

    if isinstance(dataset.X, pd.DataFrame):
        X_new: Any = dataset.X.iloc[indices].copy()
    elif isinstance(dataset.X, list):
        X_new = [dataset.X[int(i)] for i in indices.tolist()]
    else:
        X_new = dataset.X[indices]

    return RepresentationDataset(
        name=dataset.name,
        X=X_new,
        segment_ids=dataset.segment_ids[indices].copy(),
        subjects=dataset.subjects[indices].copy(),
        source_labels=dataset.source_labels[indices].copy(),
    )


def load_protocol_representations(
    cfg: ZojaExperimentConfig,
    canonical: pd.DataFrame,
) -> dict[str, RepresentationDataset]:
    """Load configured representations and align IDs against canonical segments."""

    adapter = _LoaderAdapter(
        raw=cfg.raw,
        features=cfg.features,
        paths=_PathAdapter(features_csv=cfg.paths.features_csv),
    )

    out: dict[str, RepresentationDataset] = {}

    if "raw" in cfg.representations:
        out["raw"] = load_raw_representation(canonical=canonical, cfg=adapter)

    if "features" in cfg.representations:
        if cfg.paths.features_csv is None and cfg.allow_missing_features:
            pass
        elif cfg.paths.features_csv is None and not cfg.allow_missing_features:
            raise ValueError("features representation requested but paths.features_csv is null.")
        elif cfg.paths.features_csv is not None and not cfg.paths.features_csv.exists() and cfg.allow_missing_features:
            pass
        else:
            out["features"] = load_features_representation(canonical=canonical, cfg=adapter)

    if "embeddings" in cfg.representations:
        if not cfg.embedding_variants:
            raise ValueError("embeddings representation requested but no embedding_variants are configured.")

        for variant in cfg.embedding_variants:
            payload = load_embedding_payload(variant)
            variant_canonical = build_variant_canonical_segments(
                base_canonical=canonical,
                embedding_segment_ids=payload["segment_id"],
            )
            ds = load_embeddings_representation(
                canonical=variant_canonical,
                embedding_payload=payload,
            )
            out[f"embeddings_{variant.tag}"] = ds

    if not out:
        raise ValueError("No representations were loaded.")

    return out


def align_bundle_to_common_ids(
    canonical: pd.DataFrame,
    representations: dict[str, RepresentationDataset],
) -> ProtocolDatasetBundle:
    """Intersect all representation IDs and return aligned metadata + datasets."""

    common: set[str] | None = None
    for dataset in representations.values():
        ids = set(dataset.segment_ids.astype(str).tolist())
        if common is None:
            common = ids
        else:
            common &= ids

    assert common is not None
    if not common:
        raise ValueError("No common segment IDs across requested representations.")

    canonical_common = canonical[canonical["segment_id"].isin(common)].copy()
    canonical_common.sort_values(["Subject", "file_index", "window_order"], inplace=True)
    canonical_common.reset_index(drop=True, inplace=True)
    ordered_ids = canonical_common["segment_id"].to_numpy(dtype=str)

    aligned: dict[str, RepresentationDataset] = {}
    for name, dataset in representations.items():
        id_to_pos = {segment_id: idx for idx, segment_id in enumerate(dataset.segment_ids.astype(str).tolist())}
        idx = np.asarray([id_to_pos[sid] for sid in ordered_ids], dtype=np.int64)
        aligned[name] = _subset_dataset(dataset, idx)

    return ProtocolDatasetBundle(metadata=canonical_common, representations=aligned)
