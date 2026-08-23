"""Data assembly helpers for TrustMe Tobii ET protocol evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...common.types import RepresentationDataset
from ..data_loading import (
    _apply_validity_masks,
    _normalize_window_id_value,
    _resolve_raw_target_length,
    _vectorize_window,
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


SUBJECT_TREE_FILENAMES = {
    "raw": "tobii_raw_samples.csv",
    "features": "tobii_features.csv",
    "gazemae": "tobii_gazemae_embeddings.csv",
    "moment": "tobii_moment_embeddings.csv",
}

SUBJECT_TREE_METADATA_COLUMNS = [
    "window_uid",
    "window_id",
    "subject",
    "source_file",
    "start_timestamp",
]


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

    if cfg.paths.processed_root is None:
        raise ValueError("paths.processed_root is required for processed-parquet input.")
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


def _subject_tree_export_paths(cfg: ZojaExperimentConfig) -> dict[str, dict[str, Path]]:
    """Resolve and strictly validate the frozen subject-tree cohort."""

    root = cfg.paths.subject_tree_root
    subjects = cfg.paths.subjects
    if root is None or subjects is None:
        raise ValueError("Subject-tree input requires paths.subject_tree_root and paths.subjects.")
    if not root.is_dir():
        raise ValueError(f"Subject-tree root is not a directory: {root}")

    outputs: dict[str, dict[str, Path]] = {}
    missing: list[str] = []
    for subject in subjects:
        export_dir = root / subject / "ml" / cfg.paths.subject_export_dir
        files = {key: export_dir / filename for key, filename in SUBJECT_TREE_FILENAMES.items()}
        absent = [path.name for path in files.values() if not path.is_file()]
        if absent:
            missing.append(f"{subject}: {', '.join(absent)}")
        outputs[subject] = files
    if missing:
        raise ValueError("Missing frozen-cohort exports:\n  - " + "\n  - ".join(missing))
    return outputs


def _read_subject_tree_features(
    cfg: ZojaExperimentConfig,
    paths: dict[str, dict[str, Path]],
    target_columns: list[str],
) -> tuple[pd.DataFrame, RepresentationDataset]:
    """Load canonical metadata and handcrafted-feature rows from subject exports."""

    frames: list[pd.DataFrame] = []
    for subject in cfg.paths.subjects or []:
        path = paths[subject]["features"]
        frame = pd.read_csv(path, low_memory=False)
        required = {*SUBJECT_TREE_METADATA_COLUMNS, *target_columns}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Feature export {path} is missing columns: {missing}")
        if frame["window_uid"].duplicated().any():
            raise ValueError(f"Feature export contains duplicate window_uid values: {path}")
        observed_subjects = set(frame["subject"].astype(str).unique())
        if observed_subjects != {subject}:
            raise ValueError(
                f"Feature export subject mismatch for {path}: expected={subject}, observed={sorted(observed_subjects)}"
            )
        frames.append(frame)

    features = pd.concat(frames, ignore_index=True).copy()
    if features["window_uid"].duplicated().any():
        raise ValueError("Feature exports contain duplicate window_uid values across subjects.")

    features["_subject_order"] = pd.Categorical(
        features["subject"],
        categories=cfg.paths.subjects,
        ordered=True,
    )
    features.sort_values(
        ["_subject_order", "source_file", "start_timestamp", "window_id"],
        kind="stable",
        inplace=True,
    )
    features.reset_index(drop=True, inplace=True)

    canonical = pd.DataFrame(
        {
            "segment_id": features["window_uid"].astype(str),
            "Subject": features["subject"].astype(str),
            "Label": features[target_columns[0]].astype(str),
            "file_path": features["source_file"].astype(str),
            "file_index": features.groupby("subject", sort=False)["source_file"].transform(
                lambda values: pd.factorize(values, sort=False)[0]
            ),
            "window_order": features.groupby("subject", sort=False).cumcount(),
            "start_ts": pd.to_numeric(features["start_timestamp"], errors="coerce"),
        }
    )
    for column in target_columns:
        canonical[column] = pd.to_numeric(features[column], errors="coerce")

    feature_matrix = features.drop(columns=["_subject_order"])
    ids = canonical["segment_id"].to_numpy(dtype=str)
    subjects = canonical["Subject"].to_numpy(dtype=str)
    labels = canonical["Label"].to_numpy(dtype=str)
    dataset = RepresentationDataset(
        name="features",
        X=feature_matrix,
        segment_ids=ids,
        subjects=subjects,
        source_labels=labels,
    )
    return canonical, dataset


def _read_subject_tree_embedding(
    *,
    cfg: ZojaExperimentConfig,
    paths: dict[str, dict[str, Path]],
    key: str,
    prefixes: tuple[str, ...],
) -> RepresentationDataset:
    """Load one window-level embedding family from frozen subject exports."""

    ids_parts: list[np.ndarray] = []
    matrix_parts: list[np.ndarray] = []
    expected_columns: list[str] | None = None
    for subject in cfg.paths.subjects or []:
        path = paths[subject][key]
        header = pd.read_csv(path, nrows=0).columns.tolist()
        feature_columns = [
            column for column in header if any(str(column).startswith(prefix) for prefix in prefixes)
        ]
        if not feature_columns:
            raise ValueError(f"No {key} embedding columns found in {path}")
        if expected_columns is None:
            expected_columns = feature_columns
        elif feature_columns != expected_columns:
            raise ValueError(f"Inconsistent {key} embedding columns in {path}")

        frame = pd.read_csv(path, usecols=["window_uid", *feature_columns], low_memory=False)
        if frame["window_uid"].duplicated().any():
            raise ValueError(f"Embedding export contains duplicate window_uid values: {path}")
        ids_parts.append(frame["window_uid"].astype(str).to_numpy())
        matrix_parts.append(frame[feature_columns].to_numpy(dtype=np.float32))

    ids = np.concatenate(ids_parts)
    if np.unique(ids).shape[0] != ids.shape[0]:
        raise ValueError(f"{key} exports contain duplicate window_uid values across subjects.")
    matrix = np.vstack(matrix_parts).astype(np.float32, copy=False)
    subjects = np.asarray([value.split("|", 1)[0] for value in ids], dtype=str)
    return RepresentationDataset(
        name=f"embeddings_{key}",
        X=matrix,
        segment_ids=ids,
        subjects=subjects,
        source_labels=np.full(ids.shape[0], "", dtype=str),
    )


def _iter_raw_windows(path: Path, columns: list[str], chunksize: int = 250_000):
    """Yield complete raw windows while preserving groups split across CSV chunks."""

    carry = pd.DataFrame(columns=columns)
    for chunk in pd.read_csv(path, usecols=columns, low_memory=False, chunksize=chunksize):
        combined = pd.concat([carry, chunk], ignore_index=True) if not carry.empty else chunk
        if combined.empty:
            continue
        last_id = str(combined["window_uid"].iloc[-1])
        is_last = combined["window_uid"].astype(str) == last_id
        ready = combined.loc[~is_last]
        carry = combined.loc[is_last].copy()
        for window_uid, window in ready.groupby("window_uid", sort=False):
            yield str(window_uid), window.reset_index(drop=True)
    if not carry.empty:
        for window_uid, window in carry.groupby("window_uid", sort=False):
            yield str(window_uid), window.reset_index(drop=True)


def _fast_clean_truncate_pad(values: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly impute and truncate/pad one channel without pandas overhead."""

    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(target_len, dtype=np.float32)
    if not finite.all():
        index = np.arange(array.shape[0], dtype=np.float32)
        array = np.interp(index, index[finite], array[finite]).astype(np.float32)
    if array.shape[0] >= target_len:
        return array[:target_len].astype(np.float32, copy=False)
    return np.pad(array, (0, target_len - array.shape[0]), mode="edge").astype(np.float32)


def _fast_vectorize_subject_tree_window(
    *,
    window: pd.DataFrame,
    channels: list[str],
    gaze_channels: list[str],
    pupil_channels: list[str],
    validity_gaze: list[str],
    validity_pupil: list[str],
    valid_val: int,
    target_len: int,
) -> np.ndarray:
    """Vectorize one fixed-length raw window with the standard validity policy."""

    gaze_invalid = np.zeros(len(window), dtype=bool)
    for column in validity_gaze:
        gaze_invalid |= window[column].to_numpy() != valid_val
    pupil_invalid = np.zeros(len(window), dtype=bool)
    for column in validity_pupil:
        pupil_invalid |= window[column].to_numpy() != valid_val

    vectors: list[np.ndarray] = []
    for channel in channels:
        values = window[channel].to_numpy(dtype=np.float32, copy=True)
        if channel in gaze_channels:
            values[gaze_invalid] = np.nan
        if channel in pupil_channels:
            values[pupil_invalid] = np.nan
        vectors.append(_fast_clean_truncate_pad(values, target_len))
    return np.concatenate(vectors, axis=0)


def _read_subject_tree_raw(
    *,
    cfg: ZojaExperimentConfig,
    paths: dict[str, dict[str, Path]],
    allowed_ids: set[str],
) -> RepresentationDataset:
    """Stream coordinate-normalized raw exports into fixed-width window vectors."""

    channels = cfg.raw.channels
    gaze_channels = [channel for channel in channels if "gaze" in channel.lower()]
    pupil_channels = [channel for channel in channels if "pupil" in channel.lower()]
    columns = list(
        dict.fromkeys(
            [
                "window_uid",
                *channels,
                *cfg.raw.validity_gaze_columns,
                *cfg.raw.validity_pupil_columns,
            ]
        )
    )
    target_len = _resolve_raw_target_length(cfg)
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    seen: set[str] = set()
    for subject in cfg.paths.subjects or []:
        path = paths[subject]["raw"]
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = sorted(set(columns) - header)
        if missing:
            raise ValueError(f"Raw export {path} is missing columns: {missing}")
        for window_uid, window in _iter_raw_windows(path, columns):
            if window_uid not in allowed_ids:
                continue
            if window_uid in seen:
                raise ValueError(f"Raw exports contain duplicate/non-contiguous window_uid: {window_uid}")
            seen.add(window_uid)
            ids.append(window_uid)
            if cfg.raw.length_mode == "truncate_pad":
                vectors.append(
                    _fast_vectorize_subject_tree_window(
                        window=window,
                        channels=channels,
                        gaze_channels=gaze_channels,
                        pupil_channels=pupil_channels,
                        validity_gaze=cfg.raw.validity_gaze_columns,
                        validity_pupil=cfg.raw.validity_pupil_columns,
                        valid_val=cfg.raw.validity_valid_value,
                        target_len=target_len,
                    )
                )
            else:
                masked = _apply_validity_masks(
                    window,
                    gaze_channels=gaze_channels,
                    pupil_channels=pupil_channels,
                    validity_gaze=cfg.raw.validity_gaze_columns,
                    validity_pupil=cfg.raw.validity_pupil_columns,
                    valid_val=cfg.raw.validity_valid_value,
                )
                vectors.append(
                    _vectorize_window(
                        window=masked,
                        channels=channels,
                        target_len=target_len,
                        length_mode=cfg.raw.length_mode,
                    )
                )

    missing_ids = allowed_ids - seen
    if missing_ids:
        sample = sorted(missing_ids)[:5]
        raise ValueError(f"Raw exports are missing {len(missing_ids)} common windows; sample={sample}")
    ids_array = np.asarray(ids, dtype=str)
    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    subjects = np.asarray([value.split("|", 1)[0] for value in ids_array], dtype=str)
    return RepresentationDataset(
        name="raw",
        X=matrix,
        segment_ids=ids_array,
        subjects=subjects,
        source_labels=np.full(ids_array.shape[0], "", dtype=str),
    )


def load_subject_tree_bundle(
    cfg: ZojaExperimentConfig,
    target_columns: list[str],
) -> ProtocolDatasetBundle:
    """Load and align the frozen coordinate-normalized subject-tree cohort."""

    paths = _subject_tree_export_paths(cfg)
    canonical, features = _read_subject_tree_features(cfg, paths, target_columns)
    complete_target = np.ones(len(canonical), dtype=bool)
    for column in target_columns:
        complete_target &= pd.to_numeric(canonical[column], errors="coerce").notna().to_numpy()
    target_idx = np.where(complete_target)[0].astype(np.int64)
    canonical = canonical.iloc[target_idx].copy().reset_index(drop=True)
    features = _subset_dataset(features, target_idx)
    if canonical.empty:
        raise ValueError("No labelled subject-tree windows remain for the configured target.")

    representations: dict[str, RepresentationDataset] = {}
    if "features" in cfg.representations:
        representations["features"] = features
    if "embeddings" in cfg.representations:
        representations["embeddings_gazemae"] = _read_subject_tree_embedding(
            cfg=cfg,
            paths=paths,
            key="gazemae",
            prefixes=("z_pos_", "z_vel_"),
        )
        representations["embeddings_moment"] = _read_subject_tree_embedding(
            cfg=cfg,
            paths=paths,
            key="moment",
            prefixes=("moment_",),
        )
    if "raw" in cfg.representations:
        common_ids = set(canonical["segment_id"].astype(str))
        for dataset in representations.values():
            common_ids &= set(dataset.segment_ids.astype(str))
        representations["raw"] = _read_subject_tree_raw(
            cfg=cfg,
            paths=paths,
            allowed_ids=common_ids,
        )
    if not representations:
        raise ValueError("No subject-tree representations were requested.")
    return align_bundle_to_common_ids(canonical=canonical, representations=representations)
