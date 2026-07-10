"""TrustMe data loading and representation builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.signal_processing import clean_and_resize_channel
from ..common.types import RepresentationDataset
from .types import TrustMeEmbeddingVariantConfig, TrustMeExperimentConfig


def _normalize_label_value(value: Any) -> str | None:
    """Convert one label value to canonical string, or None for missing."""

    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_window_id_value(value: Any) -> str:
    """Convert one window identifier to a stable string representation."""

    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _coerce_processed_filename(value: Any) -> str:
    """Convert filename-like values to processed parquet filename strings."""

    filename = str(value)
    if filename.endswith(".parquet"):
        return filename
    return f"{filename}.parquet"


def _iter_subject_dirs(cfg: TrustMeExperimentConfig) -> list[Path]:
    """Return selected subject directories under processed root."""

    if cfg.task.subjects is None:
        subject_dirs = sorted(path for path in cfg.paths.processed_root.iterdir() if path.is_dir())
    else:
        subject_dirs = [cfg.paths.processed_root / subject for subject in cfg.task.subjects]
    return subject_dirs


def build_base_canonical_segments(cfg: TrustMeExperimentConfig) -> pd.DataFrame:
    """Build base TrustMe segment table from processed parquet files.

    Each segment corresponds to one ``window_id`` within one processed file.
    """

    rows: list[dict[str, Any]] = []

    for subject_dir in _iter_subject_dirs(cfg):
        subject = subject_dir.name
        tobii_dir = subject_dir / "tobii"
        if not tobii_dir.exists():
            continue

        for parquet_path in sorted(tobii_dir.glob("*.parquet")):
            meta = pd.read_parquet(parquet_path, columns=["window_id", cfg.task.target_column])
            if meta.empty:
                continue

            per_window = (
                meta.groupby("window_id", sort=False)[cfg.task.target_column]
                .first()
                .reset_index()
            )

            for record in per_window.to_dict(orient="records"):
                label = _normalize_label_value(record[cfg.task.target_column])
                if label is None:
                    continue

                window_id_str = _normalize_window_id_value(record["window_id"])
                rows.append(
                    {
                        "segment_id": f"{subject}|{parquet_path.name}|{window_id_str}",
                        "Subject": subject,
                        "Label": label,
                        "file_path": str(parquet_path),
                        "window_id": window_id_str,
                    }
                )

    if not rows:
        raise ValueError(
            "No TrustMe canonical segments found. "
            "Run preprocessing first or check task.target_column."
        )

    canonical = pd.DataFrame(rows)
    canonical.sort_values(["Subject", "file_path", "window_id"], inplace=True)
    canonical.reset_index(drop=True, inplace=True)

    return canonical


def build_variant_canonical_segments(
    base_canonical: pd.DataFrame,
    embedding_segment_ids: np.ndarray,
) -> pd.DataFrame:
    """Build canonical segments by intersecting base windows with one embedding set."""

    if embedding_segment_ids.ndim != 1:
        raise ValueError("embedding_segment_ids must be a 1D array.")
    embedding_ids = embedding_segment_ids.astype(str)

    unique_ids = np.unique(embedding_ids)
    if unique_ids.shape[0] != embedding_ids.shape[0]:
        raise ValueError("Embedding payload contains duplicate segment IDs.")

    embedding_set = set(embedding_ids.tolist())
    canonical = base_canonical[base_canonical["segment_id"].isin(embedding_set)].copy()
    canonical.reset_index(drop=True, inplace=True)

    if canonical.empty:
        raise ValueError(
            "No overlap between TrustMe processed windows and embedding segment IDs."
        )
    return canonical


def _validate_embedding_payload(
    *,
    segment_ids: np.ndarray,
    embeddings: np.ndarray,
    source_name: Path | str,
) -> dict[str, np.ndarray]:
    """Validate basic payload shape and uniqueness constraints."""

    if segment_ids.ndim != 1:
        raise ValueError(f"Embedding source {source_name} has non-1D segment IDs.")
    if embeddings.ndim != 2:
        raise ValueError(f"Embedding source {source_name} has non-2D embeddings array.")
    if embeddings.shape[0] != segment_ids.shape[0]:
        raise ValueError(
            f"Embedding source {source_name} has row mismatch: embeddings={embeddings.shape[0]}, "
            f"segment_id={segment_ids.shape[0]}"
        )

    unique_ids = np.unique(segment_ids)
    if unique_ids.shape[0] != segment_ids.shape[0]:
        raise ValueError(f"Embedding source {source_name} contains duplicate segment IDs.")

    return {
        "segment_id": segment_ids.astype(str),
        "embeddings": embeddings.astype(np.float32),
    }


def _load_npz_embedding_payload(path: Path) -> dict[str, np.ndarray]:
    """Load standardized NPZ embeddings containing canonical segment IDs."""

    payload = np.load(path, allow_pickle=False)
    required = {"segment_id", "embeddings"}
    missing = sorted(required - set(payload.files))
    if missing:
        raise ValueError(f"Embedding file {path} is missing required arrays: {missing}")
    return _validate_embedding_payload(
        segment_ids=payload["segment_id"].astype(str),
        embeddings=payload["embeddings"].astype(np.float32),
        source_name=path,
    )


def _load_moment_embedding_payload(variant: TrustMeEmbeddingVariantConfig) -> dict[str, np.ndarray]:
    """Load MOMENT NPZ + metadata and map IDs to TrustMe canonical window_uids."""

    if variant.metadata_path is None:
        raise ValueError(
            f"Variant '{variant.tag}' with kind='moment_npz' requires metadata_path."
        )

    payload = np.load(variant.path, allow_pickle=False)
    required = {"segment_id", "embeddings"}
    missing = sorted(required - set(payload.files))
    if missing:
        raise ValueError(f"MOMENT embedding file {variant.path} missing arrays: {missing}")

    metadata = pd.read_parquet(
        variant.metadata_path,
        columns=["segment_id", "Subject", "Filename", "window_id"],
    )
    if len(metadata) != payload["segment_id"].shape[0]:
        raise ValueError(
            f"MOMENT metadata rows ({len(metadata)}) do not match embedding rows "
            f"({payload['segment_id'].shape[0]})."
        )

    metadata_segment_ids = metadata["segment_id"].astype(str).to_numpy()
    payload_segment_ids = payload["segment_id"].astype(str)
    if not np.array_equal(metadata_segment_ids, payload_segment_ids):
        raise ValueError(
            f"MOMENT metadata segment_id order does not match embedding NPZ ({variant.path})."
        )

    canonical_ids = np.asarray(
        [
            f"{subject}|{_coerce_processed_filename(filename)}|{_normalize_window_id_value(window_id)}"
            for subject, filename, window_id in zip(
                metadata["Subject"].astype(str),
                metadata["Filename"].astype(str),
                metadata["window_id"].tolist(),
            )
        ],
        dtype=str,
    )

    return _validate_embedding_payload(
        segment_ids=canonical_ids,
        embeddings=payload["embeddings"].astype(np.float32),
        source_name=variant.path,
    )


def _load_gazemae_parquet_payload(variant: TrustMeEmbeddingVariantConfig) -> dict[str, np.ndarray]:
    """Load GazeMAE parquet embeddings with configurable ID and feature columns."""

    id_column = variant.id_column or "window_uid"
    frame = pd.read_parquet(variant.path)
    if id_column not in frame.columns:
        raise ValueError(
            f"GazeMAE variant '{variant.tag}' missing id column '{id_column}' in {variant.path}."
        )

    prefixes = variant.feature_prefixes or ["z_pos_", "z_vel_"]
    embedding_columns = [
        col
        for col in frame.columns
        if any(col.startswith(prefix) for prefix in prefixes)
    ]
    if not embedding_columns:
        raise ValueError(
            f"GazeMAE variant '{variant.tag}' has no embedding columns for prefixes={prefixes}."
        )

    segment_ids = frame[id_column].astype(str).to_numpy()
    embeddings = frame[embedding_columns].to_numpy(dtype=np.float32)
    return _validate_embedding_payload(
        segment_ids=segment_ids,
        embeddings=embeddings,
        source_name=variant.path,
    )


def load_embedding_payload(variant: TrustMeEmbeddingVariantConfig) -> dict[str, np.ndarray]:
    """Load one embedding variant and return canonical IDs + embedding matrix."""

    kind = variant.kind.strip().lower()
    if kind == "npz":
        return _load_npz_embedding_payload(variant.path)
    if kind == "moment_npz":
        return _load_moment_embedding_payload(variant)
    if kind == "gazemae_parquet":
        return _load_gazemae_parquet_payload(variant)
    raise ValueError(
        f"Unsupported TrustMe embedding variant kind '{variant.kind}'. "
        "Supported kinds: npz, moment_npz, gazemae_parquet."
    )


def _resolve_raw_target_length(cfg: TrustMeExperimentConfig) -> int:
    """Return effective fixed output length for raw representation."""

    if cfg.raw.length_mode == "truncate_pad":
        if cfg.raw.flatten_length is None:
            raise ValueError("raw.flatten_length is required when raw.length_mode='truncate_pad'.")
        return int(cfg.raw.flatten_length)
    return int(cfg.raw.resample_length)


def _apply_validity_masks(
    window: pd.DataFrame,
    *,
    gaze_channels: list[str],
    pupil_channels: list[str],
    validity_gaze: list[str],
    validity_pupil: list[str],
    valid_val: int,
) -> pd.DataFrame:
    """Mask invalid gaze/pupil values in one window."""

    out = window.copy()
    for validity_col in validity_gaze:
        if validity_col in out.columns:
            invalid = out[validity_col] != valid_val
            for channel in gaze_channels:
                if channel in out.columns:
                    out.loc[invalid, channel] = np.nan

    for validity_col in validity_pupil:
        if validity_col in out.columns:
            invalid = out[validity_col] != valid_val
            for channel in pupil_channels:
                if channel in out.columns:
                    out.loc[invalid, channel] = np.nan

    return out


def _vectorize_window(
    window: pd.DataFrame,
    *,
    channels: list[str],
    target_len: int,
    length_mode: str,
) -> np.ndarray:
    """Convert one window table into one fixed-size flattened feature vector."""

    channel_vectors = [
        clean_and_resize_channel(
            values=window[channel].to_numpy(),
            target_length=target_len,
            length_mode=length_mode,
        )
        for channel in channels
    ]
    return np.concatenate(channel_vectors, axis=0)


def _matrixize_window_rows(
    window: pd.DataFrame,
    *,
    channels: list[str],
) -> np.ndarray:
    """Convert one window table into a cleaned variable-length row matrix."""

    if window.empty:
        return np.zeros((0, len(channels)), dtype=np.float32)

    row_count = int(len(window))
    channel_vectors = [
        clean_and_resize_channel(
            values=window[channel].to_numpy(),
            target_length=row_count,
            length_mode="truncate_pad",
        )
        for channel in channels
    ]
    return np.column_stack(channel_vectors).astype(np.float32)


def _load_raw_representation_from_parquet(
    canonical: pd.DataFrame,
    cfg: TrustMeExperimentConfig,
) -> RepresentationDataset:
    """Load raw representation from per-file parquet windows."""

    channels = cfg.raw.channels
    target_len = _resolve_raw_target_length(cfg)
    validity_gaze = cfg.raw.validity_gaze_columns
    validity_pupil = cfg.raw.validity_pupil_columns
    valid_val = cfg.raw.validity_valid_value
    length_mode = cfg.raw.length_mode

    gaze_channels = [channel for channel in channels if "gaze" in channel.lower()]
    pupil_channels = [channel for channel in channels if "pupil" in channel.lower()]
    cols_to_load = list(dict.fromkeys(channels + validity_gaze + validity_pupil + ["window_id"]))

    feature_rows: list[np.ndarray] = []
    canonical_by_file = canonical.groupby("file_path", sort=False)

    for file_path, file_rows in canonical_by_file:
        parquet_path = Path(file_path)
        ordered_window_ids = file_rows["window_id"].astype(str).tolist()
        needed_window_ids = set(ordered_window_ids)

        frame = pd.read_parquet(parquet_path, columns=cols_to_load)
        frame["window_id_norm"] = frame["window_id"].map(_normalize_window_id_value)
        frame = frame[frame["window_id_norm"].isin(needed_window_ids)].copy()

        grouped = {
            str(window_key): group.reset_index(drop=True)
            for window_key, group in frame.groupby("window_id_norm", sort=False)
        }

        for window_id in ordered_window_ids:
            if window_id not in grouped:
                raise ValueError(f"Missing window_id={window_id} in {parquet_path}")
            window = _apply_validity_masks(
                grouped[window_id].copy(),
                gaze_channels=gaze_channels,
                pupil_channels=pupil_channels,
                validity_gaze=validity_gaze,
                validity_pupil=validity_pupil,
                valid_val=valid_val,
            )
            feature_rows.append(
                _vectorize_window(
                    window=window,
                    channels=channels,
                    target_len=target_len,
                    length_mode=length_mode,
                )
            )

    matrix = np.vstack(feature_rows).astype(np.float32)
    return RepresentationDataset(
        name="raw",
        X=matrix,
        segment_ids=canonical["segment_id"].to_numpy(dtype=str),
        subjects=canonical["Subject"].to_numpy(dtype=str),
        source_labels=canonical["Label"].to_numpy(dtype=str),
    )


def _load_raw_representation_from_csv(
    canonical: pd.DataFrame,
    cfg: TrustMeExperimentConfig,
) -> RepresentationDataset:
    """Load raw representation from one CSV containing all Tobii rows."""

    if cfg.raw.csv_path is None:
        raise ValueError("raw.csv_path is required when raw.backend='csv'.")
    if not cfg.raw.csv_path.exists():
        raise FileNotFoundError(f"raw.csv_path does not exist: {cfg.raw.csv_path}")

    # In CSV backend we align windows by window_id only, so segment-level ambiguity
    # must not exist in canonical rows.
    if canonical["window_id"].duplicated().any():
        dup_ids = canonical.loc[canonical["window_id"].duplicated(), "window_id"].head(5).tolist()
        raise ValueError(
            "CSV raw backend requires unique canonical window_id values. "
            f"Found duplicates (sample): {dup_ids}"
        )

    channels = cfg.raw.channels
    target_len = _resolve_raw_target_length(cfg)
    validity_gaze = cfg.raw.validity_gaze_columns
    validity_pupil = cfg.raw.validity_pupil_columns
    valid_val = cfg.raw.validity_valid_value
    length_mode = cfg.raw.length_mode

    gaze_channels = [channel for channel in channels if "gaze" in channel.lower()]
    pupil_channels = [channel for channel in channels if "pupil" in channel.lower()]
    cols_to_load = list(dict.fromkeys(channels + validity_gaze + validity_pupil + ["window_id"]))

    frame = pd.read_csv(cfg.raw.csv_path, usecols=cols_to_load, low_memory=False)
    frame["window_id_norm"] = frame["window_id"].map(_normalize_window_id_value)
    grouped = {
        str(window_key): group.reset_index(drop=True)
        for window_key, group in frame.groupby("window_id_norm", sort=False)
    }

    ordered_window_ids = canonical["window_id"].astype(str).tolist()
    feature_rows: list[np.ndarray] = []
    for window_id in ordered_window_ids:
        if window_id not in grouped:
            raise ValueError(
                f"Missing window_id={window_id} in raw CSV backend: {cfg.raw.csv_path}"
            )
        window = _apply_validity_masks(
            grouped[window_id].copy(),
            gaze_channels=gaze_channels,
            pupil_channels=pupil_channels,
            validity_gaze=validity_gaze,
            validity_pupil=validity_pupil,
            valid_val=valid_val,
        )
        feature_rows.append(
            _vectorize_window(
                window=window,
                channels=channels,
                target_len=target_len,
                length_mode=length_mode,
            )
        )

    matrix = np.vstack(feature_rows).astype(np.float32)
    return RepresentationDataset(
        name="raw",
        X=matrix,
        segment_ids=canonical["segment_id"].to_numpy(dtype=str),
        subjects=canonical["Subject"].to_numpy(dtype=str),
        source_labels=canonical["Label"].to_numpy(dtype=str),
    )


def load_raw_representation(canonical: pd.DataFrame, cfg: TrustMeExperimentConfig) -> RepresentationDataset:
    """Load TrustMe raw windows and build fixed-size vectors."""

    backend = cfg.raw.backend.strip().lower()
    if backend == "parquet":
        return _load_raw_representation_from_parquet(canonical=canonical, cfg=cfg)
    if backend == "csv":
        return _load_raw_representation_from_csv(canonical=canonical, cfg=cfg)
    raise ValueError(f"Unsupported raw.backend={cfg.raw.backend!r}. Expected 'parquet' or 'csv'.")


def _load_raw_rows_representation_from_parquet(
    canonical: pd.DataFrame,
    cfg: TrustMeExperimentConfig,
) -> RepresentationDataset:
    """Load raw representation as variable-length row matrices per window."""

    channels = cfg.raw.channels
    validity_gaze = cfg.raw.validity_gaze_columns
    validity_pupil = cfg.raw.validity_pupil_columns
    valid_val = cfg.raw.validity_valid_value

    gaze_channels = [channel for channel in channels if "gaze" in channel.lower()]
    pupil_channels = [channel for channel in channels if "pupil" in channel.lower()]
    cols_to_load = list(dict.fromkeys(channels + validity_gaze + validity_pupil + ["window_id"]))

    window_rows: list[np.ndarray] = []
    canonical_by_file = canonical.groupby("file_path", sort=False)
    for file_path, file_rows in canonical_by_file:
        parquet_path = Path(file_path)
        ordered_window_ids = file_rows["window_id"].astype(str).tolist()
        needed_window_ids = set(ordered_window_ids)

        frame = pd.read_parquet(parquet_path, columns=cols_to_load)
        frame["window_id_norm"] = frame["window_id"].map(_normalize_window_id_value)
        frame = frame[frame["window_id_norm"].isin(needed_window_ids)].copy()
        grouped = {
            str(window_key): group.reset_index(drop=True)
            for window_key, group in frame.groupby("window_id_norm", sort=False)
        }

        for window_id in ordered_window_ids:
            if window_id not in grouped:
                raise ValueError(f"Missing window_id={window_id} in {parquet_path}")
            window = _apply_validity_masks(
                grouped[window_id].copy(),
                gaze_channels=gaze_channels,
                pupil_channels=pupil_channels,
                validity_gaze=validity_gaze,
                validity_pupil=validity_pupil,
                valid_val=valid_val,
            )
            window_rows.append(_matrixize_window_rows(window=window, channels=channels))

    return RepresentationDataset(
        name="raw_rows",
        X=window_rows,
        segment_ids=canonical["segment_id"].to_numpy(dtype=str),
        subjects=canonical["Subject"].to_numpy(dtype=str),
        source_labels=canonical["Label"].to_numpy(dtype=str),
    )


def _load_raw_rows_representation_from_csv(
    canonical: pd.DataFrame,
    cfg: TrustMeExperimentConfig,
) -> RepresentationDataset:
    """Load raw-row representation from one CSV containing all Tobii rows."""

    if cfg.raw.csv_path is None:
        raise ValueError("raw.csv_path is required when raw.backend='csv'.")
    if not cfg.raw.csv_path.exists():
        raise FileNotFoundError(f"raw.csv_path does not exist: {cfg.raw.csv_path}")
    if canonical["window_id"].duplicated().any():
        dup_ids = canonical.loc[canonical["window_id"].duplicated(), "window_id"].head(5).tolist()
        raise ValueError(
            "CSV raw backend requires unique canonical window_id values. "
            f"Found duplicates (sample): {dup_ids}"
        )

    channels = cfg.raw.channels
    validity_gaze = cfg.raw.validity_gaze_columns
    validity_pupil = cfg.raw.validity_pupil_columns
    valid_val = cfg.raw.validity_valid_value

    gaze_channels = [channel for channel in channels if "gaze" in channel.lower()]
    pupil_channels = [channel for channel in channels if "pupil" in channel.lower()]
    cols_to_load = list(dict.fromkeys(channels + validity_gaze + validity_pupil + ["window_id"]))

    frame = pd.read_csv(cfg.raw.csv_path, usecols=cols_to_load, low_memory=False)
    frame["window_id_norm"] = frame["window_id"].map(_normalize_window_id_value)
    grouped = {
        str(window_key): group.reset_index(drop=True)
        for window_key, group in frame.groupby("window_id_norm", sort=False)
    }

    ordered_window_ids = canonical["window_id"].astype(str).tolist()
    window_rows: list[np.ndarray] = []
    for window_id in ordered_window_ids:
        if window_id not in grouped:
            raise ValueError(
                f"Missing window_id={window_id} in raw CSV backend: {cfg.raw.csv_path}"
            )
        window = _apply_validity_masks(
            grouped[window_id].copy(),
            gaze_channels=gaze_channels,
            pupil_channels=pupil_channels,
            validity_gaze=validity_gaze,
            validity_pupil=validity_pupil,
            valid_val=valid_val,
        )
        window_rows.append(_matrixize_window_rows(window=window, channels=channels))

    return RepresentationDataset(
        name="raw_rows",
        X=window_rows,
        segment_ids=canonical["segment_id"].to_numpy(dtype=str),
        subjects=canonical["Subject"].to_numpy(dtype=str),
        source_labels=canonical["Label"].to_numpy(dtype=str),
    )


def load_raw_rows_representation(canonical: pd.DataFrame, cfg: TrustMeExperimentConfig) -> RepresentationDataset:
    """Load TrustMe raw windows as variable-length row matrices."""

    backend = cfg.raw.backend.strip().lower()
    if backend == "parquet":
        return _load_raw_rows_representation_from_parquet(canonical=canonical, cfg=cfg)
    if backend == "csv":
        return _load_raw_rows_representation_from_csv(canonical=canonical, cfg=cfg)
    raise ValueError(f"Unsupported raw.backend={cfg.raw.backend!r}. Expected 'parquet' or 'csv'.")


def load_embeddings_representation(
    canonical: pd.DataFrame,
    embedding_payload: dict[str, np.ndarray],
) -> RepresentationDataset:
    """Align embedding vectors from one payload to canonical segment order."""

    embeddings = embedding_payload["embeddings"].astype(np.float32)
    segment_ids = embedding_payload["segment_id"].astype(str)

    id_to_pos = {segment_id: i for i, segment_id in enumerate(segment_ids)}
    ordered_ids = canonical["segment_id"].to_numpy(dtype=str)
    order = [id_to_pos[segment_id] for segment_id in ordered_ids]
    matrix = embeddings[np.asarray(order, dtype=np.int64)]

    return RepresentationDataset(
        name="embeddings",
        X=matrix,
        segment_ids=ordered_ids,
        subjects=canonical["Subject"].to_numpy(dtype=str),
        source_labels=canonical["Label"].to_numpy(dtype=str),
    )


def _infer_feature_segment_ids(features: pd.DataFrame) -> np.ndarray:
    """Infer TrustMe feature segment IDs from known expected columns only."""

    if "segment_id" in features.columns:
        return features["segment_id"].astype(str).to_numpy()

    required = {"Subject", "Filename", "window_id"}
    if not required.issubset(set(features.columns)):
        available = sorted(features.columns.tolist())
        raise ValueError(
            "Could not infer feature segment IDs. Expected either 'segment_id' or "
            "'Subject'+'Filename'+'window_id'. "
            f"Available columns: {available}"
        )

    return np.asarray(
        [
            f"{subject}|{_coerce_processed_filename(filename)}|{_normalize_window_id_value(window_id)}"
            for subject, filename, window_id in zip(
                features["Subject"].astype(str),
                features["Filename"].astype(str),
                features["window_id"].tolist(),
            )
        ],
        dtype=str,
    )


def load_features_representation(canonical: pd.DataFrame, cfg: TrustMeExperimentConfig) -> RepresentationDataset:
    """Load handcrafted features and align rows to canonical segment IDs."""

    if cfg.paths.features_csv is None:
        raise ValueError("paths.features_csv is null; cannot load features representation.")

    features = pd.read_csv(cfg.paths.features_csv).copy()
    features["segment_id"] = _infer_feature_segment_ids(features)
    if features["segment_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate segment IDs.")

    canonical_segment_ids = canonical["segment_id"].to_numpy(dtype=str)

    features_by_id = features.set_index("segment_id", drop=False)
    ordered_ids = canonical_segment_ids
    missing_ids = sorted(set(ordered_ids.tolist()) - set(features_by_id.index.tolist()))
    if missing_ids:
        preview = missing_ids[:5]
        raise ValueError(
            "Feature table is missing canonical segment IDs. "
            f"missing_count={len(missing_ids)}, sample={preview}"
        )

    ordered = features_by_id.loc[ordered_ids].copy()
    return RepresentationDataset(
        name="features",
        X=ordered,
        segment_ids=ordered_ids,
        subjects=canonical["Subject"].to_numpy(dtype=str),
        source_labels=canonical["Label"].to_numpy(dtype=str),
    )
