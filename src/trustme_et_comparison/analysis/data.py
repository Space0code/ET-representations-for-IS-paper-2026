"""Input loading and dimensionality-reduction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import RepresentationConfig, TableConfig


def read_table(config: TableConfig, columns: list[str] | None = None) -> pd.DataFrame:
    """Read CSV or Parquet, optionally selecting columns."""

    suffix = config.path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(config.path, columns=columns)
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(config.path, usecols=columns, sep=separator, low_memory=False)
    raise ValueError(f"Unsupported table format: {config.path}")


def sample_rows(frame: pd.DataFrame, maximum: int, random_state: int) -> pd.DataFrame:
    """Return a deterministic row sample without replacement."""

    if len(frame) <= maximum:
        return frame.copy()
    return frame.sample(n=maximum, random_state=random_state).sort_index()


def read_sampled_table(
    config: TableConfig,
    columns: list[str],
    maximum: int,
    random_state: int,
) -> pd.DataFrame:
    """Read a bounded deterministic sample, streaming CSV inputs in chunks."""

    if config.path.suffix.lower() not in {".csv", ".tsv"}:
        return sample_rows(read_table(config, columns), maximum, random_state)

    separator = "\t" if config.path.suffix.lower() == ".tsv" else ","
    rng = np.random.default_rng(random_state)
    retained: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        config.path,
        usecols=columns,
        sep=separator,
        low_memory=False,
        chunksize=100_000,
    ):
        chunk = chunk.copy()
        chunk["_sample_priority"] = rng.random(len(chunk))
        retained.append(chunk)
        combined = pd.concat(retained, ignore_index=True)
        retained = [combined.nsmallest(maximum, "_sample_priority")]
    if not retained:
        return pd.DataFrame(columns=columns)
    return retained[0].drop(columns="_sample_priority").reset_index(drop=True)


def load_representation(
    config: RepresentationConfig,
    allowed_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a window ID vector and numeric feature matrix from NPZ/CSV/Parquet."""

    if config.kind == "raw_samples":
        selected_columns = [config.id_column, *config.channels]
        if config.path.suffix.lower() in {".csv", ".tsv"} and allowed_ids is not None:
            separator = "\t" if config.path.suffix.lower() == ".tsv" else ","
            chunks = pd.read_csv(
                config.path,
                usecols=selected_columns,
                sep=separator,
                low_memory=False,
                chunksize=100_000,
            )
            kept = [
                chunk[chunk[config.id_column].astype(str).isin(allowed_ids)]
                for chunk in chunks
            ]
            frame = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=selected_columns)
        else:
            frame = read_table(TableConfig(config.path, config.id_column), selected_columns)
            if allowed_ids is not None:
                frame = frame[frame[config.id_column].astype(str).isin(allowed_ids)]
        required = {config.id_column, *config.channels}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Raw representation is missing columns: {missing}")
        ids_list: list[str] = []
        sequences: list[np.ndarray] = []
        target_index = np.linspace(0.0, 1.0, config.sequence_length)
        for window_id, window in frame.groupby(config.id_column, sort=False):
            values = window[list(config.channels)].apply(pd.to_numeric, errors="coerce")
            values = values.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)
            source_index = np.linspace(0.0, 1.0, len(values))
            channels = [np.interp(target_index, source_index, values[:, idx]) for idx in range(values.shape[1])]
            ids_list.append(str(window_id))
            sequences.append(np.stack(channels, axis=1).reshape(-1))
        ids = np.asarray(ids_list, dtype=str)
        matrix = np.asarray(sequences, dtype=np.float32)
    elif config.path.suffix.lower() == ".npz":
        with np.load(config.path, allow_pickle=False) as payload:
            if config.array_key not in payload or config.id_key not in payload:
                raise ValueError(
                    f"{config.path} must contain {config.array_key!r} and {config.id_key!r}."
                )
            matrix = np.asarray(payload[config.array_key])
            ids = np.asarray(payload[config.id_key]).astype(str)
    else:
        frame = read_table(TableConfig(config.path, config.id_column))
        if config.id_column not in frame.columns:
            raise ValueError(f"Missing ID column {config.id_column!r} in {config.path}")
        if config.feature_prefixes:
            feature_columns = [
                column
                for column in frame.columns
                if any(str(column).startswith(prefix) for prefix in config.feature_prefixes)
            ]
        else:
            feature_columns = [
                column
                for column in frame.select_dtypes(include=[np.number]).columns
                if column != config.id_column
            ]
        if not feature_columns:
            raise ValueError(f"No numeric feature columns selected in {config.path}")
        ids = frame[config.id_column].astype(str).to_numpy()
        matrix = frame[feature_columns].to_numpy(dtype=np.float32)

    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.ndim != 2 or len(ids) != matrix.shape[0]:
        raise ValueError(f"Invalid representation shape for {config.name}: {matrix.shape}")
    return ids, matrix.astype(np.float32, copy=False)


def project_matrix(matrix: np.ndarray, method: str, random_state: int) -> np.ndarray:
    """Impute, standardize, and project a representation to two dimensions."""

    clean = SimpleImputer(strategy="median").fit_transform(matrix)
    clean = StandardScaler().fit_transform(clean)
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(clean)
    if method == "tsne":
        if len(clean) < 3:
            raise ValueError("t-SNE requires at least three samples.")
        perplexity = min(30.0, max(2.0, (len(clean) - 1) / 3.0))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(clean)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("UMAP requested; install the analysis extra: pip install -e '.[analysis]'") from exc
        return umap.UMAP(n_components=2, random_state=random_state).fit_transform(clean)
    raise ValueError(f"Unknown projection method: {method}")
