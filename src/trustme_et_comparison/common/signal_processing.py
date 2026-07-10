"""Shared signal cleaning and resampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_channel(values: np.ndarray) -> np.ndarray:
    """Impute NaNs in one channel and return a float32 vector."""

    series = pd.Series(values.astype(np.float32), copy=True)
    if series.isna().all():
        return np.asarray([], dtype=np.float32)

    series = series.interpolate(method="linear", limit_direction="both")
    series = series.ffill().bfill().fillna(0.0)
    return series.to_numpy(dtype=np.float32)


def clean_and_resize_channel(
    values: np.ndarray,
    target_length: int,
    length_mode: str,
) -> np.ndarray:
    """Impute NaNs and convert one channel to a fixed length.

    Args:
        values: One-dimensional channel values.
        target_length: Output number of samples.
        length_mode: ``"resample"`` or ``"truncate_pad"``.

    Returns:
        Resampled vector of shape ``(target_length,)`` and dtype ``float32``.
    """

    if target_length <= 0:
        raise ValueError("target_length must be positive.")

    cleaned = _clean_channel(values)
    if cleaned.size == 0:
        return np.zeros(target_length, dtype=np.float32)

    mode = length_mode.strip().lower()
    if mode == "truncate_pad":
        if cleaned.shape[0] >= target_length:
            return cleaned[:target_length].astype(np.float32)
        if cleaned.shape[0] == 1:
            return np.full(target_length, cleaned[0], dtype=np.float32)
        pad_len = target_length - cleaned.shape[0]
        return np.concatenate(
            [cleaned, np.full(pad_len, cleaned[-1], dtype=np.float32)],
            axis=0,
        ).astype(np.float32)

    if mode != "resample":
        raise ValueError(
            f"Unsupported length_mode={length_mode!r}. Expected 'resample' or 'truncate_pad'."
        )

    if cleaned.shape[0] == target_length:
        return cleaned.astype(np.float32)
    if cleaned.shape[0] == 1:
        return np.full(target_length, cleaned[0], dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=cleaned.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(dst, src, cleaned).astype(np.float32)


def clean_and_resample_channel(values: np.ndarray, target_length: int) -> np.ndarray:
    """Backward-compatible wrapper for linear resampling mode."""

    return clean_and_resize_channel(
        values=values,
        target_length=target_length,
        length_mode="resample",
    )
