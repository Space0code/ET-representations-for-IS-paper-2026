"""Target construction and leakage-safe label transforms for protocol runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import ZojaLabelRuleConfig, ZojaTargetConfig


def build_target_values(metadata: pd.DataFrame, target: ZojaTargetConfig) -> np.ndarray:
    """Build numeric target values from metadata according to target recipe."""

    if target.mode == "single_column":
        assert target.column is not None
        values = pd.to_numeric(metadata[target.column], errors="coerce").to_numpy(dtype=np.float32)
    elif target.mode == "mean_columns":
        assert target.columns is not None
        cols = [pd.to_numeric(metadata[col], errors="coerce") for col in target.columns]
        stack = np.vstack([col.to_numpy(dtype=np.float32) for col in cols])
        values = np.nanmean(stack, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported target mode: {target.mode}")

    if np.isnan(values).any():
        raise ValueError(
            f"Target {target.name!r} contains NaN values after numeric conversion."
        )

    return values


def centered_binary_train_test(
    *,
    values: np.ndarray,
    subjects: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_rule: ZojaLabelRuleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build train/test binary labels according to configured label rule.

    Returns:
      y_train, y_test, transformed_train_values, transformed_test_values
    """

    values = np.asarray(values, dtype=np.float32)
    subjects = np.asarray(subjects, dtype=str)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    train_values = values[train_idx]
    test_values = values[test_idx]

    if label_rule.mode == "absolute_gt_threshold":
        threshold = float(label_rule.threshold)
        y_train = (train_values > threshold).astype(np.int64)
        y_test = (test_values > threshold).astype(np.int64)
        return y_train, y_test, train_values, test_values

    if label_rule.mode != "centered_subject_train_gt0":
        raise ValueError(f"Unsupported label rule mode: {label_rule.mode}")

    train_subjects = subjects[train_idx]
    test_subjects = subjects[test_idx]
    global_mean = float(np.mean(train_values))

    subj_means: dict[str, float] = {}
    for subj in np.unique(train_subjects):
        subj_mask = train_subjects == subj
        subj_means[subj] = float(np.mean(train_values[subj_mask]))

    centered_train = np.zeros_like(train_values, dtype=np.float32)
    for i, subj in enumerate(train_subjects):
        centered_train[i] = train_values[i] - subj_means[subj]

    centered_test = np.zeros_like(test_values, dtype=np.float32)
    for i, subj in enumerate(test_subjects):
        center = subj_means.get(subj, global_mean)
        centered_test[i] = test_values[i] - center

    y_train = (centered_train > 0.0).astype(np.int64)
    y_test = (centered_test > 0.0).astype(np.int64)

    return y_train, y_test, centered_train, centered_test


def centered_binary_full_for_diagnostics(
    *,
    values: np.ndarray,
    subjects: np.ndarray,
    label_rule: ZojaLabelRuleConfig,
) -> np.ndarray:
    """Build full-series binary labels for descriptive diagnostics only."""

    values = np.asarray(values, dtype=np.float32)
    if label_rule.mode == "absolute_gt_threshold":
        return (values > float(label_rule.threshold)).astype(np.int64)
    if label_rule.mode != "centered_subject_train_gt0":
        raise ValueError(f"Unsupported label rule mode: {label_rule.mode}")

    subjects = np.asarray(subjects, dtype=str)
    centered = np.zeros_like(values, dtype=np.float32)
    for subj in np.unique(subjects):
        idx = np.where(subjects == subj)[0]
        centered[idx] = values[idx] - float(np.mean(values[idx]))

    return (centered > 0.0).astype(np.int64)
