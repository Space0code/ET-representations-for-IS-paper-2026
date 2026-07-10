"""Common evaluation helpers for Zoja-style protocol runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldEvalResult:
    """Fold-level evaluation outputs with explicit validity state."""

    status: str
    skip_reason: str
    metrics: dict[str, Any]


def _nan_metrics() -> dict[str, float]:
    """Return metric dictionary with NaN values."""

    return {
        "baseline_accuracy": np.nan,
        "baseline_balanced_accuracy": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "auc": np.nan,
        "gain": np.nan,
        "balanced_gain": np.nan,
    }


def evaluate_binary_fold(
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    one_class_policy: str,
) -> FoldEvalResult:
    """Evaluate one binary fold with consistent one-class policy handling."""

    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_train.size == 0 or y_test.size == 0:
        return FoldEvalResult(
            status="skipped",
            skip_reason="empty_partition",
            metrics=_nan_metrics(),
        )

    majority_class = int(np.bincount(y_train).argmax())
    y_pred_base = np.full(shape=y_test.shape[0], fill_value=majority_class, dtype=np.int64)

    baseline_acc = float(accuracy_score(y_test, y_pred_base))
    acc = float(accuracy_score(y_test, y_pred))

    unique_test = np.unique(y_test)
    one_class_test = unique_test.shape[0] < 2

    if one_class_test and one_class_policy == "skip":
        return FoldEvalResult(
            status="skipped",
            skip_reason="one_class_test",
            metrics={
                **_nan_metrics(),
                "baseline_accuracy": baseline_acc,
                "accuracy": acc,
                "gain": acc - baseline_acc,
            },
        )

    if one_class_test:
        baseline_bacc = np.nan
        bacc = np.nan
        macro_f1 = np.nan
        precision = np.nan
        recall = np.nan
        auc = np.nan
    else:
        baseline_bacc = float(balanced_accuracy_score(y_test, y_pred_base))
        bacc = float(balanced_accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="binary", zero_division=0))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        auc = np.nan
        if y_proba is not None:
            try:
                auc = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                auc = np.nan

    gain = acc - baseline_acc
    balanced_gain = np.nan if np.isnan(baseline_bacc) or np.isnan(bacc) else bacc - baseline_bacc

    status = "ok"
    skip_reason = ""
    if one_class_test:
        status = "ok_with_nan"
        skip_reason = "one_class_test"

    return FoldEvalResult(
        status=status,
        skip_reason=skip_reason,
        metrics={
            "baseline_accuracy": baseline_acc,
            "baseline_balanced_accuracy": baseline_bacc,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "gain": gain,
            "balanced_gain": balanced_gain,
        },
    )


def can_train_classifier(
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    min_train_samples: int,
    min_test_samples: int,
    one_class_policy: str,
    require_two_class_train: bool = False,
) -> tuple[bool, str]:
    """Validate whether fold should run supervised training."""

    if y_train.shape[0] < min_train_samples:
        return False, "train_too_small"
    if y_test.shape[0] < min_test_samples:
        return False, "test_too_small"
    if require_two_class_train and np.unique(y_train).shape[0] < 2:
        return False, "one_class_train"
    _ = one_class_policy
    return True, ""


def zscore_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    train_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Zoja-v2 LOSO normalization: train-fit median impute + standard scaling."""

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    _ = train_groups

    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    valid_cols = np.any(~np.isnan(X_train), axis=0)
    X_train = X_train[:, valid_cols]
    X_test = X_test[:, valid_cols]

    if X_train.shape[1] == 0:
        return (
            np.zeros((X_train.shape[0], 0), dtype=np.float32),
            np.zeros((X_test.shape[0], 0), dtype=np.float32),
        )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_imp)
    X_test_norm = scaler.transform(X_test_imp)

    return np.nan_to_num(X_train_norm), np.nan_to_num(X_test_norm)
