"""Evaluation metrics and artifact writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import ensure_dir


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
) -> dict[str, float | str]:
    """Compute fold metrics with robust AUC handling.

    AUC note on multi_class strategy
    ---------------------------------
    * Binary: standard ``roc_auc_score`` on the positive-class probability.
    * Multi-class: **one-vs-one (OVO)**, macro-averaged.  OVO is symmetric and
      less sensitive to class imbalance than one-vs-rest (OVR), making it the
      preferred strategy for balanced comparisons across class pairs.
    """
    metrics: dict[str, float | str] = {}
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    auc_note = "ok"
    auc_value = np.nan

    unique_true = np.unique(y_true)
    if unique_true.shape[0] < 2:
        auc_note = "only_one_class_in_test"
    elif num_classes == 2:
        if y_proba.shape[1] != 2:
            auc_note = "binary_proba_dimension_mismatch"
        else:
            auc_value = float(roc_auc_score(y_true, y_proba[:, 1]))
    else:
        if unique_true.shape[0] != num_classes:
            auc_note = "not_all_classes_present_in_test"
        elif y_proba.shape[1] != num_classes:
            auc_note = "proba_dimension_mismatch"
        else:
            auc_value = float(
                roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
            )

    metrics["auc"] = float(auc_value) if not np.isnan(auc_value) else float("nan")
    metrics["auc_note"] = auc_note

    return metrics


def save_fold_arrays(
    fold_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    segment_ids: np.ndarray,
    subjects: np.ndarray,
) -> dict[str, Path]:
    """Persist fold-level prediction arrays."""
    ensure_dir(fold_dir)

    outputs = {
        "y_true": fold_dir / "y_true.npy",
        "y_pred": fold_dir / "y_pred.npy",
        "y_proba": fold_dir / "y_proba.npy",
        "segment_id": fold_dir / "segment_id.npy",
        "subject": fold_dir / "subject.npy",
    }

    np.save(outputs["y_true"], y_true)
    np.save(outputs["y_pred"], y_pred)
    np.save(outputs["y_proba"], y_proba)
    np.save(outputs["segment_id"], segment_ids.astype(str))
    np.save(outputs["subject"], subjects.astype(str))

    return outputs


def save_aggregate_arrays(
    aggregate_dir: Path,
    all_y_true: list[np.ndarray],
    all_y_pred: list[np.ndarray],
    all_y_proba: list[np.ndarray],
    all_segment_ids: list[np.ndarray],
    all_subjects: list[np.ndarray],
) -> dict[str, Path]:
    """Persist concatenated arrays across folds."""
    ensure_dir(aggregate_dir)

    y_true = np.concatenate(all_y_true) if all_y_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_y_pred) if all_y_pred else np.array([], dtype=np.int64)
    y_proba = np.vstack(all_y_proba) if all_y_proba else np.empty((0, 0), dtype=np.float32)
    segment_ids = np.concatenate(all_segment_ids) if all_segment_ids else np.array([], dtype=str)
    subjects = np.concatenate(all_subjects) if all_subjects else np.array([], dtype=str)

    outputs = {
        "y_true": aggregate_dir / "y_true.npy",
        "y_pred": aggregate_dir / "y_pred.npy",
        "y_proba": aggregate_dir / "y_proba.npy",
        "segment_id": aggregate_dir / "segment_id.npy",
        "subject": aggregate_dir / "subject.npy",
    }

    np.save(outputs["y_true"], y_true)
    np.save(outputs["y_pred"], y_pred)
    np.save(outputs["y_proba"], y_proba)
    np.save(outputs["segment_id"], segment_ids.astype(str))
    np.save(outputs["subject"], subjects.astype(str))

    return outputs


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics to mean/std summary table."""
    summary = (
        fold_metrics.groupby(
            ["labelling", "embedding_variant", "embedding_npz", "segment_set_id", "representation", "model"],
            as_index=False,
        )
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
        )
        .sort_values(["labelling", "embedding_variant", "representation", "model"])
        .reset_index(drop=True)
    )
    return summary


def build_representation_ranking(summary_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create representation ranking per labelling scheme and embedding variant by accuracy."""
    rows: list[dict[str, object]] = []

    for (scheme, variant), frame in summary_metrics.groupby(["labelling", "embedding_variant"]):
        best_by_rep = frame.loc[frame.groupby("representation")["accuracy_mean"].idxmax()].copy()
        mean_by_rep = (
            frame.groupby("representation", as_index=False)["accuracy_mean"].mean().rename(
                columns={"accuracy_mean": "mean_model_accuracy"}
            )
        )

        merged = best_by_rep[["representation", "model", "accuracy_mean"]].merge(
            mean_by_rep,
            on="representation",
            how="left",
        )
        merged = merged.rename(
            columns={
                "model": "best_model",
                "accuracy_mean": "best_model_accuracy",
            }
        )
        merged["labelling"] = scheme
        merged["embedding_variant"] = variant
        merged = merged.sort_values(
            ["best_model_accuracy", "mean_model_accuracy"],
            ascending=[False, False],
        ).reset_index(drop=True)
        merged["rank"] = np.arange(1, len(merged) + 1)

        rows.extend(merged.to_dict(orient="records"))

    return pd.DataFrame(rows)[
        [
            "labelling",
            "embedding_variant",
            "rank",
            "representation",
            "best_model",
            "best_model_accuracy",
            "mean_model_accuracy",
        ]
    ]
