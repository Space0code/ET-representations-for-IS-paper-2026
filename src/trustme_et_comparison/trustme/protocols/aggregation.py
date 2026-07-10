"""Aggregation utilities for Zoja-style protocol outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_COLS = ["target", "model", "representation", "protocol"]
SUBJECT_GROUP_COLS = ["target", "model", "representation", "protocol", "subject"]
METRIC_COLS = [
    "baseline_accuracy",
    "baseline_balanced_accuracy",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "recall",
    "auc",
    "gain",
    "balanced_gain",
]


def build_subject_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics into per-subject protocol metrics."""

    if fold_df.empty:
        return pd.DataFrame(columns=[*SUBJECT_GROUP_COLS, *METRIC_COLS, "n_folds", "n_valid_folds", "skip_ratio"])

    work = fold_df.copy()
    if "model" not in work.columns:
        work["model"] = "unknown"
    else:
        work["model"] = work["model"].fillna("unknown")
    work["is_valid"] = work["status"].isin(["ok", "ok_with_nan"])

    agg_map: dict[str, tuple[str, str]] = {metric: (metric, "mean") for metric in METRIC_COLS}
    agg_map["n_folds"] = ("fold_id", "count")
    agg_map["n_valid_folds"] = ("is_valid", "sum")

    subject_df = (
        work.groupby(SUBJECT_GROUP_COLS, as_index=False)
        .agg(**agg_map)
        .sort_values(SUBJECT_GROUP_COLS)
        .reset_index(drop=True)
    )
    subject_df["skip_ratio"] = 1.0 - (subject_df["n_valid_folds"] / subject_df["n_folds"]).replace(0, np.nan)
    return subject_df


def build_protocol_summary(fold_df: pd.DataFrame, subject_df: pd.DataFrame) -> pd.DataFrame:
    """Build protocol summary with subject-macro and fold-weighted metrics."""

    if fold_df.empty:
        return pd.DataFrame()

    fold_work = fold_df.copy()
    if "model" not in fold_work.columns:
        fold_work["model"] = "unknown"
    else:
        fold_work["model"] = fold_work["model"].fillna("unknown")
    fold_work["is_valid"] = fold_work["status"].isin(["ok", "ok_with_nan"])

    fold_agg: dict[str, tuple[str, str]] = {
        f"fold_weighted_{metric}": (metric, "mean")
        for metric in METRIC_COLS
    }
    fold_agg["n_folds"] = ("fold_id", "count")
    fold_agg["n_valid_folds"] = ("is_valid", "sum")
    fold_summary = (
        fold_work.groupby(GROUP_COLS, as_index=False)
        .agg(**fold_agg)
        .sort_values(GROUP_COLS)
        .reset_index(drop=True)
    )
    fold_summary["skip_ratio"] = 1.0 - (fold_summary["n_valid_folds"] / fold_summary["n_folds"]).replace(0, np.nan)

    subj_work = subject_df.copy()
    if "model" not in subj_work.columns:
        subj_work["model"] = "unknown"
    else:
        subj_work["model"] = subj_work["model"].fillna("unknown")

    subj_agg: dict[str, tuple[str, str]] = {
        f"subject_macro_{metric}": (metric, "mean")
        for metric in METRIC_COLS
    }
    subj_summary = (
        subj_work.groupby(GROUP_COLS, as_index=False)
        .agg(**subj_agg)
        .sort_values(GROUP_COLS)
        .reset_index(drop=True)
    )

    merged = fold_summary.merge(subj_summary, on=GROUP_COLS, how="left")
    return merged


def build_method_comparison_tables(subject_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build protocol comparison pivots for gain and balanced gain."""

    if subject_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = subject_df.copy()
    if "model" not in work.columns:
        work["model"] = "unknown"
    else:
        work["model"] = work["model"].fillna("unknown")

    gain_tbl = (
        work.pivot_table(
            index=["target", "model", "representation", "subject"],
            columns="protocol",
            values="gain",
            aggfunc="mean",
        )
        .reset_index()
    )
    gain_tbl.columns.name = None

    balanced_tbl = (
        work.pivot_table(
            index=["target", "model", "representation", "subject"],
            columns="protocol",
            values="balanced_gain",
            aggfunc="mean",
        )
        .reset_index()
    )
    balanced_tbl.columns.name = None

    return gain_tbl, balanced_tbl


def build_representation_protocol_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rank target/representation/protocol rows by subject-macro balanced accuracy and gain."""

    if summary_df.empty:
        return pd.DataFrame()

    rank_df = summary_df.copy()
    if "model" not in rank_df.columns:
        rank_df["model"] = "unknown"
    else:
        rank_df["model"] = rank_df["model"].fillna("unknown")
    rank_df = rank_df.sort_values(
        [
            "target",
            "model",
            "subject_macro_balanced_accuracy",
            "subject_macro_gain",
            "subject_macro_macro_f1",
            "subject_macro_accuracy",
        ],
        ascending=[True, True, False, False, False, False],
    ).reset_index(drop=True)

    rank_values = rank_df.groupby(["target", "model"])["subject_macro_balanced_accuracy"].rank(
        method="first",
        ascending=False,
    )
    rank_df["rank"] = rank_values.where(
        rank_df["subject_macro_balanced_accuracy"].notna(),
        np.nan,
    ).astype("Int64")

    return rank_df[
        [
            "target",
            "model",
            "rank",
            "representation",
            "protocol",
            "subject_macro_balanced_accuracy",
            "subject_macro_gain",
            "subject_macro_macro_f1",
            "subject_macro_accuracy",
            "n_subjects",
            "n_folds",
            "n_valid_folds",
            "skip_ratio",
        ]
    ] if "n_subjects" in rank_df.columns else rank_df


def add_subject_counts(summary_df: pd.DataFrame, subject_df: pd.DataFrame) -> pd.DataFrame:
    """Attach number of contributing subjects per summary group."""

    if summary_df.empty:
        return summary_df

    work = subject_df.copy()
    if "model" not in work.columns:
        work["model"] = "unknown"
    else:
        work["model"] = work["model"].fillna("unknown")

    counts = (
        work.groupby(GROUP_COLS, as_index=False)["subject"]
        .nunique()
        .rename(columns={"subject": "n_subjects"})
    )
    return summary_df.merge(counts, on=GROUP_COLS, how="left")
