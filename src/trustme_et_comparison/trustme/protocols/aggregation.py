"""Aggregation utilities for Zoja-style protocol outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_COLS = ["target", "model", "representation", "protocol"]
SUBJECT_GROUP_COLS = ["target", "model", "representation", "protocol", "subject"]
METRIC_COLS = [
    "baseline_accuracy",
    "baseline_balanced_accuracy",
    "baseline_macro_f1",
    "baseline_auc",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "recall",
    "auc",
    "gain",
    "balanced_gain",
]

PAPER_REPRESENTATION_NAMES = {
    "raw": "raw",
    "features": "handcrafted features",
    "embeddings_gazemae": "GazeMAE embeddings",
    "embeddings_moment": "MOMENT embeddings",
}

PAPER_REPRESENTATION_ORDER = {
    "majority baseline": 0,
    "raw": 1,
    "handcrafted features": 2,
    "GazeMAE embeddings": 3,
    "MOMENT embeddings": 4,
}


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


def _metric_summary(frame: pd.DataFrame, source: str, output: str) -> dict[str, float | int]:
    """Return mean, sample SD, and available-fold count for one metric."""

    values = pd.to_numeric(frame[source], errors="coerce")
    return {
        f"{output}_mean": float(values.mean()),
        f"{output}_std": float(values.std(ddof=1)),
        f"{output}_valid_folds": int(values.notna().sum()),
    }


def build_paper_main_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Build the concise Table 1 CSV from normalized LOSO fold metrics."""

    work = fold_df[fold_df["protocol"] == "loso_normalized"].copy()
    if work.empty:
        return pd.DataFrame()

    baseline_source = work.sort_values(["model", "representation"]).drop_duplicates("fold_id")
    rows: list[dict[str, object]] = []
    baseline_row: dict[str, object] = {
        "target": str(baseline_source["target"].iloc[0]),
        "representation": "majority baseline",
        "model": "most_frequent",
        "n_subjects": int(baseline_source["subject"].nunique()),
        **_metric_summary(baseline_source, "baseline_accuracy", "accuracy"),
        "accuracy_delta_mean": 0.0,
        "accuracy_delta_std": 0.0,
        "accuracy_delta_valid_folds": int(baseline_source["baseline_accuracy"].notna().sum()),
        **_metric_summary(baseline_source, "baseline_balanced_accuracy", "balanced_accuracy"),
        "balanced_accuracy_delta_mean": 0.0,
        "balanced_accuracy_delta_std": 0.0,
        "balanced_accuracy_delta_valid_folds": int(
            baseline_source["baseline_balanced_accuracy"].notna().sum()
        ),
        **_metric_summary(baseline_source, "baseline_macro_f1", "macro_f1"),
        **_metric_summary(baseline_source, "baseline_auc", "roc_auc"),
    }
    rows.append(baseline_row)

    for (representation, model), frame in work.groupby(["representation", "model"], sort=False):
        row: dict[str, object] = {
            "target": str(frame["target"].iloc[0]),
            "representation": PAPER_REPRESENTATION_NAMES.get(str(representation), str(representation)),
            "model": str(model),
            "n_subjects": int(frame["subject"].nunique()),
            **_metric_summary(frame, "accuracy", "accuracy"),
            **_metric_summary(frame, "gain", "accuracy_delta"),
            **_metric_summary(frame, "balanced_accuracy", "balanced_accuracy"),
            **_metric_summary(frame, "balanced_gain", "balanced_accuracy_delta"),
            **_metric_summary(frame, "macro_f1", "macro_f1"),
            **_metric_summary(frame, "auc", "roc_auc"),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    result["_representation_order"] = result["representation"].map(PAPER_REPRESENTATION_ORDER)
    result["_model_order"] = result["model"].map(
        {"most_frequent": 0, "random_forest": 1, "mlp": 2}
    )
    result.sort_values(["_representation_order", "_model_order"], inplace=True)
    return result.drop(columns=["_representation_order", "_model_order"]).reset_index(drop=True)


def build_paper_persistence_results(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Build the separate subject-macro persistence-check CSV."""

    work = subject_df[subject_df["protocol"] == "persistence_baseline"].copy()
    if work.empty:
        return pd.DataFrame()
    row: dict[str, object] = {
        "target": str(work["target"].iloc[0]),
        "protocol": "one_step_persistence",
        "n_subjects": int(work["subject"].nunique()),
        "n_temporal_folds": int(work["n_folds"].sum()),
    }
    for source, output in [
        ("accuracy", "accuracy"),
        ("gain", "accuracy_delta"),
        ("balanced_accuracy", "balanced_accuracy"),
        ("balanced_gain", "balanced_accuracy_delta"),
        ("macro_f1", "macro_f1"),
    ]:
        summary = _metric_summary(work, source, output)
        row.update(
            {
                key.replace("_valid_folds", "_valid_subjects"): value
                for key, value in summary.items()
            }
        )
    return pd.DataFrame([row])


def build_cohort_results(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return per-subject and overall aligned-window counts for reporting."""

    per_subject = (
        metadata.groupby("Subject", sort=False)["segment_id"]
        .nunique()
        .rename("n_windows")
        .reset_index()
        .rename(columns={"Subject": "subject"})
    )
    total = pd.DataFrame(
        [{"subject": "ALL", "n_windows": int(metadata["segment_id"].nunique())}]
    )
    return pd.concat([total, per_subject], ignore_index=True)
