"""Plotting helpers for Zoja-style protocol outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ...common.utils import ensure_dir


def _sanitize(name: str) -> str:
    """Return filesystem-safe token for file names."""

    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def _plot_gain_barplots(
    *,
    output_dir: Path,
    subject_df: pd.DataFrame,
    metric_col: str,
) -> list[Path]:
    """Plot protocol gain bars per target with representation hues."""

    saved: list[Path] = []
    if subject_df.empty:
        return saved

    work = subject_df.copy()
    if "model" not in work.columns:
        work["model"] = "unknown"
    else:
        work["model"] = work["model"].fillna("unknown")

    for (target, model), frame in work.groupby(["target", "model"]):
        agg = (
            frame.groupby(["protocol", "representation"], as_index=False)[metric_col]
            .mean()
            .sort_values(["protocol", "representation"])
        )
        if agg.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(
            data=agg,
            x="protocol",
            y=metric_col,
            hue="representation",
            ax=ax,
            errorbar=None,
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_ylim(min(-0.2, float(agg[metric_col].min()) - 0.05), max(1.0, float(agg[metric_col].max()) + 0.05))
        ax.set_title(f"{target} / {model}: protocol comparison ({metric_col})")
        ax.set_xlabel("Protocol")
        ax.set_ylabel(metric_col)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Representation", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

        out_path = output_dir / f"barplot_{_sanitize(target)}_{_sanitize(model)}_{metric_col}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def _plot_protocol_heatmaps(
    *,
    output_dir: Path,
    summary_df: pd.DataFrame,
) -> list[Path]:
    """Plot protocol-vs-representation heatmaps per target and metric."""

    saved: list[Path] = []
    if summary_df.empty:
        return saved

    work = summary_df.copy()
    if "model" not in work.columns:
        work["model"] = "unknown"
    else:
        work["model"] = work["model"].fillna("unknown")

    metric_cols = [
        "subject_macro_accuracy",
        "subject_macro_balanced_accuracy",
        "subject_macro_macro_f1",
        "subject_macro_auc",
        "subject_macro_gain",
        "subject_macro_balanced_gain",
    ]

    for (target, model), frame in work.groupby(["target", "model"]):
        for metric in metric_cols:
            if metric not in frame.columns:
                continue
            pivot = frame.pivot_table(
                index="protocol",
                columns="representation",
                values=metric,
                aggfunc="mean",
            )
            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, max(4, 0.7 * len(pivot.index))))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="Blues",
                vmin=0.0 if "gain" not in metric else None,
                vmax=1.0 if "gain" not in metric else None,
                ax=ax,
            )
            ax.set_title(f"{target} / {model}: {metric} (protocol x representation)")
            ax.set_xlabel("Representation")
            ax.set_ylabel("Protocol")
            fig.tight_layout()

            out_path = output_dir / f"heatmap_{_sanitize(target)}_{_sanitize(model)}_{metric}.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)

    return saved


def _plot_confusions(
    *,
    output_dir: Path,
    predictions: list[dict[str, Any]],
    confusion_top_k: int,
) -> list[Path]:
    """Plot normalized confusion matrices with fixed color range [0, 1]."""

    saved: list[Path] = []
    if not predictions:
        return saved

    pred_df = pd.DataFrame(
        [
            {
                "target": row["target"],
                "model": row.get("model", "unknown"),
                "representation": row["representation"],
                "protocol": row["protocol"],
                "status": row["status"],
            }
            for row in predictions
        ]
    )

    # Rank combinations by availability of valid folds.
    rank = (
        pred_df[pred_df["status"].isin(["ok", "ok_with_nan"])]
        .groupby(["target", "model", "protocol", "representation"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["target", "model", "protocol", "n"], ascending=[True, True, True, False])
    )

    for (target, model, protocol), frame in rank.groupby(["target", "model", "protocol"]):
        top_repr = frame.head(confusion_top_k)["representation"].tolist()
        if not top_repr:
            continue

        n = len(top_repr)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, repr_name in zip(axes_flat, top_repr):
            y_true_blocks: list[np.ndarray] = []
            y_pred_blocks: list[np.ndarray] = []
            for row in predictions:
                if (
                    row["target"] != target
                    or row.get("model", "unknown") != model
                    or row["protocol"] != protocol
                    or row["representation"] != repr_name
                ):
                    continue
                if row["status"] not in {"ok", "ok_with_nan"}:
                    continue
                y_true_blocks.append(np.asarray(row["y_true"], dtype=np.int64))
                y_pred_blocks.append(np.asarray(row["y_pred"], dtype=np.int64))

            if not y_true_blocks:
                ax.axis("off")
                ax.set_title(f"{repr_name} (no valid folds)")
                continue

            y_true = np.concatenate(y_true_blocks)
            y_pred = np.concatenate(y_pred_blocks)
            cm = confusion_matrix(y_true, y_pred, labels=np.array([0, 1], dtype=np.int64)).astype(np.float64)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            cm_norm = cm / row_sums

            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
                cbar=True,
                xticklabels=["0", "1"],
                yticklabels=["0", "1"],
                ax=ax,
            )
            ax.set_title(repr_name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        for ax in axes_flat[n:]:
            ax.axis("off")

        fig.suptitle(f"Confusion matrices: {target} / {model} / {protocol}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = output_dir / f"confusion_{_sanitize(target)}_{_sanitize(model)}_{_sanitize(protocol)}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def _plot_temporal_diagnostics(
    *,
    output_dir: Path,
    metadata: pd.DataFrame,
    diagnostics_labels: dict[str, np.ndarray],
) -> list[Path]:
    """Plot rolling label distribution and entropy per subject for each target."""

    saved: list[Path] = []
    if metadata.empty:
        return saved

    subjects = metadata["Subject"].to_numpy(dtype=str)

    for target, labels in diagnostics_labels.items():
        labels = np.asarray(labels, dtype=np.int64)

        # Proportion panel.
        unique_subjects = np.unique(subjects)
        fig, axes = plt.subplots(
            nrows=len(unique_subjects),
            ncols=1,
            figsize=(11, max(3, 2.2 * len(unique_subjects))),
            sharex=False,
            squeeze=False,
        )
        for ax, subject in zip(axes.flatten(), unique_subjects):
            idx = np.where(subjects == subject)[0]
            seq = labels[idx]
            if seq.size == 0:
                ax.set_visible(False)
                continue
            window = min(100, max(5, seq.size // 8))
            smooth = pd.Series(seq).rolling(window=window, min_periods=1).mean().to_numpy()
            x = np.linspace(0.0, 100.0, num=smooth.size)
            ax.plot(x, 1.0 - smooth, label="Class 0", color="#4E79A7")
            ax.plot(x, smooth, label="Class 1", color="#E15759")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Subject {subject}")
            ax.set_ylabel("Proportion")
            ax.legend(loc="upper right")
        axes[-1, 0].set_xlabel("Recording progression (%)")
        fig.suptitle(f"{target}: temporal label proportions")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = output_dir / f"temporal_proportions_{_sanitize(target)}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

        # Entropy panel.
        fig, axes = plt.subplots(
            nrows=len(unique_subjects),
            ncols=1,
            figsize=(11, max(3, 2.2 * len(unique_subjects))),
            sharex=False,
            squeeze=False,
        )
        for ax, subject in zip(axes.flatten(), unique_subjects):
            idx = np.where(subjects == subject)[0]
            seq = labels[idx]
            if seq.size < 2:
                ax.set_visible(False)
                continue
            dummies = pd.get_dummies(pd.Series(seq))
            rolling_props = dummies.rolling(window=min(100, max(5, seq.size // 6)), min_periods=2).mean().dropna()
            if rolling_props.empty:
                ax.set_visible(False)
                continue
            probs = rolling_props.to_numpy(dtype=np.float64)
            probs = np.clip(probs, 1e-12, 1.0)
            ent = -(probs * np.log(probs)).sum(axis=1)
            x = np.linspace(0.0, 100.0, num=ent.size)
            ax.plot(x, ent, color="red", linewidth=1.8, alpha=0.8)
            ax.set_title(f"Subject {subject}")
            ax.set_ylabel("Entropy")
            ax.set_ylim(0.0, 0.75)
        axes[-1, 0].set_xlabel("Recording progression (%)")
        fig.suptitle(f"{target}: temporal label entropy")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = output_dir / f"temporal_entropy_{_sanitize(target)}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def create_protocol_plots(
    *,
    results_dir: Path,
    subject_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    predictions: list[dict[str, Any]],
    metadata: pd.DataFrame,
    diagnostics_labels: dict[str, np.ndarray],
    confusion_top_k: int,
) -> list[Path]:
    """Generate all protocol-level plots."""

    figures_dir = ensure_dir(results_dir / "figures")
    saved: list[Path] = []

    saved.extend(_plot_gain_barplots(output_dir=figures_dir, subject_df=subject_df, metric_col="gain"))
    saved.extend(
        _plot_gain_barplots(
            output_dir=figures_dir,
            subject_df=subject_df,
            metric_col="balanced_gain",
        )
    )
    saved.extend(_plot_protocol_heatmaps(output_dir=figures_dir, summary_df=summary_df))
    saved.extend(
        _plot_confusions(
            output_dir=figures_dir,
            predictions=predictions,
            confusion_top_k=confusion_top_k,
        )
    )
    saved.extend(
        _plot_temporal_diagnostics(
            output_dir=figures_dir,
            metadata=metadata,
            diagnostics_labels=diagnostics_labels,
        )
    )

    return saved
