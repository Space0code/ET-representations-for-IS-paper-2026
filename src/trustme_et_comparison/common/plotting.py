"""Plot generation for trained experiment outputs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .utils import ensure_dir


METRIC_COLUMNS = {
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
    "macro_f1": "macro_f1",
    "auc": "auc",
    "precision": "precision",
    "recall": "recall",
}

# Consistent visual identity for each representation.
# bar_color: used in barplots.  cmap: sequential colormap for confusion matrices.
REPR_STYLE: dict[str, dict[str, str]] = {
    "raw":        {"bar_color": "#4caf50", "cmap": "Greens"},   # green
    "embeddings": {"bar_color": "#2196f3", "cmap": "Blues"},    # blue
    "features":   {"bar_color": "#ff9900", "cmap": "YlOrBr"},   # yellow/amber
    "raw_rows":   {"bar_color": "#2e7d32", "cmap": "Greens"},
    "label_baseline": {"bar_color": "#6d6d6d", "cmap": "Greys"},
}

BARPLOT_BASELINE_PROXY_MODELS: tuple[str, ...] = ("lgbm", "mlp")


def _representation_sort_key(name: str) -> tuple[int, str]:
    """Sort representations into stable visual groups."""

    if name == "raw":
        return (0, name)
    if name == "raw_rows":
        return (1, name)
    if name.startswith("embeddings"):
        return (2, name)
    if name == "features" or name.startswith("features_"):
        return (3, name)
    if name.startswith("early_"):
        return (4, name)
    if name.startswith("late_soft_"):
        return (5, name)
    if name == "label_baseline":
        return (6, name)
    return (7, name)


def _ordered_representations(names: list[str]) -> list[str]:
    """Return stable representation ordering for plotting."""

    return sorted(set(names), key=_representation_sort_key)


def _repr_style(name: str) -> dict[str, str]:
    """Return plotting style for one representation."""

    if name in REPR_STYLE:
        return REPR_STYLE[name]
    palette = sns.color_palette("tab20", 20)
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    color = mcolors.to_hex(palette[int(digest[:8], 16) % len(palette)])
    return {"bar_color": color, "cmap": "viridis"}


def _remap_label_baseline_for_barplots(
    frame: pd.DataFrame,
    proxy_models: tuple[str, ...] = BARPLOT_BASELINE_PROXY_MODELS,
) -> pd.DataFrame:
    """Treat label_baseline as a representation in model-grouped barplots.

    The runner stores baseline outputs as (representation=label_baseline,
    model=label_baseline). For grouped barplots with x-axis=model, this helper
    removes the standalone baseline model and duplicates those rows under
    selected trained models (default: lgbm and mlp).
    """

    required_columns = {"representation", "model"}
    if not required_columns.issubset(frame.columns):
        return frame.copy()

    baseline_mask = (frame["representation"] == "label_baseline") & (frame["model"] == "label_baseline")
    if not baseline_mask.any():
        return frame.copy()

    # Remove standalone baseline model from the barplot input.
    remapped = frame.loc[frame["model"] != "label_baseline"].copy()
    baseline_rows = frame.loc[baseline_mask].copy()

    trained_models = sorted(set(remapped["model"].unique()))
    target_models = [model for model in proxy_models if model in trained_models]
    if not target_models:
        target_models = trained_models
    if not target_models:
        return remapped

    duplicated_rows = []
    for model in target_models:
        model_rows = baseline_rows.copy()
        model_rows["model"] = model
        duplicated_rows.append(model_rows)

    if duplicated_rows:
        remapped = pd.concat([remapped] + duplicated_rows, ignore_index=True)
    return remapped


def _load_label_names(results_dir: Path, scheme: str, variant: str) -> list[str]:
    mapping_path = results_dir / "splits" / scheme / variant / "label_mapping.json"
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    return list(payload["class_order"])


def plot_confusion_comparisons(
    results_dir: Path,
    output_dir: Path,
    confusion_top_k: int,
) -> list[Path]:
    """Plot 3-panel confusion matrices per labelling scheme and model.

    Panels are in canonical order (raw → embeddings → features) and use a
    consistent "Blues" colormap for easier cross-panel comparison.
    """
    summary = pd.read_csv(results_dir / "tables" / "summary_metrics.csv")
    saved: list[Path] = []

    for scheme, variant in sorted(
        summary[["labelling", "embedding_variant"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        label_names = _load_label_names(results_dir, scheme, variant)
        class_indices = np.arange(len(label_names), dtype=np.int64)
        scheme_frame = summary[
            (summary["labelling"] == scheme) & (summary["embedding_variant"] == variant)
        ]

        for model in sorted(scheme_frame["model"].unique()):
            model_frame = scheme_frame[scheme_frame["model"] == model].copy()
            repr_candidates = model_frame.sort_values("accuracy_mean", ascending=False)["representation"].tolist()
            top_repr_by_metric: list[str] = []
            seen: set[str] = set()
            for name in repr_candidates:
                if name in seen:
                    continue
                seen.add(name)
                top_repr_by_metric.append(name)
                if len(top_repr_by_metric) >= confusion_top_k:
                    break
            representations = _ordered_representations(top_repr_by_metric)
            if not representations:
                continue

            n_repr = len(representations)
            n_cols = min(3, n_repr)
            n_rows = int(np.ceil(n_repr / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
            axes_flat = axes.flatten()
            figure_has_content = False

            for ax, representation in zip(axes_flat, representations):
                agg_dir = results_dir / scheme / variant / representation / model / "aggregate"
                if not agg_dir.exists():
                    ax.axis("off")
                    ax.set_title(f"{representation} (missing)")
                    continue

                y_true = np.load(agg_dir / "y_true.npy")
                y_pred = np.load(agg_dir / "y_pred.npy")
                cm = confusion_matrix(y_true, y_pred, labels=class_indices).astype(np.float64)
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0.0] = 1.0
                cm_normalized = cm / row_sums

                sns.heatmap(
                    cm_normalized,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    cbar=True,
                    vmin=0.0,
                    vmax=1.0,
                    xticklabels=label_names,
                    yticklabels=label_names,
                    ax=ax,
                )
                ax.set_title(representation)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                figure_has_content = True

            for ax in axes_flat[n_repr:]:
                ax.axis("off")

            if figure_has_content:
                fig.suptitle(f"Confusion Matrices: {scheme} / {variant} / {model}")
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                out_path = output_dir / f"confusion_{scheme}_{variant}_{model}.png"
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                saved.append(out_path)
            plt.close(fig)

    return saved


def plot_metric_barplots(results_dir: Path, output_dir: Path) -> list[Path]:
    """Plot grouped barplots: x-axis = classifiers, bars grouped by representation.

    Each bar shows the cross-fold mean; error bars are ±1 SD.  The mean value
    is printed above each bar for quick visual comparison.
    """
    fold_metrics = pd.read_csv(results_dir / "tables" / "fold_metrics.csv")
    saved: list[Path] = []

    groups = fold_metrics[["labelling", "embedding_variant"]].drop_duplicates()
    for scheme, variant in sorted(groups.itertuples(index=False, name=None)):
        frame = fold_metrics[
            (fold_metrics["labelling"] == scheme) & (fold_metrics["embedding_variant"] == variant)
        ].copy()
        frame = _remap_label_baseline_for_barplots(frame)
        models = sorted(frame["model"].unique())
        representations = _ordered_representations(frame["representation"].unique().tolist())
        n_models = len(models)
        n_repr = len(representations)
        if n_models == 0 or n_repr == 0:
            continue

        for metric_name, column in METRIC_COLUMNS.items():
            if column not in frame.columns:
                continue

            x = np.arange(n_models)
            width = 0.75 / n_repr

            fig, ax = plt.subplots(figsize=(max(8, n_models * 2.0), 5))

            for i, repr_name in enumerate(representations):
                style = _repr_style(repr_name)
                vals: list[float] = []
                errs: list[float] = []
                for model in models:
                    sub = frame[
                        (frame["model"] == model) & (frame["representation"] == repr_name)
                    ]
                    if len(sub) > 0:
                        vals.append(float(sub[column].mean()))
                        errs.append(float(sub[column].std()))
                    else:
                        vals.append(0.0)
                        errs.append(0.0)

                offset = (i - n_repr / 2 + 0.5) * width
                bars = ax.bar(
                    x + offset, vals, width,
                    yerr=errs, capsize=4,
                    label=repr_name, alpha=0.85,
                    color=style.get("bar_color"),
                )

                for bar, val in zip(bars, vals):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.2f}",
                            ha="center", va="bottom", fontsize=7.5,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha="right")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{scheme} / {variant} \u2014 {metric_name}")
            ax.set_ylim(0, 1.05)
            ax.legend(title="Representation", bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()

            out_path = output_dir / f"barplot_{scheme}_{variant}_{metric_name}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)

    return saved


def create_all_plots(results_dir: Path, confusion_top_k: int = 9) -> list[Path]:
    """Generate confusion matrix and metric barplot PNG files."""
    figures_dir = ensure_dir(results_dir / "figures")
    saved = []
    saved.extend(
        plot_confusion_comparisons(
            results_dir=results_dir,
            output_dir=figures_dir,
            confusion_top_k=int(confusion_top_k),
        )
    )
    saved.extend(plot_metric_barplots(results_dir=results_dir, output_dir=figures_dir))
    return saved
