"""Paper-oriented descriptive and representation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save and close a figure."""

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(frame: pd.DataFrame, target: str, output_dir: Path) -> Path:
    """Plot counts for one target label."""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = sorted(frame[target].dropna().unique(), key=str)
    sns.countplot(data=frame, x=target, order=order, color="#4472C4", ax=ax)
    ax.set_title(f"Class distribution: {target}")
    ax.set_ylabel("Windows")
    return _save(fig, output_dir / f"class_distribution_{target}.png")


def plot_subject_class_distribution(
    frame: pd.DataFrame,
    target: str,
    subject_column: str,
    output_dir: Path,
) -> Path:
    """Plot subject-level class proportions."""

    counts = frame.groupby([subject_column, target], observed=True).size().rename("count").reset_index()
    counts["proportion"] = counts["count"] / counts.groupby(subject_column)["count"].transform("sum")
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * counts[subject_column].nunique()), 5))
    sns.barplot(data=counts, x=subject_column, y="proportion", hue=target, ax=ax)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(f"Subject-level class distribution: {target}")
    return _save(fig, output_dir / f"subject_class_distribution_{target}.png")


def plot_pupil_distributions(
    frame: pd.DataFrame,
    pupil_columns: tuple[str, ...],
    target: str,
    output_dir: Path,
) -> Path:
    """Plot sampled pupil-size distributions split by target class."""

    long = frame.melt(id_vars=[target], value_vars=list(pupil_columns), var_name="channel", value_name="pupil_size")
    long = long.dropna(subset=["pupil_size", target])
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=long, x=target, y="pupil_size", hue="channel", cut=0, inner="quart", ax=ax)
    ax.set_title(f"Pupil-size distribution by {target}")
    return _save(fig, output_dir / f"pupil_distribution_{target}.png")


def plot_gaze_density(frame: pd.DataFrame, gaze_columns: tuple[str, str], output_dir: Path) -> Path:
    """Plot gaze-coordinate density as a normalized-screen hexbin."""

    x_column, y_column = gaze_columns
    clean = frame[[x_column, y_column]].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.hexbin(clean[x_column], clean[y_column], gridsize=70, mincnt=1, cmap="magma")
    fig.colorbar(image, ax=ax, label="Samples per hexagon")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.invert_yaxis()
    ax.set_title("Gaze-coordinate density")
    return _save(fig, output_dir / "gaze_density.png")


def plot_missingness(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot missing-value fractions for the supplied raw columns."""

    missing = frame.isna().mean().sort_values(ascending=False).rename("missing_fraction").reset_index()
    missing.rename(columns={"index": "column"}, inplace=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=missing, x="column", y="missing_fraction", color="#70AD47", ax=ax)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Missingness / validity summary")
    return _save(fig, output_dir / "missingness.png")


def plot_projection(
    frame: pd.DataFrame,
    representation: str,
    method: str,
    color_column: str,
    output_dir: Path,
) -> Path:
    """Plot a two-dimensional representation projection."""

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=frame,
        x="component_1",
        y="component_2",
        hue=color_column,
        s=16,
        alpha=0.65,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(f"{representation}: {method.upper()} colored by {color_column}")
    ax.legend(title=color_column, bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=1.5)
    safe_color = "".join(char if char.isalnum() else "_" for char in color_column)
    safe_representation = representation.lower().replace(" ", "_")
    return _save(fig, output_dir / f"projection_{safe_representation}_{method}_{safe_color}.png")

