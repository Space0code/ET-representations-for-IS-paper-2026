"""Paper-oriented descriptive and representation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save and close a figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(
    frame: pd.DataFrame,
    target: str,
    target_label: str,
    output_path: Path,
) -> Path:
    """Plot counts for one target-label definition."""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = sorted(frame[target].dropna().unique(), key=str)
    sns.countplot(data=frame, x=target, order=order, color="#4472C4", ax=ax)
    ax.set_xlabel(target_label)
    ax.set_title(f"Class distribution: {target_label}")
    ax.set_ylabel("Windows")
    return _save(fig, output_path)


def plot_gaze_density(frame: pd.DataFrame, gaze_columns: tuple[str, str], output_dir: Path) -> Path:
    """Plot a density map of coordinates normalized to the display extent."""

    x_column, y_column = gaze_columns
    clean = frame[[x_column, y_column]].dropna()
    clean = clean[clean[x_column].between(0.0, 1.0) & clean[y_column].between(0.0, 1.0)]
    if clean.empty:
        raise ValueError("No valid normalized gaze coordinates are available for the density plot.")
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.hist2d(
        clean[x_column],
        clean[y_column],
        bins=(80, 45),
        range=((0.0, 1.0), (0.0, 1.0)),
        weights=[1.0 / len(clean)] * len(clean),
        cmap="magma",
    )[3]
    fig.colorbar(image, ax=ax, label="Probability mass per bin")
    ax.set_xlabel("Normalized horizontal gaze coordinate")
    ax.set_ylabel("Normalized vertical gaze coordinate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Gaze-coordinate density on the normalized display")
    return _save(fig, output_dir / "gaze_density.png")


def plot_projection(
    frame: pd.DataFrame,
    representation: str,
    method: str,
    color_column: str,
    target_label: str,
    continuous: bool,
    output_path: Path,
) -> Path:
    """Plot a two-dimensional representation projection coloured by one target."""

    fig, ax = plt.subplots(figsize=(7, 6))
    colour = pd.to_numeric(frame[color_column], errors="coerce")
    if continuous:
        scatter = ax.scatter(
            frame["component_1"],
            frame["component_2"],
            c=colour,
            cmap="viridis",
            s=16,
            alpha=0.65,
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, label=target_label)
    else:
        binary = colour.fillna(-1).astype(int)
        palette = {0: "#4C78A8", 1: "#E45756"}
        ax.scatter(
            frame["component_1"],
            frame["component_2"],
            c=binary.map(palette),
            s=16,
            alpha=0.65,
            linewidths=0,
        )
        ax.legend(
            handles=[Patch(color=palette[0], label="≤ global median"), Patch(color=palette[1], label="> global median")],
            title=target_label,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
    ax.set_title(f"{representation}: {method.upper()} coloured by {target_label}")
    safe_representation = representation.lower().replace(" ", "_")
    return _save(fig, output_path / f"projection_{safe_representation}.png")
