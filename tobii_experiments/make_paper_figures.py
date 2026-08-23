"""Build the two manuscript figures for the IS SCAI 2026 paper.

Figure 1 (single column):
  (a) gaze-coordinate density on the normalized display;
  (b) engagement-rating distribution with per-participant means.

Figure 2 (double column):
  t-SNE projections of the four representations, coloured by engagement
  rating (top row) and by participant (bottom row).

Example:
    python tobii_experiments/make_paper_figures.py \
      --config configs/final_paper_experiment.yaml \
      --projection-dir results/data_analysis_q5/raw_q5/tsne \
      --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import BoundaryNorm

GAZE_COLUMNS = ["GazePointX", "GazePointY"]
TARGET_COLUMN = "5"
REPRESENTATIONS = [
    ("projection_raw.csv", "Raw"),
    ("projection_handcrafted_features.csv", "Handcrafted"),
    ("projection_gazemae_embeddings.csv", "GazeMAE"),
    ("projection_moment_embeddings.csv", "MOMENT"),
]


def _expand(value: str) -> str:
    """Expand environment variables in a configured path."""

    return os.path.expandvars(str(value))


def load_cohort_labels(subject_root: Path, export_dir: str, subjects: list[str]) -> pd.DataFrame:
    """Return one row per modelling window with its participant and rating.

    The modelling cohort is the intersection of the four representation
    exports, restricted to windows carrying a numeric engagement rating.
    """

    rows: list[pd.DataFrame] = []
    for subject in subjects:
        base = subject_root / subject / "ml" / export_dir
        keep = set(pd.read_csv(base / "tobii_gazemae_embeddings.csv", usecols=["window_uid"])["window_uid"])
        keep &= set(pd.read_csv(base / "tobii_moment_embeddings.csv", usecols=["window_uid"])["window_uid"])
        features = pd.read_csv(base / "tobii_features.csv", usecols=["window_uid", TARGET_COLUMN])
        features = features[features["window_uid"].isin(keep)]
        features = features[pd.to_numeric(features[TARGET_COLUMN], errors="coerce").notna()]
        rows.append(pd.DataFrame({"subject": subject, "rating": features[TARGET_COLUMN].astype(float)}))
    return pd.concat(rows, ignore_index=True)


def gaze_histogram(
    subject_root: Path,
    export_dir: str,
    subjects: list[str],
    bins: int = 60,
    chunksize: int = 2_000_000,
) -> tuple[np.ndarray, int, int]:
    """Accumulate a 2-D gaze histogram on the normalized display.

    Returns the histogram, the number of on-screen samples, and the total
    number of samples with both coordinates present.
    """

    hist = np.zeros((bins, bins), dtype=np.int64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    on_screen = 0
    total = 0
    for subject in subjects:
        path = subject_root / subject / "ml" / export_dir / "tobii_raw_samples.csv"
        for chunk in pd.read_csv(path, usecols=GAZE_COLUMNS, chunksize=chunksize):
            chunk = chunk.dropna()
            total += len(chunk)
            x = chunk["GazePointX"].to_numpy(dtype=np.float64)
            y = chunk["GazePointY"].to_numpy(dtype=np.float64)
            inside = (x >= 0.0) & (x < 1.0) & (y >= 0.0) & (y < 1.0)
            on_screen += int(inside.sum())
            counts, _, _ = np.histogram2d(x[inside], y[inside], bins=[edges, edges])
            hist += counts.astype(np.int64)
    return hist, on_screen, total


def make_figure1(labels: pd.DataFrame, hist: np.ndarray, out_path: Path) -> None:
    """Draw the descriptive single-column figure."""

    plt.rcParams.update({"font.size": 6.5, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(3.34, 1.55))

    density = hist / hist.sum()
    axes[0].imshow(density.T, origin="upper", extent=(0, 1, 1, 0), cmap="magma", aspect="equal")
    axes[0].set_xticks([0, 0.5, 1])
    axes[0].set_yticks([0, 0.5, 1])
    axes[0].set_xlabel("normalised $x$", labelpad=1)
    axes[0].set_ylabel("normalised $y$", labelpad=1)
    axes[0].set_title("(a) gaze density", pad=3)

    share = labels["rating"].value_counts(normalize=True).sort_index() * 100.0
    axes[1].bar(share.index, share.values, color="#9aa7c7", width=0.8, zorder=2)
    means = labels.groupby("subject")["rating"].mean().to_numpy()
    axes[1].plot(means, np.full_like(means, 37.0), "|", color="#b2182b", markersize=6, zorder=3)
    axes[1].set_xlim(-0.7, 6.7)
    axes[1].set_ylim(0, 42)
    axes[1].set_xticks(range(7))
    axes[1].set_yticks([0, 10, 20, 30])
    axes[1].set_xlabel("engagement rating", labelpad=1)
    axes[1].set_ylabel("% of windows", labelpad=1)
    axes[1].set_title("(b) rating distribution", pad=3)
    for spine in ("top", "right"):
        axes[1].spines[spine].set_visible(False)

    fig.tight_layout(pad=0.2, w_pad=1.0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def make_figure2(projection_dir: Path, out_path: Path) -> None:
    """Draw the double-column t-SNE panel grid."""

    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.titlesize": 8})
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.55))
    rating_cmap = plt.get_cmap("viridis", 7)
    rating_norm = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), 7)
    subjects: list[str] | None = None
    scatter = None

    for column, (filename, name) in enumerate(REPRESENTATIONS):
        frame = pd.read_csv(
            projection_dir / filename,
            usecols=["subject", TARGET_COLUMN, "component_1", "component_2"],
            low_memory=False,
        )
        if subjects is None:
            subjects = sorted(frame["subject"].unique())
        subject_codes = frame["subject"].map({name_: index for index, name_ in enumerate(subjects)})

        top = axes[0, column]
        scatter = top.scatter(
            frame["component_1"], frame["component_2"], c=frame[TARGET_COLUMN],
            cmap=rating_cmap, norm=rating_norm, s=1.4, alpha=0.75, linewidths=0,
        )
        top.set_title(name, pad=3)
        axes[1, column].scatter(
            frame["component_1"], frame["component_2"], c=subject_codes,
            cmap=plt.get_cmap("tab20", len(subjects)), s=1.4, alpha=0.75, linewidths=0,
        )
        for axis in (top, axes[1, column]):
            axis.set_xticks([])
            axis.set_yticks([])

    axes[0, 0].set_ylabel("colour: rating")
    axes[1, 0].set_ylabel("colour: participant")
    fig.subplots_adjust(left=0.028, right=0.915, top=0.94, bottom=0.015, wspace=0.05, hspace=0.05)
    cax = fig.add_axes((0.928, 0.52, 0.011, 0.42))
    bar = fig.colorbar(scatter, cax=cax, ticks=range(7))
    bar.set_label("engagement rating", fontsize=6.5)
    bar.ax.tick_params(labelsize=6)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/final_paper_experiment.yaml")
    parser.add_argument("--projection-dir", default="results/data_analysis_q5/raw_q5/tsne")
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    subject_root = Path(_expand(config["paths"]["subject_tree_root"]))
    export_dir = str(config["paths"]["subject_export_dir"])
    subjects = list(config["paths"]["subjects"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_cohort_labels(subject_root, export_dir, subjects)
    hist, on_screen, total = gaze_histogram(subject_root, export_dir, subjects)
    print(f"cohort windows: {len(labels)}; participants: {labels['subject'].nunique()}")
    print(f"on-screen gaze samples: {on_screen}/{total} ({100.0 * on_screen / total:.1f}%)")

    make_figure1(labels, hist, out_dir / "fig1_descriptive.pdf")
    make_figure2(Path(args.projection_dir), out_dir / "fig2_tsne.png")
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
