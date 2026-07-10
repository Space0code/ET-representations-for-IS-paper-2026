# TrustMe Eye-Tracking Representation Comparison

This repository contains the experiment, evaluation, descriptive-analysis, and
visualization code for a controlled comparison of four Tobii eye-tracking
representations:

1. raw;
2. handcrafted features;
3. GazeMAE embeddings;
4. MOMENT embeddings.

The repository intentionally contains no participant data, derived data, model
weights, checkpoints, or experiment results. Representation computation remains
in the `TrustME-ET-end-to-end` repository; this package starts from
already-computed representations.

## Setup

The commands below assume the `trust-me-et` conda environment is active.

```bash
python -m pip install -r requirements.txt
```

UMAP is used only by the analysis CLI. If it is not needed, it may be removed
from a local installation and omitted from `projection_methods`.

## Configure the data location

The example configurations contain no machine-specific paths. Set one local
environment variable to the directory holding the latest all-window exports:

```bash
export TRUSTME_DATA_ROOT=/path/to/tobii_all
```

For the completed `s_001_gs` all-window run described in the handoff notes,
this is the local mount corresponding to its `late_2025/ml/tobii_all`
directory. The analysis example expects combined export files directly below
that directory. The experiment example expects the pipeline `_artifacts`
layout, so set `TRUSTME_DATA_ROOT` to `_artifacts` for that command. Adjust the
YAML paths if the local layout differs.

## Main experiment

The headline protocol is leakage-safe, LOSO-normalized evaluation of q5 using a
majority reference baseline, Random Forest, and MLP. Scaling and preprocessing
are fitted on training subjects and then applied to the held-out subject.

```bash
python tobii_experiments/run_zoja_protocols.py \
  --config configs/experiments.example.yaml
```

The example deliberately disables secondary protocols. They remain available
in the copied experiment code and can be enabled in YAML after the main result
is established. One-class folds are retained with undefined metrics recorded
as `NaN`. Confusion matrices are row-normalized with a fixed `[0, 1]` color
scale.

## Data analysis and visualizations

```bash
python tobii_experiments/run_data_analysis.py \
  --config configs/analysis.example.yaml
```

This produces:

- overall and subject-level class distributions;
- pupil-size distributions by class;
- a gaze-coordinate density map;
- a missingness summary;
- PCA, t-SNE, and optionally UMAP projections for each representation, colored
  by class and subject;
- projection coordinates as CSV files and a JSON output manifest.

Raw samples are streamed from CSV, resampled, and flattened per selected window
only for the representation projection. Plot sampling limits are configurable
so large all-window exports remain manageable. Dimensionality-reduction plots are illustrative and should
not be interpreted as evidence of class separability.

## Repository structure

```text
configs/                    safe example configurations
src/trustme_et_comparison/  experiment and analysis library
tobii_experiments/          command-line entrypoints
tests/                      data-free synthetic tests
PAPER_OUTLINE.md            proposed paper structure
```

Run the data-free checks with:

```bash
pytest
```
