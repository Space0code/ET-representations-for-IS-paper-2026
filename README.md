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

For descriptive analysis, set `trustme_root` in
`configs/analysis.example.yaml` to the directory containing the subject
folders:

```yaml
trustme_root: /path/to/TrustMe
```

The loader scans every `<subject>/ml/tobii/` directory and requires the four
standard CSV exports (`tobii_raw_samples.csv`, `tobii_features.csv`,
`tobii_gazemae_embeddings.csv`, and `tobii_moment_embeddings.csv`). It reports
all incomplete subjects before starting analysis. The experiment example still
uses `${TRUSTME_DATA_ROOT}` for the separate pipeline `_artifacts` layout.

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
