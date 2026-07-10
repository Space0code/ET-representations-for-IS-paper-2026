# Project memory

## 1 — Initial repository state (2026-07-10)

- Purpose: public, data-free codebase for the IS SCAI 2026 paper comparing, in fixed order, **raw**, **handcrafted features**, **GazeMAE embeddings**, and **MOMENT embeddings** on already-computed TrustMe Tobii windows; representation generation stays in `TrustME-ET-end-to-end`.
- Main question/protocol: predict the provisional primary target **q5** on unseen subjects using leakage-safe **LOSO-normalized** evaluation; preprocessing/scaling is fit on training subjects only. Compare majority baseline, RF, and MLP; report accuracy and balanced-accuracy with deltas over majority, macro-F1, and AUC (`NaN` for undefined one-class folds). Row-normalize confusion matrices with fixed `[0,1]` scale.
- Secondary/open decisions: confirm q5; decide AUC emphasis, best PCA/t-SNE/UMAP figure, whether within-subject temporal and persistence results belong in the paper, and final IS/SCAI format. These secondary protocols are disabled in the main example config.
- Repository created locally at `/home/ppg/eyetracking/ET-representations-for-IS-paper-2026`; Git history starts with `713dd63` (copied experiment/evaluation Python only), followed by `54bafc0` (analysis, safe configs, tests, README, paper outline). GitHub creation/push is handled by the user.
- Key files: `configs/experiments.example.yaml`, `configs/analysis.example.yaml`, `tobii_experiments/run_zoja_protocols.py`, `tobii_experiments/run_data_analysis.py`, `PAPER_OUTLINE.md`, and `README.md`. Paths use `${TRUSTME_DATA_ROOT}`; no participant data, outputs, weights, credentials, or sensitive server paths may be committed.
- Analysis outputs: class and subject distributions, pupil distributions, gaze density, missingness, and PCA/t-SNE/optional UMAP projections colored by class/subject; raw CSV sampling is streamed and raw windows are resampled/flattened only for visualization. Projections are illustrative, not separability evidence.
- Environment/rules: use the single `trust-me-et` conda environment and root `requirements.txt`; typed, readable Python with concise docstrings and YAML/CLI-configurable paths. Initial verification: both CLIs/configs load, synthetic suite passes (`4 passed`), and the repository contained no large/sensitive artifacts.

## 2 — Subject-tree analysis input (2026-07-10)

- Labelled per-subject exports are sufficient for current supervised/descriptive work. Analysis config now needs only `trustme_root`; code discovers every `<subject>/ml/tobii/`, strictly requires the four standard raw/features/GazeMAE/MOMENT CSVs, combines subjects in memory, and records subjects in the manifest. `paper/` is out of scope and must remain untouched.
