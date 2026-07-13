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

## 3 — Paper scaffold (2026-07-10)

- paper/main.tex is now a project-specific Information Society 2026 manuscript scaffold, replacing the pasted ACM sample. It uses references.bib (not the missing sample-base.bib), has no sample teaser asset, and contains contextual red placeholder markers for author details, dataset facts, citations, and results that are not yet known.
- The scaffold follows PAPER_OUTLINE.md: Introduction, Related Work, Dataset and Preprocessing, Representations, Experimental Setup, Results, Discussion, Conclusion, and a Reproducibility appendix. The recommended first prose is the Introduction, followed by Dataset and Preprocessing once study facts are confirmed; Results and Conclusion deliberately remain result-specific placeholders.
- Real run validated 17 subjects/68 representation CSVs and completed all PCA, t-SNE, and UMAP outputs. `umap-learn==0.5.12` was added to the existing `trust-me-et` environment (already declared in repository requirements); UMAP uses explicit `n_jobs=1` with the fixed seed for reproducibility and warning-free execution.

## 4 — Four-page paper planning (2026-07-10)

- User wants a four-page English paper including references, with neither appendix nor supplementary material. The intended core is a controlled, unseen-subject comparison of raw, handcrafted, GazeMAE, and MOMENT representations for the final selected self-report target (currently q5).
- Proposed main-results columns are representation, model, accuracy and delta over majority, balanced accuracy and delta over majority, macro-F1, and AUC. Keep deltas paired at the held-out-subject/fold level before aggregation, and keep undefined AUC values explicit for one-class test folds.
- Planned descriptive figure is a compact two-panel figure: left/right pupil-size histogram plus normalized gaze-coordinate heatmap. Screen-coordinate 0--1 normalization for the heatmap is descriptive only and must use verified per-recording screen dimensions; it is distinct from model preprocessing.
- Current `loso_normalized` code does train-fold-only median imputation followed by `StandardScaler` z-scoring per representation feature, then applies those parameters to the held-out subject. The example configuration disables the optional second model-side scaler, avoiding double scaling.

## 5 — Coordinate-normalized Tobii exports (2026-07-10)

- q5 is the numerical response to “Engagement; How immersed in your work did you feel just now?” The four-page English paper will compare representations for unseen-office-worker engagement inference; its intended title is **Comparison of Eye-Tracking Representations for Engagement Inference in Office Workers**. Frozen pretrained GazeMAE and MOMENT encoders are evaluated out of the box (only a prediction head is trained), and the intended conclusion is that embeddings do not justify their cost in this setting.
- `tobii_experiments/normalize_tobii_coordinates.py` creates each subject’s `ml/tobii_coordinate_normalized/` sibling. It copies features, GazeMAE, MOMENT, and manifest CSVs byte-for-byte and normalizes all six raw gaze columns (`GazePointX/Y`, left, right) by source-recording width/height. It maps `source_file` back from `*.tsv.parquet` to the raw TSV, reads the TSV’s `display resolution: WxH` header, and validates it against `display_resolutions.yaml`; this handles subjects with multiple displays per recording rather than selecting one subject-level resolution. It intentionally does not clip out-of-screen values, so later heatmaps must restrict to `[0,1]^2` after normalization.
- The script supports CLI/YAML configuration, chunked processing, atomic raw CSV replacement, `--dry-run`, and `--overwrite`; a synthetic test verifies per-source normalization and unchanged copied files. The 17-subject production run completed to the local raw-data tree and requires an audit before experiments/analysis point their `subject_export_dir` to `tobii_coordinate_normalized`.

## 6 — Finalized q5 analysis views (2026-07-10)

- The experimental target remains the current LOSO rule: per-training-subject mean-centering, with the held-out subject centred by the global training mean. Do not replace this with participant-median labels.
- Descriptive visualizations additionally use a clearly separate diagnostic binary label, `q5 > global median(q5)`, calculated over the entire processed dataset. The observed global median is `4.0`; this diagnostic label is never used for model training or reported LOSO metrics.
- `configs/analysis.example.yaml` now reads `tobii_coordinate_normalized` and writes `results/data_analysis_q5/`. It contains only raw-q5 and binary-q5 class distributions, the normalized gaze-density map, and PCA/t-SNE/UMAP projections; no subject-coloured, pupil/violin, or missingness figures. Outputs are grouped as `<raw|binary>_q5/<pca|tsne|umap>/` for representation-by-method comparison.

## 7 — Bullet-only manuscript map (2026-07-13)

- `paper/main.tex` is now a bullet-only writing map: empty abstract; Introduction; Related Work; Methodology with Data and Eye-Tracking Representations plus Experimental Design; Results; Discussion; Conclusion. No existing prose, appendix, or Results/Discussion subsections remain.
- The outline specifies Table 1 (main LOSO representation comparison), Figure 1 (normalized gaze-density context), and one selected continuous-q5 projection as Figure 2, including their intended interpretation. The user will write all final sentences in their own style after results are final.

## 8 — Split manuscript chapters (2026-07-13)

- `paper/main.tex` now retains only the shared template, title material, empty abstract, acknowledgements, bibliography, and `\input` directives. The six manuscript chapters reside in `paper/sections/` as separate files: introduction, related work, methodology, results, discussion, and conclusion.
