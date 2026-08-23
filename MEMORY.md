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
- The manuscript deliberately does not mention the diagnostic global-median q5 visualization split. It states only the actual fold-specific label rule: mean-center each training participant separately and center the held-out participant using the overall training-participant mean.
- `configs/analysis.example.yaml` now reads `tobii_coordinate_normalized` and writes `results/data_analysis_q5/`. It contains only raw-q5 and binary-q5 class distributions, the normalized gaze-density map, and PCA/t-SNE/UMAP projections; no subject-coloured, pupil/violin, or missingness figures. Outputs are grouped as `<raw|binary>_q5/<pca|tsne|umap>/` for representation-by-method comparison.

## 7 — Bullet-only manuscript map (2026-07-13)

- `paper/main.tex` is now a bullet-only writing map: empty abstract; Introduction; Related Work; Methodology with Data and Eye-Tracking Representations plus Experimental Design; Results; Discussion; Conclusion. No existing prose, appendix, or Results/Discussion subsections remain.
- The outline specifies Table 1 (main LOSO representation comparison), Figure 1 (normalized gaze-density context), and one selected continuous-q5 projection as Figure 2, including their intended interpretation. The user will write all final sentences in their own style after results are final.

## 8 — Split manuscript chapters (2026-07-13)

- `paper/main.tex` now retains only the shared template, title material, empty abstract, acknowledgements, bibliography, and `\input` directives. The six manuscript chapters reside in `paper/sections/` as separate files: introduction, related work, methodology, results, discussion, and conclusion.

## 9 — Related Work written (2026-08-22)

- `paper/sections/02_related_work.tex` is the first section converted from bullets to prose: 329 words, four paragraphs, no subsections. The user's constraint is **at most 350 words and only strictly needed citations**, because the four-page limit must leave room for Methodology and Results, and the literature is expanded implicitly elsewhere.
- Paragraph structure: (1) gaze/pupil as behavioural evidence and the ceiling of fixed feature sets; (2) learned representations — GazeMAE as gaze-specific, MOMENT as general-purpose frozen encoder, plus the open question of out-of-the-box transfer; (3) evaluation design — dependent within-subject windows, random splits, preprocessing fitted on the full dataset; (4) positioning as an empirical comparison extending the authors' earlier feature-based analyses, not a new encoder or fine-tuning method.
- Confirmed framing decisions: the two self-citations use **"our own prior work" / "our earlier feature-based analyses"** phrasing (user chose this over neutral third person). Related Work stays **neutral about the expected outcome** — it states the open question but does not foreshadow that frozen embeddings will underperform; the Results section must carry that argument alone.
- `paper/references.bib` holds six Crossref-verified entries: `bautista2020gazemae` (ICPR 2020 proceedings version, doi:10.1109/ICPR48806.2021.9412761), `goswami2024moment` (ICML 2024), `bozak2026workload` (AI 7(8):325, doi:10.3390/ai7080325), `bozak2024emotion` (IS 2024 SCAI, https://is.ijs.si/?p=16837), `kahneman1966pupil` (Science, doi:10.1126/science.154.3756.1583), `rosenblatt2024leakage` (Nat. Commun. 15:1829, doi:10.1038/s41467-024-46150-w).
- biblatex name form for suffixes is `Last, Jr., First` (`Naval, Jr., Prospero C.`); the BibTeX-style `Last, First, Jr.` renders incorrectly.
- Stale `paper/main.bbl-SAVE-ERROR` removed. Remaining bullet-only sections: introduction, methodology, results, conclusion.

## 10 — Dataset status: unpublished, collection ongoing (2026-08-22)

- The dataset behind this paper is **not published and still being collected**. This is therefore a **preliminary study**, and the manuscript says so explicitly in the Related Work positioning sentence.
- The data is **not the CLUES dataset** and has no data-level connection to it. An earlier note in this file wrongly claimed otherwise; that claim was retracted by the user on 2026-08-22 and the corresponding `clues2026dataset` stub was removed from `paper/references.bib`. Do not cite the IMWUT/CLUES paper as the data source.
- The only real relationship to CLUES is thematic: CLUES likewise pairs eye-tracking data with psychological labels, and the same team worked on both datasets. Any future mention must be framed as study similarity, never shared data, shared participants, or a shared collection.
- It is also **not** the same data collection as the 54-participant lab task battery in the AI 7(8):325 workload paper.
- Internal name in code and configs remains TrustMe (`trustme_root`, `TRUSTME_DATA_ROOT`); the manuscript's dataset name is still undecided and must be confirmed before Methodology prose is written.

## 11 — Performance-result audit (2026-08-22)

- The newest completed matching performance artifact found is the 15-subject run `TrustME-ET-end-to-end/results/tobii_experiments/20260514_150320/tables/protocol_summary.csv`. Its q5 `loso_normalized` rows use the intended centred-label rule, RF/MLP classifiers, and all four representations, but it predates the current 17-subject exports (July 2026) and is therefore not the final paper cohort result.
- `results/paper/table1_primary_loso_results_15_subject_snapshot.csv` is a paper-shaped, explicitly provisional extract of that run. Accuracy includes all 15 folds; balanced accuracy, macro-F1, and AUC use 14 folds because `s_004_pk` has a one-class q5 test fold. The majority baseline is common across representations/models.
- The protocol evaluator had labelled positive-class binary F1 as `macro_f1`. It now uses `average="macro"`, with a regression test. The provisional CSV's true macro-F1 values were recovered from saved fold-level sample counts, positive-class precision, and recall; no values were invented.
- Do not describe the 15-subject CSV as final. Before filling Table 1, freeze the paper cohort and run the q5-only `loso_normalized` protocol on the same aligned, coordinate-normalized cohort used by the manuscript. The current performance example points to the July all-window artifacts, while the normalized subject-tree exports contain 17 subjects; this input-path/cohort choice must be reconciled.

## 12 — Rerun scope and runtime estimate (2026-08-22)

- Coordinate normalization changes only raw gaze values. Across all 17 subjects, `tobii_features.csv`, `tobii_gazemae_embeddings.csv`, and `tobii_moment_embeddings.csv` are byte-identical between `tobii/` and `tobii_coordinate_normalized/`. For a frozen 15-subject cohort, only raw modelling needs retraining; the corrected paper extract can reuse saved predictions for the other representations.
- Moving from 15 to 17 subjects requires rerunning every representation, because the two new held-out folds are needed and adding subjects changes each LOSO training set and the fold-dependent engagement centring. Results cannot be updated by merely appending two folds.
- The 2026-05-14 RTX 4070 SUPER log records 23.0 minutes for q5 `loso_normalized` raw with MLP+RF and 83.1 minutes for all four representations with both models. Scaling from 121,176 windows/15 folds to the current 177,244-window/17-subject analysis cohort gives rough compute estimates of 38 minutes raw-only or 2.3 hours for all representations. Allow roughly 45–60 minutes and 2.5–3 hours respectively for loading/output overhead. The old full three-target/five-protocol grid took about 11 hours and would scale to roughly 18 hours; it is unnecessary for Table 1.

## 13 — Final 17-subject paper experiment locked (2026-08-22)

- User approved the current 17 named subjects, q5 only (paper name `engagement_level`), coordinate-normalized raw input, normalized LOSO only, RF+MLP, and one fixed seed (42). All other supervised/temporal/HMM protocols are disabled. A one-step persistence check is enabled once because it is inexpensive and is reported separately, never ranked against LOSO.
- `configs/final_paper_experiment.yaml` freezes the exact subject names and reads `ml/tobii_coordinate_normalized/` through `${TRUSTME_SUBJECT_ROOT}`. Newly arriving subject directories cannot silently enter the run. The common four-representation intersection has 157,685 windows; after requiring numeric q5, the final modelling cohort is exactly **140,531 windows across all 17 subjects**. Per-subject counts will be written to `cohort_window_counts.csv`.
- Final representation dimensions after alignment are raw 600 (five channels × 120 samples), handcrafted export 199 columns before fold-local metadata removal/filtering, GazeMAE 256, and MOMENT 1,024. The handcrafted structural-drop list now explicitly removes q1–q9, q5 itself, sleep feedback, prompt identifiers/times, window identifiers, subject/source fields, timestamps, and window length, preventing survey/prompt metadata leakage.
- The held-out-participant engagement centre now follows the manuscript literally: it is the unweighted mean of the training participants' individual q5 means, rather than the window-weighted global mean used by the older code. True macro-F1 and majority macro-F1/AUC are computed directly in the final run.
- Persistence is now true one-step last-observation prediction and is constrained within source recordings; it is executed once with `model=persistence`, `representation=label_history`, rather than redundantly for every representation/model pair.
- With `outputs.paper_only: true` and plotting disabled, each timestamped final result directory contains only the config snapshot and the needed CSVs: `paper_main_results.csv` (mean, sample SD, paired deltas, per-metric valid-fold counts), `loso_fold_metrics.csv`, `paper_persistence_result.csv`, `persistence_fold_metrics.csv`, and `cohort_window_counts.csv`.
- A full read-only production load passed: 140,531 aligned rows, 17 subjects, no missing q5, all four expected shapes. It took 1:52 and peaked at 3.75 GB RAM. The optimized NumPy raw vectorizer is regression-tested against the established pandas masking/interpolation semantics. The data-free suite passes 13 tests.

## 14 — Final 17-subject paper results completed (2026-08-23)

- The final run completed at `results/final_paper_experiment/20260823_101436/`. Its report-ready sources are `tables/paper_main_results.csv` for Table 1 and `tables/paper_persistence_result.csv` for the separate persistence check. Fold-level audit files and cohort counts are in the same `tables/` directory. `paper/sections/04_results.tex` now records these exact paths in source comments and explicitly warns against using the obsolete 15-subject snapshot.
- Validation passed 18 consistency checks: seed 42 and q5-only configuration; only normalized LOSO plus persistence enabled; the frozen 17-subject cohort totals 140,531 windows; all 136 expected LOSO rows (8 model–representation combinations x 17 subjects) completed without skipped folds; all summaries exactly reproduce fold-level means, sample SDs, paired deltas, and valid-fold counts; all 1,434 persistence temporal folds completed and the persistence summary exactly reproduces subject-macro aggregation.
- `s_004_pk` is the single one-class LOSO test subject, so accuracy has 17 valid folds while balanced accuracy, macro-F1, and ROC-AUC have 16. This is metric availability information only; the final performance values have not yet been interpreted in the manuscript.
