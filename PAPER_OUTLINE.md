# Proposed Paper Outline

## Working title

- **Raw Signals, Handcrafted Features, or Learned Embeddings? A Controlled
  Comparison of Eye-Tracking Representations for User-State Prediction**
- Short-title alternative: **Comparing Eye-Tracking Representations for
  User-State Prediction**

## Abstract

- Motivate eye tracking as a source of behavioral evidence for self-reported
  engagement or user state.
- State the practical uncertainty: learned representations are attractive, but
  may not outperform raw signals or handcrafted features on modest,
  subject-dependent datasets.
- Describe the controlled comparison of raw, handcrafted features, GazeMAE
  embeddings, and MOMENT embeddings.
- Name q5 as the provisional primary target and LOSO-normalized evaluation as
  the primary protocol.
- Summarize the majority baseline, Random Forest, MLP, and reported metrics.
- Insert the principal quantitative result and practical conclusion after the
  experiments are complete.

## 1. Introduction

- Explain why cognitive, affective, and engagement-related user states matter
  to interactive-system research.
- Describe the representation choice faced by practitioners:
  - raw gaze and pupil sequences retain detail but are high-dimensional;
  - handcrafted features encode domain knowledge and are interpretable;
  - GazeMAE provides gaze-specific learned embeddings;
  - MOMENT provides general-purpose time-series embeddings.
- Identify the evidence gap: these families are rarely compared with the same
  windows, labels, classifiers, and leakage-safe subject-generalization setup.
- State the research question:
  - Which eye-tracking representation is most useful for predicting the chosen
    TrustMe self-report state on unseen subjects?
- List contributions:
  - a controlled comparison across four representation families;
  - evaluation with a central LOSO-normalized protocol;
  - comparison under both Random Forest and MLP classifiers;
  - descriptive and low-dimensional analyses of data and representations;
  - practical guidance for representation selection.

## 2. Related work

- Eye tracking for user-state, engagement, cognitive-load, and affect
  prediction.
- Classical gaze and pupil feature engineering.
- Representation learning for gaze, including GazeMAE.
- General-purpose time-series foundation models, including MOMENT.
- Evaluation pitfalls in subject-dependent physiological and behavioral data:
  leakage, normalization, and overly optimistic random splits.
- Position this study as an empirical representation comparison rather than a
  new representation-learning method.

## 3. Dataset and preprocessing

- Introduce the TrustMe study and Tobii recording context.
- Report participants, sessions, recording duration, sampling characteristics,
  and exclusions once finalized.
- Describe window construction:
  - window duration and overlap;
  - canonical `window_uid` used across artifacts;
  - temporal ordering within participants.
- Explain validity filtering and missing-value handling.
- Distinguish artifact generation from supervised evaluation:
  - representations were exported for valid labelled and unlabelled windows;
  - supervised experiments use labelled windows only.
- Define q5 and justify it as the provisional primary target.
- Explain target binarization based on training data and how the held-out
  subject is handled.
- Report the common window intersection used for a fair comparison.

## 4. Eye-tracking representations

- Present representations consistently in this order:
  1. raw;
  2. handcrafted features;
  3. GazeMAE embeddings;
  4. MOMENT embeddings.
- Raw:
  - channels retained;
  - sequence length and truncate/pad or resampling policy;
  - resulting dimensionality.
- Handcrafted features:
  - feature families such as gaze, fixation/saccade, blink, and pupil metrics;
  - structural-column removal, missingness filtering, and correlation filtering;
  - fitting of data-dependent filtering on training folds only.
- GazeMAE embeddings:
  - input gaze signals, pretrained checkpoint, and embedding dimensionality;
  - gaze-specific quality-control implications.
- MOMENT embeddings:
  - multichannel input, pretrained model, pooling, and embedding dimensionality;
  - quality-control implications and the expected one-window count difference
    from GazeMAE.

## 5. Experimental setup

### 5.1 Prediction models

- Majority/most-frequent classifier as the reference baseline.
- Random Forest with class balancing and fixed random seed.
- MLP architecture, optimizer, batch size, early stopping, and validation split.
- Cross representation × model comparisons:
  - raw + RF and raw + MLP;
  - handcrafted features + RF and handcrafted features + MLP;
  - GazeMAE embeddings + RF and GazeMAE embeddings + MLP;
  - MOMENT embeddings + RF and MOMENT embeddings + MLP.

### 5.2 Evaluation protocols

- Primary: leave-one-subject-out with fold-local normalization.
- Explicitly state that imputation, feature filtering, scaling, and label
  centering are learned using training subjects only.
- Optional secondary analysis: within-subject temporal split.
- Optional discussion baseline: previous-state persistence.
- Keep secondary protocols out of the headline comparison unless results add a
  clear substantive insight.

### 5.3 Metrics and statistical reporting

- Accuracy and delta over the majority baseline.
- Balanced accuracy and delta over the majority baseline.
- Macro F1.
- AUC, with one-class folds recorded as undefined rather than silently
  discarded.
- Subject-macro summaries as the primary aggregation.
- Report variability across held-out subjects and, if feasible, paired
  uncertainty intervals or tests over subjects.
- Normalize confusion-matrix rows and fix the color scale to `[0, 1]`.

## 6. Data analysis and visualization

- Class-count distribution for q5.
- Subject-level class proportions to expose imbalance and heterogeneity.
- Pupil-size distributions by class; use subject stratification only if the
  figure remains legible.
- Gaze-coordinate density map.
- Concise missingness or validity summary.
- Representation projections:
  - run PCA, t-SNE, and UMAP with a fixed seed and common sample;
  - color separately by q5 class and participant;
  - select the clearest method for the paper and move alternatives to an
    appendix or repository;
  - describe projections as illustrative, not proof of separability.

## 7. Results

### 7.1 Main representation comparison

- Main table columns:
  - representation;
  - model;
  - accuracy and delta from majority;
  - balanced accuracy and delta from majority;
  - macro F1;
  - AUC.
- Order majority first when shown, followed by RF and MLP.
- Order representations as raw, handcrafted features, GazeMAE embeddings, and
  MOMENT embeddings.
- Emphasize unseen-subject performance and uncertainty, not only the best mean.

### 7.2 Subject-level behavior

- Show per-subject performance or distributions for selected comparisons.
- Include row-normalized confusion matrices for the most informative models.
- Discuss subjects or folds with one-class targets and their metric treatment.

### 7.3 Secondary protocol results

- Include the within-subject temporal split only if it clarifies the difference
  between personalization and subject generalization.
- Compare with persistence only if it is substantively informative.

## 8. Discussion

- Answer which representation worked best under LOSO-normalized evaluation.
- Separate representation effects from classifier effects.
- Interpret whether learned embeddings improve over handcrafted features or raw
  windows and under what conditions.
- Discuss gaze-specific versus general-purpose pretraining.
- Relate subject-colored projections to cross-subject distribution shift,
  without overinterpreting dimensionality reduction.
- Practical guidance:
  - performance versus preprocessing complexity;
  - computational and storage costs;
  - interpretability and deployability.
- Limitations:
  - sample and label size;
  - self-report noise and provisional q5 choice;
  - binary target construction;
  - dependence on pretrained checkpoints and QC rules;
  - one dataset and one windowing setup;
  - projections are exploratory.

## 9. Conclusion

- Restate the controlled subject-generalization comparison.
- Summarize the strongest empirical finding after results are available.
- Give a concise recommendation for practitioners choosing representations.
- Suggest future multi-dataset, semi-supervised, and personalization studies.

## Reproducibility and appendix material

- Exact YAML configuration and software versions.
- Per-subject sample and class counts.
- Representation dimensionalities and common-intersection counts.
- Full per-fold metric tables.
- Alternative PCA, t-SNE, and UMAP figures.
- Secondary temporal and persistence results if excluded from the main text.
- Hyperparameter details and runtime/hardware information.

## Decisions to revisit after initial results

- Confirm q5 as the final and only primary target.
- Decide how prominently to report AUC given one-class folds.
- Choose PCA, t-SNE, or UMAP for the main representation figure.
- Decide whether within-subject temporal evaluation belongs in the main paper
  or appendix.
- Decide whether persistence is a reported baseline or discussion-only check.
- Confirm the final paper length and formatting against the IS 2026 SKUI call.

