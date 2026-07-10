"""Typed data structures for the TrustME comparison pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PathsConfig:
    """Filesystem paths used by the experiment."""

    repo_root: Path
    all_segments_csv: Path
    raw_root: Path
    features_csv: Path
    results_root: Path


@dataclass
class EmbeddingVariantConfig:
    """One embedding variant used in an experiment."""

    tag: str
    npz_path: Path


@dataclass
class CVConfig:
    """Cross-validation configuration."""

    n_splits: int
    group_by: str = "subject"


@dataclass
class RawConfig:
    """Raw representation extraction configuration."""

    channels: list[str]
    resample_length: int
    # Validity flag columns from Tobii CSV (value == validity_valid_value means valid).
    validity_gaze_columns: list[str]
    validity_pupil_columns: list[str]
    validity_valid_value: int
    backend: str = "parquet"
    csv_path: Path | None = None
    length_mode: str = "resample"
    flatten_length: int | None = None


@dataclass
class LabelsConfig:
    """Label mapping and class order configuration."""

    drop_labels: list[str]
    rest_labels: list[str]
    attention_labels: list[str]
    memory_labels: list[str]
    visual_labels: list[str]
    class_order: dict[str, list[str]]


@dataclass
class FeatureStructuralDropConfig:
    """Structural feature drop configuration.

    Only metadata (non-feature) columns are removed here; all numeric feature
    columns are retained and evaluated statistically per fold.
    """

    metadata_columns: list[str]


@dataclass
class FeatureStatisticalFilterConfig:
    """Fold-specific statistical feature filtering configuration."""

    max_nan_fraction: float
    correlation_threshold: float


@dataclass
class FeaturesConfig:
    """Handcrafted feature representation configuration.

    NaN values in the retained feature columns are imputed with the per-column
    training-fold median (fit on outer-train only to avoid data leakage).
    """

    structural_drop: FeatureStructuralDropConfig
    statistical_filter: FeatureStatisticalFilterConfig


@dataclass
class MLPConfig:
    """PyTorch MLP hyperparameters."""

    hidden_dims: list[int]
    activation: str
    use_layernorm: bool
    optimizer: str
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int


@dataclass
class ModelsConfig:
    """Model-specific configuration dictionary."""

    majority: dict[str, Any]
    logistic_regression: dict[str, Any]
    lgbm: dict[str, Any]
    svm_rbf: dict[str, Any]
    mlp: MLPConfig
    random_forest: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    seed: int
    run_id: str | None
    paths: PathsConfig
    cv: CVConfig
    raw: RawConfig
    labels: LabelsConfig
    features: FeaturesConfig
    use_standard_scaler: bool
    representations: list[str]
    labelling_schemes: list[str]
    embedding_sets: dict[str, list[EmbeddingVariantConfig]]
    classifiers: list[str]
    models: ModelsConfig
    metrics: list[str]


@dataclass
class RepresentationDataset:
    """Data for one representation aligned to canonical segment IDs."""

    name: str
    X: Any
    segment_ids: np.ndarray
    subjects: np.ndarray
    source_labels: np.ndarray


@dataclass
class FoldSplit:
    """Outer test split plus inner train/validation split for MLP."""

    fold_index: int
    outer_train_idx: np.ndarray
    outer_test_idx: np.ndarray
    mlp_train_idx: np.ndarray
    mlp_val_idx: np.ndarray
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]
    train_segment_ids: list[str]
    val_segment_ids: list[str]
    test_segment_ids: list[str]
    train_groups: list[str] = field(default_factory=list)
    val_groups: list[str] = field(default_factory=list)
    test_groups: list[str] = field(default_factory=list)


@dataclass
class FoldResult:
    """Metrics and artifact paths for one fold."""

    fold_index: int
    accuracy: float
    macro_f1: float
    precision: float
    recall: float
    auc: float
    auc_note: str
    fold_dir: Path
    artifact_paths: dict[str, Path] = field(default_factory=dict)
