"""Typed configuration structures for Zoja-style TrustMe protocol runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...common.types import FeaturesConfig, ModelsConfig, RawConfig
from ..types import TrustMeEmbeddingVariantConfig


@dataclass
class ZojaPathsConfig:
    """Filesystem locations for protocol runs."""

    repo_root: Path
    processed_root: Path | None
    features_csv: Path | None
    results_root: Path
    subject_tree_root: Path | None = None
    subject_export_dir: str = "tobii"
    subjects: list[str] | None = None


@dataclass
class ZojaTargetConfig:
    """One target definition used by protocol runs."""

    name: str
    mode: str
    column: str | None = None
    columns: list[str] | None = None


@dataclass
class ZojaLabelRuleConfig:
    """Labeling behavior used to derive binary targets in each fold."""

    mode: str = "centered_subject_train_gt0"
    unseen_subject_center: str = "global_train_mean"
    threshold: float = 3.0


@dataclass
class ZojaProtocolEnabledConfig:
    """Protocol enable/disable switches."""

    loso: bool = True
    loso_normalized: bool = True
    within_subject_temporal_split: bool = True
    rolling_window: bool = True
    hybrid_loso_adaptation: bool = True
    persistence_baseline: bool = True
    hmm_rolling: bool = True


@dataclass
class ZojaProtocolParamsConfig:
    """Numeric parameters for protocol variants."""

    temporal_split_ratio: float = 0.7
    rolling_train_size: int = 120
    rolling_test_size: int = 30
    rolling_step_size: int = 15
    hybrid_calibration_size: int = 130
    hybrid_test_size: int = 30
    hybrid_step_size: int = 15
    persistence_calibration_size: int = 300
    persistence_test_size: int = 90
    persistence_step_size: int = 45
    hmm_train_size: int = 300
    hmm_test_size: int = 90
    hmm_step_size: int = 45
    hmm_n_states: int = 2
    min_train_samples: int = 10
    min_test_samples: int = 5


@dataclass
class ZojaProtocolsConfig:
    """Protocol flags and parameters."""

    enabled: ZojaProtocolEnabledConfig
    params: ZojaProtocolParamsConfig


@dataclass
class ZojaEvaluationConfig:
    """Shared evaluation behavior used across all protocols."""

    one_class_policy: str = "keep_with_nan"
    metrics: list[str] | None = None


@dataclass
class ZojaPlottingConfig:
    """Figure generation options."""

    enabled: bool = True
    confusion_top_k: int = 9


@dataclass
class ZojaExperimentConfig:
    """Top-level protocol run configuration."""

    seed: int
    run_id: str | None
    paths: ZojaPathsConfig
    targets: list[ZojaTargetConfig]
    representations: list[str]
    embedding_variants: list[TrustMeEmbeddingVariantConfig]
    allow_missing_features: bool
    model: str
    classifiers: list[str]
    label_rule: ZojaLabelRuleConfig
    protocols: ZojaProtocolsConfig
    evaluation: ZojaEvaluationConfig
    plotting: ZojaPlottingConfig
    raw: RawConfig
    features: FeaturesConfig
    models: ModelsConfig
    use_standard_scaler: bool
    paper_outputs_only: bool = False
