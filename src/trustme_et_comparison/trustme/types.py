"""Typed configuration structures for TrustMe comparison runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..common.types import CVConfig, FeaturesConfig, ModelsConfig, RawConfig


@dataclass
class TrustMePathsConfig:
    """Filesystem paths used by TrustMe comparison."""

    repo_root: Path
    processed_root: Path
    features_csv: Path | None
    results_root: Path


@dataclass
class TrustMeTaskConfig:
    """TrustMe task definition."""

    task_name: str
    target_column: str
    subjects: list[str] | None
    class_order: list[str] | None
    max_segments_per_subject: int | None
    target_mode: str = "multiclass"
    threshold_mode: str = "median"
    threshold_value: float | None = None
    threshold_scope: str = "outer_train"
    positive_rule: str = "gt"


@dataclass
class TrustMeEmbeddingVariantConfig:
    """One TrustMe embedding variant definition."""

    tag: str
    kind: str
    path: Path
    metadata_path: Path | None
    id_column: str | None
    feature_prefixes: list[str] | None


@dataclass
class TrustMeFusionComboConfig:
    """One fusion combination definition."""

    name: str
    inputs: list[str]


@dataclass
class TrustMeExperimentConfig:
    """Top-level TrustMe comparison configuration."""

    seed: int
    run_id: str | None
    paths: TrustMePathsConfig
    cv: CVConfig
    raw: RawConfig
    features: FeaturesConfig
    use_standard_scaler: bool
    representations: list[str]
    embedding_variants: list[TrustMeEmbeddingVariantConfig]
    allow_missing_features: bool
    classifiers: list[str]
    models: ModelsConfig
    metrics: list[str]
    task: TrustMeTaskConfig
    common_embedding_intersection: bool = False
    label_baseline_mode: str = "mean"
    early_fusion_combos: list[TrustMeFusionComboConfig] | None = None
    late_fusion_combos: list[TrustMeFusionComboConfig] | None = None
    late_fusion_method: str = "soft_vote"
    confusion_top_k: int = 9
