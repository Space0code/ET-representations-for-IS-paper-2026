"""Configuration loading for TrustMe representation-comparison runs."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import yaml

from ..common.types import (
    CVConfig,
    FeatureStatisticalFilterConfig,
    FeatureStructuralDropConfig,
    FeaturesConfig,
    MLPConfig,
    ModelsConfig,
    RawConfig,
)
from .types import (
    TrustMeEmbeddingVariantConfig,
    TrustMeExperimentConfig,
    TrustMeFusionComboConfig,
    TrustMePathsConfig,
    TrustMeTaskConfig,
)


def _resolve_path(repo_root: Path, maybe_relative: str) -> Path:
    """Expand user/environment tokens and resolve relative paths."""

    path = Path(os.path.expandvars(os.path.expanduser(maybe_relative)))
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _sanitize_variant_tag(raw: str) -> str:
    """Convert raw tag text into a filesystem-safe identifier."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError(f"Invalid embedding variant tag {raw!r} after sanitization.")
    return cleaned


def _as_label_string(value: Any) -> str:
    """Convert class-order value into a stable string label."""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _parse_features_cfg(cfg: dict[str, Any]) -> FeaturesConfig:
    """Parse feature filtering config with backward-compatible defaults."""

    default_cfg = {
        "structural_drop": {
            "metadata_columns": [
                "Unnamed: 0",
                "Subject",
                "subject",
                "Filename",
                "source_file",
                "Path",
                "Label",
                "window_uid",
                "window_id",
                "window_id_str",
                "Time_start",
                "Time_end",
                "Index_start",
                "Index_end",
                "Win len",
                "Overlap Percent",
                "Fs",
            ]
        },
        "statistical_filter": {
            "max_nan_fraction": 0.60,
            "correlation_threshold": 0.80,
        },
    }
    feature_cfg = cfg.get("features", default_cfg)
    structural = feature_cfg.get("structural_drop", default_cfg["structural_drop"])
    statistical = feature_cfg.get("statistical_filter", default_cfg["statistical_filter"])

    return FeaturesConfig(
        structural_drop=FeatureStructuralDropConfig(
            metadata_columns=[str(value) for value in structural["metadata_columns"]],
        ),
        statistical_filter=FeatureStatisticalFilterConfig(
            max_nan_fraction=float(statistical["max_nan_fraction"]),
            correlation_threshold=float(statistical["correlation_threshold"]),
        ),
    )


def _parse_embedding_variants(
    *,
    cfg: dict[str, Any],
    repo_root: Path,
    representations: list[str],
) -> list[TrustMeEmbeddingVariantConfig]:
    """Parse TrustMe embedding variant configuration."""

    experiment_cfg = cfg.get("experiment", {})
    raw_variants = experiment_cfg.get("embedding_variants")

    if "embeddings" not in representations:
        return []

    if raw_variants is None:
        raise ValueError(
            "Missing required 'experiment.embedding_variants' while "
            "'embeddings' is requested in experiment.representations."
        )

    if not isinstance(raw_variants, dict) or not raw_variants:
        raise ValueError("Expected non-empty mapping in 'experiment.embedding_variants'.")

    parsed: list[TrustMeEmbeddingVariantConfig] = []
    seen_tags: set[str] = set()
    for raw_tag, raw_entry in raw_variants.items():
        if not isinstance(raw_tag, str):
            raise ValueError("Embedding variant tags must be strings.")
        tag = _sanitize_variant_tag(raw_tag)
        if tag in seen_tags:
            raise ValueError(f"Duplicate embedding variant tag '{tag}'.")
        seen_tags.add(tag)

        if isinstance(raw_entry, str):
            variant = TrustMeEmbeddingVariantConfig(
                tag=tag,
                kind="npz",
                path=_resolve_path(repo_root, raw_entry),
                metadata_path=None,
                id_column=None,
                feature_prefixes=None,
            )
            parsed.append(variant)
            continue

        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"Variant '{raw_tag}' must be either a path string or a mapping."
            )

        kind = str(raw_entry.get("kind", "npz"))
        raw_path = raw_entry.get("path", raw_entry.get("embeddings_path"))
        if not isinstance(raw_path, str):
            raise ValueError(
                f"Variant '{raw_tag}' must define 'path' (or 'embeddings_path') as a string."
            )

        metadata_path_raw = raw_entry.get("metadata_path")
        metadata_path = None
        if metadata_path_raw is not None:
            if not isinstance(metadata_path_raw, str):
                raise ValueError(f"Variant '{raw_tag}' has non-string 'metadata_path'.")
            metadata_path = _resolve_path(repo_root, metadata_path_raw)

        id_column_raw = raw_entry.get("id_column")
        if id_column_raw is not None and not isinstance(id_column_raw, str):
            raise ValueError(f"Variant '{raw_tag}' has non-string 'id_column'.")

        feature_prefixes_raw = raw_entry.get("feature_prefixes")
        feature_prefixes = None
        if feature_prefixes_raw is not None:
            if not isinstance(feature_prefixes_raw, list) or not all(
                isinstance(value, str) for value in feature_prefixes_raw
            ):
                raise ValueError(
                    f"Variant '{raw_tag}' has invalid 'feature_prefixes' (expected list[str])."
                )
            feature_prefixes = [str(value) for value in feature_prefixes_raw]

        parsed.append(
            TrustMeEmbeddingVariantConfig(
                tag=tag,
                kind=kind,
                path=_resolve_path(repo_root, raw_path),
                metadata_path=metadata_path,
                id_column=id_column_raw,
                feature_prefixes=feature_prefixes,
            )
        )

    return parsed


def _parse_fusion_combos(raw_value: Any, field_name: str) -> list[TrustMeFusionComboConfig]:
    """Parse fusion combo list from config."""

    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise ValueError(f"experiment.fusion.{field_name} must be a list of mappings.")

    parsed: list[TrustMeFusionComboConfig] = []
    seen: set[str] = set()
    for idx, entry in enumerate(raw_value):
        if not isinstance(entry, dict):
            raise ValueError(
                f"experiment.fusion.{field_name}[{idx}] must be a mapping with 'name' and 'inputs'."
            )
        raw_name = entry.get("name")
        raw_inputs = entry.get("inputs")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"experiment.fusion.{field_name}[{idx}].name must be a non-empty string.")
        if not isinstance(raw_inputs, list) or not all(isinstance(v, str) for v in raw_inputs):
            raise ValueError(f"experiment.fusion.{field_name}[{idx}].inputs must be list[str].")
        if len(raw_inputs) < 2:
            raise ValueError(f"experiment.fusion.{field_name}[{idx}] needs at least 2 inputs.")

        name = _sanitize_variant_tag(raw_name)
        if name in seen:
            raise ValueError(f"Duplicate fusion combo name '{name}' in experiment.fusion.{field_name}.")
        seen.add(name)

        inputs = [str(value).strip() for value in raw_inputs]
        if any(not value for value in inputs):
            raise ValueError(f"experiment.fusion.{field_name}[{idx}] has empty input names.")
        if len(set(inputs)) != len(inputs):
            raise ValueError(
                f"experiment.fusion.{field_name}[{idx}] has duplicate input representation names."
            )
        parsed.append(TrustMeFusionComboConfig(name=name, inputs=inputs))
    return parsed


def load_trustme_experiment_config(config_path: str | Path) -> TrustMeExperimentConfig:
    """Load and validate TrustMe comparison YAML config."""

    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    repo_root = _resolve_path(config_path.parent, cfg["paths"]["repo_root"])

    task_cfg = cfg.get("task", {})
    subjects = task_cfg.get("subjects")
    if subjects is not None:
        subjects = [str(value) for value in subjects]

    class_order = task_cfg.get("class_order")
    if class_order is not None:
        class_order = [_as_label_string(value) for value in class_order]

    features_csv_raw = cfg["paths"].get("features_csv")
    if features_csv_raw is not None and not isinstance(features_csv_raw, str):
        raise ValueError("'paths.features_csv' must be a string path or null.")

    paths = TrustMePathsConfig(
        repo_root=repo_root,
        processed_root=_resolve_path(repo_root, cfg["paths"]["processed_root"]),
        features_csv=_resolve_path(repo_root, features_csv_raw) if features_csv_raw is not None else None,
        results_root=_resolve_path(repo_root, cfg["paths"]["results_root"]),
    )

    cv_cfg = cfg["cv"]
    group_by = str(cv_cfg.get("group_by", "subject")).strip().lower()
    if group_by not in {"subject", "file"}:
        raise ValueError("cv.group_by must be one of: ['subject', 'file'].")
    cv = CVConfig(
        n_splits=int(cv_cfg["n_splits"]),
        group_by=group_by,
    )

    raw_cfg = cfg["raw"]
    raw_backend = str(raw_cfg.get("backend", "parquet")).strip().lower()
    if raw_backend not in {"parquet", "csv"}:
        raise ValueError("raw.backend must be one of: ['parquet', 'csv'].")

    raw_length_mode = str(raw_cfg.get("length_mode", "resample")).strip().lower()
    if raw_length_mode not in {"resample", "truncate_pad"}:
        raise ValueError("raw.length_mode must be one of: ['resample', 'truncate_pad'].")

    raw_csv_path_raw = raw_cfg.get("csv_path")
    raw_csv_path = None
    if raw_csv_path_raw is not None:
        if not isinstance(raw_csv_path_raw, str):
            raise ValueError("'raw.csv_path' must be a string path or null.")
        raw_csv_path = _resolve_path(repo_root, raw_csv_path_raw)

    flatten_length_raw = raw_cfg.get("flatten_length")
    flatten_length = int(flatten_length_raw) if flatten_length_raw is not None else None
    if raw_length_mode == "truncate_pad" and flatten_length is None:
        raise ValueError("raw.flatten_length is required when raw.length_mode='truncate_pad'.")
    if raw_length_mode == "truncate_pad" and flatten_length is not None and flatten_length <= 0:
        raise ValueError("raw.flatten_length must be positive when provided.")
    if raw_backend == "csv" and raw_csv_path is None:
        raise ValueError("raw.csv_path is required when raw.backend='csv'.")

    raw = RawConfig(
        channels=[str(value) for value in raw_cfg["channels"]],
        resample_length=int(raw_cfg["resample_length"]),
        validity_gaze_columns=[str(value) for value in raw_cfg["validity_gaze_columns"]],
        validity_pupil_columns=[str(value) for value in raw_cfg["validity_pupil_columns"]],
        validity_valid_value=int(raw_cfg["validity_valid_value"]),
        backend=raw_backend,
        csv_path=raw_csv_path,
        length_mode=raw_length_mode,
        flatten_length=flatten_length,
    )

    features = _parse_features_cfg(cfg)

    representations = [str(value) for value in cfg.get("experiment", {}).get("representations", ["raw"])]
    valid_representations = {"raw", "raw_rows", "embeddings", "features", "label_baseline"}
    unknown = sorted(set(representations) - valid_representations)
    if unknown:
        raise ValueError(
            f"Unsupported representations requested: {unknown}. "
            f"Supported={sorted(valid_representations)}"
        )
    if not representations:
        raise ValueError("experiment.representations must not be empty.")

    embedding_variants = _parse_embedding_variants(
        cfg=cfg,
        repo_root=repo_root,
        representations=representations,
    )
    common_embedding_intersection = bool(
        cfg.get("experiment", {}).get("common_embedding_intersection", False)
    )
    label_baseline_mode = str(
        cfg.get("experiment", {}).get("label_baseline_mode", "mean")
    ).strip().lower()
    fusion_cfg = cfg.get("experiment", {}).get("fusion", {})
    if fusion_cfg is None:
        fusion_cfg = {}
    if not isinstance(fusion_cfg, dict):
        raise ValueError("experiment.fusion must be a mapping when provided.")
    early_fusion_combos = _parse_fusion_combos(
        fusion_cfg.get("early_combos"),
        field_name="early_combos",
    )
    late_fusion_combos = _parse_fusion_combos(
        fusion_cfg.get("late_combos"),
        field_name="late_combos",
    )
    late_fusion_method = str(fusion_cfg.get("late_method", "soft_vote")).strip().lower()
    allow_missing_features = bool(cfg.get("experiment", {}).get("allow_missing_features", False))

    if label_baseline_mode not in {"mean", "most_frequent"}:
        raise ValueError(
            "experiment.label_baseline_mode must be one of: ['mean', 'most_frequent']."
        )
    if late_fusion_method != "soft_vote":
        raise ValueError("experiment.fusion.late_method currently supports only 'soft_vote'.")

    if "features" in representations and paths.features_csv is None and not allow_missing_features:
        raise ValueError(
            "Representation 'features' requested but 'paths.features_csv' is null and "
            "'experiment.allow_missing_features' is false."
        )

    plotting_cfg = cfg.get("plotting", {})
    if plotting_cfg is None:
        plotting_cfg = {}
    if not isinstance(plotting_cfg, dict):
        raise ValueError("plotting must be a mapping when provided.")
    confusion_top_k = int(plotting_cfg.get("confusion_top_k", 9))
    if confusion_top_k <= 0:
        raise ValueError("plotting.confusion_top_k must be positive.")

    models = ModelsConfig(
        majority=dict(cfg["models"]["majority"]),
        logistic_regression=dict(cfg["models"]["logistic_regression"]),
        lgbm=dict(cfg["models"]["lgbm"]),
        svm_rbf=dict(cfg["models"]["svm_rbf"]),
        mlp=MLPConfig(
            hidden_dims=[int(value) for value in cfg["models"]["mlp"]["hidden_dims"]],
            activation=str(cfg["models"]["mlp"]["activation"]),
            use_layernorm=bool(cfg["models"]["mlp"]["use_layernorm"]),
            optimizer=str(cfg["models"]["mlp"]["optimizer"]),
            lr=float(cfg["models"]["mlp"]["lr"]),
            weight_decay=float(cfg["models"]["mlp"]["weight_decay"]),
            batch_size=int(cfg["models"]["mlp"]["batch_size"]),
            max_epochs=int(cfg["models"]["mlp"]["max_epochs"]),
            patience=int(cfg["models"]["mlp"]["patience"]),
            seed=int(cfg["models"]["mlp"]["seed"]),
        ),
    )

    task = TrustMeTaskConfig(
        task_name=str(task_cfg.get("task_name", "sleep_feedback")),
        target_column=str(task_cfg.get("target_column", "sleep_feedback")),
        subjects=subjects,
        class_order=class_order,
        max_segments_per_subject=(
            int(task_cfg["max_segments_per_subject"])
            if task_cfg.get("max_segments_per_subject") is not None
            else None
        ),
        target_mode=str(task_cfg.get("target_mode", "multiclass")).strip().lower(),
        threshold_mode=str(task_cfg.get("threshold_mode", "median")).strip().lower(),
        threshold_value=(
            float(task_cfg["threshold_value"])
            if task_cfg.get("threshold_value") is not None
            else None
        ),
        threshold_scope=str(task_cfg.get("threshold_scope", "outer_train")).strip().lower(),
        positive_rule=str(task_cfg.get("positive_rule", "gt")).strip().lower(),
    )

    if task.target_mode not in {"multiclass", "binary_threshold"}:
        raise ValueError(
            "task.target_mode must be one of: ['multiclass', 'binary_threshold']."
        )
    if task.threshold_mode not in {"median", "mean", "value"}:
        raise ValueError(
            "task.threshold_mode must be one of: ['median', 'mean', 'value']."
        )
    if task.threshold_mode == "value" and task.threshold_value is None:
        raise ValueError(
            "task.threshold_value is required when task.threshold_mode='value'."
        )
    if task.threshold_scope != "outer_train":
        raise ValueError("task.threshold_scope currently supports only 'outer_train'.")
    if task.positive_rule != "gt":
        raise ValueError("task.positive_rule currently supports only 'gt'.")

    return TrustMeExperimentConfig(
        seed=int(cfg["seed"]),
        run_id=cfg.get("run_id"),
        paths=paths,
        cv=cv,
        raw=raw,
        features=features,
        use_standard_scaler=bool(cfg["scaling"]["use_standard_scaler"]),
        representations=representations,
        embedding_variants=embedding_variants,
        common_embedding_intersection=common_embedding_intersection,
        allow_missing_features=allow_missing_features,
        classifiers=[str(value) for value in cfg["experiment"]["classifiers"]],
        models=models,
        metrics=[str(value) for value in cfg["metrics"]["names"]],
        task=task,
        label_baseline_mode=label_baseline_mode,
        early_fusion_combos=early_fusion_combos,
        late_fusion_combos=late_fusion_combos,
        late_fusion_method=late_fusion_method,
        confusion_top_k=confusion_top_k,
    )
