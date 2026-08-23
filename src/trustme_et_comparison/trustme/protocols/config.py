"""Configuration loading for TrustMe Tobii ET protocol runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ...common.types import MLPConfig, ModelsConfig, RawConfig
from ..config import _parse_embedding_variants, _parse_features_cfg, _resolve_path
from .types import (
    ZojaEvaluationConfig,
    ZojaExperimentConfig,
    ZojaLabelRuleConfig,
    ZojaPathsConfig,
    ZojaPlottingConfig,
    ZojaProtocolEnabledConfig,
    ZojaProtocolParamsConfig,
    ZojaProtocolsConfig,
    ZojaTargetConfig,
)


def _parse_targets(raw_targets: Any) -> list[ZojaTargetConfig]:
    """Parse target definitions from YAML.

    Expected shape:
      - name: q4
        mode: single_column
        column: "4"
      - name: engagement_mean_q4_q5
        mode: mean_columns
        columns: ["4", "5"]
    """

    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("targets must be a non-empty list.")

    parsed: list[ZojaTargetConfig] = []
    names: set[str] = set()
    for idx, item in enumerate(raw_targets):
        if not isinstance(item, dict):
            raise ValueError(f"targets[{idx}] must be a mapping.")

        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"targets[{idx}].name must be a non-empty string.")
        if name in names:
            raise ValueError(f"Duplicate target name: {name}")
        names.add(name)

        mode = str(item.get("mode", "single_column")).strip().lower()
        if mode not in {"single_column", "mean_columns"}:
            raise ValueError(
                f"targets[{idx}].mode must be one of ['single_column', 'mean_columns'], got {mode!r}."
            )

        column: str | None = None
        columns: list[str] | None = None
        if mode == "single_column":
            raw_col = item.get("column")
            if raw_col is None:
                raise ValueError(f"targets[{idx}].column is required for mode='single_column'.")
            column = str(raw_col)
        else:
            raw_cols = item.get("columns")
            if not isinstance(raw_cols, list) or len(raw_cols) < 2:
                raise ValueError(
                    f"targets[{idx}].columns must be list[str] with length>=2 for mode='mean_columns'."
                )
            columns = [str(value) for value in raw_cols]

        parsed.append(
            ZojaTargetConfig(
                name=name,
                mode=mode,
                column=column,
                columns=columns,
            )
        )

    return parsed


def _parse_protocols(cfg: dict[str, Any]) -> ZojaProtocolsConfig:
    """Parse protocol flags and numeric settings."""

    raw_protocols = cfg.get("protocols", {}) or {}
    if not isinstance(raw_protocols, dict):
        raise ValueError("protocols must be a mapping when provided.")

    raw_enabled = raw_protocols.get("enabled", {}) or {}
    if not isinstance(raw_enabled, dict):
        raise ValueError("protocols.enabled must be a mapping.")

    enabled = ZojaProtocolEnabledConfig(
        loso=bool(raw_enabled.get("loso", True)),
        loso_normalized=bool(raw_enabled.get("loso_normalized", True)),
        within_subject_temporal_split=bool(raw_enabled.get("within_subject_temporal_split", True)),
        rolling_window=bool(raw_enabled.get("rolling_window", True)),
        hybrid_loso_adaptation=bool(raw_enabled.get("hybrid_loso_adaptation", True)),
        persistence_baseline=bool(raw_enabled.get("persistence_baseline", True)),
        hmm_rolling=bool(raw_enabled.get("hmm_rolling", True)),
    )

    raw_params = raw_protocols.get("params", {}) or {}
    if not isinstance(raw_params, dict):
        raise ValueError("protocols.params must be a mapping.")

    params = ZojaProtocolParamsConfig(
        temporal_split_ratio=float(raw_params.get("temporal_split_ratio", 0.7)),
        rolling_train_size=int(raw_params.get("rolling_train_size", 120)),
        rolling_test_size=int(raw_params.get("rolling_test_size", 30)),
        rolling_step_size=int(raw_params.get("rolling_step_size", 15)),
        hybrid_calibration_size=int(raw_params.get("hybrid_calibration_size", 130)),
        hybrid_test_size=int(raw_params.get("hybrid_test_size", 30)),
        hybrid_step_size=int(raw_params.get("hybrid_step_size", 15)),
        persistence_calibration_size=int(raw_params.get("persistence_calibration_size", 300)),
        persistence_test_size=int(raw_params.get("persistence_test_size", 90)),
        persistence_step_size=int(raw_params.get("persistence_step_size", 45)),
        hmm_train_size=int(raw_params.get("hmm_train_size", 300)),
        hmm_test_size=int(raw_params.get("hmm_test_size", 90)),
        hmm_step_size=int(raw_params.get("hmm_step_size", 45)),
        hmm_n_states=int(raw_params.get("hmm_n_states", 2)),
        min_train_samples=int(raw_params.get("min_train_samples", 10)),
        min_test_samples=int(raw_params.get("min_test_samples", 5)),
    )

    if not (0.0 < params.temporal_split_ratio < 1.0):
        raise ValueError("protocols.params.temporal_split_ratio must be in (0, 1).")

    return ZojaProtocolsConfig(enabled=enabled, params=params)


def _parse_models(cfg: dict[str, Any]) -> ModelsConfig:
    """Parse model configuration."""

    return ModelsConfig(
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
        random_forest=dict(
            (cfg.get("models", {}) or {}).get(
                "random_forest",
                {
                    "n_estimators": 300,
                    "class_weight": "balanced",
                    "random_state": int(cfg.get("seed", 42)),
                    "n_jobs": -1,
                },
            )
        ),
    )


def _parse_raw(cfg: dict[str, Any], repo_root: Path) -> RawConfig:
    """Parse raw representation config with validation."""

    raw_cfg = cfg["raw"]
    raw_backend = str(raw_cfg.get("backend", "parquet")).strip().lower()
    if raw_backend not in {"parquet", "csv", "subject_tree"}:
        raise ValueError("raw.backend must be one of: ['parquet', 'csv', 'subject_tree'].")

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
    if raw_backend == "csv" and raw_csv_path is None:
        raise ValueError("raw.csv_path is required when raw.backend='csv'.")

    return RawConfig(
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


def load_zoja_protocol_config(config_path: str | Path) -> ZojaExperimentConfig:
    """Load and validate TrustMe Tobii protocol config YAML."""

    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a YAML mapping.")

    repo_root = _resolve_path(config_path.parent, cfg["paths"]["repo_root"])

    paths_cfg = cfg["paths"]
    features_csv_raw = paths_cfg.get("features_csv")
    if features_csv_raw is not None and not isinstance(features_csv_raw, str):
        raise ValueError("paths.features_csv must be a string path or null.")

    processed_root_raw = paths_cfg.get("processed_root")
    if processed_root_raw is not None and not isinstance(processed_root_raw, str):
        raise ValueError("paths.processed_root must be a string path or null.")

    subject_tree_root_raw = paths_cfg.get("subject_tree_root")
    if subject_tree_root_raw is not None and not isinstance(subject_tree_root_raw, str):
        raise ValueError("paths.subject_tree_root must be a string path or null.")
    subject_export_dir = str(paths_cfg.get("subject_export_dir", "tobii"))
    if Path(subject_export_dir).name != subject_export_dir:
        raise ValueError("paths.subject_export_dir must be a directory name, not a path.")

    raw_subjects = paths_cfg.get("subjects")
    subjects = None
    if raw_subjects is not None:
        if not isinstance(raw_subjects, list) or not raw_subjects:
            raise ValueError("paths.subjects must be a non-empty list when provided.")
        subjects = [str(value) for value in raw_subjects]
        if len(subjects) != len(set(subjects)):
            raise ValueError("paths.subjects contains duplicate subject names.")

    paths = ZojaPathsConfig(
        repo_root=repo_root,
        processed_root=(
            _resolve_path(repo_root, processed_root_raw)
            if processed_root_raw is not None
            else None
        ),
        features_csv=_resolve_path(repo_root, features_csv_raw) if features_csv_raw is not None else None,
        results_root=_resolve_path(repo_root, paths_cfg["results_root"]),
        subject_tree_root=(
            _resolve_path(repo_root, subject_tree_root_raw)
            if subject_tree_root_raw is not None
            else None
        ),
        subject_export_dir=subject_export_dir,
        subjects=subjects,
    )
    if paths.subject_tree_root is None and paths.processed_root is None:
        raise ValueError("One of paths.subject_tree_root or paths.processed_root is required.")
    if paths.subject_tree_root is not None and paths.subjects is None:
        raise ValueError(
            "paths.subjects is required with paths.subject_tree_root so the experiment cohort is frozen."
        )

    representations = [str(value) for value in cfg.get("experiment", {}).get("representations", ["raw"]) ]
    valid_representations = {"raw", "features", "embeddings"}
    unknown = sorted(set(representations) - valid_representations)
    if unknown:
        raise ValueError(f"Unsupported representations requested: {unknown}.")

    embedding_variants = (
        []
        if paths.subject_tree_root is not None
        else _parse_embedding_variants(
            cfg=cfg,
            repo_root=repo_root,
            representations=representations,
        )
    )

    targets = _parse_targets(cfg.get("targets"))

    raw_eval = cfg.get("evaluation", {}) or {}
    if not isinstance(raw_eval, dict):
        raise ValueError("evaluation must be a mapping.")
    evaluation = ZojaEvaluationConfig(
        one_class_policy=str(raw_eval.get("one_class_policy", "keep_with_nan")).strip().lower(),
        metrics=[str(value) for value in raw_eval.get("metrics", ["accuracy", "balanced_accuracy", "macro_f1", "auc", "precision", "recall"])],
    )
    if evaluation.one_class_policy not in {"keep_with_nan", "skip"}:
        raise ValueError("evaluation.one_class_policy must be one of ['keep_with_nan', 'skip'].")

    raw_plot = cfg.get("plotting", {}) or {}
    if not isinstance(raw_plot, dict):
        raise ValueError("plotting must be a mapping.")
    plotting = ZojaPlottingConfig(
        enabled=bool(raw_plot.get("enabled", True)),
        confusion_top_k=int(raw_plot.get("confusion_top_k", 9)),
    )

    raw_label_rule = cfg.get("label_rule", {}) or {}
    if not isinstance(raw_label_rule, dict):
        raise ValueError("label_rule must be a mapping.")
    label_rule = ZojaLabelRuleConfig(
        mode=str(raw_label_rule.get("mode", "centered_subject_train_gt0")).strip().lower(),
        unseen_subject_center=str(raw_label_rule.get("unseen_subject_center", "global_train_mean")).strip().lower(),
        threshold=float(raw_label_rule.get("threshold", 3.0)),
    )
    if label_rule.mode not in {"centered_subject_train_gt0", "absolute_gt_threshold"}:
        raise ValueError(
            "label_rule.mode must be one of ['centered_subject_train_gt0', 'absolute_gt_threshold']."
        )
    if label_rule.mode == "centered_subject_train_gt0":
        if label_rule.unseen_subject_center not in {"global_train_mean"}:
            raise ValueError("label_rule.unseen_subject_center currently supports only 'global_train_mean'.")
    else:
        if label_rule.threshold < 0.0:
            raise ValueError("label_rule.threshold must be >= 0.0 for mode='absolute_gt_threshold'.")

    protocols = _parse_protocols(cfg)

    allow_missing_features = bool(cfg.get("experiment", {}).get("allow_missing_features", False))
    if (
        "features" in representations
        and paths.subject_tree_root is None
        and paths.features_csv is None
        and not allow_missing_features
    ):
        raise ValueError(
            "Representation 'features' requested but paths.features_csv is null and allow_missing_features=false."
        )

    model = str(cfg.get("experiment", {}).get("model", "mlp")).strip().lower()
    if model not in {"mlp", "lgbm", "logistic_regression", "svm_rbf", "majority", "random_forest"}:
        raise ValueError(
            "experiment.model must be one of: "
            "['mlp','lgbm','logistic_regression','svm_rbf','majority','random_forest']."
        )

    raw_classifiers = cfg.get("experiment", {}).get("classifiers")
    if raw_classifiers is None:
        classifiers = [model]
    else:
        classifiers = [str(value) for value in raw_classifiers]
    valid_models = {"mlp", "lgbm", "logistic_regression", "svm_rbf", "majority", "random_forest"}
    invalid_classifiers = [name for name in classifiers if name not in valid_models]
    if invalid_classifiers:
        raise ValueError(
            "experiment.classifiers contains unsupported model names: "
            f"{invalid_classifiers}. Allowed: {sorted(valid_models)}."
        )
    if not classifiers:
        raise ValueError("experiment.classifiers must not be empty.")

    return ZojaExperimentConfig(
        seed=int(cfg["seed"]),
        run_id=cfg.get("run_id"),
        paths=paths,
        targets=targets,
        representations=representations,
        embedding_variants=embedding_variants,
        allow_missing_features=allow_missing_features,
        model=model,
        classifiers=classifiers,
        label_rule=label_rule,
        protocols=protocols,
        evaluation=evaluation,
        plotting=plotting,
        raw=_parse_raw(cfg=cfg, repo_root=repo_root),
        features=_parse_features_cfg(cfg),
        models=_parse_models(cfg),
        use_standard_scaler=bool(cfg["scaling"]["use_standard_scaler"]),
        paper_outputs_only=bool(cfg.get("outputs", {}).get("paper_only", False)),
    )
