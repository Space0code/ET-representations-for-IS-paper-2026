"""Runner for TrustMe Tobii ET protocol evaluation.

This module executes all configured protocol variants over selected ET
representations and targets, then writes standardized tables, figures, and
report artifacts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from ...common.utils import ensure_dir, make_run_id, save_yaml, set_global_seed
from ...common.types import RepresentationDataset
from .aggregation import (
    add_subject_counts,
    build_method_comparison_tables,
    build_protocol_summary,
    build_representation_protocol_ranking,
    build_subject_metrics,
)
from .config import load_zoja_protocol_config
from .data import align_bundle_to_common_ids, build_protocol_canonical, load_protocol_representations
from .labels import build_target_values, centered_binary_full_for_diagnostics
from .plotting import create_protocol_plots
from .protocols import (
    ProtocolRunOutput,
    run_hmm_rolling,
    run_hybrid_loso_adaptation,
    run_loso,
    run_persistence_baseline,
    run_rolling_window,
    run_within_subject_temporal_split,
)
from .reporting import write_protocol_report
from .types import ZojaExperimentConfig


FOLD_TABLE_COLUMNS = [
    "target",
    "representation",
    "model",
    "protocol",
    "fold_id",
    "subject",
    "n_train",
    "n_test",
    "train_pos_rate",
    "test_pos_rate",
    "status",
    "skip_reason",
    "baseline_accuracy",
    "baseline_balanced_accuracy",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "recall",
    "auc",
    "gain",
    "balanced_gain",
    "hmm_converged",
    "hmm_iterations",
    "hmm_train_log_likelihood",
]


def _required_target_columns(cfg: ZojaExperimentConfig) -> list[str]:
    """Return unique source columns required by configured targets."""

    cols: list[str] = []
    for target in cfg.targets:
        if target.mode == "single_column":
            assert target.column is not None
            cols.append(target.column)
        elif target.mode == "mean_columns":
            assert target.columns is not None
            cols.extend(target.columns)
        else:
            raise ValueError(f"Unsupported target mode: {target.mode}")

    # Preserve declaration order while removing duplicates.
    return list(dict.fromkeys(cols))


def _subject_temporal_order(metadata: pd.DataFrame) -> np.ndarray:
    """Return deterministic within-subject temporal indices."""

    if "window_order" not in metadata.columns:
        raise ValueError("metadata missing required 'window_order' for temporal protocols.")
    return metadata.groupby("Subject", sort=False).cumcount().to_numpy(dtype=np.int64)


def _validate_bundle_consistency(
    metadata: pd.DataFrame,
    representations: dict[str, Any],
) -> None:
    """Fail fast when aligned metadata and representation cohorts diverge."""

    if metadata.empty:
        raise ValueError("Aligned metadata is empty after intersection.")
    if metadata["segment_id"].duplicated().any():
        raise ValueError("Aligned metadata has duplicate segment_id values.")

    ordered_ids = metadata["segment_id"].to_numpy(dtype=str)
    ordered_subjects = metadata["Subject"].to_numpy(dtype=str)

    for rep_name, dataset in representations.items():
        if dataset.segment_ids.shape[0] != ordered_ids.shape[0]:
            raise ValueError(
                f"Representation '{rep_name}' row count mismatch: "
                f"{dataset.segment_ids.shape[0]} vs metadata {ordered_ids.shape[0]}."
            )
        if not np.array_equal(dataset.segment_ids.astype(str), ordered_ids):
            raise ValueError(f"Representation '{rep_name}' segment_id order mismatch.")
        if not np.array_equal(dataset.subjects.astype(str), ordered_subjects):
            raise ValueError(f"Representation '{rep_name}' subject order mismatch.")


def _subset_representation(dataset: RepresentationDataset, idx: np.ndarray) -> RepresentationDataset:
    """Return representation dataset subset by integer index array."""

    if isinstance(dataset.X, pd.DataFrame):
        X_new: Any = dataset.X.iloc[idx].copy()
    elif isinstance(dataset.X, list):
        X_new = [dataset.X[int(i)] for i in idx.tolist()]
    else:
        X_new = np.asarray(dataset.X)[idx]

    return RepresentationDataset(
        name=dataset.name,
        X=X_new,
        segment_ids=dataset.segment_ids[idx].copy(),
        subjects=dataset.subjects[idx].copy(),
        source_labels=dataset.source_labels[idx].copy(),
    )


def _filter_rows_with_complete_targets(
    metadata: pd.DataFrame,
    representations: dict[str, RepresentationDataset],
    target_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, RepresentationDataset], dict[str, int]]:
    """Filter aligned rows to those with valid numeric values for all target columns."""

    if not target_columns:
        return metadata, representations, {"before": len(metadata), "after": len(metadata), "dropped": 0}

    mask = np.ones(len(metadata), dtype=bool)
    for column in target_columns:
        numeric = pd.to_numeric(metadata[column], errors="coerce")
        mask &= numeric.notna().to_numpy(dtype=bool)

    idx = np.where(mask)[0].astype(np.int64)
    filtered_meta = metadata.iloc[idx].copy().reset_index(drop=True)
    filtered_reps = {name: _subset_representation(dataset, idx) for name, dataset in representations.items()}
    counts = {
        "before": int(len(metadata)),
        "after": int(len(filtered_meta)),
        "dropped": int(len(metadata) - len(filtered_meta)),
    }
    return filtered_meta, filtered_reps, counts


def _append_output(
    collected_rows: list[dict[str, Any]],
    collected_predictions: list[dict[str, Any]],
    output: ProtocolRunOutput,
) -> None:
    """Append protocol output rows and predictions into global buffers."""

    collected_rows.extend(output.rows)
    collected_predictions.extend(output.predictions)


def _run_all_protocols_for_target_representation(
    *,
    cfg: ZojaExperimentConfig,
    target_name: str,
    representation_name: str,
    X: np.ndarray | pd.DataFrame,
    values: np.ndarray,
    subjects: np.ndarray,
    subject_order: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute all enabled protocols for one target x representation pair."""

    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    def _run_protocol(name: str, fn) -> None:
        start = time.perf_counter()
        before_rows = len(rows)
        before_preds = len(predictions)
        print(
            f"[ProtocolStart] target={target_name} repr={representation_name} protocol={name}",
            flush=True,
        )
        output = fn()
        _append_output(rows, predictions, output)
        elapsed = time.perf_counter() - start
        print(
            f"[ProtocolDone] target={target_name} repr={representation_name} protocol={name} "
            f"rows+={len(rows) - before_rows} preds+={len(predictions) - before_preds} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )

    enabled = cfg.protocols.enabled
    if enabled.loso:
        _run_protocol(
            "loso",
            lambda: run_loso(
                cfg=cfg,
                target_name=target_name,
                representation_name=representation_name,
                X=X,
                values=values,
                subjects=subjects,
                normalize_loso=False,
            ),
        )

    if enabled.loso_normalized:
        _run_protocol(
            "loso_normalized",
            lambda: run_loso(
                cfg=cfg,
                target_name=target_name,
                representation_name=representation_name,
                X=X,
                values=values,
                subjects=subjects,
                normalize_loso=True,
            ),
        )

    if enabled.within_subject_temporal_split:
        _run_protocol(
            "within_subject_temporal_split",
            lambda: run_within_subject_temporal_split(
                cfg=cfg,
                target_name=target_name,
                representation_name=representation_name,
                X=X,
                values=values,
                subjects=subjects,
                subject_order=subject_order,
            ),
        )

    if enabled.rolling_window:
        _run_protocol(
            "rolling_window",
            lambda: run_rolling_window(
                cfg=cfg,
                target_name=target_name,
                representation_name=representation_name,
                X=X,
                values=values,
                subjects=subjects,
                subject_order=subject_order,
            ),
        )

    if enabled.hybrid_loso_adaptation:
        _run_protocol(
            "hybrid_loso_adaptation",
            lambda: run_hybrid_loso_adaptation(
                cfg=cfg,
                target_name=target_name,
                representation_name=representation_name,
                X=X,
                values=values,
                subjects=subjects,
                subject_order=subject_order,
            ),
        )

    if enabled.persistence_baseline:
        _run_protocol(
            "persistence_baseline",
            lambda: run_persistence_baseline(
                cfg=cfg,
                target_name=target_name,
                representation_name=representation_name,
                values=values,
                subjects=subjects,
                subject_order=subject_order,
            ),
        )

    if enabled.hmm_rolling:
        _run_protocol(
            "hmm_rolling",
            lambda: run_hmm_rolling(
                cfg=cfg,
                target_name=target_name,
                representation_name=representation_name,
                X=X,
                values=values,
                subjects=subjects,
                subject_order=subject_order,
            ),
        )

    return rows, predictions


def _scope_snapshot(cfg: ZojaExperimentConfig, metadata: pd.DataFrame) -> dict[str, Any]:
    """Build scope snapshot summary for report generation."""

    enabled_protocols = [
        name
        for name, value in vars(cfg.protocols.enabled).items()
        if bool(value)
    ]
    return {
        "model": cfg.model,
        "classifiers": cfg.classifiers,
        "representations": cfg.representations,
        "embedding_variants": [variant.tag for variant in cfg.embedding_variants],
        "targets": [target.name for target in cfg.targets],
        "label_rule": {
            "mode": cfg.label_rule.mode,
            "threshold": cfg.label_rule.threshold,
            "unseen_subject_center": cfg.label_rule.unseen_subject_center,
        },
        "enabled_protocols": enabled_protocols,
        "n_segments": int(len(metadata)),
        "n_subjects": int(metadata["Subject"].nunique()),
    }


def _ensure_fold_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return fold dataframe with stable column contract."""

    if frame.empty:
        return pd.DataFrame(columns=FOLD_TABLE_COLUMNS)

    out = frame.copy()
    for col in FOLD_TABLE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[FOLD_TABLE_COLUMNS].copy()


def run_zoja_protocols(config_path: str | Path) -> Path:
    """Run Zoja-style ET protocols and return output run directory."""

    cfg = load_zoja_protocol_config(config_path)
    set_global_seed(cfg.seed)
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        print(
            f"[Runtime] torch={torch.__version__} cuda_available={cuda_available} "
            f"cuda_device_count={torch.cuda.device_count()} cuda_device={device_name}",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - env-dependent diagnostics
        print(f"[Runtime] torch diagnostics unavailable: {type(exc).__name__}", flush=True)

    run_id = make_run_id(cfg.run_id)
    run_root = ensure_dir(cfg.paths.results_root / run_id)
    tables_dir = ensure_dir(run_root / "tables")
    _ = ensure_dir(run_root / "figures")

    save_yaml(run_root / "config_snapshot.yaml", cfg)

    target_columns = _required_target_columns(cfg)
    canonical = build_protocol_canonical(cfg, target_columns=target_columns)
    loaded = load_protocol_representations(cfg, canonical)
    bundle = align_bundle_to_common_ids(canonical=canonical, representations=loaded)

    metadata = bundle.metadata.copy()
    representations = bundle.representations
    _validate_bundle_consistency(metadata=metadata, representations=representations)

    metadata, representations, filter_counts = _filter_rows_with_complete_targets(
        metadata=metadata,
        representations=representations,
        target_columns=target_columns,
    )
    if metadata.empty:
        raise ValueError(
            "No rows remain after filtering non-numeric/missing target values "
            f"for required columns={target_columns}."
        )
    _validate_bundle_consistency(metadata=metadata, representations=representations)
    print(
        "Target availability filter:"
        f" before={filter_counts['before']}, after={filter_counts['after']}, dropped={filter_counts['dropped']}",
        flush=True,
    )

    subjects = metadata["Subject"].to_numpy(dtype=str)
    subject_order = _subject_temporal_order(metadata)

    fold_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    diagnostics_labels: dict[str, np.ndarray] = {}

    for target in cfg.targets:
        print(f"[TargetStart] target={target.name}", flush=True)
        values = build_target_values(metadata=metadata, target=target)
        diagnostics_labels[target.name] = centered_binary_full_for_diagnostics(
            values=values,
            subjects=subjects,
            label_rule=cfg.label_rule,
        )

        for model_name in cfg.classifiers:
            cfg_for_model = replace(cfg, model=model_name)
            print(f"[ModelStart] target={target.name} model={model_name}", flush=True)

            for representation_name, dataset in representations.items():
                print(
                    f"[RepresentationStart] target={target.name} model={model_name} repr={representation_name} "
                    f"n_samples={len(values)}",
                    flush=True,
                )
                out_rows, out_predictions = _run_all_protocols_for_target_representation(
                    cfg=cfg_for_model,
                    target_name=target.name,
                    representation_name=representation_name,
                    X=dataset.X,
                    values=values,
                    subjects=subjects,
                    subject_order=subject_order,
                )
                for row in out_rows:
                    row["model"] = model_name
                for pred in out_predictions:
                    pred["model"] = model_name
                fold_rows.extend(out_rows)
                predictions.extend(out_predictions)

    fold_df = _ensure_fold_schema(pd.DataFrame(fold_rows))
    subject_df = build_subject_metrics(fold_df)

    summary_df = build_protocol_summary(fold_df=fold_df, subject_df=subject_df)
    summary_df = add_subject_counts(summary_df=summary_df, subject_df=subject_df)

    gain_tbl, balanced_tbl = build_method_comparison_tables(subject_df=subject_df)
    ranking_df = build_representation_protocol_ranking(summary_df=summary_df)

    fold_df.to_csv(tables_dir / "protocol_fold_metrics.csv", index=False)
    subject_df.to_csv(tables_dir / "protocol_subject_metrics.csv", index=False)
    summary_df.to_csv(tables_dir / "protocol_summary.csv", index=False)
    gain_tbl.to_csv(tables_dir / "method_comparison_gain.csv", index=False)
    balanced_tbl.to_csv(tables_dir / "method_comparison_balanced_gain.csv", index=False)
    ranking_df.to_csv(tables_dir / "representation_protocol_ranking.csv", index=False)

    if cfg.plotting.enabled:
        create_protocol_plots(
            results_dir=run_root,
            subject_df=subject_df,
            summary_df=summary_df,
            predictions=predictions,
            metadata=metadata,
            diagnostics_labels=diagnostics_labels,
            confusion_top_k=int(cfg.plotting.confusion_top_k),
        )

    write_protocol_report(
        output_path=run_root / "report.md",
        fold_df=fold_df,
        subject_df=subject_df,
        summary_df=summary_df,
        ranking_df=ranking_df,
        run_scope=_scope_snapshot(cfg=cfg, metadata=metadata),
    )

    print(
        "Completed TrustMe Tobii ET protocols: "
        f"run_dir={run_root}, n_rows={len(fold_df)}, "
        f"n_subjects={metadata['Subject'].nunique()}, n_representations={len(representations)}",
        flush=True,
    )
    return run_root
