"""Zoja-style protocol implementations for ET representation matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from ...common.feature_filtering import FeatureFoldPreprocessor
from .evaluation import can_train_classifier, evaluate_binary_fold, zscore_train_test
from .labels import centered_binary_train_test
from .modeling import train_predict_binary
from .types import ZojaExperimentConfig

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - optional import handled by status rows.
    GaussianHMM = None


@dataclass
class ProtocolRunOutput:
    """Rows and fold predictions produced by one protocol run."""

    rows: list[dict[str, Any]]
    predictions: list[dict[str, Any]]


def _nan_metrics_row() -> dict[str, float]:
    """Return canonical metric fields with NaN defaults."""

    return {
        "baseline_accuracy": np.nan,
        "baseline_balanced_accuracy": np.nan,
        "baseline_macro_f1": np.nan,
        "baseline_auc": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "auc": np.nan,
        "gain": np.nan,
        "balanced_gain": np.nan,
    }


def _hmm_meta_row(
    *,
    converged: bool | float = np.nan,
    iterations: int | float = np.nan,
    train_log_likelihood: float = np.nan,
) -> dict[str, Any]:
    """Return canonical HMM metadata columns for one row."""

    return {
        "hmm_converged": converged,
        "hmm_iterations": iterations,
        "hmm_train_log_likelihood": train_log_likelihood,
    }


def _slice_fold_matrix(
    *,
    cfg: ZojaExperimentConfig,
    X: np.ndarray | pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice train/test data with fold-safe feature preprocessing for DataFrame inputs."""

    if isinstance(X, pd.DataFrame):
        processor = FeatureFoldPreprocessor(cfg.features).fit(X.iloc[train_idx].copy())
        X_train = processor.transform(X.iloc[train_idx].copy())
        X_test = processor.transform(X.iloc[test_idx].copy())
        return X_train, X_test

    X_np = np.asarray(X, dtype=np.float32)
    return np.asarray(X_np[train_idx], dtype=np.float32), np.asarray(X_np[test_idx], dtype=np.float32)


def _rolling_window_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create fixed-size rolling train/test splits."""

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    while start + train_size + test_size <= n_samples:
        train_idx = np.arange(start, start + train_size)
        test_idx = np.arange(start + train_size, start + train_size + test_size)
        splits.append((train_idx, test_idx))
        start += step_size
    return splits


def _build_row_base(
    *,
    target_name: str,
    representation: str,
    protocol: str,
    fold_id: str,
    subject: str,
    n_train: int,
    n_test: int,
    train_pos_rate: float,
    test_pos_rate: float,
) -> dict[str, Any]:
    """Build base row fields shared across all protocol outputs."""

    return {
        "target": target_name,
        "representation": representation,
        "protocol": protocol,
        "fold_id": fold_id,
        "subject": subject,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_pos_rate": float(train_pos_rate),
        "test_pos_rate": float(test_pos_rate),
    }


def _evaluate_supervised_fold(
    *,
    cfg: ZojaExperimentConfig,
    protocol_name: str,
    target_name: str,
    representation_name: str,
    fold_id: str,
    subject_name: str,
    X: np.ndarray | pd.DataFrame,
    values: np.ndarray,
    subjects: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    normalize_loso: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Train and evaluate one supervised fold."""

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    y_train, y_test, _, _ = centered_binary_train_test(
        values=values,
        subjects=subjects,
        train_idx=train_idx,
        test_idx=test_idx,
        label_rule=cfg.label_rule,
    )

    base = _build_row_base(
        target_name=target_name,
        representation=representation_name,
        protocol=protocol_name,
        fold_id=fold_id,
        subject=subject_name,
        n_train=train_idx.shape[0],
        n_test=test_idx.shape[0],
        train_pos_rate=float(np.mean(y_train)) if y_train.size else np.nan,
        test_pos_rate=float(np.mean(y_test)) if y_test.size else np.nan,
    )

    requires_two_class_train = cfg.model in {"logistic_regression", "lgbm", "svm_rbf"}
    ok_to_train, reason = can_train_classifier(
        y_train=y_train,
        y_test=y_test,
        min_train_samples=cfg.protocols.params.min_train_samples,
        min_test_samples=cfg.protocols.params.min_test_samples,
        one_class_policy=cfg.evaluation.one_class_policy,
        require_two_class_train=requires_two_class_train,
    )
    if not ok_to_train:
        row = {
            **base,
            "status": "skipped",
            "skip_reason": reason,
            **_nan_metrics_row(),
            **_hmm_meta_row(),
        }
        return row, None

    X_train, X_test = _slice_fold_matrix(
        cfg=cfg,
        X=X,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    if normalize_loso:
        X_train, X_test = zscore_train_test(
            X_train,
            X_test,
            train_groups=subjects[train_idx],
        )

    y_pred, y_proba = train_predict_binary(
        model_name=cfg.model,
        models_cfg=cfg.models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        use_standard_scaler=cfg.use_standard_scaler,
        random_seed=cfg.seed,
    )

    eval_result = evaluate_binary_fold(
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        one_class_policy=cfg.evaluation.one_class_policy,
    )

    row = {
        **base,
        "status": eval_result.status,
        "skip_reason": eval_result.skip_reason,
        **eval_result.metrics,
        **_hmm_meta_row(),
    }

    pred_payload = {
        "target": target_name,
        "representation": representation_name,
        "protocol": protocol_name,
        "fold_id": fold_id,
        "subject": subject_name,
        "status": eval_result.status,
        "y_true": y_test,
        "y_pred": y_pred,
    }
    return row, pred_payload


def run_loso(
    *,
    cfg: ZojaExperimentConfig,
    target_name: str,
    representation_name: str,
    X: np.ndarray | pd.DataFrame,
    values: np.ndarray,
    subjects: np.ndarray,
    normalize_loso: bool,
) -> ProtocolRunOutput:
    """Run LOSO protocol (optionally with Zoja-v2 normalization)."""

    logo = LeaveOneGroupOut()
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    total_folds = np.unique(subjects).shape[0]
    protocol_name = "loso_normalized" if normalize_loso else "loso"
    print(
        f"[FoldPlan] protocol={protocol_name} target={target_name} repr={representation_name} total_folds={total_folds}",
        flush=True,
    )

    for fold_num, (train_idx, test_idx) in enumerate(logo.split(X, groups=subjects), start=1):
        held_out_subjects = np.unique(subjects[test_idx])
        subject_name = held_out_subjects[0] if held_out_subjects.size == 1 else "multi"
        row, pred = _evaluate_supervised_fold(
            cfg=cfg,
            protocol_name=protocol_name,
            target_name=target_name,
            representation_name=representation_name,
            fold_id=f"loso_{fold_num}",
            subject_name=str(subject_name),
            X=X,
            values=values,
            subjects=subjects,
            train_idx=train_idx,
            test_idx=test_idx,
            normalize_loso=normalize_loso,
        )
        rows.append(row)
        if pred is not None:
            predictions.append(pred)

    return ProtocolRunOutput(rows=rows, predictions=predictions)


def run_within_subject_temporal_split(
    *,
    cfg: ZojaExperimentConfig,
    target_name: str,
    representation_name: str,
    X: np.ndarray | pd.DataFrame,
    values: np.ndarray,
    subjects: np.ndarray,
    subject_order: np.ndarray,
) -> ProtocolRunOutput:
    """Run pooled temporal split protocol (70/30 per subject then concatenate)."""

    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    ratio = cfg.protocols.params.temporal_split_ratio
    print(
        f"[FoldPlan] protocol=within_subject_temporal_split target={target_name} repr={representation_name} "
        f"subjects={np.unique(subjects).shape[0]} split_ratio={ratio}",
        flush=True,
    )

    train_chunks: list[np.ndarray] = []
    test_chunks: list[np.ndarray] = []
    for subject in np.unique(subjects):
        idx = np.where(subjects == subject)[0]
        idx = idx[np.argsort(subject_order[idx])]
        split = int(np.floor(idx.shape[0] * ratio))
        train_chunks.append(idx[:split])
        test_chunks.append(idx[split:])

    if not train_chunks or not test_chunks:
        return ProtocolRunOutput(rows=rows, predictions=predictions)

    train_idx = np.concatenate(train_chunks, axis=0).astype(np.int64)
    test_idx = np.concatenate(test_chunks, axis=0).astype(np.int64)

    row, pred = _evaluate_supervised_fold(
        cfg=cfg,
        protocol_name="within_subject_temporal_split",
        target_name=target_name,
        representation_name=representation_name,
        fold_id="within_all_subjects",
        subject_name="all_subjects",
        X=X,
        values=values,
        subjects=subjects,
        train_idx=train_idx,
        test_idx=test_idx,
        normalize_loso=False,
    )
    rows.append(row)
    if pred is not None:
        predictions.append(pred)

    return ProtocolRunOutput(rows=rows, predictions=predictions)


def run_rolling_window(
    *,
    cfg: ZojaExperimentConfig,
    target_name: str,
    representation_name: str,
    X: np.ndarray | pd.DataFrame,
    values: np.ndarray,
    subjects: np.ndarray,
    subject_order: np.ndarray,
) -> ProtocolRunOutput:
    """Run within-subject rolling-window protocol."""

    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    p = cfg.protocols.params
    for subject in np.unique(subjects):
        idx = np.where(subjects == subject)[0]
        idx = idx[np.argsort(subject_order[idx])]

        splits = _rolling_window_splits(
            n_samples=idx.shape[0],
            train_size=p.rolling_train_size,
            test_size=p.rolling_test_size,
            step_size=p.rolling_step_size,
        )
        print(
            f"[FoldPlan] protocol=rolling_window target={target_name} repr={representation_name} "
            f"subject={subject} folds={len(splits)}",
            flush=True,
        )

        for fold_num, (train_local, test_local) in enumerate(splits, start=1):
            train_idx = idx[train_local]
            test_idx = idx[test_local]
            row, pred = _evaluate_supervised_fold(
                cfg=cfg,
                protocol_name="rolling_window",
                target_name=target_name,
                representation_name=representation_name,
                fold_id=f"rolling_{subject}_{fold_num}",
                subject_name=str(subject),
                X=X,
                values=values,
                subjects=subjects,
                train_idx=train_idx,
                test_idx=test_idx,
                normalize_loso=False,
            )
            rows.append(row)
            if pred is not None:
                predictions.append(pred)

    return ProtocolRunOutput(rows=rows, predictions=predictions)


def run_hybrid_loso_adaptation(
    *,
    cfg: ZojaExperimentConfig,
    target_name: str,
    representation_name: str,
    X: np.ndarray | pd.DataFrame,
    values: np.ndarray,
    subjects: np.ndarray,
    subject_order: np.ndarray,
) -> ProtocolRunOutput:
    """Run hybrid LOSO+subject-adaptation protocol."""

    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    p = cfg.protocols.params
    all_idx = np.arange(X.shape[0], dtype=np.int64)

    for subject in np.unique(subjects):
        sub_idx = np.where(subjects == subject)[0]
        sub_idx = sub_idx[np.argsort(subject_order[sub_idx])]
        other_idx = all_idx[subjects != subject]
        planned = 0
        t_plan = p.hybrid_calibration_size
        while t_plan + p.hybrid_test_size <= sub_idx.shape[0]:
            planned += 1
            t_plan += p.hybrid_step_size
        print(
            f"[FoldPlan] protocol=hybrid_loso_adaptation target={target_name} repr={representation_name} "
            f"subject={subject} folds={planned}",
            flush=True,
        )

        t = p.hybrid_calibration_size
        fold_num = 0
        while t + p.hybrid_test_size <= sub_idx.shape[0]:
            fold_num += 1
            cal_idx = sub_idx[:t]
            test_idx = sub_idx[t : t + p.hybrid_test_size]
            train_idx = np.concatenate([other_idx, cal_idx], axis=0)

            row, pred = _evaluate_supervised_fold(
                cfg=cfg,
                protocol_name="hybrid_loso_adaptation",
                target_name=target_name,
                representation_name=representation_name,
                fold_id=f"hybrid_{subject}_{fold_num}",
                subject_name=str(subject),
                X=X,
                values=values,
                subjects=subjects,
                train_idx=train_idx,
                test_idx=test_idx,
                normalize_loso=False,
            )
            rows.append(row)
            if pred is not None:
                predictions.append(pred)

            t += p.hybrid_step_size

    return ProtocolRunOutput(rows=rows, predictions=predictions)


def run_persistence_baseline(
    *,
    cfg: ZojaExperimentConfig,
    target_name: str,
    representation_name: str,
    values: np.ndarray,
    subjects: np.ndarray,
    subject_order: np.ndarray,
    recordings: np.ndarray | None = None,
) -> ProtocolRunOutput:
    """Run one-step persistence within continuous subject recordings."""

    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    p = cfg.protocols.params
    recordings_array = (
        np.asarray(recordings, dtype=str)
        if recordings is not None
        else np.full(subjects.shape[0], "all_recordings", dtype=str)
    )
    for subject in np.unique(subjects):
        subject_idx = np.where(subjects == subject)[0]
        subject_recordings = recordings_array[subject_idx]
        for recording_num, recording in enumerate(pd.unique(subject_recordings), start=1):
            idx = subject_idx[subject_recordings == recording]
            idx = idx[np.argsort(subject_order[idx])]
            planned = 0
            t_plan = p.persistence_calibration_size
            while t_plan + p.persistence_test_size <= idx.shape[0]:
                planned += 1
                t_plan += p.persistence_step_size
            print(
                f"[FoldPlan] protocol=persistence_baseline target={target_name} repr={representation_name} "
                f"subject={subject} recording={recording_num} folds={planned}",
                flush=True,
            )

            t = p.persistence_calibration_size
            fold_num = 0
            while t + p.persistence_test_size <= idx.shape[0]:
                fold_num += 1
                train_idx = idx[:t]
                test_idx = idx[t : t + p.persistence_test_size]

                y_train, y_test, _, _ = centered_binary_train_test(
                    values=values,
                    subjects=subjects,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    label_rule=cfg.label_rule,
                )

                fold_id = f"persist_{subject}_{recording_num}_{fold_num}"
                base = _build_row_base(
                    target_name=target_name,
                    representation=representation_name,
                    protocol="persistence_baseline",
                    fold_id=fold_id,
                    subject=str(subject),
                    n_train=train_idx.shape[0],
                    n_test=test_idx.shape[0],
                    train_pos_rate=float(np.mean(y_train)) if y_train.size else np.nan,
                    test_pos_rate=float(np.mean(y_test)) if y_test.size else np.nan,
                )

                if y_train.shape[0] < p.min_train_samples or y_test.shape[0] < p.min_test_samples:
                    rows.append(
                        {
                            **base,
                            "status": "skipped",
                            "skip_reason": "small_partition",
                            **_nan_metrics_row(),
                            **_hmm_meta_row(),
                        }
                    )
                    t += p.persistence_step_size
                    continue

                # One-step persistence: at each test step, predict the most recently
                # observed label. The first prediction uses the last calibration
                # label; later predictions use the preceding observed test label.
                y_pred = np.concatenate(
                    [np.asarray([y_train[-1]], dtype=np.int64), y_test[:-1]]
                )
                eval_result = evaluate_binary_fold(
                    y_train=y_train,
                    y_test=y_test,
                    y_pred=y_pred,
                    y_proba=None,
                    one_class_policy=cfg.evaluation.one_class_policy,
                )
                rows.append(
                    {
                        **base,
                        "status": eval_result.status,
                        "skip_reason": eval_result.skip_reason,
                        **eval_result.metrics,
                        **_hmm_meta_row(),
                    }
                )
                predictions.append(
                    {
                        "target": target_name,
                        "representation": representation_name,
                        "protocol": "persistence_baseline",
                        "fold_id": fold_id,
                        "subject": str(subject),
                        "status": eval_result.status,
                        "y_true": y_test,
                        "y_pred": y_pred,
                    }
                )
                t += p.persistence_step_size

    return ProtocolRunOutput(rows=rows, predictions=predictions)


def _map_states_to_labels(states_train: np.ndarray, y_train: np.ndarray) -> dict[int, int]:
    """Map hidden states to majority class labels using train data only."""

    mapping: dict[int, int] = {}
    for state in np.unique(states_train):
        mask = states_train == state
        votes = y_train[mask]
        mapping[int(state)] = int(np.bincount(votes).argmax())
    return mapping


def _apply_state_mapping(
    states: np.ndarray,
    mapping: dict[int, int],
    default_label: int,
) -> np.ndarray:
    """Apply state->label mapping for test sequence."""

    return np.asarray([mapping.get(int(state), default_label) for state in states], dtype=np.int64)


def run_hmm_rolling(
    *,
    cfg: ZojaExperimentConfig,
    target_name: str,
    representation_name: str,
    X: np.ndarray | pd.DataFrame,
    values: np.ndarray,
    subjects: np.ndarray,
    subject_order: np.ndarray,
) -> ProtocolRunOutput:
    """Run HMM rolling protocol with train-only state-label mapping."""

    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    p = cfg.protocols.params

    if GaussianHMM is None:
        rows.append(
            {
                "target": target_name,
                "representation": representation_name,
                "protocol": "hmm_rolling",
                "fold_id": "hmm_missing_dependency",
                "subject": "all",
                "n_train": 0,
                "n_test": 0,
                "train_pos_rate": np.nan,
                "test_pos_rate": np.nan,
                "status": "skipped",
                "skip_reason": "hmmlearn_not_installed",
                **_nan_metrics_row(),
                **_hmm_meta_row(),
            }
        )
        return ProtocolRunOutput(rows=rows, predictions=predictions)

    for subject in np.unique(subjects):
        idx = np.where(subjects == subject)[0]
        idx = idx[np.argsort(subject_order[idx])]

        splits = _rolling_window_splits(
            n_samples=idx.shape[0],
            train_size=p.hmm_train_size,
            test_size=p.hmm_test_size,
            step_size=p.hmm_step_size,
        )
        print(
            f"[FoldPlan] protocol=hmm_rolling target={target_name} repr={representation_name} "
            f"subject={subject} folds={len(splits)}",
            flush=True,
        )

        for fold_num, (train_local, test_local) in enumerate(splits, start=1):
            train_idx = idx[train_local]
            test_idx = idx[test_local]

            y_train, y_test, _, _ = centered_binary_train_test(
                values=values,
                subjects=subjects,
                train_idx=train_idx,
                test_idx=test_idx,
                label_rule=cfg.label_rule,
            )

            base = _build_row_base(
                target_name=target_name,
                representation=representation_name,
                protocol="hmm_rolling",
                fold_id=f"hmm_{subject}_{fold_num}",
                subject=str(subject),
                n_train=train_idx.shape[0],
                n_test=test_idx.shape[0],
                train_pos_rate=float(np.mean(y_train)) if y_train.size else np.nan,
                test_pos_rate=float(np.mean(y_test)) if y_test.size else np.nan,
            )

            ok_to_train, reason = can_train_classifier(
                y_train=y_train,
                y_test=y_test,
                min_train_samples=cfg.protocols.params.min_train_samples,
                min_test_samples=cfg.protocols.params.min_test_samples,
                one_class_policy=cfg.evaluation.one_class_policy,
                require_two_class_train=True,
            )
            if not ok_to_train:
                rows.append(
                    {
                        **base,
                        "status": "skipped",
                        "skip_reason": reason,
                        **_nan_metrics_row(),
                        **_hmm_meta_row(),
                    }
                )
                continue

            X_train, X_test = _slice_fold_matrix(
                cfg=cfg,
                X=X,
                train_idx=train_idx,
                test_idx=test_idx,
            )

            # HMM preprocessing fit strictly on train.
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_scaled = scaler.transform(X_test_imp)

            try:
                model = GaussianHMM(
                    n_components=p.hmm_n_states,
                    covariance_type="diag",
                    n_iter=200,
                    random_state=cfg.seed,
                )
                model.fit(X_train_scaled)
                states_train = model.predict(X_train_scaled)
                states_test = model.predict(X_test_scaled)

                mapping = _map_states_to_labels(states_train=states_train, y_train=y_train)
                default_label = int(np.bincount(y_train).argmax())
                y_pred = _apply_state_mapping(states=states_test, mapping=mapping, default_label=default_label)

                eval_result = evaluate_binary_fold(
                    y_train=y_train,
                    y_test=y_test,
                    y_pred=y_pred,
                    y_proba=None,
                    one_class_policy=cfg.evaluation.one_class_policy,
                )

                rows.append(
                    {
                        **base,
                        "status": eval_result.status,
                        "skip_reason": eval_result.skip_reason,
                        **eval_result.metrics,
                        **_hmm_meta_row(
                            converged=bool(model.monitor_.converged),
                            iterations=int(model.monitor_.iter),
                            train_log_likelihood=float(model.score(X_train_scaled)),
                        ),
                    }
                )
                predictions.append(
                    {
                        "target": target_name,
                        "representation": representation_name,
                        "protocol": "hmm_rolling",
                        "fold_id": f"hmm_{subject}_{fold_num}",
                        "subject": str(subject),
                        "status": eval_result.status,
                        "y_true": y_test,
                        "y_pred": y_pred,
                    }
                )
            except Exception as exc:  # pragma: no cover - execution dependent
                rows.append(
                    {
                        **base,
                        "status": "failed",
                        "skip_reason": f"hmm_failed:{type(exc).__name__}",
                        **_nan_metrics_row(),
                        **_hmm_meta_row(),
                    }
                )

    return ProtocolRunOutput(rows=rows, predictions=predictions)
