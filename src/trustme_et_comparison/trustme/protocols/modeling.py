"""Model fit/predict helpers for protocol evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ...common.models import create_model


@dataclass
class _ModelCfgAdapter:
    """Minimal adapter expected by common model factory."""

    models: Any


_MLP_DEVICE_LOGGED = False


def _make_lgbm_inputs(X_train: np.ndarray, X_test: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stable-feature-name dataframes for LightGBM."""

    feature_names = [f"f_{idx}" for idx in range(X_train.shape[1])]
    return (
        pd.DataFrame(X_train, columns=feature_names),
        pd.DataFrame(X_test, columns=feature_names),
    )


def _align_binary_probabilities(model: Any, y_proba: np.ndarray) -> np.ndarray:
    """Return positive-class probability vector with class alignment safeguards."""

    y_proba = np.asarray(y_proba, dtype=np.float32)
    if y_proba.ndim == 1:
        return y_proba.astype(np.float32)

    if y_proba.shape[1] == 1:
        # Degenerate model output.
        return y_proba[:, 0].astype(np.float32)

    if hasattr(model, "classes_"):
        classes = np.asarray(getattr(model, "classes_"), dtype=np.int64)
        if classes.shape[0] == 2:
            pos_idx = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else 1
            return y_proba[:, pos_idx].astype(np.float32)

    return y_proba[:, 1].astype(np.float32)


def train_predict_binary(
    *,
    model_name: str,
    models_cfg: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    use_standard_scaler: bool,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one binary model and return test predictions + positive probabilities."""

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    scaler = None
    if use_standard_scaler:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    model = create_model(
        model_name=model_name,
        cfg=_ModelCfgAdapter(models=models_cfg),
        input_dim=int(X_train.shape[1]),
        num_classes=2,
    )

    if model_name == "mlp":
        global _MLP_DEVICE_LOGGED
        if not _MLP_DEVICE_LOGGED and hasattr(model, "device"):
            print(f"[MLP] Training device: {getattr(model, 'device')}", flush=True)
            _MLP_DEVICE_LOGGED = True
        stratify = y_train if np.unique(y_train).shape[0] > 1 else None
        val_size = 0.2
        if X_train.shape[0] < 10:
            val_size = 0.4
        try:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_size,
                random_state=random_seed,
                stratify=stratify,
            )
        except ValueError:
            X_fit, y_fit = X_train, y_train
            X_val, y_val = X_train, y_train
        if np.unique(y_fit).shape[0] < 2:
            # fallback to full set when split collapses class diversity
            X_fit, y_fit = X_train, y_train
            X_val, y_val = X_train, y_train

        model.fit(
            X_train=X_fit,
            y_train=y_fit,
            X_val=X_val,
            y_val=y_val,
        )
        y_pred = np.asarray(model.predict(X_test), dtype=np.int64)
        y_proba = _align_binary_probabilities(model, model.predict_proba(X_test))
        return y_pred, y_proba

    X_train_fit: Any = X_train
    X_test_pred: Any = X_test
    if model_name == "lgbm":
        X_train_fit, X_test_pred = _make_lgbm_inputs(X_train=X_train, X_test=X_test)

    model.fit(X_train_fit, y_train)
    y_pred = np.asarray(model.predict(X_test_pred), dtype=np.int64)
    y_proba = _align_binary_probabilities(model, model.predict_proba(X_test_pred))
    return y_pred, y_proba
