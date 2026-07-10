"""Fold-specific handcrafted feature filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import FeaturesConfig


@dataclass
class FeatureFilterReport:
    """Summary of dropped and retained handcrafted feature columns."""

    dropped_structural: list[str]
    dropped_missing: list[str]
    dropped_zero_variance: list[str]
    dropped_correlated: list[str]
    selected_columns: list[str]


class FeatureFoldPreprocessor:
    """Apply structural and fold-specific statistical filtering to feature data."""

    def __init__(self, cfg: FeaturesConfig) -> None:
        self.cfg = cfg
        self.selected_columns: list[str] = []
        # Per-column training medians, used for NaN imputation in transform().
        self.col_medians_: dict[str, float] = {}
        self.report: FeatureFilterReport | None = None

    def _structural_drop_columns(self, frame: pd.DataFrame) -> list[str]:
        """Return metadata column names to drop (non-feature identifiers)."""
        metadata = set(self.cfg.structural_drop.metadata_columns)
        return sorted(col for col in frame.columns if col in metadata)

    def fit(self, train_frame: pd.DataFrame) -> "FeatureFoldPreprocessor":
        """Fit selection pipeline on outer-train fold only.

        Pipeline:
        1. Drop metadata columns (structural drop).
        2. Drop columns with > max_nan_fraction NaN (computed on train data).
        3. Drop zero-variance / constant columns.
        4. Compute per-column medians; impute NaN with medians for correlation step.
        5. Drop highly correlated columns (Pearson r > correlation_threshold).
        6. Store selected column names and their training medians.
        """
        working = train_frame.copy()

        dropped_structural = self._structural_drop_columns(working)
        working = working.drop(columns=[c for c in dropped_structural if c in working.columns], errors="ignore")

        numeric = working.select_dtypes(include=[np.number]).copy()

        # Drop high-NaN columns (based on raw NaN fraction in training data).
        max_nan_fraction = self.cfg.statistical_filter.max_nan_fraction
        missing_fraction = numeric.isna().mean()
        dropped_missing = missing_fraction[missing_fraction > max_nan_fraction].index.tolist()
        numeric = numeric.drop(columns=dropped_missing, errors="ignore")

        # Drop zero-variance / constant columns.
        dropped_zero = [col for col in numeric.columns if numeric[col].nunique(dropna=False) <= 1]
        numeric = numeric.drop(columns=dropped_zero, errors="ignore")

        # Compute per-column medians (fit on train) and impute for correlation step.
        trial_medians = numeric.median()
        numeric_for_corr = numeric.fillna(trial_medians)

        # Drop highly correlated columns.
        dropped_correlated: list[str] = []
        if numeric_for_corr.shape[1] > 1:
            corr = numeric_for_corr.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            dropped_correlated = [
                col
                for col in upper.columns
                if any(upper[col] > self.cfg.statistical_filter.correlation_threshold)
            ]
            numeric_for_corr = numeric_for_corr.drop(columns=dropped_correlated, errors="ignore")

        self.selected_columns = list(numeric_for_corr.columns)
        if not self.selected_columns:
            raise ValueError("No feature columns left after fold filtering.")

        # Store training medians for the final selected columns (used in transform).
        self.col_medians_ = trial_medians[self.selected_columns].to_dict()

        self.report = FeatureFilterReport(
            dropped_structural=dropped_structural,
            dropped_missing=dropped_missing,
            dropped_zero_variance=dropped_zero,
            dropped_correlated=dropped_correlated,
            selected_columns=self.selected_columns,
        )
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply fitted filtering pipeline to any split.

        Columns not seen during fit are inserted as NaN and then filled with
        the training median.  NaN values are imputed with per-column training
        medians to prevent data leakage.
        """
        if not self.selected_columns:
            raise RuntimeError("FeatureFoldPreprocessor must be fit before transform.")

        working = frame.copy()
        dropped_structural = self._structural_drop_columns(working)
        working = working.drop(columns=[c for c in dropped_structural if c in working.columns], errors="ignore")

        numeric = working.select_dtypes(include=[np.number]).copy()

        # Ensure all selected columns are present (fill missing ones with NaN).
        for col in self.selected_columns:
            if col not in numeric.columns:
                numeric[col] = np.nan

        numeric = numeric[self.selected_columns]

        # Impute NaN with per-column training medians (no data leakage).
        fill_values = pd.Series(self.col_medians_)
        numeric = numeric.fillna(fill_values)

        return numeric.to_numpy(dtype=np.float32)
