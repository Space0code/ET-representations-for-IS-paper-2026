"""Tests for fold-level protocol metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from trustme_et_comparison.trustme.protocols.aggregation import build_paper_main_results
from trustme_et_comparison.trustme.data_loading import _apply_validity_masks, _vectorize_window
from trustme_et_comparison.trustme.protocols.data import (
    _fast_vectorize_subject_tree_window,
    _iter_raw_windows,
)
from trustme_et_comparison.trustme.protocols.evaluation import evaluate_binary_fold
from trustme_et_comparison.trustme.protocols.labels import centered_binary_train_test
from trustme_et_comparison.trustme.protocols.protocols import run_persistence_baseline
from trustme_et_comparison.trustme.protocols.types import ZojaLabelRuleConfig


def test_macro_f1_averages_both_binary_classes() -> None:
    """The reported macro-F1 must not equal positive-class F1."""

    result = evaluate_binary_fold(
        y_train=np.array([0, 0, 1]),
        y_test=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        y_proba=np.array([0.1, 0.8, 0.7, 0.9]),
        one_class_policy="keep_with_nan",
    )

    assert result.metrics["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.8) / 2.0)
    assert result.metrics["macro_f1"] != pytest.approx(0.8)
    assert result.metrics["baseline_macro_f1"] == pytest.approx(1.0 / 3.0)
    assert result.metrics["baseline_auc"] == pytest.approx(0.5)


def test_unseen_subject_center_weights_training_participants_equally() -> None:
    """Window-rich participants must not dominate the held-out threshold."""

    values = np.array([1.0, 5.0, 5.0, 5.0, 3.5], dtype=np.float32)
    subjects = np.array(["a", "b", "b", "b", "held_out"])
    _, y_test, _, centered_test = centered_binary_train_test(
        values=values,
        subjects=subjects,
        train_idx=np.array([0, 1, 2, 3]),
        test_idx=np.array([4]),
        label_rule=ZojaLabelRuleConfig(),
    )

    assert centered_test[0] == pytest.approx(0.5)
    assert y_test.tolist() == [1]


def test_persistence_uses_the_preceding_observed_label() -> None:
    """Persistence must update after every observed test window."""

    cfg = SimpleNamespace(
        protocols=SimpleNamespace(
            params=SimpleNamespace(
                persistence_calibration_size=3,
                persistence_test_size=3,
                persistence_step_size=3,
                min_train_samples=1,
                min_test_samples=1,
            )
        ),
        label_rule=ZojaLabelRuleConfig(mode="absolute_gt_threshold", threshold=0.5),
        evaluation=SimpleNamespace(one_class_policy="keep_with_nan"),
    )
    output = run_persistence_baseline(
        cfg=cfg,
        target_name="engagement_level",
        representation_name="label_history",
        values=np.array([0, 1, 1, 0, 1, 0], dtype=np.float32),
        subjects=np.array(["s0"] * 6),
        subject_order=np.arange(6),
    )

    assert output.predictions[0]["y_pred"].tolist() == [1, 0, 1]
    assert output.predictions[0]["y_true"].tolist() == [0, 1, 0]


def test_paper_results_include_one_baseline_and_valid_fold_counts() -> None:
    """The compact paper table has one shared baseline plus model rows."""

    folds = pd.DataFrame(
        {
            "target": ["engagement_level"] * 2,
            "model": ["mlp"] * 2,
            "representation": ["raw"] * 2,
            "protocol": ["loso_normalized"] * 2,
            "fold_id": ["loso_1", "loso_2"],
            "subject": ["s0", "s1"],
            "baseline_accuracy": [0.4, 0.6],
            "baseline_balanced_accuracy": [0.5, np.nan],
            "baseline_macro_f1": [0.3, np.nan],
            "baseline_auc": [0.5, np.nan],
            "accuracy": [0.6, 0.7],
            "gain": [0.2, 0.1],
            "balanced_accuracy": [0.55, np.nan],
            "balanced_gain": [0.05, np.nan],
            "macro_f1": [0.52, np.nan],
            "auc": [0.58, np.nan],
        }
    )
    result = build_paper_main_results(folds)

    assert result["representation"].tolist() == ["majority baseline", "raw"]
    assert result.loc[1, "accuracy_valid_folds"] == 2
    assert result.loc[1, "balanced_accuracy_valid_folds"] == 1


def test_raw_window_streaming_preserves_chunk_boundary_groups(tmp_path) -> None:
    """A raw window split between CSV chunks is yielded exactly once."""

    path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "window_uid": ["a", "a", "a", "b", "b", "c"],
            "value": [1, 2, 3, 4, 5, 6],
        }
    ).to_csv(path, index=False)

    windows = list(_iter_raw_windows(path, ["window_uid", "value"], chunksize=2))

    assert [window_id for window_id, _ in windows] == ["a", "b", "c"]
    assert [len(frame) for _, frame in windows] == [3, 2, 1]


def test_fast_raw_vectorizer_matches_standard_validity_and_interpolation() -> None:
    """The optimized final-run raw path must preserve established preprocessing."""

    window = pd.DataFrame(
        {
            "GazePointX": [0.1, 0.2, 0.3, 0.4],
            "PupilSizeLeft": [3.0, np.nan, 3.4, 3.6],
            "AverageDistance": [60.0, 61.0, np.nan, 63.0],
            "ValidityLeft": [1, 0, 1, 1],
            "ValidityRight": [1, 1, 1, 1],
            "PupilValidityLeft": [1, 1, 0, 1],
            "PupilValidityRight": [1, 1, 1, 1],
        }
    )
    channels = ["GazePointX", "PupilSizeLeft", "AverageDistance"]
    masked = _apply_validity_masks(
        window,
        gaze_channels=["GazePointX"],
        pupil_channels=["PupilSizeLeft"],
        validity_gaze=["ValidityLeft", "ValidityRight"],
        validity_pupil=["PupilValidityLeft", "PupilValidityRight"],
        valid_val=1,
    )
    expected = _vectorize_window(
        window=masked,
        channels=channels,
        target_len=6,
        length_mode="truncate_pad",
    )
    actual = _fast_vectorize_subject_tree_window(
        window=window,
        channels=channels,
        gaze_channels=["GazePointX"],
        pupil_channels=["PupilSizeLeft"],
        validity_gaze=["ValidityLeft", "ValidityRight"],
        validity_pupil=["PupilValidityLeft", "PupilValidityRight"],
        valid_val=1,
        target_len=6,
    )

    np.testing.assert_allclose(actual, expected)
