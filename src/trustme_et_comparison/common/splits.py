"""Cross-validation split utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .types import FoldSplit
from .utils import ensure_dir, save_json


def build_group_splits(
    scheme_frame: pd.DataFrame,
    n_splits: int,
) -> list[FoldSplit]:
    """Build outer GroupKFold splits and inner MLP GroupKFold validation splits."""
    groups = scheme_frame["Subject"].to_numpy(dtype=str)
    segment_ids = scheme_frame["segment_id"].to_numpy(dtype=str)

    outer = GroupKFold(n_splits=n_splits)
    fold_splits: list[FoldSplit] = []

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer.split(scheme_frame, scheme_frame["target"], groups)
    ):
        outer_train_idx = np.asarray(outer_train_idx, dtype=np.int64)
        outer_test_idx = np.asarray(outer_test_idx, dtype=np.int64)

        inner_groups = groups[outer_train_idx]
        inner_targets = scheme_frame["target"].to_numpy()[outer_train_idx]
        inner = GroupKFold(n_splits=n_splits)
        inner_train_local, inner_val_local = next(inner.split(outer_train_idx, inner_targets, inner_groups))

        mlp_train_idx = outer_train_idx[np.asarray(inner_train_local, dtype=np.int64)]
        mlp_val_idx = outer_train_idx[np.asarray(inner_val_local, dtype=np.int64)]

        split = FoldSplit(
            fold_index=fold_idx,
            outer_train_idx=outer_train_idx,
            outer_test_idx=outer_test_idx,
            mlp_train_idx=mlp_train_idx,
            mlp_val_idx=mlp_val_idx,
            train_subjects=sorted(np.unique(groups[mlp_train_idx]).tolist()),
            val_subjects=sorted(np.unique(groups[mlp_val_idx]).tolist()),
            test_subjects=sorted(np.unique(groups[outer_test_idx]).tolist()),
            train_segment_ids=segment_ids[mlp_train_idx].tolist(),
            val_segment_ids=segment_ids[mlp_val_idx].tolist(),
            test_segment_ids=segment_ids[outer_test_idx].tolist(),
        )
        fold_splits.append(split)

    return fold_splits


def save_split_manifests(
    splits: list[FoldSplit],
    output_dir: Path,
    class_order: list[str],
) -> None:
    """Persist split manifests and class mapping for one labelling scheme."""
    ensure_dir(output_dir)

    class_mapping = {
        "class_order": class_order,
        "class_to_index": {label: i for i, label in enumerate(class_order)},
    }
    save_json(output_dir / "label_mapping.json", class_mapping)

    for split in splits:
        payload = {
            "fold_index": split.fold_index,
            "train_subjects": split.train_subjects,
            "val_subjects": split.val_subjects,
            "test_subjects": split.test_subjects,
            "train_segment_ids": split.train_segment_ids,
            "val_segment_ids": split.val_segment_ids,
            "test_segment_ids": split.test_segment_ids,
        }
        save_json(output_dir / f"fold_{split.fold_index}.json", payload)
