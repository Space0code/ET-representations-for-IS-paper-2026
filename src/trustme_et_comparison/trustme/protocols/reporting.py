"""Markdown report writer for Zoja-style protocol runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _tbl(df: pd.DataFrame, n: int = 20) -> str:
    """Render small dataframe to markdown safely."""

    if df.empty:
        return "(empty)"
    safe = df.head(n).copy().astype(object)
    safe = safe.where(pd.notna(safe), None)
    return safe.to_markdown(index=False)


def write_protocol_report(
    *,
    output_path: Path,
    fold_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    run_scope: dict[str, object] | None = None,
) -> None:
    """Write concise run report with key tables and caveats."""

    fold_work = fold_df.copy()
    if "model" not in fold_work.columns:
        fold_work["model"] = "unknown"
    else:
        fold_work["model"] = fold_work["model"].fillna("unknown")

    summary_work = summary_df.copy()
    if "model" not in summary_work.columns:
        summary_work["model"] = "unknown"
    else:
        summary_work["model"] = summary_work["model"].fillna("unknown")

    target_counts = (
        fold_work.groupby("target", as_index=False)["fold_id"].count().rename(columns={"fold_id": "n_rows"})
        if not fold_work.empty
        else pd.DataFrame(columns=["target", "n_rows"])
    )

    skip_stats = (
        fold_work.groupby(["target", "model", "protocol", "status", "skip_reason"], as_index=False)["fold_id"]
        .count()
        .rename(columns={"fold_id": "n_rows"})
        .sort_values(["target", "model", "protocol", "n_rows"], ascending=[True, True, True, False])
        if not fold_work.empty
        else pd.DataFrame(columns=["target", "model", "protocol", "status", "skip_reason", "n_rows"])
    )

    top_summary = (
        summary_work.sort_values(
            ["target", "model", "subject_macro_balanced_accuracy", "subject_macro_gain"],
            ascending=[True, True, False, False],
        )
        if not summary_work.empty
        else summary_work
    )

    lines: list[str] = []
    lines.append("# Zoja-Style ET Protocol Report")
    lines.append("")
    if run_scope:
        lines.append("## Run Scope")
        for key in sorted(run_scope.keys()):
            lines.append(f"- {key}: {run_scope[key]}")
        lines.append("")
    lines.append("## Scope")
    lines.append(f"- Fold rows: {len(fold_df)}")
    lines.append(f"- Subject rows: {len(subject_df)}")
    lines.append(f"- Summary rows: {len(summary_df)}")
    lines.append("")
    lines.append("### Rows Per Target")
    lines.append(_tbl(target_counts, n=50))
    lines.append("")
    lines.append("## Top Protocol x Representation (Subject-Macro Balanced Accuracy)")
    lines.append(
        _tbl(
            top_summary[
                [
                    "target",
                    "model",
                    "representation",
                    "protocol",
                    "subject_macro_balanced_accuracy",
                    "subject_macro_gain",
                    "subject_macro_macro_f1",
                    "subject_macro_auc",
                    "n_subjects",
                    "n_folds",
                    "n_valid_folds",
                    "skip_ratio",
                ]
            ],
            n=30,
        )
        if not top_summary.empty
        else "(empty)"
    )
    lines.append("")
    lines.append("## Ranking")
    lines.append(_tbl(ranking_df, n=40))
    lines.append("")
    lines.append("## Skip/Failure Diagnostics")
    lines.append(_tbl(skip_stats, n=80))
    lines.append("")
    lines.append("## Notes")
    lines.append("- Label behavior follows `label_rule` in config (see Run Scope snapshot).")
    lines.append("- One-class behavior is controlled by `evaluation.one_class_policy` and exposed in `status`/`skip_reason`.")
    lines.append("- Balanced confusion matrices are row-normalized with fixed color scale [0, 1].")

    output_path.write_text("\n".join(lines), encoding="utf-8")
