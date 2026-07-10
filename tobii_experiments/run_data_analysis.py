#!/usr/bin/env python
"""Run paper-oriented descriptive analysis and representation visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis.example.yaml"),
        help="Analysis YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the configured analysis."""

    args = _parse_args()
    src_dir = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src_dir))
    from trustme_et_comparison.analysis import run_analysis

    output_dir = run_analysis(args.config)
    print(f"Finished. Analysis saved to: {output_dir}")


if __name__ == "__main__":
    main()
