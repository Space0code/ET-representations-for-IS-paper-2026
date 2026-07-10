#!/usr/bin/env python
"""Run Zoja-style ET protocol evaluation for TrustMe Tobii data.

Example:
  conda activate trust-me-et
  python tobii_experiments/run_zoja_protocols.py

Options:
  --config: Path to experiment YAML configuration.
  --skip-plot: Disable plot generation for this run.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import sys
import tempfile
from pathlib import Path
import traceback
from typing import TextIO


class _TeeWriter:
    """Write text to a console stream and a run log file."""

    def __init__(self, stream: TextIO, log_handle: TextIO) -> None:
        self.stream = stream
        self.log_handle = log_handle

    def write(self, text: str) -> int:
        """Write text to both destinations."""

        self.stream.write(text)
        self.log_handle.write(text)
        return len(text)

    def flush(self) -> None:
        """Flush both destinations."""

        self.stream.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        """Return whether the console stream is attached to a terminal."""

        return self.stream.isatty()


def _bootstrap_imports() -> None:
    """Insert repository ``src`` path into ``sys.path``."""

    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _repo_root() -> Path:
    """Return repository root assuming this file lives in tobii_experiments/."""

    return Path(__file__).resolve().parent.parent


def _run_log_path(root: Path) -> Path:
    """Return a timestamped master log path for this script."""

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"run_zoja_protocols_{timestamp}.log"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run TrustMe Tobii ET protocols.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments.yaml"),
        help="Path to experiment protocol YAML config.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plot generation regardless of config.plotting.enabled.",
    )
    return parser.parse_args()


def main() -> None:
    """Run protocol evaluation and print artifact directory."""

    args = _parse_args()
    _bootstrap_imports()

    from trustme_et_comparison.trustme.protocols.config import load_zoja_protocol_config
    from trustme_et_comparison.trustme.protocols.runner import run_zoja_protocols

    cfg = load_zoja_protocol_config(args.config)
    print(
        "Final arguments: "
        f"config={args.config}, skip_plot={args.skip_plot}, seed={cfg.seed}, "
        f"model={cfg.model}, classifiers={cfg.classifiers}, representations={cfg.representations}, "
        f"targets={[target.name for target in cfg.targets]}",
        flush=True,
    )

    if args.skip_plot:
        cfg.plotting.enabled = False
        from trustme_et_comparison.common.utils import save_yaml
        # Persist a temporary override config for runner compatibility.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="zoja_protocol_skip_plot_",
            delete=False,
        ) as handle:
            override_config = Path(handle.name)
        save_yaml(override_config, cfg)
        try:
            run_dir = run_zoja_protocols(config_path=override_config)
        finally:
            override_config.unlink(missing_ok=True)
    else:
        run_dir = run_zoja_protocols(config_path=args.config)

    print(f"Finished. Results saved to: {run_dir}", flush=True)


def _main_with_logging() -> int:
    """Run main while teeing console output to a timestamped master log."""

    root = _repo_root()
    log_path = _run_log_path(root)
    with log_path.open("w", encoding="utf-8") as log_handle:
        stdout_tee = _TeeWriter(sys.stdout, log_handle)
        stderr_tee = _TeeWriter(sys.stderr, log_handle)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"Master log: {log_path}", flush=True)
            try:
                main()
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_with_logging())
