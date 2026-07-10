"""General utility helpers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def make_run_id(run_id: str | None) -> str:
    """Return provided run id or generate a timestamp-based id."""
    if run_id:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for numpy, python, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_serializable(obj: Any) -> Any:
    """Convert dataclasses and numpy objects to JSON-safe values."""
    if is_dataclass(obj):
        return to_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Save dictionary to JSON file."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def save_yaml(path: Path, payload: Any) -> None:
    """Save object as YAML file."""
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(to_serializable(payload), sort_keys=False), encoding="utf-8")
