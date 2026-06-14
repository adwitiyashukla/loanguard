"""I/O helpers — save/load joblib artifacts, ensure dirs exist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .logging import get_logger

log = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_joblib(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path, compress=("zlib", 3))
    log.info(f"Saved artifact -> {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def load_joblib(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    obj = joblib.load(path)
    log.info(f"Loaded artifact <- {path}")
    return obj


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=indent, default=str)
    return path


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r") as fh:
        return json.load(fh)
