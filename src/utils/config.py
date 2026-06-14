"""Configuration loader with env-var overrides.

Convention for env overrides:  LG__SECTION__KEY=value
Example:  LG__DATA__SAMPLE_SIZE=10000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


CONFIG_ENV_PREFIX = "LG__"


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _coerce(value: str) -> Any:
    """Best-effort YAML coercion of an env-var string."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _env_overrides() -> dict:
    """Collect LG__SECTION__KEY env vars into a nested dict."""
    overrides: dict = {}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(CONFIG_ENV_PREFIX):
            continue
        parts = env_key[len(CONFIG_ENV_PREFIX):].lower().split("__")
        cursor = overrides
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce(env_val)
    return overrides


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Load YAML config, apply env overrides, return a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as fh:
        config = yaml.safe_load(fh) or {}

    overrides = _env_overrides()
    if overrides:
        _deep_update(config, overrides)

    return config


def get(config: dict, dotted_key: str, default: Any = None) -> Any:
    """Read `a.b.c` style nested keys from a config dict."""
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor
