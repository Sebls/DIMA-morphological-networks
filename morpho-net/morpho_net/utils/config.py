"""Configuration loading and merging."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load config from YAML file, resolving 'extends' for inheritance."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Resolve extends
    if "extends" in config:
        base_path = path.parent / config.pop("extends")
        base = load_config(base_path)
        config = _deep_merge(base, config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
