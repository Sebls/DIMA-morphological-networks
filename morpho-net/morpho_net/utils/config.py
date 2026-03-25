"""Configuration loading and merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from morpho_net.utils.merge import deep_merge


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
        config = deep_merge(base, config)

    return config
