"""Deep-merge helpers for nested dicts (config, initialization blocks, etc.)."""

from __future__ import annotations

import copy
from typing import Any

INIT_BLOCK_KEYS = frozenset({"block1", "block2", "block3"})


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_init_block(init_cfg: dict[str, Any], block: str | None) -> dict[str, Any]:
    """Return init options for a logical block.

    * ``block`` ``None``: single-layer models — use top-level keys only (no ``block*`` nests).
    * ``block`` ``\"block1\"`` etc.: merge shared keys with ``init_cfg[block]``.
    """
    base = {k: v for k, v in init_cfg.items() if k not in INIT_BLOCK_KEYS}
    if block and block in init_cfg and isinstance(init_cfg[block], dict):
        return deep_merge(base, init_cfg[block])
    return base
