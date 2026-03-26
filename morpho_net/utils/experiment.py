"""Experiment path and directory helpers."""

from __future__ import annotations

from pathlib import Path


def next_experiment_dir(results_base: Path | str, exp_name: str) -> Path:
    """Return next available experiment directory with sequential index (e.g. exp_name_001)."""
    results_base = Path(results_base)
    results_base.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    for p in results_base.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name == exp_name:
            max_idx = max(max_idx, 1)
        elif name.startswith(f"{exp_name}_"):
            suffix = name[len(exp_name) + 1 :]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))

    next_idx = max_idx + 1
    return results_base / f"{exp_name}_{next_idx:03d}"
