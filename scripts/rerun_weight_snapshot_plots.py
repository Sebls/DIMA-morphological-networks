#!/usr/bin/env python3
"""Regenerate the layer×time weight histogram grid from a finished experiment directory.

Reads ``config.yaml`` for ``kernel_size``, ``training.weight_snapshot_*``, and ``output.checkpoint_dir``.
Writes ``plots/weight_histogram_evolution_grid.png`` and ``plot_data/weight_histogram_evolution_grid.json``.

Example::

    uv run python scripts/rerun_weight_snapshot_plots.py experiments/exp3_4pixel_single_001
    uv run python scripts/rerun_weight_snapshot_plots.py experiments/exp3_4pixel_single_001 --max-histograms 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from morpho_net.analysis.weight_evolution import (
    generate_weight_evolution_artifacts,
    parse_weight_snapshot_plot_settings,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate weight snapshot plots from an existing experiment folder."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Experiment directory (e.g. experiments/exp3_4pixel_single_001)",
    )
    parser.add_argument(
        "--max-histograms",
        type=int,
        default=None,
        metavar="N",
        help="Max time columns in the grid (evenly spaced). 0 = no cap (all snapshots). Default: from config.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    config_path = exp_dir / "config.yaml"
    if not config_path.is_file():
        print(f"Missing config: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = dict(config.get("training") or {})
    settings = parse_weight_snapshot_plot_settings(train_cfg)

    if args.max_histograms is not None:
        settings["max_histogram_snapshots"] = (
            None if args.max_histograms <= 0 else args.max_histograms
        )

    model_cfg = config.get("model") or {}
    kernel_shape = tuple(model_cfg.get("kernel_size", [3, 3]))

    out_cfg = config.get("output") or {}
    checkpoint_dir = exp_dir / out_cfg.get("checkpoint_dir", "Checkpoint")
    weight_snapshots = checkpoint_dir / "weight_snapshots"
    manifest = weight_snapshots / "manifest.json"
    if not manifest.is_file():
        print(
            f"No {manifest} — this run did not save weight snapshots "
            f"(set training.weight_snapshot_every_k_epochs and retrain).",
            file=sys.stderr,
        )
        sys.exit(1)

    plots_dir = exp_dir / "plots"
    plot_data_dir = exp_dir / "plot_data"

    generate_weight_evolution_artifacts(
        weight_snapshots,
        plots_dir,
        plot_data_dir,
        kernel_shape=kernel_shape,
        histogram_bins=settings["histogram_bins"],
        max_histogram_snapshots=settings["max_histogram_snapshots"],
    )
    print(f"Wrote plots under {plots_dir}")
    print(f"Wrote plot data under {plot_data_dir}")


if __name__ == "__main__":
    main()
