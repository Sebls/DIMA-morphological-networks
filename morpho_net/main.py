"""Main entry point for running experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from morpho_net.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run morphological network experiments")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to experiment config YAML (from config/ or absolute)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: experiments/<name> from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global NumPy/TensorFlow seed (model, training); dataset noise uses dataset.seed in config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        pkg_dir = Path(__file__).resolve().parent.parent
        if (pkg_dir / args.config).exists():
            config_path = pkg_dir / args.config
        else:
            config_path = pkg_dir / "config" / args.config

    results = run_experiment(
        config_path=config_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
