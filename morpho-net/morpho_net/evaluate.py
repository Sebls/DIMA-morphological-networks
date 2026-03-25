"""Evaluate a trained model from experiment directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tensorflow.keras.losses import MeanSquaredError

from morpho_net.data import load_fashion_mnist_noisy, create_ground_truth_model, generate_ground_truth
from morpho_net.models import (
    build_single_sup_erosions,
    build_two_layer_sup_erosions,
    build_two_layer_receptive_field,
)


def load_model_from_experiment(exp_dir: Path):
    """Load model architecture from config and weights from checkpoint."""
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    model_cfg = config.get("model", {})
    init_cfg = config.get("initialization", {})
    arch = model_cfg.get("architecture", "single_sup_erosions")

    if arch == "single_sup_erosions":
        model = build_single_sup_erosions(
            n_erosions=model_cfg.get("n_erosions", 200),
            kernel_size=tuple(model_cfg.get("kernel_size", [3, 3])),
            minval=init_cfg.get("minval", -0.45),
            maxval=init_cfg.get("maxval", -0.15),
        )
    elif arch == "two_layer_sup_erosions":
        model = build_two_layer_sup_erosions(
            n_erosions_block1=model_cfg.get("n_erosions_block1", 50),
            n_erosions_block2=model_cfg.get("n_erosions_block2", 50),
            n_erosions_block3=model_cfg.get("n_erosions_block3", 100),
            kernel_size=tuple(model_cfg.get("kernel_size", [3, 3])),
            init_block1=(init_cfg.get("block1", {}).get("minval", -0.45), init_cfg.get("block1", {}).get("maxval", -0.15)),
            init_block2=(init_cfg.get("block2", {}).get("minval", -0.45), init_cfg.get("block2", {}).get("maxval", -0.15)),
            init_block3=(init_cfg.get("block3", {}).get("minval", -0.45), init_cfg.get("block3", {}).get("maxval", -0.15)),
        )
    elif arch == "two_layer_receptive_field":
        model = build_two_layer_receptive_field(
            n_erosions_block1=model_cfg.get("n_erosions_block1", 500),
            n_erosions_block2=model_cfg.get("n_erosions_block2", 500),
            n_erosions_block3=model_cfg.get("n_erosions_block3", 700),
            kernel_size=tuple(model_cfg.get("kernel_size", [3, 3])),
            block1_inactive_indices=model_cfg.get("block1_inactive_indices", [4, 5, 6, 7, 8]),
            block2_inactive_indices=model_cfg.get("block2_inactive_indices", [0, 1, 2, 3]),
            inactive_value=model_cfg.get("inactive_value", -10.0),
            init_block1=(init_cfg.get("block1", {}).get("minval", -0.35), init_cfg.get("block1", {}).get("maxval", 0.35)),
            init_block2=(init_cfg.get("block2", {}).get("minval", -0.35), init_cfg.get("block2", {}).get("maxval", 0.35)),
            init_block3=(init_cfg.get("block3", {}).get("minval", -0.35), init_cfg.get("block3", {}).get("maxval", 0.35)),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    ckpt = exp_dir / "Checkpoint" / "checkpoint.weights.h5"
    if ckpt.exists():
        model.load_weights(str(ckpt))
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained morphological network")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory (contains config.yaml, Checkpoint/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for dataset noise only (local RNG; pass dataset.seed from config for parity with training)",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_absolute():
        exp_dir = Path.cwd() / exp_dir
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    model, config = load_model_from_experiment(exp_dir)
    ds_cfg = config.get("dataset", {})
    train_split, val_split, test_split = load_fashion_mnist_noisy(
        size=ds_cfg.get("size", 100),
        use_noise=ds_cfg.get("use_noise", True),
        sigma=ds_cfg.get("sigma", 40),
        seed=args.seed,
    )

    gt_model = create_ground_truth_model(config.get("ground_truth", {}).get("kernel", [[0, 0.5, 0], [0, 0.5, 0], [0, 0, 0]]))
    _, y_val, y_test = generate_ground_truth(gt_model, train_split.images, val_split.images, test_split.images)

    mse_fn = MeanSquaredError()
    y_val_pred = model.predict(val_split.images, verbose=0)
    y_test_pred = model.predict(test_split.images, verbose=0)
    if y_val_pred.shape[-1] == 1:
        y_val_pred = y_val_pred[:, :, :, 0]
    if y_test_pred.shape[-1] == 1:
        y_test_pred = y_test_pred[:, :, :, 0]

    val_mse = float(mse_fn(y_val, y_val_pred).numpy())
    test_mse = float(mse_fn(y_test, y_test_pred).numpy())

    metrics = {"val_mse": val_mse, "test_mse": test_mse}
    print("\n--- Evaluation ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_path = exp_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
